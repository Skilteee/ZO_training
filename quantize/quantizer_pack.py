import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math

CLIPMIN = 1e-4




def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

# def quantize_activation_per_token_absmax(t, n_bits=4):
#     scales = t.abs().max(dim=-1, keepdim=True)[0]
#     q_max = 2 ** (n_bits - 1) - 1
#     scales.clamp_(min=1e-5).div_(q_max)
#     t.div_(scales).round_().mul_(scales)
#     return t

# def quantize_activation_per_token_absmax(t, n_bits=4):
#     scales = t.abs().max(dim=-1, keepdim=True)[0]  #
#     q_max = 2 ** (n_bits - 1) - 1  # 量化范围
#
#     # 计算量化 scale
#     scales = scales.clamp(min=1e-5) / q_max  #
#
#     # 量化激活并恢复
#     t_q = (t / scales).round() * scales  #
#
#     return t_q

def pack_intn_to_uint8(codes_uint8: torch.Tensor, nbit: int) -> torch.Tensor:

    if nbit < 1 or nbit > 8 or (8 % nbit) != 0:
        raise ValueError(f"nbit must divide 8 and be in [1,8], got {nbit}")

    flat = codes_uint8.reshape(-1).to(torch.uint8)
    group = 8 // nbit
    mask  = (1 << nbit) - 1

    pad = (-flat.numel()) % group
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8, device=flat.device)])

    x = flat.view(-1, group).to(torch.int32)
    packed32 = torch.zeros(x.size(0), dtype=torch.int32, device=x.device)

    for i in range(group):
        packed32 |= ((x[:, i] & mask) << (i * nbit))

    return packed32.to(torch.uint8).contiguous()


def unpack_uint8_to_intn_codes(packed: torch.Tensor, nbit: int, total_elems: int) -> torch.Tensor:

    if nbit < 1 or nbit > 8 or (8 % nbit) != 0:
        raise ValueError(f"nbit must divide 8 and be in [1,8], got {nbit}")

    group = 8 // nbit
    mask  = (1 << nbit) - 1

    p32 = packed.to(torch.int32)
    parts = []
    for i in range(group):
        parts.append(((p32 >> (i * nbit)) & mask).to(torch.uint8))

    # [num_bytes, group] -> [num_bytes*group]
    flat = torch.stack(parts, dim=1).reshape(-1)
    return flat[:total_elems]



class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_channel",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        
        init_value = 3.             # inti value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.register_parameter('upbound_factor', nn.Parameter(
                torch.ones((dim1, 1), device='cuda', dtype=torch.float16) * init_value))
            self.register_parameter('lowbound_factor', nn.Parameter(
                torch.ones((dim1, 1), device='cuda', dtype=torch.float16) * init_value))
            # self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            # self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = torch.round(x / scale.clamp(min=CLIPMIN))
        if round_zero_point is not None:
            x_int = x_int + round_zero_point
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant - round_zero_point
        x_dequant = x_dequant * scale
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant

    def quantize_weight_only(self, x, group_size=None, deficiency=0):

        self.per_token_dynamic_calibration(x)
        self.orig_shape = x.shape
        assert x.dim() == 2, "only support linear layer now"  # 与你的断言一致
        dim1, dim2 = x.shape

        if deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0], deficiency), dtype=x.dtype, device=x.device)
            x = torch.cat((x, pad_zeros), dim=1)

        if group_size:
            x = x.reshape(-1, group_size)

        scale_c = self.scale.clamp(min=CLIPMIN)
        x_int = torch.round(x / scale_c)

        if self.round_zero_point is not None:
            x_int = x_int + self.round_zero_point

        x_int = x_int.clamp(self.qmin, self.qmax)

        # 还原形状
        if group_size:
            x_int = x_int.reshape(dim1, -1)

        x_int = x_int.to(torch.uint8)

        x_int = pack_intn_to_uint8(x_int, self.n_bits)  # uint8, length = ceil(N/(8/n_bits)

        return x_int

    def dequant_weight_only(self, x, group_size=None, deficiency=0, dtype=torch.float16):

        assert self.scale is not None, "scale is None"

        total_elems = self.orig_shape[0] * self.orig_shape[1]

        # from quantize.quantizer import unpack_uint8_to_int2_codes
        x = unpack_uint8_to_intn_codes(x, self.n_bits, total_elems).reshape(self.orig_shape)

        xq = x.to(torch.int16).to(torch.float16)

        # 2) 如果有分组量化，先按分组形状展开，便于按组广播 scale/zp
        if group_size:
            xq = xq.reshape(-1, group_size)  # 与量化时的 reshape 对应

        s = self.scale
        if s.dim() == 1 and group_size is not None and s.numel() == xq.numel() // group_size:
            s = s.view(-1, 1)  # 每组一个 scale
        # 若 s 已与 xq 可广播，就不用改

        z = self.round_zero_point
        if isinstance(z,
                      torch.Tensor) and z.dim() == 1 and group_size is not None and z.numel() == xq.numel() // group_size:
            z = z.view(-1, 1)  # 每组一个 zp
        xq = xq - z

        # 4) 线性反量化
        x_fp = xq * s

        # 5) 还原矩阵形状，并去掉当时为了对齐 group_size 补的列
        if group_size:
            x_fp = x_fp.reshape(self.orig_shape[0], -1)
        if deficiency > 0:
            x_fp = x_fp[:, :self.orig_shape[1]]

        # 6) 转回你想要的浮点精度（例如 fp16/fp32）
        return x_fp.to(dtype)


    def forward(self, x: torch.Tensor, quant_only=False, activation=False):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()

        # x_dequant = quantize_activation_per_token_absmax(x, self.n_bits)

        if quant_only:
            if activation:
                return self.quantize_activation_only(x, self.deficiency)
            return self.quantize_weight_only(x, self.group_size, self.deficiency)
        else:
            return self.fake_quant(x, self.scale, self.round_zero_point)

    def per_token_dynamic_calibration(self, x):

        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax = x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            if self.upbound_factor.device != xmax.device:
                self.upbound_factor.data = self.upbound_factor.data.to(xmax.device)
                self.lowbound_factor.data = self.lowbound_factor.data.to(xmax.device)
            xmax = self.sigmoid(self.upbound_factor) * xmax
            xmin = self.sigmoid(self.lowbound_factor) * xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)

        # 把self.scale变成一个parameter
        self.scale = torch.nn.parameter.Parameter(self.scale, requires_grad=False)
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
            self.round_zero_point = torch.nn.parameter.Parameter(self.round_zero_point, requires_grad=False)
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point
