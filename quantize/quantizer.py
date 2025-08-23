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
            # self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            # self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            self.register_parameter('upbound_factor', nn.Parameter(torch.ones((dim1, 1), device='cuda', dtype=torch.float16) * init_value))
            self.register_parameter('lowbound_factor', nn.Parameter(torch.ones((dim1, 1), device='cuda', dtype=torch.float16) * init_value))
            # self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value).cuda()
            # self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value).cuda()
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

    def in_place_fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        # x_int = round_ste(x / scale)
        x.copy_((x/scale).round())
        # x_int = torch.round(x / scale.clamp(min=CLIPMIN))
        if round_zero_point is not None:
            # x_int = x_int.add(round_zero_point)
            # x_int = x_int + round_zero_point
            x.add_(round_zero_point)
        x.clamp_(self.qmin, self.qmax)
        # x_int = x_int.clamp(self.qmin, self.qmax)
        # x_dequant = x_int
        if round_zero_point is not None:
            # x_dequant = x_dequant.sub(round_zero_point)
            # x_dequant = x_dequant - round_zero_point
            x.sub_(round_zero_point)
        # x_dequant = x_dequant.mul(scale)
        # x_dequant = x_dequant * scale
        x.mul_(scale)
        if self.group_size:
            x = x.reshape(dim1, dim2)
        if self.deficiency > 0:
            x = x[:,:-self.deficiency]
        return x

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0], self.deficiency), dtype=x.dtype, device=x.device)
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, :-self.deficiency]
        return x_dequant
    

    def forward(self, x: torch.Tensor, inplace=False):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()

        # if torch.isnan(x).sum() != 0 or torch.isinf(x).sum() != 0:
        #     import code
        #     code.interact(local=locals())
        # x_dequant = quantize_activation_per_token_absmax(x, self.n_bits)
        # if inplace:
        #     x_dequant = self.in_place_fake_quant(x, self.scale, self.round_zero_point)
        # else:
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)

        return x_dequant

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
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin
            # xmax.mul_(self.sigmoid(self.upbound_factor))
            # xmin.mul_(self.sigmoid(self.lowbound_factor))
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
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point
