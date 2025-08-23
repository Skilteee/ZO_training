import torch
import torch.nn as nn


'''
Modify normalization layer to adapt the training of learnable equivalent transformation
'''



class OmniLayerNorm(nn.Module):
    def __init__(self, ori_layer_norm) -> None:
        super().__init__()
        self.use_act_quant = True
        # self.register_buffer('weight',ori_layer_norm.weight)
        self.register_parameter('weight', ori_layer_norm.weight)
        if ori_layer_norm.bias is not None:
            # self.register_buffer('bias',ori_layer_norm.bias)
            self.register_parameter('bias', ori_layer_norm.bias)
        else:
            self.bias = None
        self.eps = ori_layer_norm.eps
        self.norm_func = nn.functional.layer_norm
        self.normalized_shape = ori_layer_norm.normalized_shape
        self.use_temporary_parameter = False
        # self.use_weight_quant = False


    def forward(self, x):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias

        # assert (weight.to(self.weight.device) != self.weight).sum() == 0, 'break'

        # torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, eps=self.eps)
        out = self.norm_func(x,self.normalized_shape,weight, bias,eps=self.eps)

        # if torch.isnan(out).sum() != 0 or torch.isinf(out).sum() != 0:

        return out

    def set_quant_state(self, use_weight_quant, use_act_quant):
        self.use_act_quant = use_act_quant


class OmniLlamaRMSNorm(nn.Module):
    def __init__(self, ori_norm, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.register_buffer('weight',ori_norm.weight)
        self.bias = None
        self.variance_epsilon = eps
        self.use_temporary_parameter = False
        self.use_weight_quant = False


    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # if self.use_temporary_parameter:
        #     weight = self.temp_weight
        #     bias = self.temp_bias
        # else:
        weight = self.weight
        bias = self.bias if hasattr(self, 'bias') else None

        out = (weight * hidden_states+bias).to(input_dtype) if bias is not None else (weight * hidden_states).to(input_dtype)

        # if torch.isnan(out).sum() != 0 or torch.isinf(out).sum() != 0:
        #     import code
        #     code.interact(local=locals())

        return out


    def set_quant_state(self, use_weight_quant, use_act_quant):
        self.use_weight_quant = use_weight_quant
        self.use_act_quant = use_act_quant