import torch
import re
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_llama_layer import QuantLlamaDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.utils import smooth_and_quant_temporary, use_parameters, clear_temp_variable, smooth_ln

def preprocess_model(model, model_name, w_train_bits, a_train_bits, zo_eps):

    model_nick_name = model_name.split("/")[-1]
    is_llama = 'llama' in model_name.lower()

    act_scales = torch.load(f'./act_scales/{model_nick_name}.pt')
    act_shifts = torch.load(f'./act_shifts/{model_nick_name}.pt')

    quant_args = {"weight_quant_params": {'n_bits': w_train_bits, 'per_channel_axes': [0], 'symmetric': False,
                                          'dynamic_method': 'per_channel', 'group_size': False, 'lwc': False,
                                          'disable_zero_point': False},
                  "act_quant_params": {'n_bits': a_train_bits, 'per_channel_axes': [], 'symmetric': False,
                                       'dynamic_method': 'per_token'},
                  "p_quant_params": {'n_bits': 16, 'metric': 'fix0to1'}}


    if not is_llama:
        layer_name_prefix = "model.decoder.layers"
        layers = model.model.decoder.layers
        Qlayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "out_proj": "out",
            "fc1": "fc1"
        }
    else:
        layer_name_prefix = "model.layers"
        layers = model.model.layers
        Qlayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "o_proj": "out",
            "up_proj": "fc1"
        }

    named_parameters_to_optim = []


    alpha = 0.5 if not is_llama else 0.75
    qlinears = []
    for i in range(len(layers)):
        layer = layers[i]
        qlayer = Qlayer(config=model.config, ori_layer=layer, quant_args=quant_args, idx=i, zo_eps=zo_eps)
        qlayer.register_parameter("qkt_smooth_scale", torch.nn.Parameter(
            torch.ones(layer.self_attn.q_proj.out_features, device=layer.self_attn.q_proj.weight.device,
                       dtype=torch.float16)))
        for name, module in qlayer.named_modules():
            if isinstance(module, QuantLinear):
                qlinears.append(module)
                for key in pairs.keys():
                    if key in name:
                        weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                        act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=weight.device,
                                                                                         dtype=torch.float16).clamp(
                            min=1e-5)
                        scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)
                        if not is_llama:
                            shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=weight.device,
                                                                                               dtype=torch.float16)
                        else:
                            shift = torch.zeros_like(scale, device=weight.device, dtype=torch.float16)

                        if key == 'o_proj' or key == 'out_proj':
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift", torch.nn.Parameter(
                                torch.zeros(scale.shape, device=layer.self_attn.q_proj.weight.device,
                                            dtype=torch.float16)))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale", torch.nn.Parameter(
                                torch.ones(scale.shape, device=layer.self_attn.q_proj.weight.device,
                                           dtype=torch.float16)))
                        else:
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift", torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale", torch.nn.Parameter(scale))

        qlayer.init_smoothing()
        layers[i] = qlayer

    for name, param in model.named_parameters():
        param.requires_grad = False
        # if 'bias' not in name:
        if 'bias' not in name and 'norm' not in name:
            if 'scale' not in name and 'shift' not in name and 'bound' not in name:
                if 'mlp.up_proj' in name:
                    named_parameters_to_optim.insert(len(named_parameters_to_optim) - 1, (name, param))
                else:
                    named_parameters_to_optim.append((name, param))


    named_parameters_to_optim = named_parameters_to_optim[1:-1] if is_llama else named_parameters_to_optim[2:]

    if a_train_bits != 16:
        smooth_ln_pre_training(model, is_llama)
    else:
        for layer in qlinears:
            layer.use_act_quant = False

    return model, named_parameters_to_optim, qlinears


def smooth_ln_pre_training(model, is_llama):

    if is_llama:
        layers = model.model.layers
    else:
        layers = model.model.decoder.layers

    for layer in layers:
        smooth_ln(layer, isllama=is_llama)