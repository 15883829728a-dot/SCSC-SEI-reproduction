"""Model-size / computational-complexity accounting.

MACCs follow the paper's convention (Eqs. 19-23): a complex-valued conv/linear
costs 4x the multiply-accumulates of its real-valued counterpart.  Parameter
counts are the raw number of learnable parameters.
"""
import torch
import torch.nn as nn

from .complex_layers import ComplexConv1d, ComplexLinear


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def count_macc(model, input_length, in_ch=2, device="cpu"):
    was_training = model.training
    model.eval()
    total = [0]

    # inner real convs/linears of complex modules must not be double counted
    skip = set()
    for m in model.modules():
        if isinstance(m, ComplexConv1d):
            skip.add(id(m.conv_r)); skip.add(id(m.conv_i))
        elif isinstance(m, ComplexLinear):
            skip.add(id(m.lin_r)); skip.add(id(m.lin_i))

    handles = []

    def conv_hook(mult, c_in, c_out, k):
        def hook(_m, _i, out):
            total[0] += mult * k * out.shape[-1] * c_in * c_out
        return hook

    def lin_hook(mult, f_in, f_out):
        def hook(_m, _i, _o):
            total[0] += mult * f_in * f_out
        return hook

    for m in model.modules():
        if isinstance(m, ComplexConv1d):
            handles.append(m.register_forward_hook(
                conv_hook(4, m.in_ch, m.out_ch, m.conv_r.kernel_size[0])))
        elif isinstance(m, nn.Conv1d) and id(m) not in skip:
            handles.append(m.register_forward_hook(
                conv_hook(1, m.in_channels, m.out_channels, m.kernel_size[0])))
        elif isinstance(m, ComplexLinear):
            handles.append(m.register_forward_hook(lin_hook(4, m.in_features, m.out_features)))
        elif isinstance(m, nn.Linear) and id(m) not in skip:
            handles.append(m.register_forward_hook(lin_hook(1, m.in_features, m.out_features)))

    model(torch.zeros(1, in_ch, input_length, device=device))
    for h in handles:
        h.remove()
    if was_training:
        model.train()
    return int(total[0])


def complexity_report(model, input_length, in_ch=2, device="cpu"):
    return {"params": count_params(model), "macc": count_macc(model, input_length, in_ch, device)}


def compression_ratio(original, slim):
    return {
        "R_param_%": 100.0 * (original["params"] - slim["params"]) / original["params"],
        "R_macc_%": 100.0 * (original["macc"] - slim["macc"]) / original["macc"],
    }
