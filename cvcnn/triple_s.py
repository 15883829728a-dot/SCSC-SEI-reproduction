"""Triple-S (Sparse Structure Selection) pruning for CVCNN.

Weights are updated by SGD/Adam; the ternary mask layers are updated by the
proximal straight-through gradient descent (PSTGD) of the paper:

    m <- prox( m - lr * STE(dL/dm) ; lr * lambda_SR )   # L1 soft-threshold
    m <- clip(m, -1, 1)                                   # constraint |m|<=1
    m <- m / max_j |m_j|   (per layer)                    # scaling trick

After training, channels whose ternary mask is 0 are physically removed and a
smaller `SlimCVCNN` is built whose forward output is mathematically identical
to the masked model (surviving +-1 mask signs are folded into the consuming
layer's weights).
"""
import torch
import torch.nn as nn

from .model import CVCNN


def soft_threshold(x, thr):
    return torch.sign(x) * torch.clamp(x.abs() - thr, min=0.0)


def train_triple_s(model, loader, epochs, lr=1e-3, lambda_sr=1e-3,
                   device="cpu", eval_fn=None, log=print):
    assert model.masks is not None, "model must be built with with_masks=True"
    model.to(device).train()
    mask_ids = {id(mk.m) for mk in model.masks}
    weight_params = [p for p in model.parameters() if id(p) not in mask_ids]
    opt = torch.optim.Adam(weight_params, lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            for mk in model.masks:
                mk.m.grad = None
            loss = ce(model(xb), yb)            # data loss only (L1 done by prox)
            loss.backward()
            opt.step()                           # weights
            with torch.no_grad():                # PSTGD on masks
                for mk in model.masks:
                    if mk.m.grad is None:
                        continue
                    m_new = mk.m - lr * mk.m.grad
                    m_new = soft_threshold(m_new, lr * lambda_sr)
                    m_new = m_new.clamp(-1.0, 1.0)
                    denom = m_new.abs().max()
                    if float(denom) > 0:
                        m_new = m_new / denom
                    mk.m.copy_(m_new)
            running += loss.item()
        survivors = [int(mk.survivor_values()[0].numel()) for mk in model.masks]
        msg = f"[Triple-S] epoch {ep:3d}/{epochs} loss={running/len(loader):.4f} survivors={survivors}"
        if eval_fn is not None and (ep % max(1, epochs // 10) == 0 or ep == epochs):
            msg += f" acc={eval_fn(model):.4f}"
        log(msg)
    return model


def _copy_indexed(dst_tensor, src_tensor, idx):
    dst_tensor.data.copy_(src_tensor.data[idx])


@torch.no_grad()
def build_slim(model, device="cpu"):
    """Construct a SlimCVCNN with zero-mask channels removed."""
    assert model.masks is not None
    surv, vals = [], []
    for mk in model.masks:
        idx, v = mk.survivor_values()
        if idx.numel() == 0:                      # never let a layer vanish
            idx = mk.m.abs().argmax().view(1)
            v = torch.ones(1, device=mk.m.device)
        surv.append(idx.long())
        vals.append(v.to(model.fc2.weight.dtype))

    new_channels = [int(s.numel()) for s in surv]
    slim = CVCNN(num_classes=model.num_classes, channels=new_channels,
                 kernel=model.kernel, fc_dim=model.fc_dim,
                 in_complex_ch=model.in_complex_ch, input_length=model.input_length,
                 with_masks=False).to(device)

    for l, (blk_o, blk_s) in enumerate(zip(model.blocks, slim.blocks)):
        out_idx = surv[l]
        in_idx = surv[l - 1] if l > 0 else torch.arange(model.in_complex_ch)
        v_in = vals[l - 1] if l > 0 else None
        for cattr in ("conv_r", "conv_i"):
            co, cs = getattr(blk_o.conv, cattr), getattr(blk_s.conv, cattr)
            w = co.weight.data[out_idx][:, in_idx, :].clone()
            if v_in is not None:
                w = w * v_in.view(1, -1, 1)       # fold previous layer's mask signs
            cs.weight.data.copy_(w)
            cs.bias.data.copy_(co.bias.data[out_idx])
        for p in ("gamma_rr", "gamma_ii", "gamma_ri", "beta_r", "beta_i",
                  "rm_r", "rm_i", "rv_rr", "rv_ii", "rv_ri"):
            _copy_indexed(getattr(blk_s.bn, p), getattr(blk_o.bn, p), out_idx)

    # FC1: keep surviving last-layer channels (all feat_len positions) and fold v
    fl = model.feat_len
    last_idx, v_last = surv[-1], vals[-1]
    cols = torch.cat([torch.arange(int(c) * fl, int(c) * fl + fl) for c in last_idx])
    scale = torch.repeat_interleave(v_last, fl).view(1, -1)
    for lattr in ("lin_r", "lin_i"):
        lo, ls = getattr(model.fc1, lattr), getattr(slim.fc1, lattr)
        ls.weight.data.copy_(lo.weight.data[:, cols] * scale)
        ls.bias.data.copy_(lo.bias.data)
    slim.fc2.weight.data.copy_(model.fc2.weight.data)
    slim.fc2.bias.data.copy_(model.fc2.bias.data)
    return slim
