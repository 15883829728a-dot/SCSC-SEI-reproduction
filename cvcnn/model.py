"""Basic CVCNN (complex-valued CNN) and a real-valued RVCNN baseline for SEI.

Architecture follows the paper: small kernels (size 3), max pooling after every
convolution layer, increased depth (9 conv layers) and 2 fully-connected layers.
The exact per-layer channel counts in the paper's Table I were not recoverable
from the PDF, so the width is parameterized by `base_width` (the paper's "N",
e.g. 64 or 128) with a constant-width 9-layer stack; this is documented in the
README. The compression code prunes these channels per layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_layers import (ComplexConv1d, ComplexBatchNorm1d, CReLU,
                             ComplexMaxPool1d, ComplexLinear, TernaryMask, split)


class ComplexConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel=3):
        super().__init__()
        self.conv = ComplexConv1d(c_in, c_out, kernel, padding=kernel // 2)
        self.bn = ComplexBatchNorm1d(c_out)
        self.act = CReLU()
        self.pool = ComplexMaxPool1d(2)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


def _vec_magnitude(z, eps=1e-12):
    f = z.shape[1] // 2
    zr, zi = z[:, :f], z[:, f:]
    return torch.sqrt(zr * zr + zi * zi + eps)


class CVCNN(nn.Module):
    def __init__(self, num_classes, channels=None, base_width=64, n_conv=9, kernel=3,
                 fc_dim=128, in_complex_ch=1, input_length=1024, with_masks=False, mu=0.01,
                 mask_init_noise=0.05):
        super().__init__()
        channels = list(channels) if channels is not None else [base_width] * n_conv
        self.channels = channels
        self.in_complex_ch = in_complex_ch
        self.input_length = input_length
        self.num_classes = num_classes
        self.fc_dim = fc_dim
        self.kernel = kernel

        blocks, c_in = [], in_complex_ch
        for c_out in channels:
            blocks.append(ComplexConvBlock(c_in, c_out, kernel))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.masks = nn.ModuleList(
            [TernaryMask(c, mu, mask_init_noise) for c in channels]) if with_masks else None

        with torch.no_grad():
            f = torch.zeros(1, 2 * in_complex_ch, input_length)
            for blk in self.blocks:
                f = blk(f)
            self.feat_len = f.shape[-1]
        self.fc1 = ComplexLinear(channels[-1] * self.feat_len, fc_dim)
        self.fc1_act = CReLU()
        self.fc2 = nn.Linear(fc_dim, num_classes)   # real classifier on the magnitude

    def forward_features(self, x):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.masks is not None:
                x = self.masks[i](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        xr, xi = split(x)
        b = x.shape[0]
        z = torch.cat([xr.reshape(b, -1), xi.reshape(b, -1)], dim=1)
        z = self.fc1_act(self.fc1(z))
        return self.fc2(_vec_magnitude(z))

    def mask_l1(self):
        if self.masks is None:
            return torch.zeros((), device=self.fc2.weight.device)
        return sum(m.l1() for m in self.masks)


class _SafeMaxPool1d(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def forward(self, x):
        return x if x.shape[-1] < self.k else F.max_pool1d(x, self.k)


class RVCNN(nn.Module):
    """Real-valued CNN baseline with the same depth/width recipe (input = 2 real
    channels: in-phase and quadrature)."""

    def __init__(self, num_classes, base_width=128, n_conv=9, kernel=3, fc_dim=128,
                 in_ch=2, input_length=1024):
        super().__init__()
        blocks, c_in = [], in_ch
        for _ in range(n_conv):
            blocks.append(nn.Sequential(
                nn.Conv1d(c_in, base_width, kernel, padding=kernel // 2),
                nn.BatchNorm1d(base_width),
                nn.ReLU(inplace=True),
                _SafeMaxPool1d(2),
            ))
            c_in = base_width
        self.blocks = nn.Sequential(*blocks)
        with torch.no_grad():
            f = self.blocks(torch.zeros(1, in_ch, input_length))
            self.feat_len = f.shape[-1]
        self.fc1 = nn.Linear(base_width * self.feat_len, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        x = self.blocks(x).flatten(1)
        return self.fc2(F.relu(self.fc1(x)))
