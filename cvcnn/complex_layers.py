"""Complex-valued building blocks (Deep Complex Networks style).

A complex feature map with C complex channels and length L is stored as a REAL
tensor of shape (B, 2C, L): the first C channels hold the real part, the next C
hold the imaginary part.  Every layer below respects this convention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def split(x):
    """(B, 2C, L) -> (xr, xi), each (B, C, L)."""
    c = x.shape[1] // 2
    return x[:, :c], x[:, c:]


def merge(xr, xi):
    return torch.cat([xr, xi], dim=1)


def complex_magnitude(x, eps=1e-12):
    xr, xi = split(x)
    return torch.sqrt(xr * xr + xi * xi + eps)


class ComplexConv1d(nn.Module):
    """(Wr + i Wi) * (xr + i xi) = (Wr*xr - Wi*xi) + i (Wr*xi + Wi*xr)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.conv_r = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.conv_i = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        xr, xi = split(x)
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)
        return merge(yr, yi)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.lin_r = nn.Linear(in_features, out_features, bias=bias)
        self.lin_i = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        xr, xi = x[:, :self.in_features], x[:, self.in_features:]
        yr = self.lin_r(xr) - self.lin_i(xi)
        yi = self.lin_r(xi) + self.lin_i(xr)
        return torch.cat([yr, yi], dim=1)


class CReLU(nn.Module):
    """ReLU applied to real and imaginary parts independently."""

    def forward(self, x):
        return F.relu(x)


class ComplexMaxPool1d(nn.Module):
    """Magnitude-based pooling: pick, per window, the sample of largest |z| and
    keep its real and imaginary parts together (a true complex operation)."""

    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.k = kernel_size
        self.s = stride or kernel_size

    def forward(self, x):
        if x.shape[-1] < self.k:
            return x
        xr, xi = split(x)
        mag = xr * xr + xi * xi
        _, idx = F.max_pool1d(mag, self.k, self.s, return_indices=True)
        yr = torch.gather(xr, 2, idx)
        yi = torch.gather(xi, 2, idx)
        return merge(yr, yi)


class ComplexBatchNorm1d(nn.Module):
    """Complex batch normalization with 2x2 covariance whitening (Trabelsi et
    al., "Deep Complex Networks", 2018), followed by a complex affine
    transform (gamma_rr, gamma_ri, gamma_ii, beta_r, beta_i)."""

    def __init__(self, num_complex, eps=1e-5, momentum=0.1):
        super().__init__()
        c = num_complex
        self.c, self.eps, self.momentum = c, eps, momentum
        inv_sqrt2 = 2 ** -0.5
        self.gamma_rr = nn.Parameter(torch.full((c,), inv_sqrt2))
        self.gamma_ii = nn.Parameter(torch.full((c,), inv_sqrt2))
        self.gamma_ri = nn.Parameter(torch.zeros(c))
        self.beta_r = nn.Parameter(torch.zeros(c))
        self.beta_i = nn.Parameter(torch.zeros(c))
        self.register_buffer("rm_r", torch.zeros(c))
        self.register_buffer("rm_i", torch.zeros(c))
        self.register_buffer("rv_rr", torch.full((c,), inv_sqrt2))
        self.register_buffer("rv_ii", torch.full((c,), inv_sqrt2))
        self.register_buffer("rv_ri", torch.zeros(c))

    def forward(self, x):
        c = self.c
        xr, xi = split(x)
        if self.training:
            mr = xr.mean(dim=(0, 2))
            mi = xi.mean(dim=(0, 2))
        else:
            mr, mi = self.rm_r, self.rm_i
        xr = xr - mr[None, :, None]
        xi = xi - mi[None, :, None]
        if self.training:
            Vrr = (xr * xr).mean(dim=(0, 2)) + self.eps
            Vii = (xi * xi).mean(dim=(0, 2)) + self.eps
            Vri = (xr * xi).mean(dim=(0, 2))
            with torch.no_grad():
                m = self.momentum
                self.rm_r.mul_(1 - m).add_(m * mr)
                self.rm_i.mul_(1 - m).add_(m * mi)
                self.rv_rr.mul_(1 - m).add_(m * Vrr)
                self.rv_ii.mul_(1 - m).add_(m * Vii)
                self.rv_ri.mul_(1 - m).add_(m * Vri)
        else:
            Vrr, Vii, Vri = self.rv_rr + self.eps, self.rv_ii + self.eps, self.rv_ri
        # inverse square root of the 2x2 covariance [[Vrr, Vri], [Vri, Vii]]
        det = (Vrr * Vii - Vri * Vri).clamp_min(self.eps)
        s = torch.sqrt(det)
        t = torch.sqrt((Vrr + Vii + 2 * s).clamp_min(self.eps))
        inv = 1.0 / (s * t)
        Wrr = (Vii + s) * inv
        Wii = (Vrr + s) * inv
        Wri = -Vri * inv
        zr = Wrr[None, :, None] * xr + Wri[None, :, None] * xi
        zi = Wri[None, :, None] * xr + Wii[None, :, None] * xi
        outr = self.gamma_rr[None, :, None] * zr + self.gamma_ri[None, :, None] * zi + self.beta_r[None, :, None]
        outi = self.gamma_ri[None, :, None] * zr + self.gamma_ii[None, :, None] * zi + self.beta_i[None, :, None]
        return merge(outr, outi)


class _TernaryQuant(torch.autograd.Function):
    """Forward: ternary mask  sgn(m) * 1[|m| > mu]  in {0, +-1}.
    Backward: straight-through estimator through Htanh (grad passes for |m|<=1)."""

    @staticmethod
    def forward(ctx, m, mu):
        ctx.save_for_backward(m)
        return torch.sign(m) * (m.abs() > mu).to(m.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (m,) = ctx.saved_tensors
        return grad_out * (m.abs() <= 1).to(m.dtype), None


class TernaryMask(nn.Module):
    """Triple-S mask layer: one full-precision scalar per complex channel,
    quantized to {0, +-1}. Applied to both real and imaginary planes."""

    def __init__(self, num_complex, mu=0.01, init_noise=0.0):
        super().__init__()
        self.c = num_complex
        self.mu = mu
        # Paper uses ones-init. A tiny perturbation (init_noise>0) breaks the
        # degenerate all-equal symmetry under which the per-layer scaling trick
        # would exactly cancel the L1 shrink and freeze every mask at 1.0 (this
        # happens when channels are statistically identical, e.g. trivial data).
        init = torch.ones(num_complex)
        if init_noise > 0:
            init = init + init_noise * torch.randn(num_complex)
        self.m = nn.Parameter(init)

    def ternary(self):
        return _TernaryQuant.apply(self.m, self.mu)

    def forward(self, x):
        mhat = self.ternary()
        mask = torch.cat([mhat, mhat]).view(1, 2 * self.c, 1)
        return x * mask

    def l1(self):
        return self.m.abs().sum()

    @torch.no_grad()
    def survivor_values(self):
        """Return (indices_kept, signed_values) of surviving channels."""
        mhat = torch.sign(self.m) * (self.m.abs() > self.mu).float()
        idx = (mhat != 0).nonzero(as_tuple=True)[0]
        return idx, mhat[idx]
