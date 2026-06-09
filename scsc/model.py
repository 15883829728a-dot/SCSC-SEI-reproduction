"""1D-FPFE backbone + instance / cluster contrastive heads (SCSC, Sec. III-A/B)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.net(x)


class FPFE1D(nn.Module):
    """1-D Fingerprint Pyramid Feature Extractor.

    Uses *large* 1-D kernels (paper suggests 1x15 / 1x19 / 1x23) and a pyramid
    multi-scale design: every block is globally pooled and the multi-scale
    descriptors are concatenated, then projected to ``feat_dim``.
    """

    def __init__(self, in_ch=2, feat_dim=256, channels=(64, 128, 256),
                 kernels=(15, 19, 23)):
        super().__init__()
        assert len(channels) == len(kernels)
        blocks, c = [], in_ch
        for c_out, k in zip(channels, kernels):
            blocks.append(ConvBlock(c, c_out, k))
            c = c_out
        self.blocks = nn.ModuleList(blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(sum(channels), feat_dim),
            nn.ReLU(inplace=True),
        )
        self.feat_dim = feat_dim

    def forward(self, x):
        scales = []
        for blk in self.blocks:
            x = blk(x)
            scales.append(self.gap(x).squeeze(-1))   # (N, c_out)
        h = torch.cat(scales, dim=1)                 # pyramid multi-scale concat
        return self.proj(h)                          # (N, feat_dim)


def _mlp(d_in, d_hidden, d_out):
    # Paper: "linear layer, ReLu and linear layer"
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(inplace=True),
                         nn.Linear(d_hidden, d_out))


class SCSCNet(nn.Module):
    def __init__(self, in_ch=2, feat_dim=256, proj_dim=128, num_clusters=10,
                 channels=(64, 128, 256), kernels=(15, 19, 23)):
        super().__init__()
        self.backbone = FPFE1D(in_ch, feat_dim, channels, kernels)
        self.instance_head = _mlp(feat_dim, feat_dim, proj_dim)   # G_I
        self.cluster_head = _mlp(feat_dim, feat_dim, num_clusters)  # G_E
        self.num_clusters = num_clusters

    def forward(self, x):
        h = self.backbone(x)
        z = F.normalize(self.instance_head(h), dim=1)   # instance features (unit norm)
        c = F.softmax(self.cluster_head(h), dim=1)       # cluster assignment probs
        return h, z, c

    @torch.no_grad()
    def assign(self, x):
        h = self.backbone(x)
        return F.softmax(self.cluster_head(h), dim=1).argmax(dim=1)

    @torch.no_grad()
    def features(self, x):
        return self.backbone(x)
