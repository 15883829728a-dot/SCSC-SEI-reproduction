"""Instance-level and cluster-level contrastive losses (SCSC, Eqs. 8-19).

This is the Contrastive-Clustering dual-head objective adapted to RF signals:
    L = L_I + L_E
where L_I is an instance NT-Xent loss and L_E is a cluster-level NT-Xent loss
over the K cluster-assignment columns, minus a batch entropy term H(Y) that
discourages the trivial all-in-one-cluster solution.

The SCSC paper reports that a temperature coefficient is *not* required for RF
signals, so the default temperature is 1.0 (cosine similarity). It is left
configurable for experimentation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.t = temperature

    def forward(self, z_i, z_j):
        # z_i, z_j : (N, d), already L2-normalized
        n = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)              # (2N, d)
        sim = torch.matmul(z, z.t()) / self.t         # (2N, 2N)
        sim.fill_diagonal_(float("-inf"))             # mask self-similarity
        targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
        return F.cross_entropy(sim, targets)


class ClusterLoss(nn.Module):
    def __init__(self, num_clusters, temperature=1.0):
        super().__init__()
        self.k = num_clusters
        self.t = temperature

    def forward(self, c_i, c_j):
        # c_i, c_j : (N, K) softmax cluster-assignment probabilities
        # --- entropy regularizer: maximize H(cluster sizes) -> balanced clusters
        def neg_entropy(c):
            p = c.sum(dim=0)
            p = p / p.sum()
            return math.log(self.k) + (p * torch.log(p + 1e-12)).sum()
        ne = neg_entropy(c_i) + neg_entropy(c_j)

        # --- cluster-level contrast over the K assignment columns
        ci, cj = c_i.t(), c_j.t()                     # (K, N) each: a cluster = its column
        c = F.normalize(torch.cat([ci, cj], dim=0), dim=1)   # (2K, N)
        sim = torch.matmul(c, c.t()) / self.t         # (2K, 2K)
        sim.fill_diagonal_(float("-inf"))
        targets = torch.cat([torch.arange(self.k, 2 * self.k),
                             torch.arange(0, self.k)]).to(c.device)
        contrastive = F.cross_entropy(sim, targets)
        return contrastive + ne


class SCSCLoss(nn.Module):
    """Total objective  L = L_I + L_E."""

    def __init__(self, num_clusters, instance_temperature=1.0, cluster_temperature=1.0):
        super().__init__()
        self.instance = InstanceLoss(instance_temperature)
        self.cluster = ClusterLoss(num_clusters, cluster_temperature)

    def forward(self, z_i, z_j, c_i, c_j):
        li = self.instance(z_i, z_j)
        le = self.cluster(c_i, c_j)
        return li + le, li.detach(), le.detach()
