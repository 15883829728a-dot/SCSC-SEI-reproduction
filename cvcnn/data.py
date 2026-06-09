"""Supervised data helpers for the CVCNN method (train/test split).

Reuses the synthetic preamble-style emitter generator from ``scsc.datasets``
(here with a fixed SNR so accuracy-vs-SNR curves can be produced) and the
bring-your-own ``.npy`` loader.
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset

from scsc.datasets import SyntheticEmitters, NpyEmitters


def _split(X, Y, test_ratio, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Y))
    n_test = int(len(Y) * test_ratio)
    te, tr = idx[:n_test], idx[n_test:]
    to_ds = lambda i: TensorDataset(torch.from_numpy(X[i]), torch.from_numpy(Y[i]))
    return to_ds(tr), to_ds(te)


def synthetic_supervised(num_emitters=10, per_emitter=300, length=1024,
                         snr_db=30, test_ratio=0.3, seed=0):
    ds = SyntheticEmitters(num_emitters, per_emitter, length, seed=seed, snr_db=snr_db)
    X = ds.X.astype(np.float32)
    Y = ds.Y.astype(np.int64)
    train_ds, test_ds = _split(X, Y, test_ratio, seed)
    return train_ds, test_ds, num_emitters


def npy_supervised(x_path, y_path, test_ratio=0.3, seed=0):
    ds = NpyEmitters(x_path, y_path)
    train_ds, test_ds = _split(ds.X.astype(np.float32), ds.Y.astype(np.int64), test_ratio, seed)
    return train_ds, test_ds, ds.num_emitters


@torch.no_grad()
def accuracy(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += int((pred == yb).sum())
        total += yb.numel()
    return correct / max(total, 1)
