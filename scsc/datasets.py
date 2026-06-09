"""Datasets for SCSC.

The paper's own 30-emitter USRP-X310 burst dataset is NOT public, so this repo
ships:

  * SyntheticEmitters - a toy RF-fingerprint dataset (unique per-emitter hardware
    impairments: CFO, phase offset, IQ imbalance, PA nonlinearity). Runs on CPU
    in seconds and is the default for local code validation.
  * RML2016 - adapter for the public DeepSig RML2016.10a set (the SCSC paper uses
    it for the augmentation benchmark). Clusters modulation types, not emitters.
  * NpyEmitters - bring-your-own loader for real IQ data saved as .npy
    (X: (N, C, L) float32, Y: (N,) int).

All base datasets return (x, y) with x shaped (C, L) float32. Wrap any of them
in ContrastivePairs for self-supervised training.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticEmitters(Dataset):
    """Preamble-style toy RF-fingerprint dataset.

    All emitters transmit the SAME known base waveform (like a preamble, as in
    real LoRa/WiFi RFFI) and differ only by hardware impairments: well-separated
    carrier frequency offset (CFO), IQ gain/phase imbalance, power-amplifier
    nonlinearity, and carrier (DC) leakage. Each instance adds nuisances -
    random channel phase, small timing jitter, and AWGN - so that intra-emitter
    instances vary while emitter identity stays recoverable. This is a sanity
    toy for validating the SCSC pipeline on CPU, NOT a substitute for real data.
    """

    def __init__(self, num_emitters=10, per_emitter=200, length=1024, seed=0, snr_db=None):
        rng = np.random.default_rng(seed)
        self.num_emitters = num_emitters
        n = np.arange(length)
        # shared known base waveform (BPSK "preamble"), identical for all emitters
        base = (np.random.default_rng(12345).integers(0, 2, length) * 2 - 1).astype(np.float64)
        # per-emitter hardware fingerprints (well separated)
        cfo = np.linspace(-0.15, 0.15, num_emitters) + rng.uniform(-0.004, 0.004, num_emitters)
        gain = rng.uniform(0.85, 1.15, num_emitters)     # IQ gain imbalance
        phi = rng.uniform(-0.20, 0.20, num_emitters)     # IQ phase imbalance
        a3 = rng.uniform(-0.10, 0.10, num_emitters)      # PA 3rd-order nonlinearity
        dc = (rng.uniform(-0.05, 0.05, num_emitters)
              + 1j * rng.uniform(-0.05, 0.05, num_emitters))   # carrier leakage
        X, Y = [], []
        for k in range(num_emitters):
            for _ in range(per_emitter):
                theta = rng.uniform(0, 2 * np.pi)                      # channel phase (nuisance)
                b = np.roll(base, int(rng.integers(0, 8)))            # small timing jitter
                s = b * np.exp(1j * (2 * np.pi * cfo[k] * n + theta))  # CFO + phase
                s = s + a3[k] * s * np.abs(s) ** 2                     # PA nonlinearity
                s = s + dc[k]                                          # carrier leakage
                I, Q = s.real, s.imag
                Ii = gain[k] * I                                       # IQ imbalance
                Qi = Q * np.cos(phi[k]) + I * np.sin(phi[k])
                sig = np.stack([Ii, Qi], axis=0)
                snr = rng.uniform(15, 25) if snr_db is None else float(snr_db)
                p = np.mean(sig ** 2) + 1e-12
                sig = sig + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), sig.shape)
                sig = sig / (np.std(sig) + 1e-8)                       # per-instance norm
                X.append(sig.astype(np.float32))
                Y.append(k)
        self.X = np.stack(X)
        self.Y = np.asarray(Y, dtype=np.int64)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.X[i], int(self.Y[i])


class NpyEmitters(Dataset):
    """Real data loader. X: (N, C, L) float32 .npy ; Y: (N,) int .npy."""

    def __init__(self, x_path, y_path, per_instance_norm=True):
        self.X = np.load(x_path).astype(np.float32)
        self.Y = np.load(y_path).astype(np.int64)
        if self.X.ndim == 2:                       # (N, L) -> (N, 1, L)
            self.X = self.X[:, None, :]
        if per_instance_norm:
            std = self.X.std(axis=(1, 2), keepdims=True) + 1e-8
            self.X = self.X / std
        self.num_emitters = int(self.Y.max()) + 1

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.X[i], int(self.Y[i])


class RML2016(Dataset):
    """DeepSig RML2016.10a adapter (download separately: RML2016.10a_dict.pkl)."""

    def __init__(self, pkl_path, snr_min=None, per_instance_norm=True):
        import pickle
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        mods = sorted({k[0] for k in data})
        self.mod2idx = {m: i for i, m in enumerate(mods)}
        X, Y = [], []
        for (mod, snr), arr in data.items():
            if snr_min is not None and snr < snr_min:
                continue
            for s in arr:
                X.append(np.asarray(s, dtype=np.float32))   # (2, 128)
                Y.append(self.mod2idx[mod])
        self.X = np.stack(X)
        self.Y = np.asarray(Y, dtype=np.int64)
        if per_instance_norm:
            self.X = self.X / (self.X.std(axis=(1, 2), keepdims=True) + 1e-8)
        self.num_emitters = len(mods)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.X[i], int(self.Y[i])


class ContrastivePairs(Dataset):
    """Wrap a base dataset; yields two augmented views (v1, v2) and the label."""

    def __init__(self, base, augment):
        self.base = base
        self.aug = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        v1 = self.aug(x)
        v2 = self.aug(x)
        return torch.from_numpy(np.ascontiguousarray(v1)), \
            torch.from_numpy(np.ascontiguousarray(v2)), int(y)


def eval_tensors(base):
    """Stack a base dataset into (X_tensor, Y_array) for evaluation."""
    xs = np.stack([base[i][0] for i in range(len(base))])
    ys = np.asarray([base[i][1] for i in range(len(base))], dtype=np.int64)
    return torch.from_numpy(xs), ys
