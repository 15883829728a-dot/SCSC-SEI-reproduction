"""Signal data augmentation based on Bit-Pulse Selection (BPS).

Reproduces the augmentation family from the SCSC paper (Sec. III-C):
each signal is split into M temporal segments; a random M-bit pulse code
decides which segments are augmented (bit=1) and which stay unchanged
(bit=0). Four base methods are provided:

    SS  Segment Switching  - time reversal of the segment
    AJ  Amplitude Jitter   - multiply segment by sigma ~ U(0.9, 1.1)
                             (sigma -> 0 degenerates to CutOut)
    TS  Timing Skew        - cyclic shift of the segment
    RS  Random Noise       - add AWGN at SNR ~ U(1, 15) dB

All ops act on signals shaped (C, L) with C in {1, 2} (I/Q as channels).
"""
import numpy as np


def _segment_bounds(L, M):
    b = np.linspace(0, L, M + 1).astype(int)
    return [(b[j], b[j + 1]) for j in range(M)]


def aug_segment_switch(seg, rng):           # SS
    return seg[:, ::-1].copy()


def aug_amplitude_jitter(seg, rng, low=0.9, high=1.1):   # AJ
    sigma = rng.uniform(low, high)
    return seg * sigma


def aug_timing_skew(seg, rng):              # TS (cyclic shift)
    width = seg.shape[1]
    if width < 2:
        return seg
    n = int(rng.integers(1, width))
    return np.roll(seg, n, axis=1)


def aug_random_noise(seg, rng, snr_low=1.0, snr_high=15.0):   # RS
    snr = rng.uniform(snr_low, snr_high)
    power = float(np.mean(seg ** 2)) + 1e-12
    noise_power = power / (10 ** (snr / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=seg.shape)
    return seg + noise


_AUG_FUNCS = {
    "SS": aug_segment_switch,
    "AJ": aug_amplitude_jitter,
    "TS": aug_timing_skew,
    "RS": aug_random_noise,
}


class BPSAugment:
    """Bit-Pulse-Selection augmentation. Calling it on a signal returns one
    augmented *view*; call twice to build a positive pair."""

    def __init__(self, methods=("SS", "AJ", "TS", "RS"), num_segments=16, seed=None):
        unknown = set(methods) - set(_AUG_FUNCS)
        if unknown:
            raise ValueError(f"unknown augmentation(s): {unknown}")
        self.methods = list(methods)
        self.M = int(num_segments)
        self.rng = np.random.default_rng(seed)

    def _apply(self, x, method):
        C, L = x.shape
        out = x.astype(np.float32, copy=True)
        fn = _AUG_FUNCS[method]
        bits = self.rng.integers(0, 2, size=self.M)
        for (s, e), b in zip(_segment_bounds(L, self.M), bits):
            if b == 1 and e > s:
                out[:, s:e] = fn(out[:, s:e], self.rng)
        return out

    def __call__(self, x):
        method = self.methods[int(self.rng.integers(0, len(self.methods)))]
        return self._apply(np.asarray(x, dtype=np.float32), method)
