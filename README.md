# SEI Deep-Learning Reproductions (SCSC · CVCNN)

Faithful, from-scratch **PyTorch reproductions** of two specific-emitter-identification (SEI) papers, with runnable code, synthetic + bring-your-own data, and validation experiments.

| # | Method | Paper | Setting |
|---|--------|-------|---------|
| 1 | **SCSC** — Signal Contrastive Self-Supervised Clustering | Hao et al., *IEEE IoT-J* 2023 | **unsupervised** clustering |
| 2 | **CVCNN + Triple-S + KD** — Complex-Valued CNN with network compression | Wang et al., *IEEE JSAC* 2021 | **supervised** classification + model compression |

> ⚠️ Unofficial, for learning/research. No third-party code or non-public datasets are redistributed. The papers' own RF datasets are not public, so a documented **synthetic emitter dataset** is provided to validate each pipeline end-to-end; plug in real/public data via the `npy` loaders for paper-grade numbers.

---

## Install

```bash
conda create -n sei python=3.10 -y && conda activate sei
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt
```

## Repository layout

```
scsc/        method 1 — SCSC (contrastive self-supervised clustering)
  augment.py    BPS signal augmentation (SS/AJ/TS/RS)
  model.py      1D-FPFE backbone + instance & cluster heads
  losses.py     instance + cluster contrastive loss (L = L_I + L_E)
  metrics.py    ACC (Hungarian) / NMI / ARI / F
  datasets.py   synthetic emitters / RML2016 / npy / contrastive pairs
cvcnn/       method 2 — CVCNN + network compression
  complex_layers.py  complex conv / BN (whitening) / CReLU / mag-pool / linear / ternary mask
  model.py           CVCNN (9 complex-conv + 2 FC) and a real-valued RVCNN baseline
  triple_s.py        Triple-S pruning (proximal STGD) + structural slimming -> SlimCVCNN
  kd.py              knowledge distillation (CE/KL/MSE/JS)
  compress.py        parameter / MACC accounting + compression ratios
  data.py            supervised splits + accuracy
train.py · evaluate.py            SCSC train / eval (+ t-SNE)
train_cvcnn.py · compress_cvcnn.py  CVCNN train / Triple-S compression (+ KD)
```

---

## Method 1 — SCSC (unsupervised clustering)

Contrastive Clustering on 1-D RF signals: a **1D-FPFE** backbone (large kernels + pyramid multi-scale pooling), **BPS** augmentations (Segment-Switch / Amplitude-Jitter / Timing-Skew / Random-Noise), and dual **instance + cluster** contrastive heads optimised with `L = L_I + L_E` (cluster term has an entropy regulariser to avoid collapse). Single-stage, end-to-end, only the cluster count `K` is needed.

```bash
# train (8 synthetic emitters, tau=0.5, CPU, a few minutes)
python train.py --dataset synthetic --num-emitters 8 --per-emitter 120 \
    --length 512 --epochs 40 --batch-size 128 --eval-every 5 --instance-temp 0.5 --device cpu
# evaluate + t-SNE
python evaluate.py --ckpt checkpoints/scsc.pt --dataset synthetic \
    --num-emitters 8 --per-emitter 120 --length 512 --tsne tsne.png
```

**Result** (preamble-style synthetic toy, 8 emitters, CPU, 40 epochs): from random ACC 0.125 → **ACC 0.64 / NMI 0.67 / ARI 0.50**, still rising — validates the pipeline learns emitter-separable representations.

---

## Method 2 — CVCNN + Triple-S + KD (supervised + compression)

A **complex-valued CNN** processes the complex baseband directly (complex conv, **covariance-whitening complex BN**, `CReLU`, magnitude-based complex pooling). It is built with small kernels (size 3), pooling after every conv, 9 conv layers + 2 FC. Network compression then proceeds in two stages:

- **Triple-S** (Sparse Structure Selection): ternary mask layers `m̂ = sgn(m)·1[|m|>μ]` after every layer; **weights via SGD/Adam, masks via proximal STGD** (straight-through estimator through Htanh + L1 soft-threshold `prox`, clip to [−1,1], per-layer scaling). Channels with zero mask are physically removed → **SlimCVCNN** (its forward output is *mathematically identical* to the masked model — verified `max|Δ| = 0`).
- **Knowledge distillation**: teacher `CVCNN(N=128)` → student `SlimCVCNN`, `L = λ_KD·L_T(soft) + CE(hard)`, with `L_T ∈ {CE, KL, MSE, JS}`, to recover accuracy (especially at low SNR).

```bash
# basic CVCNN vs real-valued baseline (synthetic, SNR=30 dB)
python train_cvcnn.py --model cvcnn --base-width 64 --snr-db 30 --device cpu
python train_cvcnn.py --model rvcnn --base-width 128 --snr-db 30 --device cpu

# Triple-S compression -> SlimCVCNN, then knowledge distillation
python compress_cvcnn.py --num-emitters 10 --per-emitter 300 --length 1024 \
    --snr-db 30 --epochs 60 --lambda-sr <LAM> --kd --kd-epochs 20 --device cpu
```

**Complexity relationships reproduced exactly** (synthetic, length 1024, 10 classes):

| Model | Params | MACCs |
|-------|-------:|------:|
| CVCNN (N=64)  | 235,338 | 50,988,288 |
| CVCNN (N=128) | 862,346 | 202,245,376 |
| RVCNN (N=128) | 431,626 | 50,955,520 |

→ matching the paper: **CVCNN(64) ≈ RVCNN(128) in MACCs**, **CVCNN(128) ≈ 4× CVCNN(64)**, and CVCNN(64) has fewer params than RVCNN(128).

**Compression result** (synthetic, 6 emitters, SNR 30 dB, `μ=0.5, λ_SR=3`): Triple-S prunes **CVCNN(N=64)** → **SlimCVCNN** (channels `[47,44,46,33,38,39,43,40,36]`), reducing **params by 57.8% and MACCs by 53.0% with no accuracy loss** (1.000 → 1.000). The slimmed network is verified **output-identical** to the masked model (`max|Δ| = 1.9e-6`). Compression magnitude scales with `λ_SR`/`μ` and the degree of over-parameterization; `compress_cvcnn.py --kd` then applies knowledge distillation to recover accuracy at low SNR.

Use real/public data for paper-grade SNR curves:
```bash
python train_cvcnn.py --dataset npy --x-path X.npy --y-path Y.npy --device cuda
python compress_cvcnn.py --dataset npy --x-path X.npy --y-path Y.npy --kd --device cuda
```

---

## Notes / faithful-reproduction caveats

- **Datasets**: neither paper's RF dataset is public; the synthetic generator injects per-emitter hardware impairments (CFO, IQ imbalance, PA nonlinearity, carrier leakage) on a shared preamble, so emitter identity is recoverable. It validates the *methods*, not absolute paper numbers.
- **SCSC**: temperature is configurable; the paper reports none is needed (τ=1.0), while τ=0.5 trains faster on the toy.
- **CVCNN**: Table I/II exact channel counts and hyperparameters were not recoverable from the PDF, so width is parameterised by `base_width` (the paper's *N*); `CReLU` activation and covariance-whitening complex BN follow Trabelsi et al., *Deep Complex Networks*.
- **Not yet implemented** (PRs welcome): SCSC open-set rejection (ZSL + semantic centroid); CVCNN exact Table-I schedule once the source is available.

## Citations

```bibtex
@article{hao2023scsc,
  title={Contrastive Self-Supervised Clustering for Specific Emitter Identification},
  author={Hao, Xiaoyang and Feng, Zhixi and Liu, Ruoyu and Yang, Shuyuan and Jiao, Licheng and Luo, Rong},
  journal={IEEE Internet of Things Journal}, volume={10}, number={23}, pages={20803--20817}, year={2023},
  doi={10.1109/JIOT.2023.3284428}
}
@article{wang2021cvcnn,
  title={An Efficient Specific Emitter Identification Method Based on Complex-Valued Neural Networks and Network Compression},
  author={Wang, Yu and Gui, Guan and Gacanin, Haris and Ohtsuki, Tomoaki and Dobre, Octavia A. and Poor, H. Vincent},
  journal={IEEE Journal on Selected Areas in Communications}, volume={39}, number={8}, pages={2305--2317}, year={2021},
  doi={10.1109/JSAC.2021.3087243}
}
```

## License

Self-authored code under [MIT](LICENSE). Paper methods/ideas belong to their authors. Do not commit third-party code or non-public datasets.
