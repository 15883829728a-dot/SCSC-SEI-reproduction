"""Evaluate a trained SCSC checkpoint: clustering metrics + optional t-SNE plot.

Example:
    python evaluate.py --ckpt checkpoints/scsc.pt --dataset synthetic \
        --num-emitters 10 --per-emitter 120 --length 1024 --tsne tsne.png
"""
import argparse
import numpy as np
import torch

from scsc.datasets import SyntheticEmitters, NpyEmitters, RML2016, eval_tensors
from scsc.model import SCSCNet
from scsc.metrics import clustering_metrics
from scsc.utils import load_checkpoint


def build_base(args):
    if args.dataset == "synthetic":
        return SyntheticEmitters(args.num_emitters, args.per_emitter, args.length, args.seed)
    if args.dataset == "npy":
        return NpyEmitters(args.x_path, args.y_path)
    if args.dataset == "rml":
        return RML2016(args.pkl_path, snr_min=args.snr_min)
    raise ValueError(args.dataset)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", choices=["synthetic", "npy", "rml"], default="synthetic")
    ap.add_argument("--x-path"); ap.add_argument("--y-path"); ap.add_argument("--pkl-path")
    ap.add_argument("--snr-min", type=int, default=None)
    ap.add_argument("--num-emitters", type=int, default=10)
    ap.add_argument("--per-emitter", type=int, default=200)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tsne", default=None, help="path to save a t-SNE plot (optional)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    base = build_base(args)
    X, Y = eval_tensors(base)
    in_ch = int(X.shape[1])

    ckpt = torch.load(args.ckpt, map_location="cpu")
    a = ckpt.get("extra", {}).get("args", {})
    k = a.get("num_clusters") or getattr(base, "num_emitters")
    model = SCSCNet(in_ch=in_ch, feat_dim=a.get("feat_dim", 256),
                    proj_dim=a.get("proj_dim", 128), num_clusters=k).to(device)
    load_checkpoint(args.ckpt, model, map_location="cpu")
    model.eval()

    preds, feats = [], []
    for i in range(0, len(X), 256):
        xb = X[i:i + 256].to(device)
        preds.append(model.assign(xb).cpu().numpy())
        feats.append(model.features(xb).cpu().numpy())
    preds = np.concatenate(preds)
    print("clustering metrics:", clustering_metrics(Y, preds))

    if args.tsne:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        emb = TSNE(n_components=2, init="pca", random_state=args.seed).fit_transform(
            np.concatenate(feats))
        plt.figure(figsize=(6, 5))
        plt.scatter(emb[:, 0], emb[:, 1], c=Y, cmap="tab20", s=6)
        plt.title("SCSC features (t-SNE), colored by true emitter")
        plt.tight_layout(); plt.savefig(args.tsne, dpi=150)
        print("saved t-SNE plot ->", args.tsne)


if __name__ == "__main__":
    main()
