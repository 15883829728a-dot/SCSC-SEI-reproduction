"""Train SCSC (Signal Contrastive Self-Supervised Clustering) for SEI.

Examples
--------
# Local smoke test on the synthetic dataset (CPU, ~1 min):
python train.py --dataset synthetic --num-emitters 10 --per-emitter 120 \
    --length 1024 --epochs 30 --batch-size 128 --eval-every 5 --device cpu

# Real IQ data (bring your own .npy), GPU on the cloud:
python train.py --dataset npy --x-path X.npy --y-path Y.npy \
    --num-clusters 30 --length 4096 --epochs 200 --batch-size 128 --device cuda
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from scsc.augment import BPSAugment
from scsc.datasets import (SyntheticEmitters, NpyEmitters, RML2016,
                           ContrastivePairs, eval_tensors)
from scsc.model import SCSCNet
from scsc.losses import SCSCLoss
from scsc.metrics import clustering_metrics
from scsc.utils import set_seed, AverageMeter, save_checkpoint


def build_base(args):
    if args.dataset == "synthetic":
        return SyntheticEmitters(args.num_emitters, args.per_emitter, args.length, args.seed)
    if args.dataset == "npy":
        return NpyEmitters(args.x_path, args.y_path)
    if args.dataset == "rml":
        return RML2016(args.pkl_path, snr_min=args.snr_min)
    raise ValueError(args.dataset)


@torch.no_grad()
def evaluate(model, X, Y, device, batch_size=256):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        preds.append(model.assign(xb).cpu().numpy())
    return clustering_metrics(Y, np.concatenate(preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "npy", "rml"], default="synthetic")
    ap.add_argument("--x-path"); ap.add_argument("--y-path"); ap.add_argument("--pkl-path")
    ap.add_argument("--snr-min", type=int, default=None)
    ap.add_argument("--num-emitters", type=int, default=10)   # synthetic only
    ap.add_argument("--per-emitter", type=int, default=200)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--num-clusters", type=int, default=None,
                    help="defaults to the dataset's class count")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--feat-dim", type=int, default=256)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--methods", nargs="+", default=["SS", "AJ", "TS", "RS"])
    ap.add_argument("--num-segments", type=int, default=16)
    ap.add_argument("--instance-temp", type=float, default=1.0)
    ap.add_argument("--cluster-temp", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="checkpoints/scsc.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    base = build_base(args)
    in_ch = int(base[0][0].shape[0])
    n_clusters = args.num_clusters or getattr(base, "num_emitters", None)
    if n_clusters is None:
        raise ValueError("please pass --num-clusters")
    print(f"dataset={args.dataset}  samples={len(base)}  in_ch={in_ch}  K={n_clusters}")

    aug = BPSAugment(methods=args.methods, num_segments=args.num_segments, seed=args.seed)
    loader = DataLoader(ContrastivePairs(base, aug), batch_size=args.batch_size,
                        shuffle=True, drop_last=True, num_workers=args.workers)
    Xeval, Yeval = eval_tensors(base)

    model = SCSCNet(in_ch=in_ch, feat_dim=args.feat_dim, proj_dim=args.proj_dim,
                    num_clusters=n_clusters).to(device)
    criterion = SCSCLoss(n_clusters, args.instance_temp, args.cluster_temp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        lm, li_m, le_m = AverageMeter(), AverageMeter(), AverageMeter()
        for v1, v2, _ in loader:
            v1, v2 = v1.to(device), v2.to(device)
            _, z1, c1 = model(v1)
            _, z2, c2 = model(v2)
            loss, li, le = criterion(z1, z2, c1, c2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lm.update(loss.item()); li_m.update(li.item()); le_m.update(le.item())

        msg = f"epoch {epoch:3d}/{args.epochs}  loss={lm.avg:.4f} (L_I={li_m.avg:.4f} L_E={le_m.avg:.4f})"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m = evaluate(model, Xeval, Yeval, device)
            msg += f"  | ACC={m['ACC']:.4f} NMI={m['NMI']:.4f} ARI={m['ARI']:.4f}"
            if m["ACC"] > best:
                best = m["ACC"]
                save_checkpoint(args.out, model, optimizer, epoch,
                                extra={"metrics": m, "args": vars(args)})
                msg += "  [saved]"
        print(msg, flush=True)

    print(f"done. best ACC={best:.4f}  checkpoint={args.out}")


if __name__ == "__main__":
    main()
