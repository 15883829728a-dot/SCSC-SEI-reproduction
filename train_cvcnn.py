"""Train a CVCNN or RVCNN for SEI (supervised classification).

Examples
--------
# Basic CVCNN on synthetic data at SNR = 30 dB (CPU):
python train_cvcnn.py --model cvcnn --base-width 64 --num-emitters 10 \
    --per-emitter 300 --length 1024 --snr-db 30 --epochs 60 --device cpu

# Real-valued baseline for comparison:
python train_cvcnn.py --model rvcnn --base-width 128 --snr-db 30 --device cpu
"""
import argparse
import torch
from torch.utils.data import DataLoader

from cvcnn.model import CVCNN, RVCNN
from cvcnn.data import synthetic_supervised, npy_supervised, accuracy
from cvcnn.compress import complexity_report
from scsc.utils import set_seed, save_checkpoint


def build_data(args):
    if args.dataset == "synthetic":
        return synthetic_supervised(args.num_emitters, args.per_emitter, args.length,
                                    args.snr_db, seed=args.seed)
    return npy_supervised(args.x_path, args.y_path, seed=args.seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cvcnn", "rvcnn"], default="cvcnn")
    ap.add_argument("--dataset", choices=["synthetic", "npy"], default="synthetic")
    ap.add_argument("--x-path"); ap.add_argument("--y-path")
    ap.add_argument("--num-emitters", type=int, default=10)
    ap.add_argument("--per-emitter", type=int, default=300)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--snr-db", type=float, default=30)
    ap.add_argument("--base-width", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="checkpoints/cvcnn.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    tr_ds, te_ds, K = build_data(args)
    in_ch = int(tr_ds[0][0].shape[0])
    tr = DataLoader(tr_ds, args.batch_size, shuffle=True, drop_last=True)
    te = DataLoader(te_ds, 256)

    if args.model == "cvcnn":
        model = CVCNN(num_classes=K, base_width=args.base_width,
                      in_complex_ch=in_ch // 2, input_length=args.length).to(device)
    else:
        model = RVCNN(num_classes=K, base_width=args.base_width,
                      in_ch=in_ch, input_length=args.length).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        last = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            loss = ce(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss)
        acc = accuracy(model, te, device)
        if acc > best:
            best = acc
            save_checkpoint(args.out, model, opt, ep,
                            extra={"acc": acc, "args": vars(args), "num_classes": K})
        if ep % max(1, args.epochs // 10) == 0 or ep == args.epochs:
            print(f"epoch {ep:3d}/{args.epochs} loss={last:.4f} test_acc={acc:.4f}", flush=True)

    rep = complexity_report(model, args.length, in_ch=in_ch)
    print(f"done. model={args.model}(N={args.base_width}) best_acc={best:.4f} "
          f"params={rep['params']} MACC={rep['macc']} ckpt={args.out}")


if __name__ == "__main__":
    main()
