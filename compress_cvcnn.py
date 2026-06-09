"""Compress a CVCNN for SEI: Triple-S pruning -> SlimCVCNN, with optional
knowledge distillation (KD) to recover accuracy.

Pipeline (paper Fig. 4):
  1. (optional) train a teacher CVCNN(N=128) for KD;
  2. train CVCNN(N=64) with ternary masks under Triple-S (PSTGD);
  3. remove zero-mask channels  ->  SlimCVCNN;
  4. (optional) KD from the teacher to recover low-SNR accuracy.

Example (CPU):
  python compress_cvcnn.py --num-emitters 10 --per-emitter 300 --length 1024 \
      --snr-db 30 --epochs 60 --lambda-sr 2.0 --kd --kd-epochs 20 --device cpu
"""
import argparse
import torch
from torch.utils.data import DataLoader

from cvcnn.model import CVCNN
from cvcnn.triple_s import train_triple_s, build_slim
from cvcnn.kd import train_kd
from cvcnn.compress import complexity_report, compression_ratio
from cvcnn.data import synthetic_supervised, npy_supervised, accuracy
from scsc.utils import set_seed, save_checkpoint


def build_data(args):
    if args.dataset == "synthetic":
        return synthetic_supervised(args.num_emitters, args.per_emitter, args.length,
                                    args.snr_db, seed=args.seed)
    return npy_supervised(args.x_path, args.y_path, seed=args.seed)


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--lambda-sr", type=float, default=3.0, help="L1 sparsity strength")
    ap.add_argument("--mu", type=float, default=0.5, help="ternary quantization threshold")
    ap.add_argument("--mask-init-noise", type=float, default=0.3,
                    help="mask init perturbation (breaks the uniform-symmetry that blocks pruning)")
    ap.add_argument("--kd", action="store_true", help="run knowledge distillation after pruning")
    ap.add_argument("--kd-epochs", type=int, default=20)
    ap.add_argument("--kd-loss", choices=["ce", "kl", "mse", "js"], default="ce")
    ap.add_argument("--lambda-kd", type=float, default=10.0)
    ap.add_argument("--teacher-width", type=int, default=128)
    ap.add_argument("--teacher-epochs", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="checkpoints/slimcvcnn.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    tr_ds, te_ds, K = build_data(args)
    in_ch = int(tr_ds[0][0].shape[0])
    tr = DataLoader(tr_ds, args.batch_size, shuffle=True, drop_last=True)
    te = DataLoader(te_ds, 256)
    ev = lambda m: accuracy(m, te, device)

    # 2) Triple-S on CVCNN(N=base_width)
    print(f"== Triple-S on CVCNN(N={args.base_width}), lambda_sr={args.lambda_sr}, mu={args.mu} ==")
    model = CVCNN(num_classes=K, base_width=args.base_width, in_complex_ch=in_ch // 2,
                  input_length=args.length, with_masks=True, mu=args.mu,
                  mask_init_noise=args.mask_init_noise).to(device)
    train_triple_s(model, tr, epochs=args.epochs, lr=args.lr, lambda_sr=args.lambda_sr,
                   device=device, eval_fn=ev)

    # 3) build SlimCVCNN
    slim = build_slim(model, device=device)
    orig = complexity_report(CVCNN(num_classes=K, base_width=args.base_width,
                                   in_complex_ch=in_ch // 2, input_length=args.length),
                             args.length, in_ch=in_ch)
    slim_c = complexity_report(slim, args.length, in_ch=in_ch)
    ratio = compression_ratio(orig, slim_c)
    print(f"\nSlimCVCNN channels: {slim.channels}")
    print(f"CVCNN(N={args.base_width}): {orig}")
    print(f"SlimCVCNN          : {slim_c}")
    print(f"compression        : {{R_param={ratio['R_param_%']:.1f}%, R_macc={ratio['R_macc_%']:.1f}%}}")
    print(f"accuracy  slim(before KD) = {ev(slim):.4f}")

    # 4) optional KD
    if args.kd:
        print(f"\n== KD: teacher CVCNN(N={args.teacher_width}) -> SlimCVCNN "
              f"(loss={args.kd_loss}, lambda_kd={args.lambda_kd}) ==")
        teacher = CVCNN(num_classes=K, base_width=args.teacher_width,
                        in_complex_ch=in_ch // 2, input_length=args.length).to(device)
        opt = torch.optim.Adam(teacher.parameters(), lr=args.lr)
        ce = torch.nn.CrossEntropyLoss()
        for ep in range(1, args.teacher_epochs + 1):
            teacher.train()
            for xb, yb in tr:
                xb, yb = xb.to(device), yb.to(device)
                loss = ce(teacher(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        print(f"teacher trained, acc={ev(teacher):.4f}")
        train_kd(slim, teacher, tr, epochs=args.kd_epochs, lr=args.lr,
                 lambda_kd=args.lambda_kd, kind=args.kd_loss, device=device, eval_fn=ev)
        print(f"accuracy  slim(after KD)  = {ev(slim):.4f}")

    save_checkpoint(args.out, slim, None, args.epochs,
                    extra={"channels": slim.channels, "num_classes": K,
                           "args": vars(args), "compression": ratio})
    print(f"\nsaved SlimCVCNN -> {args.out}")


if __name__ == "__main__":
    main()
