"""Knowledge distillation to recover SlimCVCNN accuracy (paper Sec. IV-C).

L_KD = lambda_KD * L_T(teacher_soft, student_pred) + CE(labels, student_pred)
with L_T in {MSE, CE, KL, JS}.  The teacher is frozen.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def distill_term(student_logits, teacher_logits, kind="ce", T=1.0):
    ps_log = F.log_softmax(student_logits / T, dim=1)
    ps = ps_log.exp()
    pt = F.softmax(teacher_logits / T, dim=1)
    if kind == "ce":
        return -(pt * ps_log).sum(dim=1).mean()
    if kind == "kl":
        return F.kl_div(ps_log, pt, reduction="batchmean")
    if kind == "mse":
        return F.mse_loss(ps, pt)
    if kind == "js":
        m = 0.5 * (ps + pt)
        m_log = m.clamp_min(1e-12).log()
        return 0.5 * F.kl_div(m_log, pt, reduction="batchmean") + \
            0.5 * F.kl_div(m_log, ps, reduction="batchmean")
    raise ValueError(kind)


def train_kd(student, teacher, loader, epochs=20, lr=1e-3, lambda_kd=10.0,
             kind="ce", T=1.0, device="cpu", eval_fn=None, log=print):
    student.to(device).train()
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        student.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                t_logits = teacher(xb)
            s_logits = student(xb)
            loss = ce(s_logits, yb) + lambda_kd * distill_term(s_logits, t_logits, kind, T)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        msg = f"[KD] epoch {ep:3d}/{epochs} loss={running/len(loader):.4f}"
        if eval_fn is not None:
            msg += f" acc={eval_fn(student):.4f}"
        log(msg)
    return student
