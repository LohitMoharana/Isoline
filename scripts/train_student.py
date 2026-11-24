"""Teacher -> student distillation + BN folding + int8 power-of-two QAT.

Development (default):  trains on DS1 minus a record-wise validation hold-out
and reports VALIDATION metrics only. DS2 is not even loaded.

Final (--final):  evaluates on DS2, once per seed, with a frozen configuration.

    python scripts/train_student.py --delta 0.1 --seeds 0            # dev
    python scripts/train_student.py --delta 0.1 --seeds 0 1 2 3 4 --final
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEV = torch.device("cpu")      # set in main() from --device

from isoline import data  # noqa: E402
from isoline.config import CLASSES, DS1_VAL, AFEConfig  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402
from isoline.models import (StudentSiliconTCN, TeacherSEResNet1D, prep_dense,  # noqa: E402
                            prep_spike)
from isoline.pipeline import build_dataset  # noqa: E402
from isoline.quant import FoldedStudent  # noqa: E402


def class_balanced_weights(y: np.ndarray, beta: float) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64).clip(min=1)
    w = (1 - beta) / (1 - beta ** counts)
    return torch.tensor(w / w.sum() * len(CLASSES), dtype=torch.float32)


def focal_loss(logits, y, cls_w, gamma: float, smooth: float = 0.0):
    ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=smooth)
    pt = torch.exp(-ce)
    return (cls_w[y] * (1 - pt) ** gamma * ce).mean()


def kd_loss(s, t, T: float):
    return F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction="batchmean") * T * T


@torch.no_grad()
def predict(model, xs, ss, bs: int = 4096):
    model.eval()
    return torch.cat([model(xs[i:i + bs], ss[i:i + bs]) for i in range(0, len(xs), bs)])


def evaluate(model, xs, ss, y):
    return report(y.cpu().numpy(), predict(model, xs, ss).argmax(1).cpu().numpy())


def fit(model, xs, ss, y, val, epochs, lr, cls_w, a, teacher_logits=None, wd=1e-4,
        smooth=0.0, noise=0.0, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    n = len(xs)
    steps = n // a.batch
    best, best_state = -1.0, copy.deepcopy(model.state_dict())
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), torch.zeros((), device=xs.device)
        perm = torch.randperm(n, device=xs.device)
        for k in range(steps):
            idx = perm[k * a.batch:(k + 1) * a.batch]
            xb, sb, yb = xs[idx], ss[idx], y[idx]
            if noise > 0:
                xb = xb + noise * torch.randn_like(xb)
            logits = model(xb, sb)
            loss = focal_loss(logits, yb, cls_w, a.gamma, smooth)
            if teacher_logits is not None:
                loss = (1 - a.alpha) * loss + a.alpha * kd_loss(logits, teacher_logits[idx], a.temp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach()
        sched.step()
        rep = evaluate(model, *val)
        if rep[a.select_by] > best:
            best, best_state = rep[a.select_by], copy.deepcopy(model.state_dict())
        print(f"[{tag}] ep {ep + 1}/{epochs} loss {float(tot) / steps:.4f} "
              f"val acc {rep['accuracy']:.4f} macroF1 {rep['macro_f1']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model


def run_seed(a, seed, T, te):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    xs_tr, ss_tr, xd_tr, y_tr = T["tr"]
    xs_va, ss_va, xd_va, y_va = T["va"]
    cls_w = class_balanced_weights(y_tr.cpu().numpy(), a.cb_beta).to(DEV)
    print(f"--- seed {seed}  class weights {dict(zip(CLASSES, np.round(cls_w.tolist(), 3)))}")

    teacher = TeacherSEResNet1D(nscal=a.ncol, dropout=a.teacher_dropout).to(DEV)
    teacher = fit(teacher, xd_tr, ss_tr, y_tr, (xd_va, ss_va, y_va), a.teacher_epochs, 2e-3,
                  cls_w, a, wd=a.teacher_wd, smooth=a.smooth, noise=a.noise, tag=f"s{seed} teacher")
    t_logits = predict(teacher, xd_tr, ss_tr)

    student = StudentSiliconTCN(width=a.width, nscal=a.ncol).to(DEV)
    student = fit(student, xs_tr, ss_tr, y_tr, (xs_va, ss_va, y_va), a.student_epochs, 3e-3,
                  cls_w, a, teacher_logits=t_logits, wd=a.student_wd, tag=f"s{seed} student")

    folded = FoldedStudent(student).to(DEV)
    folded.set_qat(False)
    with torch.no_grad():
        student.eval(); folded.eval()
        d = (student(xs_va[:512], ss_va[:512]) - folded(xs_va[:512], ss_va[:512])).abs().max()
    assert float(d) < 1e-3, f"BN folding mismatch {float(d)}"
    folded.train()
    with torch.no_grad():
        for i in range(0, min(len(xs_tr), 20480), 1024):
            folded(xs_tr[i:i + 1024], ss_tr[i:i + 1024])
    folded.set_qat(True)
    folded = fit(folded, xs_tr, ss_tr, y_tr, (xs_va, ss_va, y_va), a.qat_epochs, 5e-4,
                 cls_w, a, teacher_logits=t_logits, wd=a.student_wd, tag=f"s{seed} qat")

    out = dict(seed=seed, footprint_bytes=folded.footprint_bytes(),
               val=dict(teacher=evaluate(teacher, xd_va, ss_va, y_va),
                        student_float=evaluate(student, xs_va, ss_va, y_va),
                        student_int8=evaluate(folded, xs_va, ss_va, y_va)))
    if te is not None:                      # --final only
        xs_te, ss_te, xd_te, y_te = te
        out["test"] = dict(teacher=evaluate(teacher, xd_te, ss_te, y_te),
                           student_float=evaluate(student, xs_te, ss_te, y_te),
                           student_int8=evaluate(folded, xs_te, ss_te, y_te))
    torch.save(dict(student=student.state_dict(), folded=folded.state_dict(), width=a.width,
                    delta_mv=a.delta, seed=seed), Path(a.out) / f"student_{a.tag}_seed{seed}.pt")
    return out


def summarize(runs, split):
    keys = ["accuracy", "macro_f1"]
    rows = {}
    for model in ("teacher", "student_float", "student_int8"):
        vals = {k: [r[split][model][k] for r in runs] for k in keys}
        for c in CLASSES:
            vals[f"f1_{c}"] = [r[split][model][c]["f1"] for r in runs]
            vals[f"se_{c}"] = [r[split][model][c]["se"] for r in runs]
        rows[model] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in vals.items()}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "mitdb"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--hp", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--final", action="store_true", help="evaluate on DS2 (frozen config only)")
    ap.add_argument("--jitter", type=int, default=5)
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--teacher-epochs", type=int, default=6)
    ap.add_argument("--student-epochs", type=int, default=30)
    ap.add_argument("--qat-epochs", type=int, default=6)
    ap.add_argument("--width", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cb-beta", type=float, default=0.999)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--teacher-dropout", type=float, default=0.3)
    ap.add_argument("--teacher-wd", type=float, default=5e-2)
    ap.add_argument("--student-wd", type=float, default=1e-4)
    ap.add_argument("--select-by", default="accuracy", choices=["accuracy", "macro_f1"])
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--svdb-dir", default="", help="extra TRAINING data (never evaluated); e.g. data/svdb")
    ap.add_argument("--delayed", action="store_true", help="use non-causal next-RR columns (off by default)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--subsample", type=int, default=0, help="debug: random subset of train beats")
    a = ap.parse_args()
    global DEV
    a.ncol = 16 if a.delayed else 13
    DEV = torch.device("cuda" if (a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())) else "cpu")
    print("device:", DEV, torch.cuda.get_device_name(0) if DEV.type == "cuda" else "")
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ds1, ds2 = data.get_splits(a.data_dir)
    val_recs = [r for r in DS1_VAL if r in ds1]
    train_recs = [r for r in ds1 if r not in val_recs]
    print(f"train {len(train_recs)} rec | val {val_recs} | DS2 used: {a.final}")
    cfg = replace(AFEConfig(), delta_mv=a.delta, hp_hz=a.hp)

    def tens(ds):
        xs, ss = prep_spike(ds["hist"], ds["res"], ds["scal"][:, :a.ncol])
        y = torch.from_numpy(ds["y"].astype(np.int64))
        return tuple(t.to(DEV) for t in (xs, ss, prep_dense(ds["filt"]), y))

    tr = build_dataset(train_recs, a.data_dir, cfg, jitter=a.jitter, copies=a.copies,
                       want_dense=True, cache_dir=a.cache, seed=0, verbose=False)
    if a.svdb_dir:
        sv_recs = sorted(p.stem for p in Path(a.svdb_dir).glob("*.hea"))
        sv = build_dataset(sv_recs, a.svdb_dir, cfg, jitter=a.jitter, copies=1, want_dense=True,
                           cache_dir=a.cache, seed=1, verbose=False, lead="ECG1")
        print(f"extra training data: SVDB {len(sv['y'])} beats, classes "
              f"{np.bincount(sv['y'].astype(int), minlength=4).tolist()}")
        tr = {k: np.concatenate([tr[k], sv[k]]) for k in tr if k != "stats"}
    va = build_dataset(val_recs, a.data_dir, cfg, want_dense=True, cache_dir=a.cache, verbose=False)
    T = dict(tr=tens(tr), va=tens(va))
    if a.subsample and a.subsample < len(T["tr"][3]):
        idx = torch.randperm(len(T["tr"][3]), device=DEV)[: a.subsample]
        T["tr"] = tuple(t[idx] for t in T["tr"])
    te = None
    te_stats = None
    if a.final:
        tds = build_dataset(ds2, a.data_dir, cfg, want_dense=True, cache_dir=a.cache, verbose=False)
        te, te_stats = tens(tds), tds["stats"]

    runs = [run_seed(a, s, T, te) for s in a.seeds]
    split = "test" if a.final else "val"
    summ = summarize(runs, split)
    for model, rows in summ.items():
        print(f"\n== {model} [{split}] over seeds {a.seeds}")
        for k in ["accuracy", "macro_f1", "f1_N", "f1_S", "f1_V", "f1_F", "se_S", "se_V"]:
            print(f"  {k:9s} {rows[k][0]:.4f} +/- {rows[k][1]:.4f}")
    if len(runs) == 1:
        print(format_report(runs[0][split]["student_int8"], f"== int8 student seed {a.seeds[0]} [{split}]"))
    (out / f"train_{a.tag}.json").write_text(json.dumps(dict(
        args=vars(a), split=split, test_stats=te_stats, summary=summ, runs=runs), indent=2))
    print(f"saved {out / f'train_{a.tag}.json'}")


if __name__ == "__main__":
    main()
