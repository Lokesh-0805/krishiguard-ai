"""Fine-tune a pretrained backbone on the leakage-free PlantVillage split.

Features: seeded, MLflow-tracked, cosine LR with warmup, differential LR (head x10), class-weighted
label-smoothed CE, early stopping on validation macro-F1, best/last checkpoints, curves.

Example:
  python ml/training/train.py --arch efficientnet_b0 --img-size 224 --epochs 6 --lr 1e-3 --run-name effb0
  python ml/training/train.py --arch resnet50 --init-from ml/artifacts/models/x/best.pt --extra-train domain_train ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ROOT,
    basic_metrics,
    build_model,
    eval_transform,
    get_device,
    load_classes,
    load_split,
    make_loader,
    predict_logits,
    seed_everything,
    train_transform,
)

MODELS_DIR = ROOT / "ml/artifacts/models"
MLRUNS = ROOT / "ml/experiments"  # mlflow.db (sqlite) + mlartifacts/


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="head LR; backbone uses lr * backbone_lr_mult")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-val-samples", type=int, default=None)
    ap.add_argument("--extra-train", nargs="*", default=[], help="additional split manifests to train on")
    ap.add_argument("--extra-train-repeat", type=int, default=1, help="oversampling factor for extra splits")
    ap.add_argument("--init-from", default=None, help="checkpoint to initialise weights from")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--experiment", default="krishiguard")
    ap.add_argument("--select-metric", default="val_macro_f1", choices=["val_macro_f1", "extval_macro_f1", "mean_macro_f1"],
                    help="checkpoint/early-stopping criterion; mean = average of PlantVillage-val and PlantDoc ext-val macro-F1")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    return ap.parse_args(argv)


def class_weights(labels: np.ndarray, n: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def smoothed_weighted_ce(logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Class-weighted, label-smoothed CE (same definition as F.cross_entropy(weight=, label_smoothing=)).
    Written with gather/sum only: torch's MPS path uses masked_select -> nonzero, which forces a GPU sync
    every step and was observed to hang under memory pressure."""
    logp = F.log_softmax(logits.float(), dim=1)
    w_y = weight[y]
    nll = -logp.gather(1, y[:, None]).squeeze(1) * w_y
    smooth = -(logp * weight[None, :]).sum(1) / logits.shape[1]
    return ((1 - eps) * nll + eps * smooth).sum() / w_y.sum()


def lr_lambda(total_steps: int, warmup: int):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    return f


def head_params(model):
    head = model.get_classifier()
    ids = {id(p) for p in head.parameters()}
    return [p for p in model.parameters() if id(p) in ids], [p for p in model.parameters() if id(p) not in ids]


def save_curves(history, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], "o-", label="validation")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend(frameon=False)
    axes[1].plot(ep, [h["train_acc"] for h in history], "o-", label="train acc (augmented)")
    axes[1].plot(ep, [h["val_accuracy"] for h in history], "o-", label="val acc")
    axes[1].plot(ep, [h["val_macro_f1"] for h in history], "s--", label="val macro-F1")
    if "extval_macro_f1" in history[0]:
        axes[1].plot(ep, [h["extval_macro_f1"] for h in history], "^:", label="ext-val macro-F1 (PlantDoc)")
    axes[1].set_title("Accuracy / F1"); axes[1].set_xlabel("epoch"); axes[1].legend(frameon=False, fontsize=7)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def run(args) -> dict:
    import mlflow

    seed_everything(args.seed)
    device = get_device()
    classes = load_classes()
    nc = len(classes)

    import pandas as pd

    train_df = load_split("train", args.max_train_samples, args.seed)
    for extra in args.extra_train:
        train_df = pd.concat([train_df] + [load_split(extra)] * args.extra_train_repeat, ignore_index=True)
    val_df = load_split("validation", args.max_val_samples, args.seed)
    extval_df = load_split("external_validation")

    tl = make_loader(train_df, train_transform(args.img_size), args.batch_size, True, args.workers, args.seed)
    # eval loaders are non-persistent: each worker holds ~300 MB (torch import) and RAM is 8 GB
    vl = make_loader(val_df, eval_transform(args.img_size), 64, False, 2, persistent=False)
    el = make_loader(extval_df, eval_transform(args.img_size), 64, False, 2, persistent=False)

    model = build_model(args.arch, nc, pretrained=args.init_from is None)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
    model.to(device)
    hp, bp = head_params(model)
    opt = torch.optim.AdamW(
        [{"params": bp, "lr": args.lr * args.backbone_lr_mult}, {"params": hp, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    total = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, int(args.warmup_frac * total)))
    cw = class_weights(train_df.label_idx.to_numpy(), nc).to(device)
    use_amp = device.type == "cuda"  # fp16 autocast on MPS measured ~15x slower (see EXPERIMENT_LOG)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = MODELS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    MLRUNS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS.resolve() / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(args.experiment) is None:
        mlflow.create_experiment(args.experiment, artifact_location=(MLRUNS / "mlartifacts" / args.experiment).resolve().as_uri())
    mlflow.set_experiment(args.experiment)

    history, best, bad_epochs = [], {args.select_metric: -1}, 0
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({k: (v if not isinstance(v, list) else ",".join(v)) for k, v in vars(args).items()})
        mlflow.log_params({"device": device.type, "n_train": len(train_df), "n_val": len(val_df), "amp": use_amp})
        step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, seen = time.time(), 0
            run_loss = torch.zeros((), device=device); run_correct = torch.zeros((), device=device)
            for x, y in tl:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(x)
                    loss = smoothed_weighted_ce(logits, y, cw, args.label_smoothing)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                run_loss += loss.detach() * len(y)
                run_correct += (logits.detach().argmax(1) == y).sum()
                seen += len(y)
                if step % args.log_every == 0:
                    ips = seen / (time.time() - t0)
                    print(f"[{args.run_name}] ep{epoch} step {step}/{total} loss {run_loss.item()/seen:.4f} acc {run_correct.item()/seen:.4f} {ips:.1f} img/s", flush=True)
            train_time = time.time() - t0
            run_loss, run_correct = run_loss.item(), run_correct.item()
            vlog, vy = predict_logits(model, vl, device)
            elog, ey = predict_logits(model, el, device)
            vm = basic_metrics(vy, vlog.argmax(1))
            em = basic_metrics(ey, elog.argmax(1))
            rec = {
                "epoch": epoch,
                "train_loss": run_loss / seen,
                "train_acc": run_correct / seen,
                "val_loss": float(F.cross_entropy(torch.from_numpy(vlog), torch.from_numpy(vy)).item()),
                **{f"val_{k}": v for k, v in vm.items()},
                **{f"extval_{k}": v for k, v in em.items()},
                "epoch_seconds": train_time,
                "lr_head": opt.param_groups[1]["lr"],
            }
            rec["mean_macro_f1"] = (rec["val_macro_f1"] + rec["extval_macro_f1"]) / 2
            history.append(rec)
            mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=epoch)
            print(f"[{args.run_name}] EPOCH {epoch} " + json.dumps({k: round(v, 4) for k, v in rec.items()}), flush=True)
            improved = rec[args.select_metric] > best[args.select_metric] + 1e-4
            ckpt = {
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "img_size": args.img_size,
                "classes": classes,
                "epoch": epoch,
                "metrics": rec,
                "config": vars(args),
            }
            if not args.no_save:
                torch.save(ckpt, out_dir / "last.pt")
            if improved:
                best, bad_epochs = rec, 0
                if not args.no_save:
                    torch.save(ckpt, out_dir / "best.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"[{args.run_name}] early stopping at epoch {epoch}", flush=True)
                    break
        summary = {"run_name": args.run_name, "arch": args.arch, "img_size": args.img_size, "best": best, "history": history, "config": vars(args)}
        (out_dir / "history.json").write_text(json.dumps(summary, indent=2))
        save_curves(history, out_dir / "curves.png")
        mlflow.log_artifact(str(out_dir / "curves.png"))
        mlflow.log_metrics({f"best_{k}": v for k, v in best.items() if isinstance(v, float)})
    if not args.no_save and (out_dir / "last.pt").exists():
        (out_dir / "last.pt").unlink()  # disk budget: keep only best
    return summary


if __name__ == "__main__":
    run(parse())
