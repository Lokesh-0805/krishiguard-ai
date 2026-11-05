"""Resumable experiment queue. Each stage is skipped when its output already exists, so the script can
be re-run after an interruption (or after datasets are placed) and continues where it stopped.

  python ml/training/run_experiments.py            # run all stages
  python ml/training/run_experiments.py --only resnet50
Runs strictly one GPU job at a time (8 GB unified memory).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
MODELS = ROOT / "ml/artifacts/models"
EXP = ROOT / "ml/experiments"


def sh(args: list[str]) -> None:
    print(">>", " ".join(args), flush=True)
    subprocess.run([PY, *args], check=True, cwd=ROOT)


def done(run: str) -> bool:
    return (MODELS / run / "history.json").exists()


def train(run, arch, *extra):
    if done(run):
        print(f"skip {run} (exists)", flush=True)
        return
    sh(["ml/training/train.py", "--arch", arch, "--run-name", run, *extra])


def best_params(arch):
    p = EXP / f"optuna_{arch}.json"
    bp = json.loads(p.read_text())["best_params"]
    return ["--lr", f"{bp['lr']:.6g}", "--backbone-lr-mult", f"{bp['backbone_lr_mult']:.6g}",
            "--weight-decay", f"{bp['weight_decay']:.6g}", "--label-smoothing", f"{bp['label_smoothing']:.4g}"]


def lr_probe(arch, lrs, samples=6000):
    """Short learning-rate comparison on a stratified subset (1 epoch); returns best head LR."""
    out = EXP / f"lr_probe_{arch}.json"
    if out.exists():
        return json.loads(out.read_text())["best_lr"]
    res = {}
    for lr in lrs:
        run = f"lrprobe_{arch}_{lr:g}"
        sh(["ml/training/train.py", "--arch", arch, "--run-name", run, "--epochs", "1", "--lr", str(lr),
            "--max-train-samples", str(samples), "--max-val-samples", "1500", "--log-every", "1000"])
        h = json.loads((MODELS / run / "history.json").read_text())
        res[str(lr)] = h["best"]
        for f in ("best.pt",):
            (MODELS / run / f).unlink(missing_ok=True)
    best = max(res, key=lambda k: res[k]["val_macro_f1"])
    out.write_text(json.dumps({"arch": arch, "samples": samples, "results": res, "best_lr": float(best)}, indent=2))
    return float(best)


STAGES = ["effb0_tune", "effb0", "resnet50", "vit_small", "effb0_domain", "effb2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=STAGES)
    a = ap.parse_args()
    for st in STAGES:
        if st not in a.only:
            continue
        print(f"===== STAGE {st} =====", flush=True)
        if st == "effb0_tune" and not (EXP / "optuna_efficientnet_b0.json").exists():
            sh(["ml/training/tune.py", "--arch", "efficientnet_b0", "--trials", "8"])
        elif st == "effb0":
            train("effnet_b0_224", "efficientnet_b0", "--img-size", "224", "--epochs", "5", "--patience", "2", *best_params("efficientnet_b0"))
        elif st == "resnet50":
            lr = lr_probe("resnet50", [1e-3, 3e-3])
            train("resnet50_224", "resnet50", "--img-size", "224", "--epochs", "3", "--patience", "2", "--lr", str(lr), "--backbone-lr-mult", "0.1")
        elif st == "vit_small":
            # standard ViT fine-tuning recipe: low backbone LR, strong weight decay
            train("vit_small_224", "vit_small_patch16_224.augreg_in21k_ft_in1k", "--img-size", "224", "--epochs", "3", "--patience", "2",
                  "--lr", "1e-3", "--backbone-lr-mult", "0.05", "--weight-decay", "0.05", "--batch-size", "32")
        elif st == "effb0_domain":
            # domain-adaptation experiment: continue from the PlantVillage model with PlantDoc domain_train mixed in
            if done("effnet_b0_224"):
                train("effnet_b0_224_domain", "efficientnet_b0", "--img-size", "224", "--epochs", "3", "--patience", "2",
                      "--init-from", str(MODELS / "effnet_b0_224/best.pt"), "--max-train-samples", "10000",
                      "--extra-train", "domain_train", "--extra-train-repeat", "3", "--lr", "3e-4", "--backbone-lr-mult", "0.3",
                      "--warmup-frac", "0.02", "--select-metric", "mean_macro_f1")
        elif st == "effb2":
            train("effnet_b2_256", "efficientnet_b2", "--img-size", "256", "--epochs", "2", "--patience", "1", *best_params("efficientnet_b0"))
    print("ALL_STAGES_DONE", flush=True)


if __name__ == "__main__":
    main()
