"""Optuna search over fine-tuning hyper-parameters on a stratified subset (budget-limited proxy task).

Each trial fine-tunes from ImageNet weights for 1 epoch on --train-samples images and is scored on a
stratified validation subset (macro-F1). Best params are written to ml/experiments/optuna_<arch>.json.
The proxy measures early convergence, not final accuracy; this limitation is documented in
docs/ML_METHODOLOGY.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402
import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--train-samples", type=int, default=4000)
    ap.add_argument("--val-samples", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 3e-4, 5e-3, log=True)
        mult = trial.suggest_float("backbone_lr_mult", 0.03, 0.5, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
        ls = trial.suggest_float("label_smoothing", 0.0, 0.15)
        args = train.parse([
            "--arch", a.arch, "--img-size", str(a.img_size), "--epochs", "1",
            "--lr", str(lr), "--backbone-lr-mult", str(mult), "--weight-decay", str(wd),
            "--label-smoothing", str(ls), "--max-train-samples", str(a.train_samples),
            "--max-val-samples", str(a.val_samples), "--seed", str(a.seed), "--no-save",
            "--run-name", f"optuna_{a.arch}_t{trial.number}", "--experiment", "krishiguard-optuna",
            "--patience", "1", "--log-every", "1000",
        ])
        summary = train.run(args)
        trial.set_user_attr("val_accuracy", summary["best"]["val_accuracy"])
        trial.set_user_attr("extval_macro_f1", summary["best"]["extval_macro_f1"])
        return summary["best"]["val_macro_f1"]

    storage = f"sqlite:///{ROOT / 'ml/experiments/optuna.db'}"
    study = optuna.create_study(
        study_name=f"{a.arch}_finetune", storage=storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=a.seed, n_startup_trials=4),
    )
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    study.optimize(objective, n_trials=max(0, a.trials - done))
    res = {
        "arch": a.arch,
        "best_value_val_macro_f1": study.best_value,
        "best_params": study.best_params,
        "proxy": {"train_samples": a.train_samples, "val_samples": a.val_samples, "epochs": 1},
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params, **t.user_attrs}
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ],
    }
    out = ROOT / f"ml/experiments/optuna_{a.arch}.json"
    out.write_text(json.dumps(res, indent=2))
    print("OPTUNA_BEST", json.dumps(res["best_params"]), res["best_value_val_macro_f1"], flush=True)


if __name__ == "__main__":
    main()
