"""Compare evaluated runs, select the deployed model using VALIDATION data only, export serving artefacts.

  python ml/evaluation/compare_and_export.py [--force-run NAME]

Selection rule: maximise mean(PlantVillage-validation macro-F1, PlantDoc external-validation macro-F1);
runs within 0.005 of the best are tie-broken by CPU batch-1 latency. Test sets are not used for selection.

Writes:
  ml/artifacts/evaluation/summary.json      (consumed by GET /api/metrics and the dashboard)
  ml/artifacts/evaluation/model_comparison.png, training_curves.png, gradcam_examples.png
  ml/artifacts/serving/model.pt + model_card.json  (consumed by the backend)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml/training"))
sys.path.insert(0, str(ROOT / "backend"))
from common import RAW, build_model, load_split  # noqa: E402

EVAL = ROOT / "ml/artifacts/evaluation"
MODELS = ROOT / "ml/artifacts/models"
SERVING = ROOT / "ml/artifacts/serving"
META = ROOT / "ml/datasets/metadata"
GRADCAM_LAYER = {"resnet": "layer4", "efficientnet": "bn2", "vit": "blocks.-1.norm1"}

PRETTY = {
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b2": "EfficientNet-B2",
    "resnet50": "ResNet-50",
    "vit_small_patch16_224.augreg_in21k_ft_in1k": "ViT-S/16 (IN-21k)",
}


def headline(r):
    keep = ("n", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "crop_accuracy", "top3_accuracy",
            "ece_uncalibrated", "ece_calibrated", "selective_at_threshold", "unrecognized_rate")
    return {k: r[k] for k in keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-run", default=None, help="override automatic selection (documented in summary)")
    a = ap.parse_args()
    runs = {}
    for p in sorted(EVAL.glob("*/metrics.json")):
        runs[p.parent.name] = json.loads(p.read_text())
    if not runs:
        sys.exit("no evaluated runs found")

    scores = {k: (v["results"]["validation"]["macro_f1"] + v["results"]["external_validation"]["macro_f1"]) / 2 for k, v in runs.items()}
    best_score = max(scores.values())
    near = [k for k, s in scores.items() if s >= best_score - 0.005]
    selected = min(near, key=lambda k: runs[k]["latency"]["cpu"]["batch1_ms_mean"])
    rule = ("argmax mean(validation macro-F1, external-validation macro-F1); ties within 0.005 broken by CPU latency. "
            "Test and external-test sets were not used for selection.")
    if a.force_run:
        selected, rule = a.force_run, f"manually forced: {a.force_run} (automatic choice would follow: {rule})"
    sel = runs[selected]

    # ------------------------------------------------------------- charts
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(runs)
    sets = [("test", "PlantVillage test"), ("external_test_plantdoc", "PlantDoc test (field)"), ("external_test_tomato_pk_field", "Tomato field (PK)")]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.8 / len(sets)
    for i, (k, lab) in enumerate(sets):
        vals = [runs[n]["results"][k]["macro_f1"] for n in names]
        bars = ax.bar(np.arange(len(names)) + i * w, vals, w, label=lab, color=["#2f7d4f", "#d9a441", "#4a6fa5"][i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(names)) + w, names, fontsize=8)
    ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1.08); ax.legend(frameon=False, fontsize=7, ncols=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False); ax.set_title("Internal vs external generalisation", fontsize=10)
    fig.tight_layout(); fig.savefig(EVAL / "model_comparison.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for n in names:
        h = runs[n]["history"]
        axes[0].plot([x["epoch"] for x in h], [x["val_loss"] for x in h], "o-", label=n)
        axes[1].plot([x["epoch"] for x in h], [x["val_macro_f1"] for x in h], "o-", label=n)
    axes[0].set_title("validation loss"); axes[1].set_title("validation macro-F1")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.legend(frameon=False, fontsize=7); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(EVAL / "training_curves.png", dpi=140); plt.close(fig)

    # ------------------------------------------------------------- export serving artefacts
    from app.ml.inference import blur_score  # same function the API uses

    ck = torch.load(MODELS / selected / "best.pt", map_location="cpu", weights_only=False)
    SERVING.mkdir(parents=True, exist_ok=True)
    torch.save(ck["state_dict"], SERVING / "model.pt")
    sha = hashlib.sha256((SERVING / "model.pt").read_bytes()).hexdigest()
    family = next(k for k in GRADCAM_LAYER if ck["arch"].startswith(k))
    # blur warning threshold: 1st percentile of sharpness over the validation images (both domains)
    from PIL import Image

    vdf = pd.concat([load_split("validation", 1500, 0), load_split("external_validation")])
    blur = np.array([blur_score(Image.open(RAW / p).convert("RGB")) for p in vdf.path])
    blur_thr = float(np.percentile(blur, 1))
    card = {
        "model_version": f"{selected}@{sha[:8]}",
        "run_name": selected,
        "arch": ck["arch"],
        "img_size": ck["img_size"],
        "classes": ck["classes"],
        "temperature": sel["temperature"],
        "confidence_threshold": sel["confidence_threshold"],
        "reject_threshold": sel["reject_threshold"],
        "blur_threshold": round(blur_thr, 2),
        "checkpoint": "model.pt",
        "checkpoint_sha256": sha,
        "gradcam_target_layer": GRADCAM_LAYER[family],
        "trained_on": sel["trained_on"],
        "parameters_m": round(sel["parameters_m"], 2),
        "size_mb": round((SERVING / "model.pt").stat().st_size / 1e6, 1),
        "selected_because": rule,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (SERVING / "model_card.json").write_text(json.dumps(card, indent=2))

    gradcam_examples(ck, card)

    stats = json.loads((META / "dataset_statistics.json").read_text())
    summary = {
        "generated_at": card["created_at"],
        "selection": {"rule": rule, "selected_run": selected, "validation_scores": scores},
        "serving": {k: card[k] for k in ("model_version", "arch", "img_size", "temperature", "confidence_threshold", "reject_threshold", "blur_threshold", "size_mb", "parameters_m")},
        "models": [
            {
                "run": n,
                "arch": r["arch"],
                "display_name": PRETTY.get(r["arch"], r["arch"]) + (" + PlantDoc adaptation" if "domain" in n else ""),
                "img_size": r["img_size"],
                "trained_on": r["trained_on"],
                "parameters_m": r["parameters_m"],
                "checkpoint_mb": r["checkpoint_mb"],
                "epochs_trained": r["epochs_trained"],
                "best_epoch": r["best_epoch"],
                "train_minutes": r["train_seconds"] / 60,
                "latency": r["latency"],
                "temperature": r["temperature"],
                "confidence_threshold": r["confidence_threshold"],
                "results": {k: headline(v) for k, v in r["results"].items()},
                "history": r["history"],
                "selected": n == selected,
            }
            for n, r in runs.items()
        ],
        "selected_detail": {
            "per_class": {k: sel["results"][k]["per_class"] for k in ("test", "external_test_plantdoc", "external_test_tomato_pk_field")},
            "confusion_matrix_test": sel["results"]["test"]["confusion_matrix"],
            "top_confusions": {k: sel["results"][k]["top_confusions"] for k in ("test", "external_test_plantdoc", "external_test_tomato_pk_field")},
            "risk_coverage": sel["risk_coverage"],
            "classes": ck["classes"],
        },
        "dataset": {
            "splits": {k: v["images"] for k, v in stats["splits"].items()},
            "plantvillage_images": stats["plantvillage"]["images"],
            "num_classes": stats["plantvillage"]["classes"],
            "num_crops": stats["plantvillage"]["crops"],
            "leakage_groups": stats["plantvillage"]["leakage_groups"],
            "external_test_by_source": {k: v["images"] for k, v in stats["external_test_by_source"].items()},
        },
    }
    (EVAL / "summary.json").write_text(json.dumps(summary, indent=1))
    print("SELECTED", selected, json.dumps({k: round(v, 4) for k, v in scores.items()}))
    print(json.dumps({k: card[k] for k in ("model_version", "temperature", "confidence_threshold", "reject_threshold", "blur_threshold")}))


def gradcam_examples(ck, card):
    """Grad-CAM on a fixed set of test + field images using the API's own implementation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from app.ml.gradcam import GradCAM, render
    from app.ml.inference import model_view, to_array

    model = build_model(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"]); model.eval()
    cam = GradCAM(model, card["gradcam_target_layer"], getattr(model, "num_prefix_tokens", 0))
    rng = np.random.default_rng(3)
    test = load_split("test"); ext = load_split("external_test")
    picks = list(test.sample(4, random_state=1).path) + list(ext[ext.source == "plantdoc"].sample(2, random_state=1).path) + \
        list(ext[ext.source == "tomato_pk_field"].sample(2, random_state=1).path)
    fig, axes = plt.subplots(len(picks), 3, figsize=(6.6, 2.3 * len(picks)))
    for r, p in enumerate(picks):
        img = Image.open(RAW / p).convert("RGB")
        view = model_view(img, card["img_size"])
        x = torch.from_numpy(to_array(view))[None]
        with torch.no_grad():
            pr = torch.softmax(model(x) / card["temperature"], 1)[0]
        k = int(pr.argmax())
        heat, overlay = render(cam(x, k), view)
        import io

        for c, (im, t) in enumerate([(view, "model input"), (Image.open(io.BytesIO(heat)), "Grad-CAM"), (Image.open(io.BytesIO(overlay)), "overlay")]):
            axes[r, c].imshow(im); axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_title(f"{Path(p).parts[0]}: true {Path(p).parent.name[:26]}", fontsize=6, loc="left")
            if c == 2:
                axes[r, c].set_title(f"pred {ck['classes'][k].split('___')[1][:24]} ({pr[k]:.2f})", fontsize=6)
    fig.tight_layout(); fig.savefig(EVAL / "gradcam_examples.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
