"""Full evaluation of one trained run.

  python ml/evaluation/evaluate.py --run effnet_b0_224 [--device mps]

Outputs ml/artifacts/evaluation/<run>/: metrics.json, logits cache, confusion matrices, reliability and
risk-coverage plots, misclassification grid.

Protocol (no test-set peeking):
  * temperature T is fitted on the validation sets only (domain-balanced NLL: PlantVillage-val and
    PlantDoc external-validation weighted equally);
  * the "confident" threshold is selected on the same validation sets: the lowest confidence threshold whose
    accepted predictions reach >= TARGET_SELECTIVE_ACC on BOTH validation domains, falling back to the
    threshold maximising the balanced selective accuracy at >= 10% coverage;
  * the "unrecognized" threshold is the 2nd percentile of calibrated confidence over correctly classified
    validation images (below it, the model's own validation behaviour gives little support to any class);
  * test / external-test are evaluated once with these frozen values.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from common import RAW, ROOT, build_model, eval_transform, get_device, load_split, make_loader, predict_logits  # noqa: E402

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support  # noqa: E402

MODELS = ROOT / "ml/artifacts/models"
EVAL = ROOT / "ml/artifacts/evaluation"
TARGET_SELECTIVE_ACC = 0.90


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(logits, y, T):
    p = softmax(logits / T)
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()


def fit_temperature(sets):
    """Grid + golden refinement of T on the mean of per-domain NLLs."""
    grid = np.exp(np.linspace(np.log(0.3), np.log(10), 200))
    losses = [np.mean([nll(l, y, t) for l, y in sets]) for t in grid]
    t = grid[int(np.argmin(losses))]
    lo, hi = t / 1.05, t * 1.05
    for _ in range(40):
        a, b = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        if np.mean([nll(l, y, a) for l, y in sets]) < np.mean([nll(l, y, b) for l, y in sets]):
            hi = b
        else:
            lo = a
    return float((lo + hi) / 2)


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    total, out = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
            out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "acc": float(correct[m].mean()), "conf": float(conf[m].mean())})
    return float(total), out


def selective(conf, correct, thr):
    m = conf >= thr
    return {"threshold": float(thr), "coverage": float(m.mean()), "selective_accuracy": float(correct[m].mean()) if m.any() else None,
            "rejected_accuracy": float(correct[~m].mean()) if (~m).any() else None}


def risk_coverage(conf, correct):
    o = np.argsort(-conf)
    c = np.cumsum(correct[o]) / np.arange(1, len(o) + 1)
    cov = np.arange(1, len(o) + 1) / len(o)
    idx = np.unique(np.linspace(0, len(o) - 1, 60).astype(int))
    return {"coverage": cov[idx].round(4).tolist(), "selective_accuracy": c[idx].round(4).tolist(), "aurc": float(np.mean(1 - c))}


def choose_thresholds(val_sets):
    cands = np.unique(np.concatenate([c for c, _ in val_sets]))
    cands = np.concatenate([[0.0], np.quantile(cands, np.linspace(0, 0.995, 400))])
    best = None
    for t in np.sort(cands):
        accs = []
        for c, k in val_sets:
            m = c >= t
            accs.append(k[m].mean() if m.sum() >= 5 else np.nan)
        if all(np.isfinite(a) and a >= TARGET_SELECTIVE_ACC for a in accs):
            best = float(t)
            rule = f"lowest threshold with selective accuracy >= {TARGET_SELECTIVE_ACC:.2f} on every validation domain"
            break
    if best is None:
        scores = []
        for t in np.sort(cands):
            covs = [(c >= t).mean() for c, _ in val_sets]
            if min(covs) < 0.10:
                continue
            scores.append((np.mean([k[c >= t].mean() for c, k in val_sets]), t))
        best = float(max(scores)[1])
        rule = "target unreachable on some domain -> threshold maximising balanced selective accuracy at >=10% coverage"
    correct_conf = np.concatenate([c[k.astype(bool)] for c, k in val_sets])
    reject = float(min(best, np.quantile(correct_conf, 0.02)))
    return best, reject, rule


def plot_cm(cm, classes, path, title, normalize):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = cm.astype(float)
    if normalize:
        m = m / np.maximum(m.sum(1, keepdims=True), 1)
    short = [c.replace("_(including_sour)", "").replace("_(maize)", "").replace(",_bell", "").replace("___", ": ").replace("_", " ")[:34] for c in classes]
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.32 + 3), max(5, n * 0.32 + 2.5)))
    im = ax.imshow(m, cmap="Greens", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n), short, rotation=90, fontsize=6)
    ax.set_yticks(range(n), short, fontsize=6)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)
    if n <= 40:
        for i in range(n):
            for j in range(n):
                v = m[i, j]
                if v > (0.005 if normalize else 0):
                    ax.text(j, i, f"{v:.2f}" if normalize else int(v), ha="center", va="center", fontsize=4.2,
                            color="white" if v > (0.6 if normalize else m.max() * 0.6) else "black")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_reliability(bins_before, bins_after, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for b, lab, col in ((bins_before, "uncalibrated", "#c0392b"), (bins_after, "temperature-scaled", "#2f7d4f")):
        ax.plot([x["conf"] for x in b], [x["acc"] for x in b], "o-", label=lab, color=col, ms=3)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title(title, fontsize=9); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_risk_coverage(curves, path, thr_cov):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for name, rc in curves.items():
        ax.plot(rc["coverage"], rc["selective_accuracy"], label=name)
    for name, cov in thr_cov.items():
        ax.axvline(cov, ls=":", lw=0.8, color="grey")
    ax.set_xlabel("coverage (fraction of images answered)"); ax.set_ylabel("accuracy on answered images")
    ax.set_title("Risk-coverage (calibrated confidence)", fontsize=9); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def misclassified_grid(df, y, pred, conf, classes, path, n=16):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    wrong = np.nonzero(y != pred)[0]
    if len(wrong) == 0:
        return
    wrong = wrong[np.argsort(-conf[wrong])][:n]  # most confident mistakes first
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), wrong):
        ax.imshow(Image.open(RAW / df.path.iloc[i]).convert("RGB").resize((200, 200)))
        t = classes[y[i]].split("___")[1][:22]
        p = classes[pred[i]].split("___")
        ax.set_title(f"true: {classes[y[i]].split('___')[0][:10]} {t}\npred: {p[0][:10]} {p[1][:22]} ({conf[i]:.2f})", fontsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def latency(model, size, device, runs=40):
    out = {}
    x = torch.randn(1, 3, size, size)
    for dev in sorted({"cpu", device.type}):
        m = model.to(dev).eval()
        xx = x.to(dev)
        times = []
        with torch.no_grad():
            for i in range(runs + 5):
                if dev == "mps":
                    torch.mps.synchronize()
                t0 = time.perf_counter()
                m(xx)
                if dev == "mps":
                    torch.mps.synchronize()
                if i >= 5:
                    times.append((time.perf_counter() - t0) * 1000)
        out[dev] = {"batch1_ms_mean": float(np.mean(times)), "batch1_ms_p50": float(np.median(times)), "batch1_ms_p95": float(np.percentile(times, 95)),
                    "threads": torch.get_num_threads() if dev == "cpu" else None}
    model.to(device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: stratified subsample per split")
    a = ap.parse_args()
    device = torch.device(a.device) if a.device else get_device()
    ck_path = MODELS / a.run / "best.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    classes, size, arch = ck["classes"], ck["img_size"], ck["arch"]
    out = EVAL / a.run
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(arch, len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    splits = {k: load_split(k, a.limit) for k in ("validation", "external_validation", "test", "external_test")}
    logits = {}
    cache = out / "logits.npz"
    if cache.exists():
        z = np.load(cache)
        logits = {k: (z[f"{k}_logits"], z[f"{k}_y"]) for k in splits}
    else:
        for k, df in splits.items():
            t0 = time.time()
            logits[k] = predict_logits(model, make_loader(df, eval_transform(size), 64, False, a.workers, persistent=False), device)
            print(f"[{a.run}] {k}: {len(df)} images in {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(cache, **{f"{k}_logits": v[0] for k, v in logits.items()}, **{f"{k}_y": v[1] for k, v in logits.items()})

    T = fit_temperature([logits["validation"], logits["external_validation"]])

    def calib(k):
        lg, y = logits[k]
        p = softmax(lg / T)
        return p, p.max(1), p.argmax(1), y

    val_sets = []
    for k in ("validation", "external_validation"):
        p, c, pr, y = calib(k)
        val_sets.append((c, (pr == y).astype(float)))
    thr, reject_thr, rule = choose_thresholds(val_sets)

    crop_of = np.array([c.split("___")[0] for c in classes])
    results = {}
    curves, thr_cov = {}, {}
    ext_df = splits["external_test"]
    eval_sets = {
        "validation": (splits["validation"], None),
        "external_validation": (splits["external_validation"], None),
        "test": (splits["test"], None),
        "external_test_plantdoc": (ext_df, (ext_df.source == "plantdoc").to_numpy()),
        "external_test_tomato_pk_field": (ext_df, (ext_df.source == "tomato_pk_field").to_numpy()),
        "external_test_all": (ext_df, None),
    }
    for name, (df, mask) in eval_sets.items():
        base = "external_test" if name.startswith("external_test") else name
        p, conf, pred, y = calib(base)
        raw_conf = softmax(logits[base][0]).max(1)
        if mask is not None:
            p, conf, pred, y, raw_conf, df = p[mask], conf[mask], pred[mask], y[mask], raw_conf[mask], df[mask].reset_index(drop=True)
        correct = (pred == y).astype(float)
        present = np.unique(y)
        P, R, F, S = precision_recall_fscore_support(y, pred, labels=present, zero_division=0)
        ece_b, bins_b = ece(raw_conf, correct)
        ece_a, bins_a = ece(conf, correct)
        sel = selective(conf, correct, thr)
        rej = float((conf < reject_thr).mean())
        res = {
            "n": int(len(y)),
            "classes_present": int(len(present)),
            "accuracy": float(correct.mean()),
            "macro_precision": float(P.mean()),
            "macro_recall": float(R.mean()),
            "macro_f1": float(F.mean()),
            "weighted_f1": float(np.average(F, weights=S)),
            "crop_accuracy": float((crop_of[pred] == crop_of[y]).mean()),
            "top3_accuracy": float(np.mean([y[i] in np.argsort(-p[i])[:3] for i in range(len(y))])),
            "ece_uncalibrated": ece_b,
            "ece_calibrated": ece_a,
            "selective_at_threshold": sel,
            "unrecognized_rate": rej,
            "mean_confidence": float(conf.mean()),
            "per_class": {
                classes[c]: {"precision": float(P[i]), "recall": float(R[i]), "f1": float(F[i]), "support": int(S[i])}
                for i, c in enumerate(present)
            },
        }
        cm = confusion_matrix(y, pred, labels=range(len(classes)))
        res["confusion_matrix"] = cm.tolist()
        # confused pairs
        off = cm.copy(); np.fill_diagonal(off, 0)
        pairs = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:10], off.shape))[0]
        res["top_confusions"] = [{"true": classes[i], "pred": classes[j], "count": int(off[i, j])} for i, j in pairs if off[i, j] > 0]
        results[name] = res
        if name in ("test", "external_test_plantdoc", "external_test_tomato_pk_field"):
            if name == "test":
                plot_cm(cm, classes, out / "confusion_matrix_test.png", f"{a.run} — PlantVillage test (counts)", False)
                plot_cm(cm, classes, out / "confusion_matrix_test_normalized.png", f"{a.run} — PlantVillage test (row-normalised)", True)
            else:
                rows = sorted(set(present) | set(np.unique(pred)))
                sub = cm[np.ix_(rows, rows)]
                plot_cm(sub, [classes[i] for i in rows], out / f"confusion_matrix_{name}_normalized.png", f"{a.run} — {name} (row-normalised)", True)
            plot_reliability(bins_b, bins_a, out / f"reliability_{name}.png", f"Reliability — {name}")
            misclassified_grid(df, y, pred, conf, classes, out / f"misclassified_{name}.png")
            report = classification_report(y, pred, labels=present, target_names=[classes[c] for c in present], zero_division=0, digits=4)
            (out / f"classification_report_{name}.txt").write_text(report)
        curves[name] = risk_coverage(conf, correct)
        thr_cov[name] = sel["coverage"]
    plot_risk_coverage({k: v for k, v in curves.items() if k in ("test", "external_test_plantdoc", "external_test_tomato_pk_field")}, out / "risk_coverage.png", thr_cov)

    size_mb = ck_path.stat().st_size / 1e6
    params = sum(p.numel() for p in model.parameters()) / 1e6
    lat = latency(model, size, device)
    hist = json.loads((MODELS / a.run / "history.json").read_text())
    summary = {
        "run": a.run,
        "arch": arch,
        "img_size": size,
        "trained_on": "PlantVillage train" + (" + PlantDoc domain_train" if "domain_train" in hist["config"].get("extra_train", []) else ""),
        "epochs_trained": len(hist["history"]),
        "train_seconds": float(sum(h["epoch_seconds"] for h in hist["history"])),
        "best_epoch": hist["best"]["epoch"],
        "parameters_m": params,
        "checkpoint_mb": size_mb,
        "state_dict_fp32_mb": params * 4,  # params in millions x 4 bytes
        "temperature": T,
        "confidence_threshold": thr,
        "reject_threshold": reject_thr,
        "threshold_rule": rule,
        "latency": lat,
        "results": results,
        "risk_coverage": curves,
        "history": hist["history"],
        "config": hist["config"],
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    brief = {k: {m: round(v[m], 4) for m in ("accuracy", "macro_f1", "crop_accuracy")} for k, v in results.items()}
    print(f"[{a.run}] T={T:.3f} thr={thr:.3f} reject={reject_thr:.3f} lat={lat}", flush=True)
    print(json.dumps(brief, indent=1), flush=True)


if __name__ == "__main__":
    main()
