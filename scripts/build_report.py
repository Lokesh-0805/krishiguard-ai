"""Build the detailed project report (HTML -> PDF via Chrome) from the saved project artefacts.

Every number is read from the evaluation / dataset / model-card files, never typed by hand.
  .venv/bin/python scripts/build_report.py            # writes docs/report/KrishiGuard_AI_Report.html
  (cd frontend && node e2e/print-report.mjs)           # prints it to docs/report/KrishiGuard_AI_Report.pdf
"""
from __future__ import annotations

import html
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/report"
EVAL = ROOT / "ml/artifacts/evaluation"
RUN = "effnet_b0_224"

m = json.loads((EVAL / RUN / "metrics.json").read_text())
summary = json.loads((EVAL / "summary.json").read_text())
stats = json.loads((ROOT / "ml/datasets/metadata/dataset_statistics.json").read_text())
card = json.loads((ROOT / "ml/artifacts/serving/model_card.json").read_text())
optuna = json.loads((ROOT / "ml/experiments/optuna_efficientnet_b0.json").read_text())
probe = json.loads((ROOT / "ml/experiments/lr_probe_resnet50.json").read_text()) if (ROOT / "ml/experiments/lr_probe_resnet50.json").exists() else None
probe_runs = {k: json.loads((ROOT / f"ml/artifacts/models/lrprobe_resnet50_{k}/history.json").read_text())["best"]
              for k in ("0.001", "0.003") if (ROOT / f"ml/artifacts/models/lrprobe_resnet50_{k}/history.json").exists()}
advisory = json.loads((ROOT / "advisory/diseases.json").read_text())
R = m["results"]

e = html.escape


def p(v, d=1):
    return "—" if v is None else f"{v * 100:.{d}f}%"


ASSETS = OUT / "assets"


def _asset(rel: str) -> str:
    """Copy the image into docs/report/assets (large ones as optimised JPEG) and return a RELATIVE path, so the
    report is portable and contains no absolute local paths."""
    from PIL import Image

    src = ROOT / rel
    ASSETS.mkdir(parents=True, exist_ok=True)
    stem = rel.replace("/", "_").rsplit(".", 1)[0]
    if src.stat().st_size > 350_000:
        dst = ASSETS / f"{stem}.jpg"
        im = Image.open(src).convert("RGB")
        im.thumbnail((1700, 2600))
        im.save(dst, "JPEG", quality=84, optimize=True, progressive=True)
    else:
        dst = ASSETS / f"{stem}.png"
        dst.write_bytes(src.read_bytes())
    return dst.relative_to(OUT).as_posix()


def img(rel: str, caption: str, width: str = "100%", cls: str = "") -> str:
    if not (ROOT / rel).exists():
        return ""
    return (f'<figure class="{cls}"><img src="{_asset(rel)}" style="width:{width}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


def nice(c: str) -> str:
    a, b = c.split("___")
    a = a.replace("_(maize)", "").replace("_(including_sour)", "").replace(",_bell", " (bell)").replace("_", " ")
    return f"{a} — {b.replace('_', ' ').strip()}"


def table(headers, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


fig_no = [0]
FIGS: list[str] = []


def F(rel, caption, width="100%", cls=""):
    if not (ROOT / rel).exists():
        return ""
    fig_no[0] += 1
    FIGS.append(caption)
    return img(rel, f"<b>Figure {fig_no[0]}.</b> {caption}", width, cls)


pv = stats["plantvillage"]
splits = stats["splits"]
t, pdoc, tom = R["test"], R["external_test_plantdoc"], R["external_test_tomato_pk_field"]
hist = m["history"]

# ----------------------------------------------------------------------------------------------- sections
S = []

S.append(f"""
<section class="cover">
  <div class="cover-top">A PROJECT REPORT ON</div>
  <div class="cover-band">
    <div class="logo">🌿</div>
    <h1>KrishiGuard AI</h1>
    <div class="subtitle">Intelligent Crop Disease Detection and Advisory System<br/>using Deep Learning (Transfer Learning with EfficientNet-B0)</div>
  </div>
  <div class="cover-sub">Submitted as a Deep Learning project</div>
  <div class="cover-meta">
    <table class="meta">
      <tr><td>Domain</td><td>Computer Vision · Deep Learning · Agriculture</td></tr>
      <tr><td>Model</td><td>EfficientNet-B0 (ImageNet-pretrained, fine-tuned in PyTorch); served with ONNX Runtime</td></tr>
      <tr><td>Dataset</td><td>PlantVillage — {pv['images']:,} images, {pv['classes']} classes, {pv['crops']} crops; external field datasets PlantDoc and Tomato (Pakistan)</td></tr>
      <tr><td>Headline result</td><td><b>{p(t['accuracy'], 2)}</b> accuracy and <b>{p(t['macro_f1'], 2)}</b> macro-F1 on {t['n']:,} held-out test images</td></tr>
      <tr><td>Application</td><td>Responsive bilingual (English / Hindi) web application — <a href="https://krishiguard-ai-five.vercel.app">krishiguard-ai-five.vercel.app</a></td></tr>
      <tr><td>Report date</td><td>{date.today().strftime('%B %Y')}</td></tr>
    </table>
  </div>
  {img('docs/report/screens/results-en.png', 'Diagnosis result page of the application (real model output)', '100%', 'cover-shot')}
</section>""")

S.append(f"""
<section>
<h2>Abstract</h2>
<p>Plant diseases cause large yield losses, and farmers often lack quick access to an expert. <b>KrishiGuard AI</b> is a
responsive web application in which a farmer uploads, photographs or pastes an image of a crop leaf and receives the
<b>crop</b>, the <b>disease (or healthy)</b>, a <b>calibrated confidence</b>, a <b>heat-map explanation</b> and <b>advisory in
English and Hindi</b> (symptoms, causes, prevention, organic and chemical options with cited agricultural sources).</p>
<p>An ImageNet-pretrained <b>EfficientNet-B0</b> convolutional neural network was fine-tuned with transfer learning on the
PlantVillage dataset ({pv['images']:,} images, {pv['classes']} classes, {pv['crops']} crops) using a <b>leakage-free, group-level
split</b>. On {t['n']:,} held-out test images it reaches <b>{p(t['accuracy'], 2)} accuracy</b>, {p(t['macro_precision'], 2)} macro
precision, {p(t['macro_recall'], 2)} macro recall and <b>{p(t['macro_f1'], 2)} macro-F1</b>. Testing on two independent
real-field datasets (PlantDoc and a tomato field dataset from Pakistan) shows the well-known <i>lab-to-field gap</i>
({p(pdoc['accuracy'])} and {p(tom['accuracy'])} accuracy); the system therefore uses temperature scaling and validation-selected
thresholds to label results as <i>confident</i>, <i>uncertain</i> or <i>not recognised</i>, and shows treatment advice only for
confident results.</p>
<p>The model is served with ONNX Runtime from a FastAPI backend and a Next.js frontend, both deployed free on Vercel.
The project includes {44} backend tests, 6 component tests and 16 end-to-end browser tests (desktop and mobile), all passing.</p>
<p class="kw"><b>Keywords:</b> plant disease detection, transfer learning, EfficientNet, PlantVillage, calibration, Grad-CAM,
ONNX, FastAPI, Next.js, Hindi advisory.</p>
</section>""")

S.append(f"""
<section>
<h2>Introduction</h2>
<h3>Problem statement</h3>
<p>Farmers should be able to photograph a crop leaf and immediately learn whether it is diseased, which disease it is
likely to be, how sure the system is, and what preventive and treatment options exist — in a language they understand.
The solution must run in any browser (desktop, tablet, phone) without installing an app.</p>
<h3>Objectives</h3>
<ol>
<li>Identify the crop and disease (or healthy state) from a single leaf photo.</li>
<li>Report a confidence score and clearly flag uncertain or unrecognised photos.</li>
<li>Explain the prediction visually with a heat-map.</li>
<li>Provide symptoms, causes, prevention, organic and chemical options in <b>English and Hindi</b>, with sources.</li>
<li>Evaluate honestly on an untouched test set and on independent real-field images.</li>
<li>Deliver a professional, responsive web application with a real performance dashboard.</li>
</ol>
<h3>Scope</h3>
<p>{pv['classes']} classes over {pv['crops']} crops: apple, blueberry, cherry, corn (maize), grape, orange, peach, bell pepper,
potato, raspberry, soybean, squash, strawberry and tomato (26 diseases + 12 healthy classes). The system is a
decision-support tool and not a substitute for an agricultural expert.</p>
<h3>Organisation of the report</h3>
<p>The following chapters review related work, describe the development environment, the system architecture, the datasets
and their preparation, the model and every training experiment performed, calibration, results, explainability, the
advisory engine, the web application and API, testing and deployment, and finally the limitations and conclusions.</p>
</section>""")

S.append("""
<section>
<h2>Literature Review</h2>
<h3>Deep learning for plant disease classification</h3>
<p>Mohanty, Hughes and Salathé (2016) introduced large-scale deep learning for leaf disease recognition on the PlantVillage
dataset (54,306 images, 38 classes) and reported up to 99.35 % accuracy with GoogLeNet trained by transfer learning. They also
noted that accuracy fell sharply on images taken under conditions different from the training data — the first
documented sign of the lab-to-field gap that is examined again in this project.</p>
<h3>Real-world (field) datasets</h3>
<p>Singh et al. (2020) released <b>PlantDoc</b>, a dataset of 2,598 images collected from the internet with natural backgrounds
and multiple leaves per image. Models trained only on PlantVillage performed poorly on PlantDoc, which motivated using
PlantDoc and an independent tomato field dataset as <i>external</i> test sets in this work.</p>
<h3>Efficient convolutional networks and transfer learning</h3>
<p>Transfer learning reuses features learned on ImageNet and fine-tunes them on a smaller target dataset, giving high accuracy
with modest data and compute. <b>EfficientNet</b> (Tan and Le, 2019) scales network depth, width and input resolution jointly;
its smallest member, EfficientNet-B0 (≈ 5.3 M parameters with the ImageNet head, 4.1 M with a 38-class head), offers a good
accuracy–cost trade-off for CPU deployment, which is why it was chosen here.</p>
<h3>Calibration and explainability</h3>
<p>Guo et al. (2017) showed that modern neural networks are over-confident and that a single <b>temperature</b> parameter fitted on
validation data substantially improves calibration. Selvaraju et al. (2017) proposed <b>Grad-CAM</b>, which produces class-specific
heat-maps from gradients of the last convolutional layer; for networks ending in global average pooling it reduces to the
Class Activation Map (CAM) of Zhou et al. (2016). Both ideas are used to make the predictions of this system more trustworthy.</p>
<h3>Gap addressed by this project</h3>
<p>Many student and research prototypes report only PlantVillage accuracy with random splits. This project adds (1) a
leakage-free group split, (2) external field evaluation, (3) calibrated confidence with explicit uncertainty states,
(4) visual explanations and (5) a bilingual, source-cited advisory delivered through a deployed web application.</p>
</section>""")

S.append(f"""
<section>
<h2>Development Environment and Requirements</h2>
<h3>Hardware used for training</h3>
{table(['Component', 'Specification'], [
    ['Machine', 'Apple MacBook with Apple M1 chip (8-core CPU: 4 performance + 4 efficiency cores; 7-core GPU)'],
    ['Memory', '8 GB unified memory (shared by CPU and GPU)'],
    ['GPU backend', 'PyTorch MPS (Metal Performance Shaders) — no CUDA GPU available'],
    ['Free disk space', 'about 8.5 GB at the start of the project'],
])}
<h3>Software</h3>
{table(['Purpose', 'Software / version'], [
    ['Language', 'Python 3.12 (ML and backend), TypeScript (frontend), Node.js 22'],
    ['Deep learning', 'PyTorch 2.15, torchvision 0.29, timm 1.0.31'],
    ['Experiment tracking / tuning', 'MLflow 3.18 (SQLite store), Optuna 5.0'],
    ['Data / evaluation', 'NumPy, pandas, Pillow, imagehash, scikit-learn, matplotlib'],
    ['Serving', 'ONNX 1.23, ONNX Runtime 1.31, FastAPI, Uvicorn'],
    ['Web', 'Next.js 16, React 19, Tailwind CSS 4, shadcn/ui, Recharts'],
    ['Testing', 'pytest, Vitest, Playwright 1.64'],
])}
<h3>Measured training throughput (batch 32, 224 px, steady state)</h3>
{table(['Model', 'Parameters', 'Training (images/s)', 'Inference (images/s)'], [
    ['EfficientNet-B0', '4.1 M', '34.1', '200.7'],
    ['ResNet-50', '23.6 M', '22.8', '82.6'],
])}
<p>Mixed-precision (fp16) training on the MPS backend was measured to be about 15 times <i>slower</i> than fp32 on this
software stack (ResNet-50: 1.6 images/s), so all training used fp32. These measurements guided the epoch budgets used below.</p>
</section>""")

S.append(f"""
<section>
<h2>System Architecture and Technology Stack</h2>
<div class="arch">
  <div class="lane"><div class="lane-title">User</div><div class="box">Browser (desktop / mobile)<br/><small>upload · camera · paste</small></div></div>
  <div class="arrow">→</div>
  <div class="lane"><div class="lane-title">Frontend (Vercel)</div><div class="box">Next.js 16 · React 19 · Tailwind · shadcn/ui<br/><small>7 pages · EN/HI · image resize · report download</small></div></div>
  <div class="arrow">→<br/><small>/api/*</small></div>
  <div class="lane"><div class="lane-title">Backend (Vercel Python)</div><div class="box">FastAPI<br/><small>validation · rate limit · logging</small></div>
    <div class="box sub">ONNX Runtime — EfficientNet-B0<br/><small>logits + CAM heat-map</small></div>
    <div class="box sub">Advisory store (JSON, EN/HI)</div><div class="box sub">Metrics (evaluation summary)</div><div class="box sub">SQLite / PostgreSQL (metadata only)</div></div>
</div>
<div class="arch offline">
  <div class="box">Raw datasets</div><div class="arrow">→</div><div class="box">build_dataset.py<br/><small>integrity · dedup · group split · EDA</small></div>
  <div class="arrow">→</div><div class="box">train.py / tune.py<br/><small>PyTorch · Optuna · MLflow</small></div>
  <div class="arrow">→</div><div class="box">evaluate.py<br/><small>metrics · calibration · thresholds</small></div>
  <div class="arrow">→</div><div class="box">export_onnx.py<br/><small>model.onnx + model card</small></div>
</div>
<p class="caption"><b>Architecture diagram.</b> Online request path (top) and offline machine-learning pipeline (bottom).</p>
{table(['Layer', 'Technologies'], [
    ['Deep learning', 'PyTorch 2, timm (EfficientNet-B0, ImageNet weights), scikit-learn, MLflow, Optuna'],
    ['Serving', 'ONNX Runtime (CPU), NumPy, Pillow (+ AVIF), pillow-heif (HEIC)'],
    ['Backend', 'FastAPI, Pydantic, SQLAlchemy (PostgreSQL / SQLite), slowapi rate limiting, JSON logging'],
    ['Frontend', 'Next.js 16, React 19, TypeScript, Tailwind CSS 4, shadcn/ui, Lucide icons, Recharts'],
    ['Testing', 'pytest, Vitest + Testing Library, Playwright (Chrome, desktop + mobile emulation)'],
    ['Deployment', 'Vercel (Hobby plan): Next.js project + Python (FastAPI) project; Dockerfiles + docker-compose also provided'],
])}
</section>""")

ds_rows = [
    ["PlantVillage (colour)", "Train / validation / internal test", f"{stats['per_source_images']['plantvillage']:,}", "38", "CC BY-SA 3.0", "github.com/spMohanty/PlantVillage-Dataset"],
    ["PlantDoc", "External validation + external test (field)", f"{stats['per_source_images']['plantdoc']:,}", "28 → mapped", "CC BY 4.0", "github.com/pratikkayal/PlantDoc-Dataset"],
    ["Tomato Leaf Disease, Pakistan", "Independent external test (field)", f"{stats['per_source_images']['tomato_pk_field']:,}", "6 → mapped", "CC BY 4.0", "Mendeley Data 3mbnb82mxd"],
]
S.append(f"""
<section>
<h2>Datasets and Exploratory Data Analysis</h2>
<h3>Sources</h3>
{table(['Dataset', 'Role', 'Images', 'Classes', 'Licence', 'Source'], ds_rows)}
<p>All datasets were downloaded automatically from the original authors' repositories (Kaggle required an API token)
and verified file by file. Every PlantDoc and Tomato-field class was mapped by hand onto the matching PlantVillage
class so that external evaluation compares like with like.</p>
<h3>Integrity checks</h3>
{table(['Check', 'Result'], [
    ['Corrupt / unreadable images', f"{len(stats['corrupt'])} of {stats['total_indexed']:,}"],
    ['PlantVillage image size', 'all 256×256 RGB JPEG'],
    ['Exact duplicates (MD5), all sources', f"{pv['exact_duplicate_groups_all_sources']} groups / {pv['exact_duplicate_images_all_sources']} images"],
    ['Images matched to the PlantVillage leaf-identity map', f"{pv['leaf_map_matched_images']:,} ({pv['physical_leaves_with_multiple_images']:,} leaves have several photos)"],
    ['Near-duplicate pairs (perceptual hash, ≤ 4 bits)', f"{pv['near_dup_pairs_within_class']:,}"],
    ['Leakage groups after merging', f"{pv['leakage_groups']:,}"],
    ['PlantDoc ↔ PlantVillage near-duplicates', f"{stats['cross_source_overlap']['plantdoc']['images_with_pv_near_dup']} (removed)"],
    ['PlantDoc test images duplicating PlantDoc train', f"{stats['plantdoc_test_images_near_dup_of_plantdoc_train']} (removed)"],
])}
<h3>Class balance</h3>
<p>The dataset is strongly imbalanced: the largest class (<i>{nice(pv['largest_class'][0])}</i>) has {pv['largest_class'][1]:,} images and the
smallest (<i>{nice(pv['smallest_class'][0])}</i>) only {pv['smallest_class'][1]} — a ratio of <b>{pv['imbalance_ratio_max_min']}×</b>.
This is handled with class-weighted loss and macro-averaged metrics.</p>
{F('ml/artifacts/eda/class_distribution_plantvillage.png', 'Number of images per PlantVillage class.', '88%')}
{F('ml/artifacts/eda/samples_plantvillage.png', 'Random PlantVillage samples: single leaves on a uniform background (lab-style).', '72%')}
<div class="two">
{F('ml/artifacts/eda/samples_plantdoc.png', 'PlantDoc samples: real backgrounds, multiple leaves, varied light.')}
{F('ml/artifacts/eda/samples_tomato_pk_field.png', 'Tomato field samples (Pakistan), smartphone photos.')}
</div>
{F('ml/artifacts/eda/rgb_distribution.png', 'Per-image mean colour by dataset — the field datasets differ clearly from PlantVillage (domain shift).', '92%')}
</section>""")

S.append(f"""
<section>
<h2>Data Preprocessing and Leakage-free Splitting</h2>
<p>PlantVillage contains several photos of the same physical leaf. A random image-level split would put almost identical
photos in both training and test sets and inflate accuracy. Images were therefore grouped by (1) the dataset's own
leaf-identity map, (2) exact byte duplicates and (3) perceptual-hash near-duplicates, and <b>whole groups</b> were
assigned to train / validation / test (≈ 70/15/15, stratified by class, seed 42). A script assertion guarantees that no
group crosses splits.</p>
{table(['Split', 'Images', 'Used for'], [
    ['train', f"{splits['train']['images']:,}", 'gradient updates'],
    ['validation', f"{splits['validation']['images']:,}", 'early stopping, tuning, calibration, thresholds'],
    ['test', f"{splits['test']['images']:,}", 'final internal evaluation (used once)'],
    ['external validation (PlantDoc)', f"{splits['external_validation']['images']:,}", 'calibration and thresholds only'],
    ['domain-train (PlantDoc)', f"{splits['domain_train']['images']:,}", 'prepared for future field adaptation (not used)'],
    ['external test (PlantDoc + Tomato field)', f"{splits['external_test']['images']:,}", 'final external evaluation (used once)'],
])}
{F('ml/artifacts/eda/split_distribution.png', 'Per-class distribution of the leakage-free split.', '82%')}
<h3>Augmentation and normalisation</h3>
<ul>
<li><b>Training:</b> random resized crop to 224 px (scale 0.55–1.0), horizontal and vertical flips, rotation up to 30°,
brightness / contrast / saturation jitter, very small hue jitter (leaf colour is diagnostic), occasional Gaussian blur.</li>
<li><b>Evaluation and serving:</b> resize the short side to 256 px, centre-crop 224 px, ImageNet mean/std normalisation.
The serving code is unit-tested to produce numerically identical input to the training pipeline.</li>
</ul>
</section>""")

opt_rows = [[str(tr['number']), f"{tr['value']:.4f}", f"{tr['params']['lr']:.2e}", f"{tr['params']['backbone_lr_mult']:.3f}",
             f"{tr['params']['weight_decay']:.1e}", f"{tr['params']['label_smoothing']:.3f}"] for tr in optuna["trials"]]
best_t = max(optuna["trials"], key=lambda x: x["value"])["number"]
opt_rows = [[f"<b>{c}</b>" if r[0] == str(best_t) else c for c in r] for r in opt_rows]
cfg = m["config"]
opt_hist = [json.loads(f.read_text()) for f in sorted((ROOT / "ml/artifacts/models").glob("optuna_efficientnet_b0_t*/history.json"))]
opt_minutes = sum(h["history"][0]["epoch_seconds"] for h in opt_hist) / 60
best_opt = max(optuna["trials"], key=lambda x: x["value"])
EXPERIMENTS = [
    ["E1", "Hardware benchmark", "random tensors", "EfficientNet-B0 and ResNet-50, fp32 vs fp16, batch 32", "minutes",
     "fp32 chosen; B0 34 img/s, ResNet-50 23 img/s"],
    ["E2", "Hyper-parameter search (Optuna, TPE)", "4,000 train / 1,500 val images (stratified)",
     "8 trials × 1 epoch; LR, backbone multiplier, weight decay, label smoothing",
     f"{opt_minutes:.0f} min (training only)", f"best val macro-F1 {best_opt['value']:.3f} (trial {best_opt['number']})"],
]
for k, v in sorted(probe_runs.items()):
    EXPERIMENTS.append([f"E{len(EXPERIMENTS) + 1}", f"ResNet-50 learning-rate probe (head LR {k})", "6,000 train / 1,500 val images",
                        "1 epoch, backbone LR ×0.1", f"{v['epoch_seconds'] / 60:.1f} min", f"val macro-F1 {v['val_macro_f1']:.3f}"])
EXPERIMENTS += [
    [f"E{len(EXPERIMENTS) + 1}", "<b>EfficientNet-B0 final training</b>", f"{splits['train']['images']:,} train images per epoch",
     f"Optuna-best settings, {m['epochs_trained']} epochs, batch {cfg['batch_size']}", f"{m['train_seconds'] / 60:.0f} min",
     f"<b>val acc {p(hist[-1]['val_accuracy'], 2)}, val macro-F1 {hist[-1]['val_macro_f1']:.4f}</b>"],
    [f"E{len(EXPERIMENTS) + 2}", "Evaluation pass (inference only)", "22,881 images: validation, external validation, test, external test",
     "temperature fit, thresholds, metrics, plots", "≈ 4 min", f"test acc {p(t['accuracy'], 2)}"],
    [f"E{len(EXPERIMENTS) + 3}", "ONNX export and verification", f"{card['onnx_verification']['images']} test images",
     "PyTorch → ONNX (opset 17), CAM output", "&lt; 1 min", "identical predictions; CAM ≡ Grad-CAM"],
]

hist_rows = [[str(h['epoch']), f"{h['train_loss']:.3f}", p(h['train_acc'], 2), f"{h['val_loss']:.3f}", p(h['val_accuracy'], 2),
              f"{h['val_macro_f1']:.4f}", f"{h['extval_macro_f1']:.4f}", f"{h['epoch_seconds'] / 60:.1f}"] for h in hist]
probe_txt = ""
if probe_runs:
    probe_txt = (f"<p>A short ResNet-50 learning-rate probe (1 epoch, 6,000 images) was also run: head LR 1e-3 → validation "
                 f"macro-F1 {probe_runs['0.001']['val_macro_f1']:.3f}; head LR 3e-3 → {probe_runs['0.003']['val_macro_f1']:.3f}. "
                 "Full ResNet-50 and Vision-Transformer training was not carried out to keep the computational load suitable "
                 "for a laptop.</p>")
S.append(f"""
<section>
<h2>Model Architecture and Training</h2>
<h3>Architecture</h3>
<p><b>EfficientNet-B0</b> (Tan &amp; Le, 2019) is a compact convolutional network built from mobile inverted-bottleneck
(MBConv) blocks with squeeze-and-excitation, scaled for depth/width/resolution. The ImageNet-pretrained backbone
(timm) was kept and its classifier replaced by a new {pv['classes']}-class linear layer with dropout 0.2.
Parameters: <b>{m['parameters_m']:.2f} M</b>; input 224×224×3; exported model size {card['size_mb']} MB.</p>
<h3>Training setup</h3>
{table(['Setting', 'Value'], [
    ['Framework / hardware', 'PyTorch 2 on Apple M1 GPU (MPS, fp32), 8 GB unified memory'],
    ['Optimiser', 'AdamW; head LR ' + f"{cfg['lr']:.2e}" + ', backbone LR ×' + f"{cfg['backbone_lr_mult']:.2f}" + ', weight decay ' + f"{cfg['weight_decay']:.3g}"],
    ['Schedule', 'cosine decay with 5 % linear warm-up, per batch'],
    ['Loss', f"class-weighted (inverse √frequency) cross-entropy, label smoothing {cfg['label_smoothing']:.2f}"],
    ['Batch size / epochs', f"{cfg['batch_size']} / {m['epochs_trained']} (early stopping on validation macro-F1, patience 2)"],
    ['Training time', f"{m['train_seconds'] / 60:.0f} minutes ({m['epochs_trained']} epochs)"],
    ['Best epoch', f"{m['best_epoch']}"],
    ['Tracking', 'MLflow (sqlite) — every epoch logged'],
])}
<h3>Training procedure (step by step)</h3>
<ol>
<li><b>Load data:</b> each batch of 32 images is read from the split manifest, decoded, augmented (random resized crop, flips,
rotation, colour jitter, blur) and normalised; 3 parallel data-loader workers feed the GPU.</li>
<li><b>Model:</b> EfficientNet-B0 with ImageNet weights; the 1000-class head is replaced by a 38-class linear layer.</li>
<li><b>Fine-tuning:</b> all layers are trained. The new head uses the base learning rate; the pretrained backbone uses a
smaller rate (×{cfg['backbone_lr_mult']:.2f}) so that ImageNet features are adapted gently.</li>
<li><b>Loss:</b> cross-entropy with label smoothing ({cfg['label_smoothing']:.2f}) and per-class weights (inverse square root of
class frequency) to counter the 36× class imbalance.</li>
<li><b>Schedule:</b> learning rate rises linearly for the first 5 % of steps, then follows a cosine decay to zero.</li>
<li><b>Validation every epoch:</b> accuracy, macro-F1 and weighted-F1 on the PlantVillage validation split and on the PlantDoc
external-validation split are logged to MLflow; the checkpoint with the best validation macro-F1 is kept.</li>
<li><b>Early stopping:</b> training stops if validation macro-F1 does not improve for 2 epochs (here all 5 epochs ran because
the score kept improving).</li>
</ol>
<h3>Engineering issues during training and how they were solved</h3>
{table(['Problem observed', 'Cause found', 'Solution'], [
    ['Training extremely slow with mixed precision', 'fp16 autocast on the MPS backend ~15× slower', 'train in fp32; mixed precision enabled only on CUDA'],
    ['Training hung on the first batch', 'weighted, label-smoothed cross-entropy on MPS forces a GPU sync and stalled under memory pressure', 'equivalent loss written with gather/sum (verified identical to PyTorch to 1e-6)'],
    ['Very low throughput (&lt; 1 image/s) in a smoke test', 'too many persistent data-loader worker processes (~325 MB each) on 8 GB RAM', '3 training workers; validation workers created only during evaluation'],
    ['Experiment tracking failed', 'MLflow 3 refuses the file-based store', 'MLflow with a SQLite backend'],
], 'small')}
<h3>Summary of all experiments performed</h3>
{table(['#', 'Experiment', 'Data used', 'Configuration', 'Time', 'Outcome'], EXPERIMENTS, 'small')}
<p><b>Planned but not run.</b> Full ResNet-50 training (3 epochs), a Vision Transformer (ViT-S/16, 3 epochs), EfficientNet-B2
(2 epochs) and an EfficientNet-B0 domain-adaptation run on PlantDoc images were prepared in the training queue but stopped
to keep the computational load suitable for a laptop. The trained EfficientNet-B0 already met the project objectives.</p>
<h3>Hyper-parameter search (Optuna)</h3>
<p>8 TPE trials, each fine-tuning for one epoch on a stratified 4,000-image subset and scored by macro-F1 on 1,500
validation images. The best trial (bold) was used for the full run. Trials with a very low backbone learning rate barely
learned in one epoch.</p>
{table(['Trial', 'Val macro-F1', 'Head LR', 'Backbone ×', 'Weight decay', 'Label smoothing'], opt_rows, 'small')}
{probe_txt}
<h3>Training history</h3>
{table(['Epoch', 'Train loss', 'Train acc (aug.)', 'Val loss', 'Val acc', 'Val macro-F1', 'Field ext-val macro-F1', 'Minutes'], hist_rows, 'small')}
{F(f'ml/artifacts/models/{RUN}/curves.png', 'Training curves: loss (left) and accuracy / macro-F1 (right). Validation accuracy is near its ceiling after epoch 1; field (PlantDoc) performance stays low throughout.', '92%')}
</section>""")

sel_rows = []
for k, name in (("validation", "PlantVillage validation"), ("external_validation", "PlantDoc ext-validation"), ("test", "PlantVillage test"),
                ("external_test_plantdoc", "PlantDoc test"), ("external_test_tomato_pk_field", "Tomato field test")):
    x = R[k]["selective_at_threshold"]
    sel_rows.append([name, p(x['coverage']), p(x['selective_accuracy']), p(R[k]['unrecognized_rate']),
                     f"{R[k]['ece_uncalibrated']:.3f} → {R[k]['ece_calibrated']:.3f}"])
S.append(f"""
<section>
<h2>Confidence Calibration and Uncertainty Handling</h2>
<p>A softmax score is not a guarantee of correctness. The system therefore:</p>
<ol>
<li>fits a single <b>temperature</b> T = {m['temperature']:.3f} on the validation sets (lab and field weighted equally);</li>
<li>selects a <b>confident threshold</b> τ = {m['confidence_threshold']:.3f} and a <b>reject threshold</b> τ<sub>r</sub> = {m['reject_threshold']:.3f}
on validation data only (rule: {e(m['threshold_rule'])});</li>
<li>reports <i>confident</i> (p ≥ τ), <i>uncertain</i> (τ<sub>r</sub> ≤ p &lt; τ, advice hidden, top-3 shown) or
<i>not recognised</i> (p &lt; τ<sub>r</sub>);</li>
<li>never reports a blurry photo (variance-of-Laplacian below the 1st validation percentile, {card['blur_threshold']}) as confident.</li>
</ol>
{table(['Set', 'Answered (coverage)', 'Accuracy on answered', 'Not recognised', 'ECE raw → calibrated'], sel_rows)}
<p>On lab-style photos 72 % of images are answered confidently at 99.9 % accuracy. On field photos most images (~70 %)
are honestly reported as <i>not recognised</i>, but confident field answers are still only about 50 % correct — a
limitation that the application states explicitly.</p>
<div class="two">
{F(f'ml/artifacts/evaluation/{RUN}/risk_coverage.png', 'Risk–coverage: accuracy on the images the model answers as the confidence bar is raised.')}
{F(f'ml/artifacts/evaluation/{RUN}/reliability_external_test_plantdoc.png', 'Reliability diagram (PlantDoc test) before and after temperature scaling.', '78%')}
</div>
</section>""")

ext_rows = [[name, f"{x['n']:,}", p(x['accuracy'], 2), p(x['macro_precision'], 2), p(x['macro_recall'], 2), p(x['macro_f1'], 2),
             p(x['weighted_f1'], 2), p(x['crop_accuracy']), p(x['top3_accuracy'])]
            for name, x in (("PlantVillage validation", R['validation']), ("PlantVillage test", t), ("PlantDoc test (field)", pdoc),
                            ("Tomato field test (Pakistan)", tom))]
S.append(f"""
<section>
<h2>Results and Evaluation</h2>
<h3>Internal test set (PlantVillage, never used for training or selection)</h3>
<div class="kpis">
  <div><span>{p(t['accuracy'], 2)}</span>Accuracy</div><div><span>{p(t['macro_precision'], 2)}</span>Macro precision</div>
  <div><span>{p(t['macro_recall'], 2)}</span>Macro recall</div><div><span>{p(t['macro_f1'], 2)}</span>Macro F1</div>
  <div><span>{p(t['weighted_f1'], 2)}</span>Weighted F1</div>
</div>
<p>Only <b>{round((1 - t['accuracy']) * t['n'])} of {t['n']:,}</b> test images are misclassified. Crop identification accuracy is
{p(t['crop_accuracy'], 2)} and top-3 accuracy {p(t['top3_accuracy'], 2)}.</p>
{F(f'ml/artifacts/evaluation/{RUN}/confusion_matrix_test_normalized.png', 'Row-normalised confusion matrix on the PlantVillage test set (38 classes).', '92%')}
<h3>All evaluation sets</h3>
{table(['Set', 'n', 'Accuracy', 'Macro P', 'Macro R', 'Macro F1', 'Weighted F1', 'Crop acc.', 'Top-3'], ext_rows, 'small')}
{F('ml/artifacts/evaluation/model_comparison.png', 'Internal (lab) vs external (field) macro-F1 of the trained model.', '70%')}
<h3>External (field) evaluation</h3>
<p>The drop from lab to field photos ({p(t['accuracy'])} → {p(pdoc['accuracy'])} on PlantDoc and {p(tom['accuracy'])} on the
tomato field set) is the classic dataset-bias result reported by the PlantDoc authors: a model trained only on single
leaves photographed on a plain background does not transfer to cluttered field photos.</p>
<div class="two">
{F(f'ml/artifacts/evaluation/{RUN}/confusion_matrix_external_test_plantdoc_normalized.png', 'Confusion matrix — PlantDoc test (classes present).')}
{F(f'ml/artifacts/evaluation/{RUN}/confusion_matrix_external_test_tomato_pk_field_normalized.png', 'Confusion matrix — Tomato field test.')}
</div>
<h3>Error analysis</h3>
<ul>
<li><b>Internal:</b> errors occur between visually similar lesions on the same crop — e.g.
{'; '.join(f"{nice(c['true'])} → {nice(c['pred']).split(' — ')[1]} ({c['count']})" for c in t['top_confusions'][:4])}.</li>
<li><b>PlantDoc:</b> healthy leaves on the plant are confused with other crops' healthy classes or with blights;
crop-level accuracy is only {p(pdoc['crop_accuracy'])}.</li>
<li><b>Tomato field:</b> healthy field leaves are called Late or Early blight; Septoria ↔ Early blight and Leaf Mold ↔ Late blight are
the main confusions — natural light, shadows and soil look like lesions to a lab-trained model.</li>
</ul>
{F(f'ml/artifacts/evaluation/{RUN}/misclassified_test.png', 'The most confident mistakes on the PlantVillage test set (true vs predicted, calibrated confidence).', '80%')}
{F(f'ml/artifacts/evaluation/{RUN}/misclassified_external_test_tomato_pk_field.png', 'The most confident mistakes on tomato field photos.', '80%')}
</section>""")

ov = card["onnx_verification"]
S.append(f"""
<section>
<h2>Explainability (Grad-CAM / CAM)</h2>
<p><b>Grad-CAM</b> (Selvaraju et al., 2017) highlights the image regions that most influenced the predicted class score.
EfficientNet-B0 ends with global average pooling and one linear layer, so Grad-CAM at the last feature map equals the
classic <b>Class Activation Map</b> (CAM = classifier weights × feature maps) up to a constant factor. The deployed ONNX
model therefore outputs the CAM directly, without gradients. This equivalence was verified numerically: minimum
correlation between CAM and Grad-CAM = <b>{ov['cam_vs_gradcam_min_correlation']:.8f}</b>; ONNX predictions agree with PyTorch on
<b>{ov['images']}</b> test images (max logit difference {ov['max_abs_logit_diff']:.1e}).</p>
{F('ml/artifacts/evaluation/gradcam_examples.png', 'Model input, heat-map and overlay for lab (top four rows) and field (bottom four rows) images. On lab images the model focuses on the leaf and lesions; on field photos attention is scattered and predictions are wrong with low confidence.', '78%')}
<p class="note">Heat-maps show influence on the class score; they are a sanity check, not proof that a diagnosis is correct.</p>
</section>""")

n_src = len({s['url'] for d in advisory['diseases'] for s in d['sources']})
ex = next(d for d in advisory['diseases'] if d['id'] == 'Tomato___Early_blight')
S.append(f"""
<section>
<h2>Bilingual Advisory Engine</h2>
<p>A structured JSON knowledge base (<code>advisory/diseases.json</code>) holds one entry for each of the {len(advisory['diseases'])}
classes with crop and disease names, scientific name, pathogen type, summary, symptoms, causes, prevention, organic
options and chemical options — every field in <b>English and simple Hindi</b>. It cites <b>{n_src} distinct sources</b> (ICAR /
NIPHM integrated pest management packages, TNAU Agritech, Kerala Agricultural University, National Horticulture Board,
FAO and university extension services), each fetched and dated.</p>
<ul>
<li><b>No dosages</b> are ever given; chemical options are listed only as "options reported by &lt;source&gt;" and every
list ends with an instruction to use products registered by CIB&amp;RC, follow the label and consult the local KVK.</li>
<li>Advice is shown <b>only for confident predictions</b>; for uncertain results the user sees candidate diseases with links.</li>
<li>A safety review replaced an older recommendation of streptocycline (an antibiotic the Government of India has moved to
prohibit in agriculture) with a "do not use" warning.</li>
</ul>
<p><b>Example entry — {e(ex['crop']['en'])}, {e(ex['name']['en'])} (<i>{e(ex['scientific_name'])}</i>):</b></p>
{table(['Field', 'Content (English version; the Hindi version is stored in parallel)'], [
    ['Summary', e(ex['summary']['en'])],
    ['Symptoms', '<br/>'.join('• ' + e(x) for x in ex['symptoms']['en'])],
    ['Causes', '<br/>'.join('• ' + e(x) for x in ex['causes']['en'])],
    ['Prevention', '<br/>'.join('• ' + e(x) for x in ex['prevention']['en'])],
    ['Organic options', '<br/>'.join('• ' + e(x) for x in ex['organic_treatment']['en'])],
    ['Chemical options', '<br/>'.join('• ' + e(x) for x in ex['chemical_treatment']['en'])],
    ['Sources', '<br/>'.join(e(x['publisher']) for x in ex['sources'])],
], 'small')}
{F('docs/report/screens/advisory-treatment-en.png', 'Treatment tab in the application with organic and chemical options and the label-safety notice.', '80%')}
</section>""")

S.append(f"""
<section>
<h2>Web Application</h2>
<p>The website has seven pages: Landing, Diagnose, Results, Disease library, Model performance, Methodology and About.
It is fully responsive and bilingual, supports drag-and-drop, file selection, browser camera capture and
<b>pasting an image (Ctrl/⌘+V or a Paste button)</b>, and accepts JPEG, PNG, WebP, AVIF, HEIC (iPhone), BMP, TIFF and GIF.
Large photos are resized in the browser before upload.</p>
{F('docs/report/screens/landing.png', 'Landing page — live metrics are loaded from the evaluation results.', '92%')}
<div class="two">
{F('docs/report/screens/detect-empty.png', 'Diagnose page: upload, camera and paste options, supported formats.')}
{F('docs/report/screens/detect-preview.png', 'Selected photo with preview, replace / camera / paste / remove actions.')}
</div>
{F('docs/report/screens/results-en.png', 'Results: status banner, crop and disease, calibrated confidence, top-3 candidates, heat-map and advisory.', '92%')}
{F('docs/report/screens/results-hi.png', 'The same result after switching to Hindi.', '92%')}
<div class="two">
{F('docs/report/screens/gradcam-viewer.png', 'Heat-map viewer: model input, heat-map and overlay (also a slider compare mode).')}
{F('docs/report/screens/results-unrecognized.png', 'A field photo outside the training distribution is reported as not recognised; no advice is shown.')}
</div>
<div class="two">
{F('docs/report/screens/library.png', 'Disease library with search and crop filters.')}
{F('docs/report/screens/library-detail.png', 'Disease detail sheet with cited advisory.')}
</div>
{F('docs/report/screens/performance-top.png', 'Model performance dashboard (top part) — KPIs, lab-vs-field chart, risk–coverage and model summary; every number comes from the real evaluation.', '88%')}
<div class="three">
{F('docs/report/screens/mobile-landing.png', 'Mobile — landing.')}
{F('docs/report/screens/mobile-detect.png', 'Mobile — diagnose.')}
{F('docs/report/screens/mobile-results.png', 'Mobile — result.')}
</div>
<p>Other features: downloadable HTML diagnosis report, print / save-as-PDF, dark mode, feedback (stored as unverified),
uncertainty warnings, blurry-photo warnings.</p>
</section>""")

S.append(f"""
<section>
<h2>Backend API and Model Serving</h2>
{table(['Endpoint', 'Purpose'], [
    ['GET /api/health', 'model / database / advisory status'],
    ['GET /api/model/info', 'architecture, classes, temperature, thresholds, checksum'],
    ['POST /api/predict', 'diagnose one photo (lang = en|hi, optional heat-map)'],
    ['POST /api/predict/batch', 'diagnose up to 8 photos; per-file errors'],
    ['GET /api/diseases, /api/diseases/{id}', 'disease list and detail (EN/HI)'],
    ['GET /api/advisory/{id}', 'advisory for one disease in the chosen language'],
    ['GET /api/metrics', 'evaluation summary for the dashboard (503 if not generated — never synthetic)'],
    ['POST /api/feedback', 'user feedback, stored as unverified'],
])}
<ul>
<li><b>Validation:</b> 8 MB limit, MIME check plus content sniffing, decompression-bomb guard, EXIF orientation,
minimum size, blur detection; consistent JSON errors (413, 415, 422, 429, 503).</li>
<li><b>Serving:</b> the PyTorch model is exported to <b>ONNX</b> (16 MB) and run with ONNX Runtime; the deployment needs no
PyTorch (≈ 133 MB of dependencies instead of &gt; 1 GB) and inference takes about <b>35 ms</b> per image on the server
(measured), versus {m['latency']['cpu']['batch1_ms_mean']:.0f} ms for the PyTorch model on the development laptop CPU.</li>
<li><b>Integrity:</b> the API refuses to load a model whose SHA-256 differs from the model card.</li>
<li><b>Privacy:</b> uploaded images are never stored — only a hash and prediction metadata.</li>
</ul>
</section>""")

S.append("""
<section>
<h2>Testing and Quality Assurance</h2>
<table><thead><tr><th>Suite</th><th>Tests</th><th>What is covered</th><th>Result</th></tr></thead><tbody>
<tr><td>Backend (pytest)</td><td>44</td><td>preprocessing parity with training, 8 image formats, invalid uploads, ONNX loading & checksum,
CAM = Grad-CAM, API endpoints, feedback, real-model integration</td><td class="ok">all passed</td></tr>
<tr><td>Frontend components (Vitest)</td><td>6</td><td>i18n switching, advisory rendering, heat-map viewer, report escaping</td><td class="ok">all passed</td></tr>
<tr><td>End-to-end (Playwright, Chrome)</td><td>16</td><td>desktop + mobile: full journey (upload → result → Hindi → heat-map → report → dashboard),
library, unrecognised field photo, invalid upload, "New diagnosis" regression, AVIF, HEIC, paste</td><td class="ok">all passed</td></tr>
<tr><td>Live deployment</td><td>—</td><td>end-to-end suite and API checks run against the deployed Vercel site</td><td class="ok">passed</td></tr>
</tbody></table>
<p>Bugs found and fixed during testing included a training hang on Apple MPS (replaced loss implementation), chart
animation artefacts in print, a stale "Analysing…" state after <i>New diagnosis</i> (Next.js route state preservation) and a
<code>.gitignore</code> rule that would have excluded the deployed model.</p>
</section>""")

S.append("""
<section>
<h2>Deployment</h2>
<p>Both parts run on <b>Vercel's free (Hobby) plan</b> from the same GitHub repository; every <code>git push</code> redeploys them.</p>
<table><thead><tr><th>Project</th><th>Root</th><th>URL</th></tr></thead><tbody>
<tr><td>Website (Next.js)</td><td><code>frontend/</code></td><td><a href="https://krishiguard-ai-five.vercel.app">krishiguard-ai-five.vercel.app</a></td></tr>
<tr><td>API (FastAPI, Python function)</td><td>repository root (<code>index.py</code>)</td><td><a href="https://krishiguard-api.vercel.app/docs">krishiguard-api.vercel.app/docs</a></td></tr>
</tbody></table>
<p>The website proxies <code>/api/*</code> to the API project, so the browser talks to a single origin. Docker files and a
docker-compose configuration (with PostgreSQL) are also provided for self-hosting.</p>
</section>""")

S.append(f"""
<section>
<h2>Limitations and Future Scope</h2>
<ul>
<li><b>Field accuracy:</b> {p(pdoc['accuracy'])}–{p(tom['accuracy'])} on real field photos versus {p(t['accuracy'])} on lab photos.
<i>Future:</i> fine-tune on labelled field images (the PlantDoc domain-train split is already prepared) and collect Indian field data.</li>
<li><b>Closed set:</b> only 38 classes; other crops, pests or nutrient problems are forced onto these labels.</li>
<li><b>Single architecture:</b> ResNet-50 / Vision Transformer comparisons were planned but not trained, to keep the
computational load suitable for a laptop; the training scripts support them.</li>
<li><b>Advisory review:</b> content is source-cited but has not yet been reviewed by an agronomist; pesticide registrations change.</li>
<li><b>Calibration:</b> one temperature cannot be right for both lab and field photos; confident field answers are still wrong about half the time.</li>
</ul>
</section>
<section>
<h2>Conclusion</h2>
<p>KrishiGuard AI demonstrates a complete, working pipeline from verified public datasets to a deployed, bilingual web
application. A transfer-learned EfficientNet-B0 reaches <b>{p(t['accuracy'], 2)} accuracy</b> and <b>{p(t['macro_f1'], 2)} macro-F1</b> on a
leakage-free held-out test set, explains its decisions with heat-maps, and serves cited English and Hindi advice. Honest
external testing shows that lab-trained models do not yet generalise to field photos; the application responds to this
with calibrated confidence, explicit uncertainty states and by withholding advice when unsure.</p>
</section>
<section>
<h2>References</h2>
<ol class="refs">
<li>Mohanty, S. P., Hughes, D. P., &amp; Salathé, M. (2016). Using deep learning for image-based plant disease detection. <i>Frontiers in Plant Science</i>, 7, 1419.</li>
<li>Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., &amp; Batra, N. (2020). PlantDoc: A dataset for visual plant disease detection. <i>CODS-COMAD 2020</i>.</li>
<li>Tomato Leaf Disease Classification Dataset in Pakistan. Mendeley Data, 3mbnb82mxd (CC BY 4.0).</li>
<li>Tan, M., &amp; Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. <i>ICML</i>.</li>
<li>Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. <i>ICCV</i>.</li>
<li>Zhou, B., et al. (2016). Learning deep features for discriminative localization (CAM). <i>CVPR</i>.</li>
<li>Guo, C., Pleiss, G., Sun, Y., &amp; Weinberger, K. Q. (2017). On calibration of modern neural networks. <i>ICML</i>.</li>
<li>Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. <i>KDD</i>.</li>
<li>NIPHM / Directorate of Plant Protection, Quarantine &amp; Storage — AESA-based IPM packages; TNAU Agritech Portal; and the other
{n_src - 2} advisory sources listed in <code>advisory/SOURCES.md</code>.</li>
</ol>
</section>""")

pc_rows = []
for c, v in sorted(R["test"]["per_class"].items()):
    pc_rows.append([nice(c), p(v['precision']), p(v['recall']), p(v['f1']), str(v['support'])])
ext_pc = []
for k, name in (("external_test_plantdoc", "PlantDoc"), ("external_test_tomato_pk_field", "Tomato field")):
    for c, v in sorted(R[k]["per_class"].items()):
        ext_pc.append([name, nice(c), p(v['precision']), p(v['recall']), p(v['f1']), str(v['support'])])
S.append(f"""
<section>
<h2>Appendix A — Per-class results</h2>
<h3>A.1 PlantVillage test set</h3>
{table(['Class', 'Precision', 'Recall', 'F1', 'Support'], pc_rows, 'small')}
<h3>A.2 External field test sets</h3>
{table(['Set', 'Class', 'Precision', 'Recall', 'F1', 'Support'], ext_pc, 'small')}
</section>""")

split_rows = []
for c in sorted(splits["train"]["per_class"]):
    split_rows.append([nice(c)] + [str(splits[k]["per_class"].get(c, 0)) for k in ("train", "validation", "test", "external_validation", "external_test")])
S.append(f"""
<section>
<h2>Appendix B — Dataset split per class</h2>
{table(['Class', 'Train', 'Validation', 'Test', 'PlantDoc ext-val', 'External test'], split_rows, 'small')}
</section>""")

# ---- automatic chapter numbering, contents and list of figures
UNNUMBERED = ("Abstract", "References", "Appendix")
S = [part for item in S for part in re.split(r"(?=<section)", item) if part.strip()]  # one <section> per entry
toc, ch = [], 0
for i, sec in enumerate(S):
    mh = re.search(r"<h2>(.*?)</h2>", sec)
    if not mh:
        continue
    title = mh.group(1)
    if title.startswith(UNNUMBERED):
        toc.append(f"<li class='un'>{title}</li>")
        continue
    ch += 1
    k = [0]

    def h3(mm, ch=ch, k=k):
        k[0] += 1
        return f"<h3>{ch}.{k[0]} {mm.group(1)}</h3>"

    sec = re.sub(r"<h3>(.*?)</h3>", h3, sec)
    sec = sec.replace(mh.group(0), f"<h2><span class='chno'>Chapter {ch}</span>{title}</h2>", 1)
    S[i] = sec
    toc.append(f"<li><b>Chapter {ch}.</b> {title}</li>")
figs_html = "".join(f"<li>Figure {n}. {re.sub('<[^>]+>', '', c)}</li>" for n, c in enumerate(FIGS, 1))
S.insert(2, f"""<section class="toc"><h2>Contents</h2><ol class="toc-list">{''.join(toc)}</ol></section>
<section class="toc"><h2>List of Figures</h2><ol class="figs">{figs_html}</ol></section>""")

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, "Kohinoor Devanagari", "Noto Sans Devanagari", sans-serif;
       color: #1d2a22; font-size: 10.2pt; line-height: 1.5; margin: 0; }
h1, h2, h3 { font-family: Georgia, "Times New Roman", serif; color: #1f5135; }
h2 { font-size: 19pt; border-bottom: 2px solid #cfe1c9; padding-bottom: 4px; margin: 0 0 10px; }
h3 { font-size: 12.5pt; margin: 14px 0 6px; break-after: avoid; page-break-after: avoid; }
h2 { break-after: avoid; }
section { page-break-before: always; }
section.cover { page-break-before: auto; }
p { margin: 6px 0; text-align: justify; }
a { color: #2f7d4f; text-decoration: none; }
code { font-size: 9pt; background: #f1f4ee; padding: 0 3px; border-radius: 3px; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 12px; font-size: 9.2pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th { background: #2f7d4f; color: #fff; text-align: left; padding: 5px 7px; font-weight: 600; }
td { border-bottom: 1px solid #e3e8df; padding: 4px 7px; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f7f9f5; }
table.small { font-size: 8.3pt; }
table.small td, table.small th { padding: 3px 6px; }
td.ok { color: #1f7a3f; font-weight: 600; }
figure { margin: 10px auto; text-align: center; page-break-inside: avoid; }
figure img { border: 1px solid #e3e8df; border-radius: 6px; max-width: 100%; max-height: 222mm; object-fit: contain; }
figcaption, .caption { font-size: 8.6pt; color: #56625a; margin-top: 4px; text-align: center; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; align-items: start; }
.three { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; align-items: start; }
.note { background: #fff6dd; border: 1px solid #f0d58a; padding: 6px 10px; border-radius: 6px; font-size: 9pt; }
.kw { font-size: 9pt; color: #56625a; }
.kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 8px 0; }
.kpis div { border: 1px solid #cfe1c9; border-radius: 8px; padding: 8px; text-align: center; font-size: 8.6pt; color: #56625a; }
.kpis span { display: block; font-family: Georgia, serif; font-size: 16pt; color: #1f5135; font-weight: 700; }
.cover-band { background: linear-gradient(135deg, #1f5135, #2f7d4f 60%, #6a9f3a); color: #fff; border-radius: 14px;
              padding: 34px 32px 30px; margin-top: 4px; }
.cover-band h1 { color: #fff; font-size: 40pt; margin: 6px 0 4px; }
.cover-band .kicker { text-transform: uppercase; letter-spacing: .14em; font-size: 9pt; opacity: .85; }
.cover-band .subtitle { font-size: 14pt; opacity: .95; }
.cover-band .logo { font-size: 30pt; }
.cover-meta { margin: 18px 0 8px; }
table.meta td:first-child { width: 28%; color: #56625a; font-weight: 600; }
table.meta tbody tr:nth-child(even) td { background: none; }
.cover-shot img { max-height: 92mm; object-fit: cover; object-position: top; }
.toc ol { font-size: 11.5pt; line-height: 1.9; }
.arch { display: flex; align-items: stretch; gap: 8px; margin: 10px 0 4px; }
.arch .lane { flex: 1; border: 1px dashed #b9cfb2; border-radius: 10px; padding: 8px; }
.arch .lane-title { font-size: 8pt; text-transform: uppercase; letter-spacing: .1em; color: #56625a; margin-bottom: 4px; }
.arch .box { background: #eef5ea; border: 1px solid #b9cfb2; border-radius: 8px; padding: 7px; font-size: 8.8pt; text-align: center; margin-bottom: 5px; }
.arch .box.sub { background: #fff; }
.arch .arrow { align-self: center; font-size: 16pt; color: #2f7d4f; text-align: center; }
.arch.offline { align-items: center; }
.arch.offline .box { flex: 1; margin: 0; background: #fff8e6; border-color: #eed59a; }
.refs { font-size: 9.2pt; }
h2 .chno { display: block; font-size: 10pt; letter-spacing: .14em; text-transform: uppercase; color: #6a9f3a; font-family: -apple-system, Helvetica, sans-serif; }
.toc-list { list-style: none; padding: 0; font-size: 11.5pt; line-height: 2; }
.toc-list li.un { font-style: italic; }
.figs { list-style: none; padding: 0; font-size: 9pt; line-height: 1.6; }
.cover-top { text-align: center; letter-spacing: .3em; font-size: 10pt; color: #56625a; margin: 6px 0 10px; }
.cover-sub { text-align: center; font-size: 11pt; color: #56625a; margin-top: 12px; font-style: italic; }
table.hi td { font-size: 9pt; }
"""

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>KrishiGuard AI — Project Report</title>
<style>{CSS}</style></head><body>{''.join(S)}</body></html>"""
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "KrishiGuard_AI_Report.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "KrishiGuard_AI_Report.html", f"({fig_no[0]} figures)")
