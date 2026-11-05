# Experiment Log

Chronological record of what was run, what happened and what was decided. Times are IST on the development machine
(Apple M1, 8 GB). Raw logs: `ml/experiments/logs/`, MLflow: `ml/experiments/mlflow.db`, Optuna: `ml/experiments/optuna.db`.

## 2026-11-04

### Phase 1 — Environment (22:10)
* M1 / 8 GB / MPS / 8.5 GB free disk / no CUDA / no Kaggle token / Docker daemon stopped / Postgres running.
  → RESOURCE_REPORT.md. Project venv created with `--system-site-packages`; uv nevertheless installed its own
  torch 2.15 (1 GB) — accepted.

### Phase 2–3 — Data acquisition (22:15–22:32)
* `kaggle datasets list -s plantvillage` → fails at import (`api.authenticate()`, no token). Kaggle abandoned after one
  attempt; authors' repositories used.
* PlantVillage: `git clone --filter=blob:none --sparse` + `sparse-checkout set raw/color` (avoids the 2.1 GB full repo),
  `.git` removed afterwards to recover disk. **Verified 38 classes / 54,305 images.** `leaf-map.json` and
  `CITATION.cff` fetched at the same commit.
* PlantDoc: shallow clone; **2,336 train + 236 test images, 28 classes** (paper reports 2,598; the public repo has 2,572).
* Field images: Mendeley *Tomato Leaf Disease Classification Dataset in Pakistan* (CC BY 4.0), 99.5 MB zip,
  sha256 recorded; **7,200 images, 6 classes**, all 256×256 with visible rotation-augmentation artefacts.
* Plant Pathology 2020 (apple, field) — Kaggle-competition-only; not acquired (optional).

### Phase 4 — EDA & splits (22:35–22:50)
* `build_dataset.py` indexes 64,077 images in ~1 min (6 processes): 0 corrupt, 381 exact duplicates (182 groups),
  1,165 within-class pHash near-duplicate pairs, 41,111 images matched to the leaf map → **19,988 leakage groups**.
* Cross-source: 1 PlantDoc image near-duplicates PlantVillage; 0 for Tomato-PK; 14 PlantDoc-test images duplicate
  PlantDoc-train → all removed from external sets.
* Decision: split PlantDoc-train into external_validation (463) and domain_train (1,872) so calibration/selection can
  see field data without touching any test set.

### Phase 5 — Training infrastructure (22:50–23:12)
| Issue | Evidence | Fix |
|---|---|---|
| fp16 autocast on MPS extremely slow | ResNet-50 train 1.6 img/s with AMP vs 22.8 img/s fp32 | fp32 on MPS; AMP only on CUDA |
| Training hang | `sample` stack: `cross_entropy(weight, label_smoothing)` → `masked_select_mps` → `nonzero` → `MPSStream::commitAndWait` blocked | gather-based `smoothed_weighted_ce` (equal to torch to 1e-6), no per-step `.item()` syncs |
| 0.4–0.8 img/s in smoke run | 10 persistent DataLoader workers × ~325 MB each → swap > 10 GB | eval loaders non-persistent, 3 train workers; profiled: data 0.01 s vs GPU 0.9 s per batch |
| MLflow file store refused | MLflow 3.18 maintenance-mode exception | sqlite backend `ml/experiments/mlflow.db` |
| macOS `spawn` + no `__main__` guard | profiler hung at DataLoader | guard added (train.py already had it) |

### Phase 5–6 — Optuna, EfficientNet-B0 (23:12–00:02)
Proxy task: 1 epoch, 4,000 stratified train images, 1,500 stratified validation images; TPE (4 random start-up trials).

| Trial | Val macro-F1 | head LR | backbone mult | weight decay | label smoothing | Ext-val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8272 | 8.6e-4 | 0.435 | 5.1e-3 | 0.090 | 0.136 |
| 1 | 0.1727 | 4.7e-4 | 0.047 | 2e-5 | 0.130 | 0.046 |
| 2 | 0.8167 | 1.63e-3 | 0.220 | 1e-5 | 0.145 | 0.137 |
| 3 | 0.7854 | 3.12e-3 | 0.055 | 5e-5 | 0.028 | 0.141 |
| **4** | **0.8895** | **1.95e-3** | **0.401** | **1.81e-2** | **0.120** | 0.146 |
| 5 | 0.1658 | 4.1e-4 | 0.060 | 8.1e-3 | 0.138 | 0.039 |
| 6 | 0.8670 | 4.46e-3 | 0.236 | 3.8e-3 | 0.088 | 0.159 |
| 7 | 0.2419 | 4.2e-4 | 0.091 | 3.8e-2 | 0.023 | 0.044 |

Observations: the effective **backbone** learning rate dominates — trials with backbone LR ≲ 4e-5 barely learn in one
epoch. External-validation (PlantDoc) macro-F1 stays at 0.04–0.16 for every configuration: the domain gap is not a
hyper-parameter problem. Trial 4 adopted for EfficientNet-B0 (and reused for B2).

<!-- RUNS:START -->
## 2026-11-05

### Phase 5 — EfficientNet-B0 full training (00:02–01:55)
Config: Optuna trial-4 hyper-parameters, 5 epochs, batch 32, 224 px, full train split (39,309 images), fp32 on MPS (~33 img/s).

| Epoch | Train loss | Train acc (augmented) | Val loss | Val acc | Val macro-F1 | Ext-val (PlantDoc) macro-F1 | Minutes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.464 | 88.95% | 0.218 | 98.80% | 0.9858 | 0.1971 | 19.3 |
| 2 | 1.123 | 97.94% | 0.262 | 98.48% | 0.9797 | 0.1746 | 23.7 |
| 3 | 1.074 | 98.92% | 0.212 | 99.19% | 0.9887 | 0.2164 | 19.4 |
| 4 | 1.045 | 99.45% | 0.189 | 99.68% | 0.9958 | 0.1805 | 22.6 |
| 5 | 1.031 | 99.74% | 0.192 | 99.72% | 0.9963 | 0.1676 | 18.4 |

Epoch 2 dipped while the learning rate was still high, then improved as the cosine schedule annealed; best = epoch 5.
External-validation macro-F1 peaked at epoch 3 (0.216) and declined as the model specialised further on lab images.
Checkpoint: `ml/artifacts/models/effnet_b0_224/best.pt`.

### ResNet-50 learning-rate probe (01:55–02:15)
1 epoch on 6,000 images: head LR 1e-3 → val macro-F1 0.400; head LR 3e-3 → 0.863.

### Scope change (02:20)
At the user's request ("college project submission level"; laptop overheating after ~4 h of continuous GPU load) the
remaining queue was **stopped**: full ResNet-50 (3 epochs), ViT-S/16, EfficientNet-B0 domain adaptation and
EfficientNet-B2 were **not trained**. A short 1-epoch ResNet-50 run was started and then interrupted by a session
restart before saving anything; it was not retried. The CPU evaluation watcher was stopped because CPU inference
during GPU training ran at only ~7 img/s.

### Phase 7 — Evaluation of the trained model (23:16, single pass on MPS, ~4 min)
`evaluate.py --run effnet_b0_224 --device mps`: 22,881 images (validation, external validation, test, external test).
* Temperature T = 1.055, confident threshold 0.730, reject threshold 0.647 (rule: target ≥ 90 % selective accuracy
  on both validation domains was unreachable on PlantDoc → threshold maximising balanced selective accuracy).
* PlantVillage test: accuracy 99.56 %, macro-F1 99.49 % (34 errors / 7,751).
* PlantDoc test: accuracy 31.1 %, macro-F1 28.3 %; Tomato field: accuracy 26.5 %, macro-F1 24.2 %.
* Note: label smoothing (0.12) keeps confidences on correct lab images around 0.7–0.85, so 72 % of PlantVillage test
  images clear the confident threshold (at 99.9 % accuracy) and ~26 % are shown as *uncertain*. A single domain-balanced
  temperature cannot simultaneously sharpen lab predictions and soften field predictions.
* Exported `ml/artifacts/serving/model.pt` (16.5 MB, sha256 in `model_card.json`), Grad-CAM examples, summary.json.

### Phase 11–12 — Integration and testing (23:17–23:30)
* Backend pytest: 33 passed (incl. 3 real-model integration tests). Vitest: 6 passed. Playwright: 8 passed
  (desktop + Pixel 7 emulation): upload → prediction → advisory → Hindi → Grad-CAM → report download → dashboard;
  library search; unrecognised field photo; invalid upload.
* Bugs fixed: duplicated "advice withheld" message; long filename overflow on mobile; Recharts entry animation
  produced truncated charts in screenshots/print (animation disabled); duplicate y-axis tick labels; Grad-CAM note
  not localised in Hindi.
* Local PostgreSQL failed to start after a restart (stale `postmaster.pid` pointing at an unrelated PID); the
  documented SQLite fallback was used for the test run. The fix is left to the machine owner.
<!-- RUNS:END -->
