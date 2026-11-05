# Resource Report (Phase 1 — environment inspection, 2026-11-04)

| Resource | Finding | Consequence |
|---|---|---|
| OS | macOS 26.5.2 (Darwin 25.5.0), arm64 | — |
| CPU | Apple M1, 8 cores (4 performance + 4 efficiency) | DataLoader limited to 3 train workers |
| RAM | **8 GB unified** (shared by CPU and GPU); 5–11 GB swap in use during the session because of other open apps | Biggest constraint: one GPU job at a time, non-persistent eval workers, batch 32 |
| GPU | Apple M1 integrated GPU (7 cores), Metal 4, PyTorch **MPS** backend available | Used for training; **no CUDA** |
| Disk | 228 GB volume, **8.5 GB free** at start (96 % full) | Partial/sparse git clones; split manifests instead of image copies; only `best.pt` checkpoints kept |
| Python | 3.12.1 (python.org); project venv `.venv` (uv) with torch 2.15.0, torchvision 0.29.2, timm 1.0.31, MLflow 3.18, Optuna 5.0 | — |
| Node.js | v18.20.8 default; **v22.23.4** via nvm used for the frontend (Next.js 16 needs ≥ 20.9) | — |
| Package managers | pip, uv 0.9.27, npm 10.9, Homebrew; no conda/pnpm/yarn | — |
| Kaggle CLI | installed (1.7.4.5) but **no API token** (`~/.kaggle` empty) | Kaggle search/download impossible → authors' GitHub / Mendeley sources used |
| Docker | 28.1.1 installed, **daemon not running** | Images not built locally (see DEPLOYMENT_GUIDE) |
| PostgreSQL | 14.18 (Homebrew) running on :5432 | Dedicated role + database `krishiguard` created |
| Browser automation | Google Chrome + Playwright 1.64 (uses installed Chrome, no browser download); Claude-in-Chrome extension available | E2E tests and screenshots |
| Project location | `…/Deep Learning Project/Code/krishiguard-ai` (own git repo; the parent home directory is itself a git repo, so the project was `git init`-ed to keep it isolated — nothing committed) | — |

## Measured accelerator throughput (MPS, batch 32, 224 px, steady state)

Script: `scripts/benchmark_hw.py`, raw output `ml/experiments/hw_benchmark.json`.

| Model | Params | Train img/s (fp32) | Inference img/s |
|---|---:|---:|---:|
| EfficientNet-B0 | 4.1 M | 34.1 | 200.7 |
| ResNet-50 | 23.6 M | 22.8 | 82.6 |

* **fp16 autocast on MPS was ~15× slower** (ResNet-50: 1.6 img/s train) on this torch build → training runs in fp32;
  AMP is enabled automatically only on CUDA.
* `F.cross_entropy(weight=…, label_smoothing=…)` on MPS routes through `masked_select → nonzero`, forcing a GPU
  sync each step; under memory pressure this **hung** the GPU command queue. Replaced with an equivalent
  gather-based implementation (`smoothed_weighted_ce`, verified equal to 1e-6).
* Per-step profiling showed data loading at ~0.01 s vs ~0.9 s GPU time per batch: training is GPU-bound.

## Adaptations made because of these limits

1. Short, early-stopped fine-tuning schedules (≤ 5 epochs) instead of long runs; hyper-parameters tuned on a
   4,000-image proxy task with Optuna rather than full-data trials.
2. EfficientNet-B2 and ViT trained with reduced epoch budgets; recorded in `docs/EXPERIMENT_LOG.md`.
3. Strictly sequential experiment queue (`ml/training/run_experiments.py`), resumable after interruption.
4. CPU inference in the API (the deployed model is chosen partly for CPU latency).
