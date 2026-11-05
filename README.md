# KrishiGuard AI — Crop Disease Detection & Advisory (College Project)

A responsive web application where a farmer uploads or photographs a crop leaf and gets:

1. the **crop** and the **disease** (or *healthy*) — 14 crops, 38 classes,
2. a **confidence score**, with clear *uncertain* / *not recognised* states,
3. a **Grad-CAM heat-map** showing where the model looked,
4. **advisory in English and Hindi** — symptoms, causes, prevention, organic and chemical options with sources,
5. a downloadable **diagnosis report**,

plus a **disease library**, a **model performance dashboard** with real evaluation metrics and a confusion matrix,
and **Methodology / About** pages.

> Decision support only — not a substitute for an agricultural expert. Advice is shown only for confident
> results and never includes pesticide dosages.

![results page](docs/screenshots/desktop-03-results-en.png)

## Technologies

| Layer | Stack |
|---|---|
| Deep learning | PyTorch 2, `timm` (EfficientNet-B0, ImageNet-pretrained), scikit-learn metrics, MLflow (experiment tracking), Optuna (small hyper-parameter search) |
| Backend | FastAPI, Pydantic, SQLAlchemy (PostgreSQL, SQLite fallback), slowapi rate limiting, JSON logging |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4, shadcn/ui, Lucide icons, Recharts |
| Testing | pytest, Vitest + Testing Library, Playwright (desktop + mobile browser E2E) |
| Deployment | Dockerfiles for frontend and backend, docker-compose with Postgres |

## Dataset

| Dataset | Used for | Images | Licence |
|---|---|---:|---|
| [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) (colour) | training / validation / test | 54,305 (38 classes) | CC BY-SA 3.0 |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) | external field test (+ calibration) | 2,572 (28 classes, mapped to PlantVillage) | CC BY 4.0 |
| [Tomato Leaf Disease, Pakistan](https://data.mendeley.com/datasets/3mbnb82mxd) | external field test | 7,200 (6 tomato classes) | CC BY 4.0 |

All three were downloaded automatically (Kaggle needed an API token, so the original authors' repositories were
used) and verified (`scripts/verify_datasets.py`). Splits are **leakage-free**: PlantVillage photographs of the same
physical leaf, exact duplicates and near-duplicates (perceptual hash) are always kept in the same split
(≈ 70 / 15 / 15 → 39,309 train, 7,245 validation, 7,751 test). Details: [docs/DATASET_REPORT.md](docs/DATASET_REPORT.md).

## Model and training

* **Architecture:** EfficientNet-B0 (4.1 M parameters) pretrained on ImageNet, classifier head replaced with a
  38-class layer (dropout 0.2). Input 224×224.
* **Transfer learning:** whole network fine-tuned, head learning rate 1.95e-3, backbone 0.4× that; AdamW,
  cosine schedule with warm-up, label smoothing 0.12, class-weighted loss for the 36× class imbalance.
* **Augmentation:** random resized crop, flips, rotation, brightness/contrast/saturation jitter, light blur.
* **Tuning:** 8-trial Optuna search on a 4,000-image subset (1 epoch per trial).
* **Training:** 5 epochs on the full training split (~1 h 40 min on an Apple M1 GPU), early stopping on validation
  macro-F1, best epoch = 5 (validation accuracy 99.72%).
* **Confidence:** temperature scaling plus a confidence threshold chosen on validation data. Results with confidence
  ≥ 73% are *confident*; 65–73% are *uncertain* (advice hidden, top-3 shown); below 65% the photo is *not recognised*.
  Blurry photos are never reported as confident.

Details: [docs/ML_METHODOLOGY.md](docs/ML_METHODOLOGY.md) · run log: [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)

<!-- RESULTS:START -->
## Results (held-out data)

Deployed: **EfficientNet-B0** — selected on validation data only.

| Model | PlantVillage test (internal) acc / macro-F1 | PlantDoc test (field) acc / macro-F1 | Tomato field, Pakistan (field) acc / macro-F1 | CPU ms |
|---|---:|---:|---:|---:|
| EfficientNet-B0 ⭐ | 99.6% / 99.5% | 31.1% / 28.3% | 26.5% / 24.2% | 133 |

Lab-style accuracy does **not** carry over to field photos; the app therefore flags uncertain results and withholds advice below a validation-selected confidence threshold. Details: [docs/MODEL_EVALUATION_REPORT.md](docs/MODEL_EVALUATION_REPORT.md).

![screens](docs/screenshots/desktop-03-results-en.png)
<!-- RESULTS:END -->

**PlantVillage test set (7,751 images, never used in training):** accuracy **99.56%**, macro precision **99.46%**,
macro recall **99.52%**, macro F1 **99.49%** — 34 errors in total, mostly between similar lesions on the same crop.
Full per-class report, confusion matrices, calibration and error analysis:
[docs/MODEL_EVALUATION_REPORT.md](docs/MODEL_EVALUATION_REPORT.md).

## Run the project

The trained model, evaluation results and advisory data are already in the project folders, so **no training is
needed** to run the app.

**Requirements:** Python 3.12, Node.js ≥ 20.9 (22 recommended).

```bash
cd krishiguard-ai

# 1) Backend  →  http://localhost:8000/docs
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt
cd backend
KG_DATABASE_URL=sqlite:///./krishiguard_demo.db ../.venv/bin/uvicorn app.main:app --port 8000
#   (omit KG_DATABASE_URL to use PostgreSQL: postgresql+psycopg://krishiguard:krishiguard_dev@localhost:5432/krishiguard)

# 2) Frontend →  http://localhost:3000   (new terminal)
cd frontend
npm ci
npm run build && npm start        # or: npm run dev
```

Open http://localhost:3000 → **Diagnose a leaf** → upload a leaf photo (for a quick demo use any image from
`ml/datasets/raw/plantvillage/color/<class>/`) → see the result, switch **हिंदी / English**, inspect Grad-CAM,
download the report, then open **Model performance**.

Docker alternative: `cp .env.example .env && docker compose up --build`.

**Free online deployment:** frontend on **Vercel** (root directory `frontend`, env `BACKEND_URL`) + backend on a
**Hugging Face Space** (`scripts/deploy_hf_space.py`). Step-by-step: [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md).

### Tests (all passing on 2026-11-05)

```bash
cd backend  && ../.venv/bin/python -m pytest   # 33 passed: preprocessing, model, API, real-model integration
cd frontend && npm test                         # 6 passed: Vitest component tests
cd frontend && npx playwright test              # 8 passed: full journey on desktop + mobile (servers must be running)
```

### Re-creating the ML results (optional, heavy)

```bash
.venv/bin/pip install timm mlflow-skinny optuna imagehash pandas scikit-learn matplotlib seaborn
.venv/bin/python scripts/verify_datasets.py              # check the datasets
.venv/bin/python -I ml/preprocessing/build_dataset.py     # splits + EDA (~1 min)
.venv/bin/python ml/training/train.py --arch efficientnet_b0 --run-name effnet_b0_224 --epochs 5 \
    --lr 0.00195 --backbone-lr-mult 0.401 --weight-decay 0.0181 --label-smoothing 0.12   # ~1h40 on M1
.venv/bin/python ml/evaluation/evaluate.py --run effnet_b0_224                           # ~4 min
.venv/bin/python ml/evaluation/compare_and_export.py && .venv/bin/python ml/evaluation/write_report.py
```

## Project structure

```
krishiguard-ai/
├── frontend/      Next.js app (7 pages), component tests, Playwright E2E
├── backend/       FastAPI app (predict, batch, diseases, advisory, metrics, feedback, health) + tests
├── ml/            datasets/, preprocessing/, training/, evaluation/, experiments/, artifacts/
├── advisory/      diseases.json (38 entries, English + Hindi, 51 cited sources) + SOURCES.md
├── docs/          reports, guides, screenshots
├── scripts/       dataset verification, figure sync, benchmarks
└── docker-compose.yml, .env.example
```

## Limitations

* Trained on **lab photos** (single leaf, plain background): accuracy on real field photos is about 26–31%.
  The app marks most field photos as *not recognised* rather than giving a wrong answer, but some confident field
  answers are still wrong.
* Only the 38 PlantVillage classes; other crops, pests or nutrient problems are forced onto these labels.
* Advisory content is compiled from cited sources but has not been reviewed by an agronomist; pesticide
  registrations change — always follow the product label and your local KVK.
* Full list: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Planned but not completed (scope reduced for a laptop-friendly college project)

| Item | Status |
|---|---|
| ResNet-50 and Vision Transformer comparison | Not trained (only a 1-epoch ResNet-50 learning-rate probe was run). The training script supports them (`--arch resnet50`, `--arch vit_small_patch16_224.augreg_in21k_ft_in1k`). |
| EfficientNet-B2, domain adaptation with PlantDoc images | Not run; stages exist in `ml/training/run_experiments.py`. |
| Docker images | Dockerfiles and compose written and validated (`docker compose config`), **not built** (Docker daemon not running, low disk). |
| Public cloud deployment | Prepared (Vercel + Hugging Face Space files and guide) but not deployed — needs your Vercel/HF accounts. |

## Documentation

[Resource report](RESOURCE_REPORT.md) · [Architecture](docs/PROJECT_ARCHITECTURE.md) · [Dataset report](docs/DATASET_REPORT.md) ·
[ML methodology](docs/ML_METHODOLOGY.md) · [Evaluation report](docs/MODEL_EVALUATION_REPORT.md) ·
[Experiment log](docs/EXPERIMENT_LOG.md) · [API](docs/API_DOCUMENTATION.md) · [Deployment](docs/DEPLOYMENT_GUIDE.md) ·
[User guide (EN/HI)](docs/USER_GUIDE.md) · [Limitations](docs/LIMITATIONS.md) · [Advisory sources](advisory/SOURCES.md)

## Credits

PlantVillage — Mohanty, Hughes & Salathé (2016), *Frontiers in Plant Science* 7:1419. PlantDoc — Singh et al. (2020),
CODS-COMAD. Tomato Leaf Disease Dataset (Pakistan) — Mendeley Data 3mbnb82mxd. Datasets and model training
checkpoints are not committed to git (see `.gitignore`).
