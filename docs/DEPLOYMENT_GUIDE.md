# Deployment Guide

## What ships

| Component | Image | Notes |
|---|---|---|
| `frontend` | `frontend/Dockerfile` (node:22-alpine, Next.js standalone) | Port 3000, public. Proxies `/api/*`, `/docs` to the backend. `BACKEND_URL` is a **build arg** (rewrites are resolved at build time). |
| `backend` | `backend/Dockerfile` (python:3.12-slim, CPU-only torch) | Port 8000, internal only. Bundles `advisory/diseases.json`, `ml/artifacts/serving/{model.pt,model_card.json}`, `ml/artifacts/evaluation/summary.json`. Non-root user, healthcheck on `/api/health`. |
| `db` | `postgres:16-alpine` | Volume `pgdata`, healthcheck with `pg_isready`. |

Datasets, training checkpoints and MLflow/Optuna stores are excluded by `.gitignore` / `.dockerignore`; only the
exported serving model (~16–100 MB depending on the selected architecture) enters the backend image.

## Prerequisites

1. Serving artefacts exist: `ml/artifacts/serving/model.pt`, `model_card.json`, `ml/artifacts/evaluation/summary.json`
   (produced by `python ml/evaluation/compare_and_export.py`).
2. Docker Engine ≥ 24 with Compose v2 (≥ 2.24 for optional `env_file`).

## Run with Docker Compose

```bash
cp .env.example .env               # set POSTGRES_PASSWORD
docker compose up --build -d
docker compose ps                  # all three should become "healthy"
open http://localhost:3000
```

Verified in this environment: `docker compose config` validates. **Not verified:** image builds/runs — the Docker
daemon on the development machine was not running and the disk had < 8 GB free (a CPU torch image is ~1 GB).
Start Docker Desktop and run the commands above to verify.

## Run locally without Docker (verified)

```bash
# backend (from repo root)
python -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt
cd backend && KG_DATABASE_URL=postgresql+psycopg://krishiguard:krishiguard_dev@localhost:5432/krishiguard \
  ../.venv/bin/uvicorn app.main:app --port 8000
# frontend (Node ≥ 20.9)
cd frontend && npm ci && npm run build && npm start      # http://localhost:3000
```
Without Postgres the API still diagnoses (health = `degraded`, feedback returns 503). For a quick local DB use
`KG_DATABASE_URL=sqlite:///./krishiguard.db`.

## Recommended free deployment: everything on Vercel (Hobby plan, no credit card)

The app is deployed as **two Vercel projects from the same GitHub repository**:

| Project | Root Directory | What it runs |
|---|---|---|
| `krishiguard-api` | `.` (repository root) | FastAPI as a Python function (`index.py`, auto-detected). Inference uses **onnxruntime** (`ml/artifacts/serving/model.onnx`) — no PyTorch — so the function is ~150 MB instead of > 1 GB. |
| `krishiguard-ai` (web) | `frontend` | Next.js site; `/api/*` and `/docs` are proxied to the API project. |

Files used by the API project: `index.py`, `requirements.txt`, `.python-version` (3.12), `vercel.json`
(function max duration 60 s) and `.vercelignore` (keeps datasets, frontend, docs and training code out of the upload).

**1. API project**
1. https://vercel.com → *Add New → Project* → import `krishiguard-ai` from GitHub.
2. Project name `krishiguard-api`; **Root Directory: leave as `./`**; Framework preset: *FastAPI* (or *Other*).
3. Deploy. Check `https://krishiguard-api.vercel.app/api/health` → `"model_loaded": true` (use the domain Vercel shows).

**2. Web project**
1. *Add New → Project* → import the **same** repository again.
2. Project name e.g. `krishiguard-ai`; **Root Directory: `frontend`** (Next.js is detected).
3. Environment variable `BACKEND_URL` = the API URL from step 1 (e.g. `https://krishiguard-api.vercel.app`, no trailing slash).
4. Deploy and open the site.

Both projects redeploy automatically on every `git push`. If `BACKEND_URL` changes, redeploy the web project
(it is read at build time).

**Free-plan behaviour**
* No sleeping: the first request after a quiet period takes a few seconds (cold start: model load ≈ 1–2 s),
  later requests ~50 ms of inference.
* Uploads larger than ~4.5 MB would exceed Vercel's request limit, so the browser downscales photos to 1,600 px first.
* The API uses SQLite in `/tmp` on Vercel, so feedback/prediction logs are temporary (fine for a demo). Set
  `KG_DATABASE_URL` in the API project's environment variables to a free hosted Postgres (e.g. Neon) to keep them.
* Heat-maps are computed as **CAM** inside the ONNX graph. For EfficientNet-B0 (global average pooling + linear
  head) this equals Grad-CAM up to scale; `ml/evaluation/export_onnx.py` verifies it (correlation 0.99999999) and
  that ONNX predictions match PyTorch on 105 test images (max logit difference 1e-5).

## Cloud deployment options

### Option A — single VM (simplest; recommended for a demo)
Any 2 vCPU / 4 GB VM (e.g. AWS Lightsail, DigitalOcean, GCP e2-medium):
```bash
sudo apt install docker.io docker-compose-v2 && git clone <repo> && cd krishiguard-ai
# copy ml/artifacts/serving + ml/artifacts/evaluation/summary.json to the VM (not in git)
cp .env.example .env && nano .env && sudo docker compose up --build -d
```
Put Caddy/Nginx in front for TLS (e.g. Caddy: `reverse_proxy localhost:3000`). Set `KG_CORS_ORIGINS` to your domain.

### Option B — managed containers
* **Backend:** Google Cloud Run / AWS App Runner / Render / Fly.io from `backend/Dockerfile` (build context = repo root).
  Give it ≥ 1 GiB memory, 1 instance minimum to avoid cold-start model loading (~2 s), env `KG_DATABASE_URL`.
* **Database:** managed Postgres (Cloud SQL, RDS, Neon, Supabase). Tables are created automatically on startup.
* **Frontend:** Vercel or the same container platform; build with `BACKEND_URL=https://<backend-host>`.

### Production checklist
- [ ] `POSTGRES_PASSWORD` strong, `.env` not committed
- [ ] `KG_CORS_ORIGINS` restricted to the real origin
- [ ] TLS terminated at the proxy; `--proxy-headers` already set so rate limiting sees client IPs (trust only your proxy)
- [ ] Rate limits tuned (`KG_RATE_LIMIT_PREDICT`); limiter is in-memory per process — use Redis storage if scaling out
- [ ] Logs (JSON lines on stdout) shipped to your log platform; alert on `level=ERROR`
- [ ] `model_card.json` SHA-256 matches `model.pt` (the API refuses to start serving otherwise)
- [ ] Advisory content reviewed by a local agronomist before public launch (see LIMITATIONS)

## Updating the model
1. Train/evaluate (`ml/training/run_experiments.py`, `ml/evaluation/evaluate.py --run …`).
2. `python ml/evaluation/compare_and_export.py` → new `model.pt`, `model_card.json` (new version string = run@sha).
3. `scripts/sync_figures.sh` (methodology figures), rebuild images. The new version is registered in
   `model_versions` and its evaluation in `evaluation_runs` at backend start-up.
