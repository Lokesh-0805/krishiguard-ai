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

## Recommended free deployment: Vercel (frontend) + Hugging Face Spaces (backend)

Vercel cannot host the backend (its Python functions are limited to ~250 MB, PyTorch alone is larger), so the API
runs on a free Hugging Face Docker Space (2 vCPU / 16 GB RAM) and Vercel proxies `/api/*` to it.

**1. Backend → Hugging Face Space**
1. Create a free account at https://huggingface.co and a **write** token at https://huggingface.co/settings/tokens.
2. From the project folder:
   ```bash
   export HF_TOKEN=hf_xxxxxxxx
   .venv/bin/python scripts/deploy_hf_space.py <your-hf-username>/krishiguard-api
   ```
   This uploads only what the API needs (`backend/app`, advisory, the 16.5 MB model, metrics) using
   `infrastructure/hf-space/Dockerfile`. The first build takes ~5–10 minutes (watch the logs on the Space page).
3. Check `https://<your-hf-username>-krishiguard-api.hf.space/api/health` → `"model_loaded": true`.

**2. Frontend → Vercel**
1. Push the repository to GitHub, then on https://vercel.com → *Add New Project* → import the repo.
2. **Root Directory:** `frontend` (framework is detected as Next.js; keep default build settings).
3. **Environment variable:** `BACKEND_URL = https://<your-hf-username>-krishiguard-api.hf.space` (no trailing slash).
   It is used at build time by the `/api/*` rewrite, so redeploy after changing it.
4. Deploy and open the Vercel URL.

Notes: the free Space sleeps after ~48 h without traffic — open the health URL a minute before a demo to wake it.
The Space uses SQLite in `/tmp`, so feedback/prediction logs reset on restart (fine for a demo). Large phone photos
are downscaled in the browser to 1,600 px before upload, keeping requests under Vercel's ~4.5 MB proxy limit.

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
