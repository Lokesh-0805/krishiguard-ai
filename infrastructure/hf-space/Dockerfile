# Hugging Face Spaces (Docker SDK) image for the KrishiGuard API.
# Build context = the folder assembled by scripts/deploy_hf_space.py (backend/app, advisory, ml artefacts).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.*"
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY --chown=user backend/app backend/app
COPY --chown=user advisory advisory
COPY --chown=user ml ml

USER user
WORKDIR /app/backend
# Demo settings: SQLite in /tmp (resets on restart), CPU inference, all browser traffic arrives via the Vercel proxy
ENV KG_DATABASE_URL=sqlite:////tmp/krishiguard.db \
    KG_DEVICE=cpu \
    KG_TORCH_THREADS=2 \
    KG_RATE_LIMIT_PREDICT=120/minute \
    KG_RATE_LIMIT_DEFAULT=600/minute
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
