# syntax=docker/dockerfile:1
# Build context: repository root (needs backend/, advisory/, ml/artifacts/{serving,evaluation}, class mapping).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only torch keeps the image several times smaller than the default CUDA wheels
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.*" && \
    useradd --create-home --uid 10001 appuser
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

COPY backend/app /app/backend/app
COPY advisory/diseases.json /app/advisory/diseases.json
COPY ml/datasets/metadata/class_mapping.json /app/ml/datasets/metadata/class_mapping.json
COPY ml/artifacts/serving /app/ml/artifacts/serving
COPY ml/artifacts/evaluation/summary.json /app/ml/artifacts/evaluation/summary.json

WORKDIR /app/backend
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
