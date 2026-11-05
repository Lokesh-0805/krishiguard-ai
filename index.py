"""Vercel entrypoint for the KrishiGuard API (Vercel project with Root Directory = repository root).

Vercel auto-detects `index.py` and serves the ASGI `app` below. The real application lives in backend/app.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

if os.environ.get("VERCEL"):
    # Serverless defaults (each can be overridden in the Vercel project's Environment Variables):
    os.environ.setdefault("KG_DATABASE_URL", "sqlite:////tmp/krishiguard.db")  # only /tmp is writable; resets on cold start
    os.environ.setdefault("KG_INFERENCE_THREADS", "1")                          # Hobby functions get 1 vCPU
    os.environ.setdefault("KG_RATE_LIMIT_PREDICT", "120/minute")                # all browser traffic arrives via the frontend proxy
    os.environ.setdefault("KG_RATE_LIMIT_DEFAULT", "600/minute")

from app.main import app  # noqa: E402,F401
