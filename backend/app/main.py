"""KrishiGuard AI — FastAPI application."""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.errors import AppError, app_error_handler
from app.core.logging import log, setup_logging
from app.core.ratelimit import limiter
from app.db.models import AdvisorySnapshot, EvaluationRun, ModelVersion
from app.db.session import Database
from app.ml.inference import ModelService
from app.services.advisory import AdvisoryStore

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("krishiguard")


def _register_metadata(app: FastAPI) -> None:
    """Record model version, evaluation summary and advisory snapshot (idempotent)."""
    session = app.state.db.session()
    if session is None:
        return
    try:
        with session:
            m = app.state.model
            if m.ready and session.get(ModelVersion, m.cfg.model_version) is None:
                session.add(ModelVersion(version=m.cfg.model_version, arch=m.cfg.arch, checkpoint_sha256=m.cfg.checkpoint_sha256,
                                         card={**m.cfg.extra, "classes": m.cfg.classes, "temperature": m.cfg.temperature}))
                session.flush()
            summary = settings.evaluation_dir / "summary.json"
            if m.ready and summary.exists():
                raw = summary.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if session.query(EvaluationRun).filter_by(summary_sha256=digest).first() is None:
                    session.add(EvaluationRun(model_version=m.cfg.model_version, summary_sha256=digest, metrics=json.loads(raw)))
            a = app.state.advisory
            if a is not None and session.get(AdvisorySnapshot, a.version_hash) is None:
                session.add(AdvisorySnapshot(version_hash=a.version_hash, version=a.version,
                                             content=json.loads(settings.advisory_path.read_text())))
            session.commit()
    except Exception as e:  # noqa: BLE001
        log(logger, logging.WARNING, "metadata registration failed", error=repr(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ModelService(settings.serving_dir, settings.device, settings.torch_threads)
    app.state.model.load()
    app.state.advisory = AdvisoryStore(settings.advisory_path)
    app.state.db = Database(settings.database_url)
    app.state.db.connect()
    _register_metadata(app)
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Crop leaf disease detection with calibrated uncertainty, Grad-CAM explanations and bilingual (English/Hindi) advisory.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"], max_age=600,
)
app.add_exception_handler(AppError, app_error_handler)


@app.exception_handler(RateLimitExceeded)
async def _rate_limited(_: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": {"code": "rate_limited", "message": f"Too many requests: {exc.detail}", "details": {}}})


@app.exception_handler(RequestValidationError)
async def _validation(_: Request, exc: RequestValidationError):
    errs = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": {"code": "validation_error", "message": "Invalid request.", "details": {"errors": errs}}})


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log(logger, logging.ERROR, "unhandled error", path=request.url.path, error=repr(exc))
    return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "Something went wrong.", "details": {}}})


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    log(logger, logging.INFO, "request", rid=rid, method=request.method, path=request.url.path,
        status=response.status_code, ms=round((time.perf_counter() - t0) * 1000, 1))
    return response


app.include_router(router)
