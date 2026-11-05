"""KrishiGuard AI — FastAPI application."""
from __future__ import annotations

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
from app.bootstrap import init_state

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("krishiguard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_state(app)  # also done lazily on first request for serverless runtimes without lifespan events
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
