"""One-time application state: model, advisory store, database and metadata registration.

Called from the FastAPI lifespan hook and, lazily, on the first request (serverless platforms such as Vercel may
not run lifespan events). A lock makes it safe under concurrent first requests."""
from __future__ import annotations

import hashlib
import json
import logging
import threading

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import log
from app.db.models import AdvisorySnapshot, EvaluationRun, ModelVersion
from app.db.session import Database
from app.ml.inference import ModelService
from app.services.advisory import AdvisoryStore

logger = logging.getLogger("krishiguard")
_lock = threading.Lock()


def init_state(app: FastAPI) -> None:
    if getattr(app.state, "initialized", False):
        return
    with _lock:
        if getattr(app.state, "initialized", False):
            return
        settings = get_settings()
        app.state.model = ModelService(settings.serving_dir, settings.inference_threads)
        app.state.model.load()
        app.state.advisory = AdvisoryStore(settings.advisory_path)
        app.state.db = Database(settings.database_url)
        app.state.db.connect()
        _register_metadata(app)
        app.state.initialized = True


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
            summary = get_settings().evaluation_dir / "summary.json"
            if m.ready and summary.exists():
                raw = summary.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if session.query(EvaluationRun).filter_by(summary_sha256=digest).first() is None:
                    session.add(EvaluationRun(model_version=m.cfg.model_version, summary_sha256=digest, metrics=json.loads(raw)))
            a = app.state.advisory
            if a is not None and session.get(AdvisorySnapshot, a.version_hash) is None:
                session.add(AdvisorySnapshot(version_hash=a.version_hash, version=a.version,
                                             content=json.loads(get_settings().advisory_path.read_text())))
            session.commit()
    except Exception as e:  # noqa: BLE001
        log(logger, logging.WARNING, "metadata registration failed", error=repr(e))
