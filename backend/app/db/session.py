from __future__ import annotations

import logging

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.logging import log
from app.db.models import Base

logger = logging.getLogger(__name__)


class Database:
    """Fail-soft database: if it is unreachable, inference keeps working and persistence endpoints
    return 503 instead of crashing the service."""

    def __init__(self, url: str):
        self.url = url
        self.engine: Engine | None = None
        self.SessionLocal: sessionmaker | None = None
        self.error: str | None = None

    def connect(self) -> None:
        try:
            kw = {"connect_args": {"check_same_thread": False}} if self.url.startswith("sqlite") else {"pool_pre_ping": True}
            if self.url == "sqlite://":
                from sqlalchemy.pool import StaticPool

                kw["poolclass"] = StaticPool
            engine = create_engine(self.url, **kw)
            Base.metadata.create_all(engine)
            self.engine, self.SessionLocal, self.error = engine, sessionmaker(engine, expire_on_commit=False), None
            log(logger, logging.INFO, "database ready", dialect=engine.dialect.name)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {str(e).splitlines()[0]}"
            log(logger, logging.WARNING, "database unavailable", error=self.error)

    @property
    def ready(self) -> bool:
        return self.SessionLocal is not None

    def session(self) -> Session | None:
        return self.SessionLocal() if self.SessionLocal else None
