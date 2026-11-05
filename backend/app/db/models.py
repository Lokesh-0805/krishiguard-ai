"""Persistence: prediction metadata, unverified user feedback, model versions, evaluation runs,
advisory snapshots. Uploaded images are NOT stored (only a SHA-256 for de-duplication/audit)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ModelVersion(Base):
    __tablename__ = "model_versions"
    version: Mapped[str] = mapped_column(String(80), primary_key=True)
    arch: Mapped[str] = mapped_column(String(80))
    checkpoint_sha256: Mapped[str] = mapped_column(String(64))
    card: Mapped[dict] = mapped_column(JSON)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.version"))
    summary_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    metrics: Mapped[dict] = mapped_column(JSON)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AdvisorySnapshot(Base):
    __tablename__ = "advisory_snapshots"
    version_hash: Mapped[str] = mapped_column(String(12), primary_key=True)
    version: Mapped[str] = mapped_column(String(20))
    content: Mapped[dict] = mapped_column(JSON)
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    model_version: Mapped[str] = mapped_column(String(80))
    image_sha256: Mapped[str] = mapped_column(String(64), index=True)
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    predicted_class: Mapped[str] = mapped_column(String(120))
    confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20))
    top_k: Mapped[list] = mapped_column(JSON)
    quality_flags: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    prediction_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    suggested_class: Mapped[str | None] = mapped_column(String(120), nullable=True)
    advisory_helpful: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    # feedback is user-reported and never used as ground truth without expert review
    verification_status: Mapped[str] = mapped_column(String(20), default="unverified")
