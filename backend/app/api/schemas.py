from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Lang = Literal["en", "hi"]
Status = Literal["confident", "uncertain", "unrecognized"]


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    database: bool
    advisory_entries: int
    detail: dict = {}


class TopK(BaseModel):
    class_id: str
    crop: str
    disease: str
    probability: float


class QualityReport(BaseModel):
    width: int
    height: int
    blur_score: float
    flags: list[str]


class Explanation(BaseModel):
    method: str
    target_layer: str
    model_input_png: str = Field(description="base64 PNG of the exact (cropped) region the model saw")
    heatmap_png: str
    overlay_png: str
    note: str


class PredictionOut(BaseModel):
    prediction_id: str | None
    model_version: str
    status: Status
    status_message: str
    class_id: str
    crop: str
    disease: str
    healthy: bool
    confidence: float
    crop_confidence: float
    top_k: list[TopK]
    quality: QualityReport
    latency_ms: float
    advisory: dict | None
    explanation: Explanation | None = None
    filename: str | None = None


class BatchItemError(BaseModel):
    filename: str | None
    error: dict


class BatchOut(BaseModel):
    results: list[PredictionOut | BatchItemError]


class FeedbackIn(BaseModel):
    prediction_id: str = Field(min_length=36, max_length=36)
    prediction_correct: bool | None = None
    suggested_class: str | None = Field(default=None, max_length=120)
    advisory_helpful: bool | None = None
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackOut(BaseModel):
    id: int
    verification_status: str
