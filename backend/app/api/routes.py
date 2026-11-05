from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, File, Form, Query, Request, UploadFile

from app.bootstrap import init_state
from app.api.schemas import BatchItemError, BatchOut, FeedbackIn, FeedbackOut, HealthOut, PredictionOut
from app.core.config import get_settings
from app.core.errors import AppError, InvalidImageError, NotFoundError, ServiceUnavailableError
from app.core.logging import log
from app.core.ratelimit import limiter
from app.db.models import Feedback, Prediction
from app.ml.inference import decode_upload
from app.services.metrics import load_metrics

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)
settings = get_settings()

STATUS_MESSAGES = {
    "confident": {
        "en": "The model is confident about this result. Confirm with a local expert before spraying.",
        "hi": "मॉडल इस परिणाम को लेकर आश्वस्त है। छिड़काव से पहले स्थानीय विशेषज्ञ से पुष्टि करें।",
    },
    "uncertain": {
        "en": "The model is not sure. Possible matches are shown below — retake a clear photo of a single leaf or consult your local KVK.",
        "hi": "मॉडल निश्चित नहीं है। संभावित परिणाम नीचे दिए हैं — एक पत्ती की साफ़ फ़ोटो दोबारा लें या अपने नज़दीकी KVK से सलाह लें।",
    },
    "unrecognized": {
        "en": "This image could not be matched to a supported crop or disease. It may be an unsupported plant, not a leaf, or a poor-quality photo.",
        "hi": "यह फ़ोटो किसी समर्थित फसल या रोग से मेल नहीं खाती। यह असमर्थित पौधा हो सकता है, पत्ती न हो, या फ़ोटो साफ़ न हो।",
    },
}


def _svc(request: Request):
    init_state(request.app)
    return request.app.state.model, request.app.state.advisory, request.app.state.db


def _split(class_id: str) -> tuple[str, str]:
    crop, disease = class_id.split("___", 1)
    return crop, disease


def _labels(advisory, class_id: str, lang: str) -> tuple[str, str, bool]:
    try:
        e = advisory.resolve(class_id)
        return e["crop"][lang], e["name"][lang], bool(e["healthy"])
    except NotFoundError:
        crop, disease = _split(class_id)
        return crop.replace("_", " "), disease.replace("_", " "), disease == "healthy"


def _build_prediction(request: Request, upload_bytes: bytes, content_type: str | None, filename: str | None, lang: str, explain: bool) -> PredictionOut:
    model, advisory, db = _svc(request)
    if not model.ready:
        raise ServiceUnavailableError("The diagnosis model is not loaded.", details={"reason": model.load_error})
    img = decode_upload(
        upload_bytes, content_type,
        max_bytes=int(settings.max_upload_mb * 1024 * 1024), allowed=settings.allowed_mime, min_side=settings.min_image_side,
    )
    cfg = model.cfg
    blur_thr = settings.blur_warning_threshold if settings.blur_warning_threshold is not None else cfg.blur_threshold
    flags = []
    if img.blur_score < blur_thr:
        flags.append("blurry")
    if min(img.width, img.height) < cfg.img_size:
        flags.append("low_resolution")
    try:
        r = model.predict([img.image])[0]
    except AppError:
        raise
    except Exception as e:  # noqa: BLE001
        log(logger, logging.ERROR, "inference failed", error=repr(e))
        raise AppError("Inference failed. Please try again.", code="inference_failed", status_code=500) from e

    status = r["status"]
    if status == "confident" and "blurry" in flags:
        status = "uncertain"  # do not present a confident diagnosis on a blurry photo
    crop, disease, healthy = _labels(advisory, r["class_id"], lang)
    top_k = []
    for t in r["top_k"]:
        c, d, _ = _labels(advisory, t["class_id"], lang)
        top_k.append({"class_id": t["class_id"], "crop": c, "disease": d, "probability": t["probability"]})
    adv = None
    if status == "confident":
        try:
            adv = advisory.advisory(advisory.resolve(r["class_id"]), lang)
        except NotFoundError:
            adv = None

    explanation = None
    if explain and status != "unrecognized":
        model_in, heat, overlay = model.explain(img.image, r["class_index"])
        b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
        explanation = {
            "method": "CAM (equivalent to Grad-CAM for this model)",
            "target_layer": cfg.gradcam_target_layer or "default",
            "model_input_png": b64(model_in),
            "heatmap_png": b64(heat),
            "overlay_png": b64(overlay),
            "note": "Highlights image regions that most influenced the predicted class score. It is not proof that the diagnosis is correct.",
        }

    pred_id = None
    session = db.session()
    if session is not None:
        try:
            with session:
                row = Prediction(
                    model_version=cfg.model_version, image_sha256=img.sha256, image_width=img.width, image_height=img.height,
                    predicted_class=r["class_id"], confidence=r["confidence"], status=status, top_k=r["top_k"],
                    quality_flags=flags, latency_ms=r["latency_ms"],
                )
                session.add(row)
                session.commit()
                pred_id = row.id
        except Exception as e:  # noqa: BLE001 - persistence must never break a diagnosis
            log(logger, logging.WARNING, "prediction not persisted", error=repr(e))

    log(logger, logging.INFO, "prediction", class_id=r["class_id"], confidence=round(r["confidence"], 4), status=status,
        flags=flags, latency_ms=round(r["latency_ms"], 1))
    crop_c, _, _ = _labels(advisory, r["class_id"], lang)
    return PredictionOut(
        prediction_id=pred_id, model_version=cfg.model_version, status=status, status_message=STATUS_MESSAGES[status][lang],
        class_id=r["class_id"], crop=crop, disease=disease, healthy=healthy, confidence=r["confidence"],
        crop_confidence=r["crop_confidence"], top_k=top_k,
        quality={"width": img.width, "height": img.height, "blur_score": round(img.blur_score, 1), "flags": flags},
        latency_ms=r["latency_ms"], advisory=adv, explanation=explanation, filename=filename,
    )


def _read_upload(f: UploadFile) -> bytes:
    limit = int(settings.max_upload_mb * 1024 * 1024)
    return f.file.read(limit + 1)  # read one byte past the limit so oversize is detected without reading everything


@router.get("/health", response_model=HealthOut, tags=["system"])
def health(request: Request):
    model, advisory, db = _svc(request)
    ok = model.ready and advisory is not None
    return HealthOut(
        status="ok" if ok and db.ready else "degraded",
        model_loaded=model.ready,
        model_version=model.cfg.model_version if model.cfg else None,
        database=db.ready,
        advisory_entries=len(advisory.entries) if advisory else 0,
        detail={k: v for k, v in {"model_error": model.load_error, "database_error": db.error}.items() if v},
    )


@router.get("/model/info", tags=["model"])
def model_info(request: Request):
    model, _, _ = _svc(request)
    if not model.ready:
        raise ServiceUnavailableError("The diagnosis model is not loaded.", details={"reason": model.load_error})
    c = model.cfg
    return {
        "model_version": c.model_version, "architecture": c.arch, "input_size": c.img_size, "num_classes": len(c.classes),
        "classes": c.classes, "temperature": c.temperature, "confidence_threshold": c.confidence_threshold,
        "reject_threshold": c.reject_threshold, "blur_threshold": c.blur_threshold, "checkpoint_sha256": c.checkpoint_sha256,
        "gradcam_target_layer": c.gradcam_target_layer, **{k: v for k, v in c.extra.items() if k in ("trained_on", "parameters_m", "size_mb", "selected_because", "created_at", "run_name")},
    }


@router.post("/predict", response_model=PredictionOut, tags=["inference"])
@limiter.limit(settings.rate_limit_predict)
def predict(
    request: Request,
    file: UploadFile = File(..., description="Leaf photo (JPEG/PNG/WebP)"),
    lang: str = Form("en", pattern="^(en|hi)$"),
    explain: bool = Form(True),
):
    return _build_prediction(request, _read_upload(file), file.content_type, file.filename, lang, explain)


@router.post("/predict/batch", response_model=BatchOut, tags=["inference"])
@limiter.limit(settings.rate_limit_predict)
def predict_batch(request: Request, files: list[UploadFile] = File(...), lang: str = Form("en", pattern="^(en|hi)$")):
    if len(files) > settings.max_batch_files:
        raise InvalidImageError(f"At most {settings.max_batch_files} files per batch.", code="too_many_files")
    results = []
    for f in files:
        try:
            results.append(_build_prediction(request, _read_upload(f), f.content_type, f.filename, lang, explain=False))
        except ServiceUnavailableError:
            raise
        except AppError as e:
            results.append(BatchItemError(filename=f.filename, error={"code": e.code, "message": e.message}))
    return BatchOut(results=results)


@router.get("/diseases", tags=["advisory"])
def list_diseases(request: Request, lang: str = Query("en", pattern="^(en|hi)$"), crop: str | None = None):
    _, advisory, _ = _svc(request)
    items = [advisory.summary(e, lang) for e in advisory.entries.values()]
    if crop:
        items = [i for i in items if i["id"].lower().startswith(crop.lower())]
    return {"advisory_version": advisory.version, "count": len(items), "items": items}


@router.get("/diseases/{disease_id}", tags=["advisory"])
def get_disease(request: Request, disease_id: str, lang: str = Query("en", pattern="^(en|hi)$")):
    _, advisory, _ = _svc(request)
    entry = advisory.resolve(disease_id)
    return {**advisory.advisory(entry, lang), "names": {"en": entry["name"]["en"], "hi": entry["name"]["hi"]},
            "crops": entry["crop"]}


@router.get("/advisory/{disease_id}", tags=["advisory"])
def get_advisory(request: Request, disease_id: str, lang: str = Query("en", pattern="^(en|hi)$")):
    _, advisory, _ = _svc(request)
    return advisory.advisory(advisory.resolve(disease_id), lang)


@router.get("/metrics", tags=["model"])
def metrics():
    return load_metrics(settings.evaluation_dir)


@router.post("/feedback", response_model=FeedbackOut, status_code=201, tags=["feedback"])
@limiter.limit("20/minute")
def feedback(request: Request, body: FeedbackIn):
    _, advisory, db = _svc(request)
    session = db.session()
    if session is None:
        raise ServiceUnavailableError("Feedback storage is unavailable.", details={"reason": db.error})
    if body.suggested_class is not None and body.suggested_class not in advisory.entries:
        raise InvalidImageError("suggested_class must be one of the supported class ids.", code="invalid_class")
    with session:
        if session.get(Prediction, body.prediction_id) is None:
            raise NotFoundError("Prediction not found.")
        fb = Feedback(**body.model_dump())
        session.add(fb)
        session.commit()
        return FeedbackOut(id=fb.id, verification_status=fb.verification_status)
