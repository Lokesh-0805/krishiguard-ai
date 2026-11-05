"""Model loading, image validation/preprocessing, calibrated prediction and uncertainty handling."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.errors import InvalidImageError, PayloadTooLargeError, ServiceUnavailableError, UnsupportedMediaError
from app.core.logging import log
from app.ml.gradcam import GradCAM, default_target_layer, render

logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PIL_FORMAT_TO_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "MPO": "image/jpeg"}


@dataclass
class DecodedImage:
    image: Image.Image  # RGB, EXIF-orientation corrected
    mime: str
    sha256: str
    width: int
    height: int
    blur_score: float


def decode_upload(data: bytes, declared_mime: str | None, *, max_bytes: int, allowed: tuple[str, ...], min_side: int) -> DecodedImage:
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise PayloadTooLargeError(f"File is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    if declared_mime and declared_mime.split(";")[0].strip().lower() not in allowed + ("application/octet-stream",):
        raise UnsupportedMediaError(f"Unsupported file type '{declared_mime}'. Upload a JPEG, PNG or WebP photo.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise InvalidImageError("The file could not be read as an image.", details={"reason": type(e).__name__}) from e
    mime = PIL_FORMAT_TO_MIME.get(fmt or "")
    if mime not in allowed:  # content sniffing beats the declared header
        raise UnsupportedMediaError(f"Detected format '{fmt}' is not supported. Upload a JPEG, PNG or WebP photo.")
    img = ImageOps.exif_transpose(img).convert("RGB")
    if min(img.size) < min_side:
        raise InvalidImageError(f"Image is too small ({img.width}x{img.height}). Minimum side is {min_side}px.")
    return DecodedImage(img, mime, hashlib.sha256(data).hexdigest(), img.width, img.height, blur_score(img))


def blur_score(img: Image.Image) -> float:
    """Variance of a 3x3 Laplacian on a 256x256 grayscale version (resolution-normalised sharpness)."""
    g = np.asarray(img.convert("L").resize((256, 256), Image.BILINEAR), dtype=np.float32)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def model_view(img: Image.Image, size: int) -> Image.Image:
    """Exactly the evaluation transform used in training: resize shorter side to size/0.875, center crop."""
    short = int(round(size / 0.875))
    w, h = img.size
    if w <= h:
        nw, nh = short, int(short * h / w)
    else:
        nw, nh = int(short * w / h), short
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = int(round((nw - size) / 2.0)), int(round((nh - size) / 2.0))
    return img.crop((left, top, left + size, top + size))


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


@dataclass
class ServingConfig:
    model_version: str
    arch: str
    img_size: int
    classes: list[str]
    temperature: float
    confidence_threshold: float
    reject_threshold: float
    blur_threshold: float
    checkpoint: str
    checkpoint_sha256: str
    gradcam_target_layer: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, serving_dir: Path) -> "ServingConfig":
        raw = json.loads((serving_dir / "model_card.json").read_text())
        known = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw and k != "extra"}
        return cls(**known, extra={k: v for k, v in raw.items() if k not in known})


class ModelService:
    """Holds the single deployed model. Loaded once at startup; thread-safe for inference."""

    def __init__(self, serving_dir: Path, device: str = "cpu", threads: int = 4):
        self.serving_dir = serving_dir
        self.device = torch.device(device)
        self.cfg: ServingConfig | None = None
        self.model: torch.nn.Module | None = None
        self.cam: GradCAM | None = None
        self.load_error: str | None = None
        torch.set_num_threads(threads)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        import timm

        t0 = time.time()
        try:
            cfg = ServingConfig.load(self.serving_dir)
            ckpt_path = self.serving_dir / cfg.checkpoint
            digest = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
            if digest != cfg.checkpoint_sha256:
                raise RuntimeError("checkpoint sha256 mismatch with model_card.json")
            model = timm.create_model(cfg.arch, pretrained=False, num_classes=len(cfg.classes))
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
            model.eval().to(self.device)
            target = cfg.gradcam_target_layer or default_target_layer(cfg.arch)
            self.cam = GradCAM(model, target, getattr(model, "num_prefix_tokens", 0))
            with torch.no_grad():  # warm-up
                model(torch.zeros(1, 3, cfg.img_size, cfg.img_size, device=self.device))
            self.cfg, self.model, self.load_error = cfg, model, None
            log(logger, logging.INFO, "model loaded", version=cfg.model_version, arch=cfg.arch, seconds=round(time.time() - t0, 2))
        except FileNotFoundError as e:
            self.load_error = f"model artefacts not found: {e.filename}"
            log(logger, logging.ERROR, "model not loaded", error=self.load_error)
        except Exception as e:  # noqa: BLE001 - surface any loading failure through /api/health
            self.load_error = f"{type(e).__name__}: {e}"
            log(logger, logging.ERROR, "model not loaded", error=self.load_error)

    def _require(self) -> tuple[torch.nn.Module, ServingConfig]:
        if self.model is None or self.cfg is None:
            raise ServiceUnavailableError("The diagnosis model is not loaded.", details={"reason": self.load_error})
        return self.model, self.cfg

    @torch.no_grad()
    def predict(self, images: list[Image.Image]) -> list[dict]:
        model, cfg = self._require()
        views = [model_view(im, cfg.img_size) for im in images]
        x = torch.stack([to_tensor(v) for v in views]).to(self.device)
        t0 = time.perf_counter()
        logits = model(x).float().cpu()
        latency_ms = (time.perf_counter() - t0) * 1000 / len(images)
        probs = torch.softmax(logits / cfg.temperature, dim=1).numpy()
        return [self._interpret(p, latency_ms) for p in probs]

    def _interpret(self, p: np.ndarray, latency_ms: float) -> dict:
        cfg = self.cfg
        order = np.argsort(-p)
        top = [{"class_id": cfg.classes[i], "probability": float(p[i])} for i in order[:3]]
        crops: dict[str, float] = {}
        for i, c in enumerate(cfg.classes):
            crops[c.split("___")[0]] = crops.get(c.split("___")[0], 0.0) + float(p[i])
        best_crop = max(crops, key=crops.get)
        conf = float(p[order[0]])
        if conf >= cfg.confidence_threshold:
            status = "confident"
        elif conf >= cfg.reject_threshold:
            status = "uncertain"
        else:
            status = "unrecognized"
        return {
            "class_index": int(order[0]),
            "class_id": cfg.classes[order[0]],
            "confidence": conf,
            "crop_key": best_crop,
            "crop_confidence": crops[best_crop],
            "status": status,
            "top_k": top,
            "latency_ms": latency_ms,
        }

    def explain(self, image: Image.Image, class_idx: int) -> tuple[bytes, bytes, bytes]:
        model, cfg = self._require()
        view = model_view(image, cfg.img_size)
        cam = self.cam(to_tensor(view)[None].to(self.device), class_idx)
        heat, overlay = render(cam, view)
        buf = io.BytesIO()
        v = view.copy()
        v.thumbnail((448, 448))
        v.save(buf, format="PNG")
        return buf.getvalue(), heat, overlay
