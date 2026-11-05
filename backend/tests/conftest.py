"""Test fixtures.

`fake_serving` builds a randomly-initialised EfficientNet-B0 serving bundle in a temp dir: it exercises the
real loading / preprocessing / Grad-CAM / API plumbing without depending on training outputs.
Tests that need the REAL trained model are marked `real_model` and skip when artefacts are absent.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

CLASSES = [c["id"] for c in json.loads((ROOT / "ml/datasets/metadata/class_mapping.json").read_text())["classes"]] \
    if (ROOT / "ml/datasets/metadata/class_mapping.json").exists() else [f"Crop{i}___d{i}" for i in range(38)]
REAL_SERVING = ROOT / "ml/artifacts/serving"


def jpeg_bytes(size=(320, 280), color=(40, 140, 60), noise=True, fmt="JPEG") -> bytes:
    rng = np.random.default_rng(0)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8) + np.array(color, dtype=np.uint8)
    if noise:
        arr = np.clip(arr.astype(int) + rng.integers(-60, 60, arr.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="session")
def fake_serving(tmp_path_factory) -> Path:
    import timm
    import torch

    d = tmp_path_factory.mktemp("serving")
    torch.manual_seed(0)
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASSES))
    torch.save(m.state_dict(), d / "model.pt")
    card = {
        "model_version": "test@00000000", "run_name": "test", "arch": "efficientnet_b0", "img_size": 224, "classes": CLASSES,
        "temperature": 1.0, "confidence_threshold": 0.6, "reject_threshold": 0.2, "blur_threshold": 5.0,
        "checkpoint": "model.pt", "checkpoint_sha256": hashlib.sha256((d / "model.pt").read_bytes()).hexdigest(),
        "gradcam_target_layer": "bn2",
    }
    (d / "model_card.json").write_text(json.dumps(card))
    return d


def _make_client(serving_dir: Path, eval_dir: Path | None = None):
    os.environ["KG_SERVING_DIR"] = str(serving_dir)
    os.environ["KG_DATABASE_URL"] = "sqlite://"
    os.environ["KG_RATE_LIMIT_ENABLED"] = "false"
    if eval_dir is not None:
        os.environ["KG_EVALUATION_DIR"] = str(eval_dir)
    from app.core.config import get_settings

    get_settings.cache_clear()
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def client(fake_serving, tmp_path_factory):
    with _make_client(fake_serving, tmp_path_factory.mktemp("eval_empty")) as c:
        yield c


@pytest.fixture(scope="module")
def real_client():
    if not (REAL_SERVING / "model_card.json").exists():
        pytest.skip("real serving artefacts not exported yet")
    with _make_client(REAL_SERVING, ROOT / "ml/artifacts/evaluation") as c:
        yield c
