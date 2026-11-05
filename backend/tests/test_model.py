import io
import json
import shutil

import numpy as np
import pytest
import torch
from PIL import Image

from app.core.errors import ServiceUnavailableError
from app.ml.gradcam import GradCAM, render
from app.ml.inference import ModelService
from tests.conftest import CLASSES, jpeg_bytes


@pytest.fixture(scope="module")
def svc(fake_serving):
    s = ModelService(fake_serving)
    s.load()
    assert s.ready, s.load_error
    return s


def test_predict_outputs_are_valid_distribution(svc):
    img = Image.open(io.BytesIO(jpeg_bytes())).convert("RGB")
    out = svc.predict([img, img])
    assert len(out) == 2
    r = out[0]
    assert r["class_id"] in CLASSES and 0 <= r["confidence"] <= 1
    assert r["status"] in {"confident", "uncertain", "unrecognized"}
    assert [t["probability"] for t in r["top_k"]] == sorted([t["probability"] for t in r["top_k"]], reverse=True)
    assert r["top_k"][0]["class_id"] == r["class_id"]
    assert r["crop_confidence"] >= r["confidence"] - 1e-6
    assert out[0]["class_id"] == out[1]["class_id"]  # deterministic


def test_status_thresholds(svc):
    p = np.zeros(len(CLASSES)); p[0] = 0.9; p[1:] = 0.1 / (len(CLASSES) - 1)
    assert svc._interpret(p, 1.0)["status"] == "confident"
    p[0] = 0.4
    assert svc._interpret(p, 1.0)["status"] == "uncertain"
    p[0] = 0.1
    assert svc._interpret(p, 1.0)["status"] == "unrecognized"


def test_cam_matches_gradcam(svc, fake_serving):
    """The API's ONNX CAM must equal gradient-based Grad-CAM (GAP + linear head)."""
    import timm

    from app.ml.inference import model_view, to_array

    img = Image.open(io.BytesIO(jpeg_bytes())).convert("RGB")
    x = to_array(model_view(img, 224))[None]
    (cam,) = svc.session.run(["cam"], {"input": x})
    torch.manual_seed(0)
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASSES)).eval()
    g = GradCAM(m, "bn2")(torch.from_numpy(x), 3)
    c = np.maximum(cam[0, 3], 0)
    c = (c - c.min()) / (c.max() - c.min())
    assert np.corrcoef(g.ravel(), c.ravel())[0, 1] > 0.999


def test_gradcam_cnn(svc):
    img = Image.open(io.BytesIO(jpeg_bytes())).convert("RGB")
    model_in, heat, overlay = svc.explain(img, 3)
    for b in (model_in, heat, overlay):
        im = Image.open(io.BytesIO(b))
        assert im.format == "PNG" and im.size == (224, 224)


def test_gradcam_vit_tokens():
    import timm

    m = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=5).eval()
    cam = GradCAM(m, "blocks.-1.norm1", m.num_prefix_tokens)(torch.randn(1, 3, 224, 224), 2)
    assert cam.shape == (14, 14) and cam.min() >= 0 and cam.max() <= 1


def test_checksum_mismatch_refuses_to_load(fake_serving, tmp_path):
    d = tmp_path / "s"
    shutil.copytree(fake_serving, d)
    card = json.loads((d / "model_card.json").read_text())
    card["onnx_sha256"] = "0" * 64
    (d / "model_card.json").write_text(json.dumps(card))
    s = ModelService(d)
    s.load()
    assert not s.ready and "sha256" in s.load_error
    with pytest.raises(ServiceUnavailableError):
        s.predict([Image.new("RGB", (100, 100))])


def test_missing_artifacts(tmp_path):
    s = ModelService(tmp_path)
    s.load()
    assert not s.ready and "not found" in s.load_error
