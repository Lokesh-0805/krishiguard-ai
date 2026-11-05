import io
import sys

import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter

from app.core.errors import InvalidImageError, PayloadTooLargeError, UnsupportedMediaError
from app.ml.inference import blur_score, decode_upload, model_view, to_array
from tests.conftest import ROOT, jpeg_bytes

from app.core.config import Settings

KW = dict(max_bytes=8 * 1024 * 1024, allowed=Settings().allowed_mime, min_side=64)


def test_matches_training_eval_transform():
    """Serving preprocessing must be numerically identical to the transform used for validation/test."""
    sys.path.insert(0, str(ROOT / "ml/training"))
    from common import eval_transform

    img = Image.open(io.BytesIO(jpeg_bytes((517, 389)))).convert("RGB")
    ours = torch.from_numpy(to_array(model_view(img, 224)))
    ref = eval_transform(224)(img)
    assert ours.shape == ref.shape == (3, 224, 224)
    assert torch.allclose(ours, ref, atol=1e-5)


@pytest.mark.parametrize(
    "fmt,mime,expected",
    [
        ("JPEG", "image/jpeg", "image/jpeg"),
        ("PNG", "image/png", "image/png"),
        ("WEBP", "image/webp", "image/webp"),
        ("AVIF", "image/avif", "image/avif"),
        ("HEIF", "image/heic", "image/heic"),   # iPhone photos
        ("HEIF", "", "image/heic"),             # some browsers send no type for .heic
        ("BMP", "image/bmp", "image/bmp"),
        ("TIFF", "image/tiff", "image/tiff"),
        ("GIF", "image/gif", "image/gif"),
        ("JPEG", "application/octet-stream", "image/jpeg"),
    ],
)
def test_decode_supported_formats(fmt, mime, expected):
    d = decode_upload(jpeg_bytes(fmt=fmt), mime, **KW)
    assert d.mime == expected and (d.width, d.height) == (320, 280) and d.image.mode == "RGB" and len(d.sha256) == 64


@pytest.mark.parametrize(
    "data,mime,exc",
    [
        (b"", "image/jpeg", InvalidImageError),
        (b"not an image at all", "image/jpeg", InvalidImageError),
        (jpeg_bytes(), "text/plain", UnsupportedMediaError),
        (jpeg_bytes(fmt="ICO"), "image/x-icon", UnsupportedMediaError),
        (jpeg_bytes(fmt="ICO"), "image/jpeg", UnsupportedMediaError),  # header lies -> content sniffing wins
        (b"%PDF-1.4 not an image", "image/jpeg", InvalidImageError),
        (jpeg_bytes((40, 40)), "image/jpeg", InvalidImageError),
    ],
)
def test_decode_rejections(data, mime, exc):
    with pytest.raises(exc):
        decode_upload(data, mime, **KW)


def test_size_limit():
    with pytest.raises(PayloadTooLargeError):
        decode_upload(jpeg_bytes(), "image/jpeg", **{**KW, "max_bytes": 100})


def test_exif_orientation_applied():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    d = decode_upload(buf.getvalue(), "image/jpeg", **KW)
    assert (d.width, d.height) == (100, 200)


def test_blur_score_detects_blur():
    img = Image.open(io.BytesIO(jpeg_bytes((300, 300)))).convert("RGB")
    assert blur_score(img.filter(ImageFilter.GaussianBlur(6))) < blur_score(img) / 5
