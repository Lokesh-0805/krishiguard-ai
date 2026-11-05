"""Integration tests against the REAL exported model (skipped until ml/artifacts/serving exists)."""
import pandas as pd
import pytest

from tests.conftest import ROOT

RAW = ROOT / "ml/datasets/raw"


@pytest.fixture(scope="module")
def test_samples():
    p = ROOT / "ml/datasets/processed/test/manifest.csv"
    if not p.exists():
        pytest.skip("test manifest missing")
    df = pd.read_csv(p)
    return df.groupby("label").head(1).sample(12, random_state=0)


def test_real_model_on_plantvillage_test(real_client, test_samples):
    hits = 0
    for _, row in test_samples.iterrows():
        data = (RAW / row.path).read_bytes()
        b = real_client.post("/api/predict", files={"file": ("x.jpg", data, "image/jpeg")}, data={"explain": "false"}).json()
        hits += b["class_id"] == row.label
    assert hits >= 10  # held-out PlantVillage test images (expected near-perfect)


def test_real_model_confident_has_advisory_and_gradcam(real_client, test_samples):
    row = test_samples.iloc[0]
    b = real_client.post("/api/predict", files={"file": ("x.jpg", (RAW / row.path).read_bytes(), "image/jpeg")}, data={"lang": "hi"}).json()
    assert b["explanation"]["overlay_png"]
    if b["status"] == "confident":
        assert b["advisory"]["language"] == "hi"


def test_real_metrics_endpoint(real_client):
    m = real_client.get("/api/metrics").json()
    assert m["selection"]["selected_run"] and m["models"]
    assert 0 <= m["models"][0]["results"]["test"]["accuracy"] <= 1
