from tests.conftest import jpeg_bytes


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] and body["database"] and body["advisory_entries"] == 38
    assert body["status"] == "ok"
    assert r.headers.get("x-request-id")


def test_model_info(client):
    b = client.get("/api/model/info").json()
    assert b["architecture"] == "efficientnet_b0" and b["num_classes"] == 38 and b["input_size"] == 224


def test_predict_success_with_explanation(client):
    r = client.post("/api/predict", files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")}, data={"lang": "en"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["prediction_id"] and len(b["top_k"]) == 3
    assert b["status"] in {"confident", "uncertain", "unrecognized"}
    if b["status"] != "unrecognized":
        assert b["explanation"]["method"].startswith("CAM") and len(b["explanation"]["overlay_png"]) > 100
    if b["status"] != "confident":
        assert b["advisory"] is None  # never attach treatment advice to an uncertain guess


def test_predict_hindi_labels(client):
    b = client.post("/api/predict", files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")}, data={"lang": "hi", "explain": "false"}).json()
    assert b["explanation"] is None
    assert any("ऀ" <= ch <= "ॿ" for ch in b["status_message"])  # Devanagari


def test_predict_rejects_non_image(client):
    r = client.post("/api/predict", files={"file": ("x.jpg", b"hello", "image/jpeg")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_image"


def test_predict_rejects_wrong_type(client):
    r = client.post("/api/predict", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 415


def test_predict_requires_file(client):
    r = client.post("/api/predict")
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_predict_oversize(client):
    big = b"\xff\xd8" + b"0" * (8 * 1024 * 1024 + 10)
    r = client.post("/api/predict", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_batch_mixed(client):
    files = [("files", ("a.jpg", jpeg_bytes(), "image/jpeg")), ("files", ("b.txt", b"nope", "image/jpeg"))]
    b = client.post("/api/predict/batch", files=files).json()
    assert len(b["results"]) == 2
    assert "class_id" in b["results"][0] and b["results"][1]["error"]["code"] == "invalid_image"


def test_batch_limit(client):
    files = [("files", (f"{i}.jpg", jpeg_bytes(), "image/jpeg")) for i in range(9)]
    assert client.post("/api/predict/batch", files=files).status_code == 422


def test_diseases_and_advisory(client):
    items = client.get("/api/diseases").json()["items"]
    assert len(items) == 38
    hi = client.get("/api/diseases", params={"lang": "hi"}).json()["items"]
    assert any("ऀ" <= ch <= "ॿ" for ch in hi[0]["name"] + hi[0]["crop"])
    first = items[0]
    by_id = client.get(f"/api/diseases/{first['id']}").json()
    by_slug = client.get(f"/api/diseases/{first['slug']}").json()
    assert by_id["id"] == by_slug["id"] and by_id["sources"]
    adv = client.get(f"/api/advisory/{first['slug']}", params={"lang": "hi"}).json()
    assert adv["language"] == "hi" and adv["disclaimer"]
    assert client.get("/api/advisory/does-not-exist").status_code == 404
    assert client.get("/api/diseases", params={"lang": "fr"}).status_code == 422


def test_metrics_unavailable_is_explicit(client):
    r = client.get("/api/metrics")
    assert r.status_code == 503 and r.json()["error"]["code"] == "service_unavailable"


def test_feedback_flow(client):
    pid = client.post("/api/predict", files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")}, data={"explain": "false"}).json()["prediction_id"]
    r = client.post("/api/feedback", json={"prediction_id": pid, "prediction_correct": False, "suggested_class": "Tomato___Late_blight",
                                           "advisory_helpful": True, "comment": "spots look different"})
    assert r.status_code == 201 and r.json()["verification_status"] == "unverified"
    assert client.post("/api/feedback", json={"prediction_id": "0" * 36, "prediction_correct": True}).status_code == 404
    assert client.post("/api/feedback", json={"prediction_id": pid, "suggested_class": "Banana___x"}).status_code == 422
    assert client.post("/api/feedback", json={"prediction_id": pid, "comment": "x" * 1001}).status_code == 422
