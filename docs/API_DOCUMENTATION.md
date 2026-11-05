# API Documentation

Base URL (local): `http://localhost:8000` — interactive OpenAPI docs at `/docs`, schema at `/openapi.json`.
Behind the frontend the same endpoints are reachable at `http://localhost:3000/api/*` (Next.js rewrite).

All errors share one shape:

```json
{ "error": { "code": "invalid_image", "message": "The file could not be read as an image.", "details": {} } }
```

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 422 | missing/invalid fields (details.errors lists them) |
| `invalid_image` | 422 | empty, undecodable, too small, too many batch files, unknown feedback class |
| `unsupported_media_type` | 415 | not JPEG/PNG/WebP/AVIF/HEIC/BMP/TIFF/GIF (declared **or** sniffed content) |
| `payload_too_large` | 413 | file > `KG_MAX_UPLOAD_MB` (default 8 MB) |
| `not_found` | 404 | unknown disease id / prediction id |
| `rate_limited` | 429 | default 120/min per IP; predict 30/min; feedback 20/min |
| `service_unavailable` | 503 | model not loaded, metrics not generated, database down (feedback) |
| `inference_failed` / `internal_error` | 500 | unexpected failure (logged with request id) |

Every response carries an `x-request-id` header (echoed if the client sends one).

---

## `GET /api/health`
```json
{ "status": "ok", "model_loaded": true, "model_version": "effnet_b0_224@1a2b3c4d", "database": true, "advisory_entries": 38, "detail": {} }
```
`status` is `degraded` when the model or database is unavailable (`detail` explains why).

## `GET /api/model/info`
Architecture, input size, class list, temperature, thresholds, checkpoint SHA-256, training data description and
the model-selection rule.

## `POST /api/predict`  (multipart/form-data)
| Field | Type | Default | |
|---|---|---|---|
| `file` | file | — | leaf photo: JPEG, PNG, WebP, AVIF, HEIC/HEIF, BMP, TIFF or GIF (first frame) |
| `lang` | `en` \| `hi` | `en` | language of labels, status message and advisory |
| `explain` | bool | `true` | include Grad-CAM images |

```bash
curl -s -F file=@leaf.jpg -F lang=hi http://localhost:8000/api/predict | jq '.status, .class_id, .confidence'
```
Response (abridged):
```json
{
  "prediction_id": "5e0b…", "model_version": "…", "status": "confident",
  "status_message": "मॉडल इस परिणाम को लेकर आश्वस्त है…",
  "class_id": "Tomato___Early_blight", "crop": "टमाटर", "disease": "अगेती झुलसा", "healthy": false,
  "confidence": 0.97, "crop_confidence": 0.99,
  "top_k": [{ "class_id": "Tomato___Early_blight", "crop": "टमाटर", "disease": "अगेती झुलसा", "probability": 0.97 }, …],
  "quality": { "width": 1024, "height": 768, "blur_score": 812.4, "flags": [] },
  "latency_ms": 41.3,
  "advisory": { …see /api/advisory… },
  "explanation": { "method": "Grad-CAM", "target_layer": "bn2", "model_input_png": "<base64>", "heatmap_png": "<base64>", "overlay_png": "<base64>", "note": "…" }
}
```
* `status`: `confident` | `uncertain` | `unrecognized` (thresholds selected on validation data — see MODEL_EVALUATION_REPORT).
* `advisory` is **null unless `confident`**. `explanation` is null for `unrecognized` or when `explain=false`.
* `quality.flags`: `blurry`, `low_resolution`. A blurry photo is never reported as `confident`.
* `confidence` is a temperature-scaled probability, **not** a guarantee of correctness.

## `POST /api/predict/batch`  (multipart, repeated `files`, max 8)
Returns `{ "results": [ <prediction> | { "filename": "x.txt", "error": { "code", "message" } } ] }` — one bad file does not fail the batch. No Grad-CAM in batch mode.

## `GET /api/diseases?lang=en|hi[&crop=Tomato]`
`{ "advisory_version": "1.0.0", "count": 38, "items": [{ "id", "slug", "crop", "name", "healthy", "pathogen_type", "scientific_name" }] }`

## `GET /api/diseases/{id_or_slug}?lang=`
Full advisory plus `names` (en+hi) and `crops` (en+hi).

## `GET /api/advisory/{id_or_slug}?lang=`
```json
{ "id": "...", "slug": "...", "crop": "...", "name": "...", "healthy": false, "pathogen_type": "fungus", "scientific_name": "...",
  "summary": "...", "symptoms": ["..."], "causes": ["..."], "prevention": ["..."], "organic_treatment": ["..."],
  "chemical_treatment": { "items": ["..."], "active_ingredients": ["mancozeb"] },
  "sources": [{ "title", "publisher", "url", "accessed" }], "disclaimer": "...", "chemical_label_notice": "...", "language": "en" }
```
No dosages are ever returned; chemical items end with a CIB&RC registration / label / KVK instruction.

## `GET /api/metrics`
The evaluation summary written by `ml/evaluation/compare_and_export.py` (all models, per-set metrics, per-class
metrics, confusion matrix, risk–coverage, selection rule). Returns **503** if evaluation has not been run — the
API never synthesises metrics.

## `POST /api/feedback`  (JSON)
```json
{ "prediction_id": "<uuid from /predict>", "prediction_correct": false, "suggested_class": "Tomato___Late_blight",
  "advisory_helpful": true, "comment": "≤ 1000 chars" }
```
→ `201 { "id": 12, "verification_status": "unverified" }`. Feedback is stored as **unverified** and is never used
as ground truth without expert review.
