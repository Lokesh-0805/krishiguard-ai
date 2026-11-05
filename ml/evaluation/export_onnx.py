"""Export the deployed model to ONNX for lightweight serving (onnxruntime, no PyTorch) and verify it.

The ONNX graph has two outputs:
  logits [B, 38]          - identical to the PyTorch model
  cam    [B, 38, h, w]    - class activation maps: CAM_c = sum_k W[c, k] * A_k, where A = last feature map (bn2)
                            and W = classifier weights.
For a network that ends in global-average-pooling + one linear layer (EfficientNet-B0), Grad-CAM at the last
feature map uses weights mean(d logit_c / d A_k) = W[c, k] / (h*w), so ReLU(CAM_c) equals Grad-CAM up to a constant
factor, which disappears after min-max normalisation. This script checks that numerically.

  python ml/evaluation/export_onnx.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml/training"))
sys.path.insert(0, str(ROOT / "backend"))
from common import RAW, build_model, load_split  # noqa: E402

SERVING = ROOT / "ml/artifacts/serving"


class LogitsAndCam(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model.forward_features(x)                # [B, K, h, w] (after bn2 + activation)
        logits = self.model.forward_head(feats)               # global pool -> classifier
        w = self.model.get_classifier().weight                # [C, K]
        cam = torch.einsum("ck,bkhw->bchw", w, feats)         # [B, C, h, w]
        return logits, cam


def main():
    from PIL import Image

    from app.ml.gradcam import GradCAM
    from app.ml.inference import model_view, to_array

    card = json.loads((SERVING / "model_card.json").read_text())
    model = build_model(card["arch"], len(card["classes"]), pretrained=False)
    model.load_state_dict(torch.load(SERVING / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    wrapper = LogitsAndCam(model).eval()
    size = card["img_size"]
    out = SERVING / "model.onnx"
    torch.onnx.export(
        wrapper, (torch.randn(1, 3, size, size),), str(out), input_names=["input"], output_names=["logits", "cam"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "cam": {0: "batch"}}, opset_version=17, dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    test = load_split("test", 64, seed=1)
    ext = load_split("external_test", 16, seed=1)
    paths = list(test.path) + list(ext.path)
    x = np.stack([to_array(model_view(Image.open(RAW / p).convert("RGB"), size)) for p in paths])
    o_logits, o_cam = sess.run(None, {"input": x})
    with torch.no_grad():
        t_logits = model(torch.from_numpy(x)).numpy()
    max_diff = float(np.abs(o_logits - t_logits).max())
    agree = float((o_logits.argmax(1) == t_logits.argmax(1)).mean())

    gc = GradCAM(model, "bn2")
    corrs = []
    for i in range(12):
        k = int(t_logits[i].argmax())
        g = gc(torch.from_numpy(x[i : i + 1]), k)
        c = np.maximum(o_cam[i, k], 0)
        c = (c - c.min()) / (c.max() - c.min() + 1e-12)
        corrs.append(float(np.corrcoef(g.ravel(), c.ravel())[0, 1]))
    report = {"images": len(paths), "max_abs_logit_diff": max_diff, "argmax_agreement": agree,
              "cam_vs_gradcam_min_correlation": min(corrs)}
    print(json.dumps(report, indent=1))
    assert agree == 1.0 and max_diff < 1e-3, "ONNX output does not match PyTorch"
    assert min(corrs) > 0.99, "CAM does not match Grad-CAM"

    card["onnx"] = "model.onnx"
    card["onnx_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    card["explanation_method"] = "CAM (equivalent to Grad-CAM at the last feature map for GAP + linear head)"
    card["onnx_verification"] = report
    (SERVING / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and updated model_card.json")


if __name__ == "__main__":
    main()
