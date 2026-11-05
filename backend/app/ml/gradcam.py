"""Grad-CAM for the deployed model (CNN feature maps or ViT patch tokens).

Grad-CAM (Selvaraju et al., 2017): weights = spatial mean of d(score_c)/d(A); cam = ReLU(sum_k w_k A_k).
For ViTs the token sequence of the chosen block is reshaped to the patch grid (prefix tokens dropped),
the common adaptation used by e.g. pytorch-grad-cam. The heatmap shows which regions influenced the
score — it is NOT evidence that the diagnosis is correct.
"""
from __future__ import annotations

import io
import threading

import numpy as np
import torch
from PIL import Image


def resolve_module(model: torch.nn.Module, dotted: str) -> torch.nn.Module:
    m = model
    for part in dotted.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def default_target_layer(arch: str) -> str:
    if arch.startswith("resnet"):
        return "layer4"
    if arch.startswith("efficientnet"):
        return "bn2"  # BatchNormAct after conv_head: last spatial feature map
    if arch.startswith(("vit", "deit")):
        return "blocks.-1.norm1"
    raise ValueError(f"no default Grad-CAM layer for {arch}")


class GradCAM:
    def __init__(self, model: torch.nn.Module, target: str, num_prefix_tokens: int = 0):
        self.model = model
        if ".-1" in target:  # allow "blocks.-1.norm1"
            head, tail = target.split(".-1", 1)
            seq = resolve_module(model, head)
            mod = seq[len(seq) - 1]
            self.layer = resolve_module(mod, tail.lstrip(".")) if tail.strip(".") else mod
        else:
            self.layer = resolve_module(model, target)
        self.prefix = num_prefix_tokens
        self._lock = threading.Lock()  # hooks are stateful -> one explanation at a time

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        acts, grads = {}, {}

        def fwd_hook(_, __, out):
            acts["v"] = out
            out.register_hook(lambda g: grads.__setitem__("v", g))

        with self._lock:
            h = self.layer.register_forward_hook(fwd_hook)
            try:
                self.model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    x = x.clone().requires_grad_(True)
                    logits = self.model(x)
                    logits[0, class_idx].backward()
            finally:
                h.remove()
            a, g = acts["v"].detach(), grads["v"].detach()
        if a.dim() == 3:  # ViT tokens [B, N, C] -> [B, C, h, w]
            a, g = a[:, self.prefix :], g[:, self.prefix :]
            side = int(round(a.shape[1] ** 0.5))
            a = a.transpose(1, 2).reshape(a.shape[0], -1, side, side)
            g = g.transpose(1, 2).reshape(g.shape[0], -1, side, side)
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * a).sum(1))[0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.cpu().numpy()


def _jet(v: np.ndarray) -> np.ndarray:
    """Matplotlib-free 'jet'-like colormap: v in [0,1] -> uint8 RGB."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def render(cam: np.ndarray, image: Image.Image, alpha: float = 0.45) -> tuple[bytes, bytes]:
    """Return (heatmap PNG, overlay PNG) at the input image's (display-sized) resolution."""
    base = image.convert("RGB")
    base.thumbnail((448, 448))
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(base.size, Image.BICUBIC)
    heat = _jet(np.asarray(cam_img, dtype=np.float32) / 255.0)
    overlay = (np.asarray(base, dtype=np.float32) * (1 - alpha) + heat * alpha).astype(np.uint8)

    def png(arr) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    return png(heat), png(overlay)
