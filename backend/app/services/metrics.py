"""Serves real evaluation outputs written by ml/evaluation (never synthesised)."""
from __future__ import annotations

import json
from pathlib import Path

from app.core.errors import ServiceUnavailableError


def load_metrics(evaluation_dir: Path) -> dict:
    p = evaluation_dir / "summary.json"
    if not p.exists():
        raise ServiceUnavailableError("Evaluation results are not available yet. Run ml/evaluation/evaluate.py.")
    return json.loads(p.read_text())
