"""Re-measure batch-1 latency of every evaluated run under identical, quiet conditions (run after all training has
finished) and update each run's metrics.json in place. The latencies recorded during evaluation were measured while
another model was training, so they are not comparable across runs.

  python ml/evaluation/measure_latency.py [--threads 4]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from common import build_model, get_device  # noqa: E402
from evaluate import EVAL, MODELS, latency  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = get_device()
    for mpath in sorted(EVAL.glob("*/metrics.json")):
        run = mpath.parent.name
        ck = torch.load(MODELS / run / "best.pt", map_location="cpu", weights_only=False)
        model = build_model(ck["arch"], len(ck["classes"]), pretrained=False)
        model.load_state_dict(ck["state_dict"])
        model.to(device).eval()
        lat = latency(model, ck["img_size"], device, runs=60)
        m = json.loads(mpath.read_text())
        m["latency"] = lat
        m["latency_conditions"] = f"quiet machine, torch threads={a.threads}, 60 timed runs after 5 warm-up, batch 1"
        mpath.write_text(json.dumps(m, indent=2))
        print(run, {k: round(v["batch1_ms_mean"], 1) for k, v in lat.items()}, flush=True)


if __name__ == "__main__":
    main()
