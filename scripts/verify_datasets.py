"""Verify that the raw datasets are present and complete. Exit code 0 = PlantVillage ready.

  python scripts/verify_datasets.py            # report
  python scripts/verify_datasets.py --resume   # if ready, rebuild splits and resume the (resumable) training queue
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "ml/datasets/raw"
EXPECTED = {
    "plantvillage": (RAW / "plantvillage/color", 54305, 38),
    "plantdoc": (RAW / "plantdoc", 2572, 28),
    "tomato_pk_field": (RAW / "field_images/tomato_pakistan_mendeley/Tomato_Leaf_Dataset", 7200, 6),
}
EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    report = {}
    for name, (path, n_exp, c_exp) in EXPECTED.items():
        imgs = [p for p in path.rglob("*") if p.suffix.lower() in EXT] if path.exists() else []
        if name == "plantdoc":
            imgs = [p for p in imgs if p.parent.parent.name.lower() in ("train", "test")]
        classes = {p.parent.name for p in imgs}
        report[name] = {
            "path": str(path.relative_to(ROOT)), "images": len(imgs), "expected": n_exp, "classes": len(classes),
            "ok": len(imgs) >= n_exp * 0.98 and len(classes) >= c_exp,
        }
    print(json.dumps(report, indent=2))
    if not report["plantvillage"]["ok"]:
        print("PlantVillage missing/incomplete - see README 'Datasets'. External sets are optional for training.")
        sys.exit(1)
    if a.resume:
        py = sys.executable
        subprocess.run([py, "-I", "ml/preprocessing/build_dataset.py"], check=True, cwd=ROOT)
        subprocess.run([py, "ml/training/run_experiments.py"], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
