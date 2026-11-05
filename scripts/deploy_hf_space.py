"""Deploy the backend API to a Hugging Face Space (Docker SDK).

Assembles only the files the API needs (no datasets, no training checkpoints) and uploads them with the
huggingface_hub client, which also handles the 16.5 MB model file without git-lfs.

  export HF_TOKEN=hf_xxx                      # a *write* token from https://huggingface.co/settings/tokens
  .venv/bin/python scripts/deploy_hf_space.py <hf-username>/krishiguard-api [--private]

The Space builds automatically; the API is then at https://<hf-username>-krishiguard-api.hf.space
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ("infrastructure/hf-space/Dockerfile", "Dockerfile"),
    ("infrastructure/hf-space/README.md", "README.md"),
    ("backend/requirements.txt", "backend/requirements.txt"),
    ("advisory/diseases.json", "advisory/diseases.json"),
    ("ml/datasets/metadata/class_mapping.json", "ml/datasets/metadata/class_mapping.json"),
    ("ml/artifacts/serving/model.pt", "ml/artifacts/serving/model.pt"),
    ("ml/artifacts/serving/model_card.json", "ml/artifacts/serving/model_card.json"),
    ("ml/artifacts/evaluation/summary.json", "ml/artifacts/evaluation/summary.json"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("space_id", help="e.g. your-hf-username/krishiguard-api")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN to a Hugging Face write token first.")

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for src, dst in FILES:
            (tmp / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / src, tmp / dst)
        shutil.copytree(ROOT / "backend/app", tmp / "backend/app", ignore=shutil.ignore_patterns("__pycache__"))
        api = HfApi(token=token)
        api.create_repo(a.space_id, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
        api.upload_folder(folder_path=str(tmp), repo_id=a.space_id, repo_type="space",
                          commit_message="Deploy KrishiGuard API")
    owner, name = a.space_id.split("/")
    print(f"Uploaded. Build logs: https://huggingface.co/spaces/{a.space_id}")
    print(f"API (after build): https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space/api/health")


if __name__ == "__main__":
    main()
