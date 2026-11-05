"""Shared data / model / metric utilities for training, tuning and evaluation."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "ml/datasets/raw"
PROC = ROOT / "ml/datasets/processed"
META = ROOT / "ml/datasets/metadata"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_classes() -> list[str]:
    return [c["id"] for c in json.loads((META / "class_mapping.json").read_text())["classes"]]


def load_split(name: str, max_samples: int | None = None, seed: int = 0) -> pd.DataFrame:
    df = pd.read_csv(PROC / name / "manifest.csv")
    if max_samples and len(df) > max_samples:
        # stratified subsample, keeps every class represented
        frac = max_samples / len(df)
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=seed))
            .reset_index(drop=True)
        )
    return df


def train_transform(size: int) -> T.Compose:
    """Augmentations chosen to keep lesion colour/texture intact: geometric + mild photometric only.
    Hue jitter is kept tiny because chlorosis (yellowing) and necrosis colour are diagnostic."""
    return T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation(30)], p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.02),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize(int(round(size / 0.875))),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df.path.tolist()
        self.labels = df.label_idx.astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(RAW / self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.labels[i]


def make_loader(df, transform, batch_size, shuffle, workers=4, seed=0, persistent=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ManifestDataset(df, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=persistent and workers > 0,
        pin_memory=False,
        drop_last=shuffle,
        generator=g,
    )


def build_model(arch: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.2):
    import timm

    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


@torch.no_grad()
def predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def basic_metrics(y_true, y_pred) -> dict:
    # macro averages are taken over the classes present in y_true (external sets cover a subset of
    # the 38 classes); predictions into absent classes still count as errors via recall/accuracy.
    labels = np.unique(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }
