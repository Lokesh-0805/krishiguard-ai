"""Reproducible dataset indexing, integrity checks, leakage-free splitting and EDA.

Reads ONLY from ml/datasets/raw (never modified). Writes:
  ml/datasets/processed/{train,validation,test,external_test}/manifest.csv
  ml/datasets/metadata/{class_mapping,dataset_statistics,dataset_manifest}.json
  ml/datasets/metadata/image_index.parquet-like CSV (full per-image index)
  ml/artifacts/eda/*.png

Images are not copied (disk budget); split manifests reference raw files by relative path.

Usage:  python -I ml/preprocessing/build_dataset.py [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "ml/datasets/raw"
PROC = ROOT / "ml/datasets/processed"
META = ROOT / "ml/datasets/metadata"
EDA = ROOT / "ml/artifacts/eda"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "validation", "test", "external_validation", "domain_train", "external_test")
NEAR_DUP_HAMMING = 4  # pHash (64-bit) distance treated as near-duplicate

# PlantDoc folder -> PlantVillage class (all 28 PlantDoc classes have a PV counterpart)
PLANTDOC_TO_PV = {
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
}
TOMATO_PK_TO_PV = {
    "Tomato_Early_blight": "Tomato___Early_blight",
    "Tomato_Healthy": "Tomato___healthy",
    "Tomato_leaf_late_blight": "Tomato___Late_blight",
    "Tomato_leaf_yellow_curl_virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mold_leaf": "Tomato___Leaf_Mold",
    "Tomato_septora_leaf_spot": "Tomato___Septoria_leaf_spot",
}


def phash64(img: Image.Image) -> int:
    """DCT perceptual hash (same algorithm as imagehash.phash, returned as int)."""
    import imagehash

    return int(str(imagehash.phash(img)), 16)


def inspect(path: str) -> dict:
    rec = {"path": path, "corrupt": False, "error": ""}
    p = RAW / path
    try:
        raw = p.read_bytes()
        rec["bytes"] = len(raw)
        rec["md5"] = hashlib.md5(raw).hexdigest()
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            rec["width"], rec["height"], rec["mode"] = im.width, im.height, im.mode
            rgb = im.convert("RGB")
            small = np.asarray(rgb.resize((64, 64)), dtype=np.float32) / 255.0
            rec["mean_r"], rec["mean_g"], rec["mean_b"] = small.reshape(-1, 3).mean(0).tolist()
            rec["std_r"], rec["std_g"], rec["std_b"] = small.reshape(-1, 3).std(0).tolist()
            rec["phash"] = phash64(rgb)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        rec["corrupt"], rec["error"] = True, f"{type(e).__name__}: {e}"
    return rec


def list_images(sub: str) -> list[str]:
    base = RAW / sub
    return sorted(
        str(p.relative_to(RAW)) for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT
    )


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def near_dup_pairs(h_a: np.ndarray, h_b: np.ndarray | None, thr: int, chunk: int = 1024):
    """Yield (i, j) index pairs whose 64-bit hashes differ by <= thr bits."""
    same = h_b is None
    h_b = h_a if same else h_b
    for s in range(0, len(h_a), chunk):
        block = h_a[s : s + chunk, None] ^ h_b[None, :]
        d = np.bitwise_count(block)
        ii, jj = np.nonzero(d <= thr)
        ii = ii + s
        if same:
            keep = ii < jj
            ii, jj = ii[keep], jj[keep]
        yield from zip(ii.tolist(), jj.tolist())


def pv_original_id(fname: str) -> str:
    stem = fname.split("___", 1)[-1].rsplit(".", 1)[0]
    return stem.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    for d in (PROC, META, EDA, *(PROC / s for s in SPLITS)):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ index
    sources = {
        "plantvillage": list_images("plantvillage/color"),
        "plantdoc": list_images("plantdoc/train") + list_images("plantdoc/test"),
        "tomato_pk_field": list_images("field_images/tomato_pakistan_mendeley"),
    }
    if not sources["plantvillage"]:
        sys.exit("PlantVillage not found under ml/datasets/raw/plantvillage/color — run scripts/verify_datasets.py")
    all_paths = [p for v in sources.values() for p in v]
    print(f"indexing {len(all_paths)} images with {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        recs = pool.map(inspect, all_paths, chunksize=256)
    df = pd.DataFrame(recs)
    src_of = {p: s for s, v in sources.items() for p in v}
    df["source"] = df.path.map(src_of)

    def label_of(row):
        parts = Path(row.path).parts
        if row.source == "plantvillage":
            return parts[2], parts[2]
        if row.source == "plantdoc":
            return parts[2], PLANTDOC_TO_PV.get(parts[2])
        return parts[3], TOMATO_PK_TO_PV.get(parts[3])

    df[["source_label", "label"]] = df.apply(lambda r: pd.Series(label_of(r)), axis=1)
    df["source_split"] = df.path.map(
        lambda p: Path(p).parts[1].lower() if p.startswith("plantdoc/") else ""
    )
    unmapped = df[df.label.isna()]
    assert unmapped.empty, f"unmapped labels: {unmapped.source_label.unique()}"

    classes = sorted(df.loc[df.source == "plantvillage", "label"].unique())
    cls_idx = {c: i for i, c in enumerate(classes)}
    df["crop"] = df.label.str.split("___").str[0]
    df["disease"] = df.label.str.split("___").str[1]

    # --------------------------------------------------------- dedup / leakage
    ok = df[~df.corrupt].reset_index(drop=True)
    n = len(ok)
    dsu = DSU(n)
    # 1) exact duplicates (same bytes)
    for _, idx in ok.groupby("md5").groups.items():
        idx = list(idx)
        for j in idx[1:]:
            dsu.union(idx[0], j)
    exact_dup_groups = int((ok.groupby("md5").size() > 1).sum())
    exact_dup_images = int(ok.duplicated("md5", keep=False).sum())
    # 2) PlantVillage leaf identity (same physical leaf photographed several times)
    leaf_map = json.loads((RAW / "plantvillage/leaf-map.json").read_text())
    pv_mask = ok.source == "plantvillage"
    leaf_ids = {}
    matched = 0
    for i in np.nonzero(pv_mask.to_numpy())[0]:
        oid = pv_original_id(Path(ok.path[i]).name)
        entry = leaf_map.get(oid)
        if entry:
            matched += 1
            leaf_ids.setdefault(entry[0], []).append(int(i))
    for members in leaf_ids.values():
        for j in members[1:]:
            dsu.union(members[0], j)
    multi_image_leaves = sum(len(v) > 1 for v in leaf_ids.values())
    # 3) near duplicates by pHash, within each PV class and across sources
    hashes = ok.phash.astype("uint64").to_numpy()
    near_pairs_within = 0
    for lbl, idx in ok[pv_mask].groupby("label").groups.items():
        idx = np.asarray(list(idx))
        for a, b in near_dup_pairs(hashes[idx], None, NEAR_DUP_HAMMING):
            dsu.union(int(idx[a]), int(idx[b]))
            near_pairs_within += 1
    cross = {}
    pv_idx = np.nonzero(pv_mask.to_numpy())[0]
    for src in ("plantdoc", "tomato_pk_field"):
        sidx = np.nonzero((ok.source == src).to_numpy())[0]
        pairs = [(int(sidx[a]), int(pv_idx[b])) for a, b in near_dup_pairs(hashes[sidx], hashes[pv_idx], NEAR_DUP_HAMMING)]
        exact = len(set(ok.md5[sidx]) & set(ok.md5[pv_idx]))
        cross[src] = {
            "near_dup_pairs_vs_plantvillage": len(pairs),
            "images_with_pv_near_dup": len({a for a, _ in pairs}),
            "exact_md5_overlap_with_plantvillage": exact,
        }
        ok.loc[[a for a, _ in pairs], "pv_near_dup"] = True
    # PlantDoc train/test overlap (known issue in the public release)
    pd_tr = np.nonzero(((ok.source == "plantdoc") & (ok.source_split == "train")).to_numpy())[0]
    pd_te = np.nonzero(((ok.source == "plantdoc") & (ok.source_split == "test")).to_numpy())[0]
    pd_overlap = {(int(pd_te[a])) for a, _ in near_dup_pairs(hashes[pd_te], hashes[pd_tr], NEAR_DUP_HAMMING)}
    ok["pv_near_dup"] = ok.get("pv_near_dup", False)
    ok["pv_near_dup"] = ok["pv_near_dup"].fillna(False).astype(bool)
    pd_all = np.nonzero((ok.source == "plantdoc").to_numpy())[0]
    for a, b in near_dup_pairs(hashes[pd_all], None, NEAR_DUP_HAMMING):
        dsu.union(int(pd_all[a]), int(pd_all[b]))
    ok["group"] = [dsu.find(i) for i in range(n)]

    # ---------------------------------------------------------------- splits
    rng = np.random.default_rng(args.seed)
    pv = ok[pv_mask].copy()
    # assign whole groups to splits, stratified by each group's (majority) label
    grp_label = pv.groupby("group").label.agg(lambda s: s.mode().iloc[0])
    grp_size = pv.groupby("group").size()
    split_of_group = {}
    for lbl in classes:
        gs = grp_label[grp_label == lbl].index.to_numpy()
        gs = gs[rng.permutation(len(gs))]
        total = grp_size[gs].sum()
        acc = 0
        for g in gs:
            frac = acc / total
            split_of_group[g] = "train" if frac < 0.70 else ("validation" if frac < 0.85 else "test")
            acc += grp_size[g]
    pv["split"] = pv.group.map(split_of_group)
    # sanity: no group crosses splits
    assert (pv.groupby("group").split.nunique() == 1).all()

    ext = ok[~pv_mask].copy()
    ext["split"] = "external_test"
    # PlantDoc's own train release -> 80% domain_train (only used by the explicitly labelled
    # domain-adaptation experiment) / 20% external_validation (calibration + threshold selection only).
    pd_train_mask = (ext.source == "plantdoc") & (ext.source_split == "train")
    pd_groups = ext[pd_train_mask].groupby("group").label.agg(lambda s: s.mode().iloc[0])
    for lbl in sorted(pd_groups.unique()):
        gs = pd_groups[pd_groups == lbl].index.to_numpy()
        gs = gs[rng.permutation(len(gs))]
        n_val = max(1, int(round(0.2 * len(gs)))) if len(gs) > 1 else 0
        val_g = set(gs[:n_val].tolist())
        sel = pd_train_mask & ext.label.eq(lbl)
        ext.loc[sel, "split"] = np.where(ext.loc[sel, "group"].isin(val_g), "external_validation", "domain_train")
    # PlantDoc test images that near-duplicate PlantDoc train images are not independent -> excluded
    ext = ext[~ext.index.isin(list(pd_overlap))]
    # an external image that is a near-copy of a PlantVillage image is not independent -> excluded
    ext = ext[~ext.pv_near_dup]

    cols = ["path", "label", "source", "source_label", "source_split", "crop", "disease", "group"]
    out = pd.concat([pv, ext])
    out["label_idx"] = out.label.map(cls_idx)
    for split in SPLITS:
        sub = out[out.split == split][cols + ["label_idx"]].sort_values("path")
        sub.to_csv(PROC / split / "manifest.csv", index=False)
    df.drop(columns=["phash"]).to_csv(META / "image_index.csv", index=False)

    # ------------------------------------------------------------ metadata
    def crop_display(c):
        return c.replace("_(maize)", "").replace("_(including_sour)", "").replace(",_bell", " (bell)").replace("_", " ")

    mapping = {
        "num_classes": len(classes),
        "classes": [
            {
                "index": i,
                "id": c,
                "crop": crop_display(c.split("___")[0]),
                "disease": c.split("___")[1].replace("_", " ").strip(),
                "healthy": c.endswith("healthy"),
            }
            for i, c in enumerate(classes)
        ],
        "external_label_maps": {"plantdoc": PLANTDOC_TO_PV, "tomato_pk_field": TOMATO_PK_TO_PV},
    }
    (META / "class_mapping.json").write_text(json.dumps(mapping, indent=2))

    def counts(frame):
        return {k: int(v) for k, v in frame.label.value_counts().sort_index().items()}

    pvc = pv.label.value_counts()
    stats = {
        "seed": args.seed,
        "near_dup_hamming_threshold": NEAR_DUP_HAMMING,
        "total_indexed": int(len(df)),
        "corrupt": df[df.corrupt][["path", "error"]].to_dict("records"),
        "per_source_images": {k: int(v) for k, v in df.source.value_counts().items()},
        "plantvillage": {
            "images": int(len(pv)),
            "classes": len(classes),
            "crops": int(pv.crop.nunique()),
            "imbalance_ratio_max_min": round(float(pvc.max() / pvc.min()), 2),
            "largest_class": [pvc.idxmax(), int(pvc.max())],
            "smallest_class": [pvc.idxmin(), int(pvc.min())],
            "dimensions": {f"{w}x{h}": int(c) for (w, h), c in pv.groupby(["width", "height"]).size().items()},
            "leaf_map_matched_images": int(matched),
            "physical_leaves_with_multiple_images": int(multi_image_leaves),
            "exact_duplicate_groups_all_sources": exact_dup_groups,
            "exact_duplicate_images_all_sources": exact_dup_images,
            "near_dup_pairs_within_class": int(near_pairs_within),
            "leakage_groups": int(pv.group.nunique()),
            "rgb_mean": pv[["mean_r", "mean_g", "mean_b"]].mean().round(4).tolist(),
            "rgb_std": pv[["std_r", "std_g", "std_b"]].mean().round(4).tolist(),
        },
        "cross_source_overlap": cross,
        "plantdoc_test_images_near_dup_of_plantdoc_train": len(pd_overlap),
        "splits": {s: {"images": int((out.split == s).sum()), "per_class": counts(out[out.split == s])} for s in SPLITS},
        "external_test_by_source": {s: {"images": int(((ext.source == s) & (ext.split == "external_test")).sum()), "per_class": counts(ext[(ext.source == s) & (ext.split == "external_test")])} for s in ext.source.unique()},
        "external_dimensions_summary": {
            s: {"w_median": float(g.width.median()), "h_median": float(g.height.median()), "w_min": int(g.width.min()), "w_max": int(g.width.max())}
            for s, g in ext.groupby("source")
        },
    }
    (META / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))

    manifest_path = META / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"datasets": {}}
    for key, d in manifest.get("datasets", {}).items():
        d["image_count_verified"] = int((df.source == key).sum())
        d["corrupt_images"] = int(((df.source == key) & df.corrupt).sum())
        d["processing_status"] = "indexed, deduplicated, split manifests written"
        d["processed_on"] = str(date.today())
    manifest["splits"] = {s: str((PROC / s / "manifest.csv").relative_to(ROOT)) for s in SPLITS}
    manifest_path.write_text(json.dumps(manifest, indent=2))

    make_plots(df, pv, ext, classes)
    print(json.dumps({k: stats[k] for k in ("per_source_images", "cross_source_overlap", "plantdoc_test_images_near_dup_of_plantdoc_train")}, indent=2))
    print(json.dumps({s: v["images"] for s, v in stats["splits"].items()}))


def make_plots(df, pv, ext, classes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    vc = pv.label.value_counts().reindex(classes)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(range(len(vc)), vc.values, color="#2f7d4f")
    ax.set_yticks(range(len(vc)), [c.replace("___", " — ").replace("_", " ") for c in vc.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("images")
    ax.set_title("PlantVillage: images per class")
    for i, v in enumerate(vc.values):
        ax.text(v + 30, i, str(v), va="center", fontsize=6)
    fig.tight_layout(); fig.savefig(EDA / "class_distribution_plantvillage.png", dpi=130); plt.close(fig)

    split_ct = pv.groupby(["label", "split"]).size().unstack(fill_value=0).reindex(classes)[["train", "validation", "test"]]
    fig, ax = plt.subplots(figsize=(10, 9))
    left = np.zeros(len(split_ct))
    for col, colr in zip(split_ct.columns, ["#2f7d4f", "#d9a441", "#4a6fa5"]):
        ax.barh(range(len(split_ct)), split_ct[col], left=left, label=col, color=colr)
        left += split_ct[col].values
    ax.set_yticks(range(len(split_ct)), [c.replace("___", " — ").replace("_", " ") for c in split_ct.index], fontsize=7)
    ax.invert_yaxis(); ax.legend(frameon=False); ax.set_title("Leakage-free group split per class")
    fig.tight_layout(); fig.savefig(EDA / "split_distribution.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, (src, g) in zip(axes, ext.groupby("source")):
        ax.scatter(g.width, g.height, s=4, alpha=0.4, color="#4a6fa5")
        ax.set_title(f"{src}: image dimensions"); ax.set_xlabel("width"); ax.set_ylabel("height")
    fig.tight_layout(); fig.savefig(EDA / "external_dimensions.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, ch, col in zip(axes, ["r", "g", "b"], ["#c0392b", "#27ae60", "#2e6fbf"]):
        for src, g in df[~df.corrupt].groupby("source"):
            ax.hist(g[f"mean_{ch}"], bins=50, histtype="step", density=True, label=src)
        ax.set_title(f"mean {ch.upper()} per image")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(EDA / "rgb_distribution.png", dpi=130); plt.close(fig)

    rng = np.random.default_rng(0)
    for name, frame, ncls in (("plantvillage", pv, 12), ("plantdoc", ext[ext.source == "plantdoc"], 8), ("tomato_pk_field", ext[ext.source == "tomato_pk_field"], 6)):
        labels = sorted(frame.label.unique())[:ncls] if name != "plantvillage" else list(rng.choice(classes, ncls, replace=False))
        fig, axes = plt.subplots(len(labels), 4, figsize=(7, 1.9 * len(labels)))
        for r, lbl in enumerate(labels):
            sub = frame[frame.label == lbl]
            for c in range(4):
                ax = axes[r, c]; ax.axis("off")
                if c < len(sub):
                    p = sub.path.iloc[rng.integers(len(sub))]
                    ax.imshow(Image.open(RAW / p).convert("RGB").resize((160, 160)))
                if c == 0:
                    ax.set_title(lbl.replace("___", " — ").replace("_", " ")[:48], fontsize=7, loc="left")
        fig.tight_layout(); fig.savefig(EDA / f"samples_{name}.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
