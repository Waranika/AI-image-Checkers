"""Phase 2, step 1 — de-confounded dataset preparation.

The single biggest failure mode in AI-image detection training is the classifier
learning compression/resolution confounds instead of generation artifacts (real
photos arrive as Q~75 JPEGs at camera resolutions; generated sets are often Q=96
PNGs at 512/1024px). The GenImage literature's recommendation: apply IDENTICAL
preprocessing to both classes and sample JPEG quality from the SAME distribution.

This module implements that: every image (real or fake) is resized into a shared
resolution range and re-encoded at a Q sampled from one shared distribution, with
the sampling seeded per-file so the dataset is reproducible.

Usage (Colab or local):
    from training.prepare_data import prepare_split
    manifest = prepare_split(
        real_dir="genimage/imagenet_val", fake_dir="genimage/sdv14/fake",
        out_dir="data/train", n_per_class=50_000, seed=0,
    )
"""
from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Shared distributions (both classes sample from these — that's the whole point).
Q_CHOICES = list(range(70, 96))            # JPEG quality 70..95
SIZE_CHOICES = [256, 320, 384, 448, 512]   # short-side targets


def _seed_for(path: Path, seed: int) -> int:
    h = hashlib.sha256(f"{seed}:{path.name}".encode()).hexdigest()
    return int(h[:8], 16)


def _process_one(src: Path, dst: Path, seed: int) -> dict:
    rng = random.Random(_seed_for(src, seed))
    q = rng.choice(Q_CHOICES)
    short = rng.choice(SIZE_CHOICES)

    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = short / min(w, h)
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
        im.save(dst, format="JPEG", quality=q)

    return {"file": dst.name, "src": str(src), "jpeg_q": q, "short_side": short}


def prepare_split(
    real_dir: str | Path,
    fake_dir: str | Path,
    out_dir: str | Path,
    n_per_class: int = 10_000,
    seed: int = 0,
) -> Path:
    """Build a balanced, confound-matched split. Returns path to manifest.csv."""
    out = Path(out_dir)
    rows: list[dict] = []

    for label, src_dir in (("real", Path(real_dir)), ("fake", Path(fake_dir))):
        if not src_dir.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {src_dir}")
        files = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not files:
            raise ValueError(f"no images found in {label} directory: {src_dir}")
        random.Random(seed).shuffle(files)
        files = files[:n_per_class]
        if len(files) < n_per_class:
            print(f"warning: only {len(files)} {label} images available")

        dst_dir = out / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        for k, src in enumerate(files):
            row = _process_one(src, dst_dir / f"{label}_{k:06d}.jpg", seed)
            row["label"] = label
            rows.append(row)

    manifest = out / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "src", "jpeg_q", "short_side"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest
