"""Phase 2, step 2 — precompute frozen-backbone embeddings (GPU recommended).

One pass of DINOv2 over the dataset; afterwards, all head training/calibration
runs in minutes on CPU. Saves patch-token embeddings (needed for attention
pooling) as one compressed .npz shard per batch, plus labels.

Colab usage:
    from training.embed import precompute
    precompute("data/train/manifest.csv", "data/train", "emb/train",
               n_aug=2, device="cuda")

`n_aug` extra robustness-augmented variants per image (JPEG/resize/blur) are
encoded alongside the clean one, so the head sees transformed inputs without
re-running the backbone at training time.
"""
from __future__ import annotations

import csv
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

INPUT_SIZE = 224  # dinov2 ViT-*/14 with 224 -> 16x16 = 256 patch tokens


def _augment(im: Image.Image, rng: random.Random) -> Image.Image:
    """One random robustness transform (NTIRE-style: JPEG / resize / blur)."""
    t = rng.choice(["jpeg", "resize", "blur"])
    if t == "jpeg":
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=rng.randint(30, 90))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if t == "resize":
        f = rng.uniform(0.5, 1.5)
        w, h = im.size
        return im.resize((max(32, int(w * f)), max(32, int(h * f))), Image.BICUBIC)
    return im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.0)))


def _to_tensor(im: Image.Image):
    import torch

    im = im.resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (x - mean) / std


def precompute(
    manifest_csv: str | Path,
    img_root: str | Path,
    out_dir: str | Path,
    n_aug: int = 1,
    batch_size: int = 32,
    device: str = "cuda",
    model_name: str = "dinov2_vits14",  # vits14 fast; vitl14 stronger
    seed: int = 0,
) -> Path:
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()

    with open(manifest_csv) as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(seed)

    tensors, labels, shard = [], [], 0

    def flush():
        nonlocal tensors, labels, shard
        if not tensors:
            return
        with torch.no_grad():
            x = torch.stack(tensors).to(device)
            feats = model.forward_features(x)
            patch = feats["x_norm_patchtokens"].cpu().numpy().astype(np.float16)
            cls = feats["x_norm_clstoken"].cpu().numpy().astype(np.float16)
        np.savez_compressed(
            out / f"shard_{shard:05d}.npz",
            patch=patch, cls=cls, label=np.array(labels, dtype=np.int8),
        )
        tensors, labels, shard = [], [], shard + 1

    for row in rows:
        path = Path(img_root) / row["label"] / row["file"]
        y = 1 if row["label"] == "fake" else 0
        with Image.open(path) as im:
            im = im.convert("RGB")
            variants = [im] + [_augment(im, rng) for _ in range(n_aug)]
            for v in variants:
                tensors.append(_to_tensor(v))
                labels.append(y)
                if len(tensors) >= batch_size:
                    flush()
    flush()
    print(f"wrote {shard} shards to {out}")
    return out
