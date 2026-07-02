"""Ingest: hashing + decoding. Keeps the original bytes untouched for metadata tools."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image


@dataclass
class IngestedImage:
    path: Path            # original file on disk (metadata tools need the raw file)
    sha256: str
    phash: str
    rgb: np.ndarray       # HxWx3 uint8, for pixel-level analysis


def ingest(path: str | Path) -> IngestedImage:
    path = Path(path)
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()

    with Image.open(path) as im:
        im = im.convert("RGB")
        phash = str(imagehash.phash(im))
        rgb = np.asarray(im, dtype=np.uint8)

    return IngestedImage(path=path, sha256=sha256, phash=phash, rgb=rgb)
