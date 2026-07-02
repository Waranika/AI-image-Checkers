"""End-to-end tests: watermark round-trip -> verified; IPTC tag -> likely;
clean photo-like image -> inconclusive."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from ai_image_id.main import analyze_image
from ai_image_id.schema import Verdict
from ai_image_id.watermark import SDXL_BITS
from ai_image_id.watermark import dwt_dct


def _make_photo_like(path: Path, size: int = 512) -> np.ndarray:
    """Smooth gradient + mild noise, saved as high-quality JPEG."""
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:size, 0:size]
    base = np.stack(
        [
            120 + 60 * np.sin(x / 97.0),
            100 + 50 * np.cos(y / 83.0),
            90 + 40 * np.sin((x + y) / 71.0),
        ],
        axis=-1,
    )
    img = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)
    return img


def test_watermarked_image_is_verified(tmp_path: Path):
    clean = _make_photo_like(tmp_path / "clean.jpg")
    marked = dwt_dct.embed(clean, SDXL_BITS)
    out = tmp_path / "marked.png"  # lossless: watermark survives
    Image.fromarray(marked).save(out)

    result = analyze_image(out)
    assert result.ai_verdict == Verdict.VERIFIED
    wm = next(w for w in result.evidence.watermarks if w.scheme == "dwtDct")
    assert wm.detected and wm.matched_payload == "sdxl-48bit"


def test_iptc_tagged_image_is_likely(tmp_path: Path):
    path = tmp_path / "tagged.jpg"
    _make_photo_like(path)
    subprocess.run(
        ["exiftool", "-overwrite_original",
         "-XMP-iptcExt:DigitalSourceType=http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
         str(path)],
        check=True, capture_output=True,
    )
    result = analyze_image(path)
    assert result.ai_verdict == Verdict.LIKELY
    assert result.evidence.provenance.iptc_digital_source_type == "trainedAlgorithmicMedia"


def test_clean_image_is_inconclusive(tmp_path: Path):
    path = tmp_path / "clean.jpg"
    _make_photo_like(path)
    result = analyze_image(path)
    assert result.ai_verdict == Verdict.INCONCLUSIVE
    assert any("non-evidence" in n for n in result.notes)
