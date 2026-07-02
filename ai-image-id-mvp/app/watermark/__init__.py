"""Step 3.1 (subset) — invisible watermark detection.

Checks the image against known public payloads used by Stable Diffusion pipelines:
  * "StableDiffusionV1" ASCII bytes (SD 1.x/2.x reference pipelines, dwtDct)
  * SDXL's 48-bit WATERMARK_MESSAGE (0b1011001111101100100100000111101110110001100111 10)

Uses the real `imwatermark` decoder when installed (payload-compatible with SD
outputs); otherwise falls back to the vendored codec in dwt_dct.py.

IMPORTANT: absence of a watermark is NON-evidence — most generators embed nothing,
and watermarks are stripped by screenshots/heavy edits. SynthID has no public
detector API: verification is a manual step via the Gemini app / Google portal.
"""
from __future__ import annotations

import numpy as np

from ..schema import WatermarkEvidence
from . import dwt_dct

SDXL_MESSAGE = 0b101100111110110010010000011110111011000110011110  # 48 bits
SDXL_BITS = [(SDXL_MESSAGE >> (47 - i)) & 1 for i in range(48)]
SD_V1_BYTES = b"StableDiffusionV1"
SD_V1_BITS = dwt_dct._bits_from_bytes(SD_V1_BYTES)

KNOWN_PAYLOADS = {
    "sdxl-48bit": SDXL_BITS,
    "stable-diffusion-v1-text": SD_V1_BITS,
}

DETECTION_THRESHOLD = 0.90  # bit accuracy


def _decode(rgb: np.ndarray, wm_len: int) -> list[int]:
    try:
        from imwatermark import WatermarkDecoder  # optional, torch-free for dwtDct? prefer if present

        decoder = WatermarkDecoder("bits", wm_len)
        import cv2

        bits = decoder.decode(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "dwtDct")
        return [int(b) for b in bits]
    except Exception:
        return dwt_dct.decode_bits(rgb, wm_len)


def analyze_watermarks(rgb: np.ndarray) -> list[WatermarkEvidence]:
    results: list[WatermarkEvidence] = []

    if min(rgb.shape[0], rgb.shape[1]) < 256:
        results.append(
            WatermarkEvidence(
                scheme="dwtDct", applicable=False,
                notes="image too small for reliable blind-watermark decoding",
            )
        )
        return results

    best: WatermarkEvidence | None = None
    for name, payload in KNOWN_PAYLOADS.items():
        decoded = _decode(rgb, len(payload))
        acc = float(np.mean([d == p for d, p in zip(decoded, payload)]))
        ev = WatermarkEvidence(
            scheme="dwtDct",
            detected=acc >= DETECTION_THRESHOLD,
            matched_payload=name if acc >= DETECTION_THRESHOLD else None,
            bit_accuracy=round(acc, 3),
        )
        if best is None or (ev.bit_accuracy or 0) > (best.bit_accuracy or 0):
            best = ev
    results.append(best)  # report best-matching payload only, keeps output compact

    results.append(
        WatermarkEvidence(
            scheme="synthid", applicable=False,
            notes="no public detector API; verify manually via the Gemini app",
        )
    )
    return results
