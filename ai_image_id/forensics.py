"""Step 3.3 (subset) — frequency-domain heuristic.

Natural photos show a smooth power-law roll-off in the radial power spectrum of the
high-pass residual. Generative upsampling (GAN transposed conv, some diffusion
decoders) can leave periodic peaks in mid/high frequencies.

This is a WEAK, heuristic signal: it must never upgrade a verdict past "likely",
and it is down-weighted for recompressed/small images. The learned detector (M4,
phase 2 of the roadmap) will replace/augment this.
"""
from __future__ import annotations

import cv2
import numpy as np

from .schema import SpectrumEvidence


def _radial_profile(mag: np.ndarray) -> np.ndarray:
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    profile = np.bincount(r.ravel(), mag.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    return profile[: min(cy, cx)]


def analyze_spectrum(rgb: np.ndarray) -> SpectrumEvidence:
    if min(rgb.shape[0], rgb.shape[1]) < 256:
        return SpectrumEvidence(valid=False, notes="image too small for spectral analysis")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    residual = gray - cv2.medianBlur(gray, 3)  # high-pass residual

    f = np.fft.fftshift(np.fft.fft2(residual))
    log_mag = np.log1p(np.abs(f))
    profile = _radial_profile(log_mag)

    # JPEG's 8x8 block grid creates spectral peaks at harmonics of dim/8 —
    # compression artifacts, not generation artifacts. Mask those radii out.
    h, w = residual.shape
    jpeg_mask = np.ones(len(profile), dtype=bool)
    for dim in (h, w):
        for k in range(1, 5):
            r = round(k * dim / 8)
            jpeg_mask[max(0, r - 3) : r + 4] = False

    # Look for peaks in the upper half of the radial profile relative to a smoothed
    # baseline: natural roll-off is smooth; periodic artifacts create bumps.
    half_start = len(profile) // 2
    half = profile[half_start:]
    kernel = np.ones(9) / 9.0
    baseline = np.convolve(half, kernel, mode="same")
    excess = half - baseline
    sigma = float(np.std(excess)) or 1e-6
    peak_hits = (excess > 4 * sigma) & jpeg_mask[half_start:][: len(excess)]
    peaks = int(np.sum(peak_hits))

    # Map peak count to a bounded anomaly score.
    score = float(min(1.0, peaks / 6.0))
    return SpectrumEvidence(anomaly_score=round(score, 3), n_peaks=peaks)
