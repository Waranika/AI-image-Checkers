"""SynthID presence detector — learned CNN ensemble (surrogate, not cryptographic).

Detects whether an image carries a SynthID-like watermark pattern, as embedded
by Google Gemini/Imagen and (since May 2026) OpenAI GPT-Image-2.

Architecture: 3-model ensemble (ResNet-18 + ResNet-34 + EfficientNet-B0),
each fine-tuned on paired watermarked/clean images with shared augmentation
to prevent shortcut learning. Ensemble prediction = mean P(watermarked).

What this IS:  a learned binary classifier that detects the watermark's
               statistical signature. Reliable on in-distribution images.
What this ISN'T: a cryptographic verifier. It cannot read the watermark's
                 payload or prove who embedded it. A false positive on a
                 non-watermarked image is possible (~3% on the author's
                 validation set).

Fusion tier: `likely` (not `verified`) — same as the M4 classifier, because
learned detectors are not cryptographic proof.

Weights: ~140 MB total, downloaded from GitHub on first use and cached locally.
Requires: torch, torchvision (already needed for M4 detector / training).
Source: https://github.com/newideas99/gpt-image-synthid-detector (MIT license).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────────────── constants ──

WEIGHTS_REPO = "newideas99/gpt-image-synthid-detector"
WEIGHTS_BRANCH = "main"
ARCHS = ["resnet18", "resnet34", "efficientnet_b0"]
DETECTION_THRESHOLD = 0.5   # ensemble mean P(watermarked) > 0.5
WEIGHTS_DIR = Path.home() / ".cache" / "ai-image-id" / "synthid-cnn"

# ────────────────────────────────────────────────── model building ──

_ENSEMBLE: Optional[dict] = None   # cached across calls


def _build_model(arch: str):
    """Build the architecture with a 1-output head (matching the training code)."""
    import torch.nn as nn
    import torchvision as tv

    if arch == "resnet18":
        m = tv.models.resnet18()
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif arch == "resnet34":
        m = tv.models.resnet34()
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        m = tv.models.efficientnet_b0()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    else:
        raise ValueError(f"unknown arch: {arch}")
    return m


def _download_weights() -> Path:
    """Download weights from GitHub if not cached locally."""
    import urllib.request

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    base_url = f"https://raw.githubusercontent.com/{WEIGHTS_REPO}/{WEIGHTS_BRANCH}/weights"
    for arch in ARCHS:
        dst = WEIGHTS_DIR / f"surrogate_{arch}.pt"
        if dst.exists():
            continue
        url = f"{base_url}/surrogate_{arch}.pt"
        print(f"synthid-cnn: downloading {arch} weights ({url})...")
        urllib.request.urlretrieve(url, dst)
    return WEIGHTS_DIR


def _load_ensemble(device: str) -> dict:
    """Load or return cached ensemble. Downloads weights on first call."""
    global _ENSEMBLE
    if _ENSEMBLE is not None:
        return _ENSEMBLE

    import torch

    weights_dir = _download_weights()
    models = {}
    for arch in ARCHS:
        ckpt = weights_dir / f"surrogate_{arch}.pt"
        if not ckpt.exists():
            continue
        m = _build_model(arch)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models[arch] = m

    if not models:
        raise RuntimeError("no SynthID CNN weights found after download")
    _ENSEMBLE = models
    return models


# ──────────────────────────────────────────────────── inference ──

def _preprocess(rgb: np.ndarray):
    """Resize → center-crop → normalize, matching the training pipeline exactly."""
    import torch
    from PIL import Image
    from torchvision.transforms import v2

    tf = v2.Compose([
        v2.Resize(640, antialias=True),
        v2.CenterCrop(512),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil = Image.fromarray(rgb)
    return tf(pil).unsqueeze(0)


def detect_synthid(rgb: np.ndarray, device: str = "cpu") -> tuple[bool, float, dict[str, float]]:
    """Run the ensemble. Returns (detected, ensemble_p, per_model_scores).

    ensemble_p is the mean P(watermarked) across the three models.
    detected is True when ensemble_p > DETECTION_THRESHOLD.
    per_model_scores maps arch name → individual P(watermarked).
    """
    import torch

    models = _load_ensemble(device)
    x = _preprocess(rgb).to(device)
    per_model: dict[str, float] = {}
    with torch.no_grad():
        for arch, m in models.items():
            per_model[arch] = float(torch.sigmoid(m(x)).item())
    ensemble_p = sum(per_model.values()) / len(per_model)
    return ensemble_p > DETECTION_THRESHOLD, round(ensemble_p, 4), per_model
