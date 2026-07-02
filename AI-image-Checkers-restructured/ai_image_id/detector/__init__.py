"""M4 inference wrapper. Activates only when a trained checkpoint is available
(env var AI_IMAGE_ID_HEAD or explicit path); otherwise the pipeline runs without
detector evidence, exactly as before. Torch is imported lazily so the core
package stays torch-free.

Inference does two forward passes: clean input and a JPEG-recompressed copy
(Q=75). The absolute probability difference is reported as `robustness_drift`
and used by the fusion engine to down-weight unstable scores.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ..schema import DetectorEvidence

_MODEL_CACHE: dict = {}


def _checkpoint_path() -> Path | None:
    p = os.environ.get("AI_IMAGE_ID_HEAD")
    return Path(p) if p and Path(p).exists() else None


def _load(ckpt: Path, backbone: str, device: str):
    key = (str(ckpt), backbone)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import torch

    from training.train_head import build_head  # repo layout: training/ alongside package

    bb = torch.hub.load("facebookresearch/dinov2", backbone).to(device).eval()
    state = torch.load(ckpt, map_location=device)
    head = build_head(state["dim"]).to(device).eval()
    head.load_state_dict(state["state_dict"])

    calib = ckpt.with_suffix(".calibration.json")
    T = json.loads(calib.read_text())["temperature"] if calib.exists() else 1.0

    _MODEL_CACHE[key] = (bb, head, T)
    return _MODEL_CACHE[key]


def _predict(rgb: np.ndarray, bb, head, T: float, device: str) -> float:
    import torch

    from training.embed import _to_tensor

    x = _to_tensor(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = bb.forward_features(x)
        logit = head(feats["x_norm_patchtokens"], feats["x_norm_clstoken"])
    return float(1.0 / (1.0 + np.exp(-float(logit) / T)))


def analyze_detector(
    rgb: np.ndarray,
    ckpt: str | Path | None = None,
    backbone: str = "dinov2_vits14",
    device: str = "cpu",
) -> DetectorEvidence | None:
    ckpt = Path(ckpt) if ckpt else _checkpoint_path()
    if ckpt is None:
        return None  # no trained model configured — pipeline runs without M4
    try:
        bb, head, T = _load(ckpt, backbone, device)
        p_clean = _predict(rgb, bb, head, T, device)

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=75)
        buf.seek(0)
        recompressed = np.asarray(Image.open(buf).convert("RGB"))
        p_jpeg = _predict(recompressed, bb, head, T, device)

        return DetectorEvidence(
            model=f"{backbone}+attnpool",
            p_calibrated=round(p_clean, 4),
            robustness_drift=round(abs(p_clean - p_jpeg), 4),
        )
    except Exception as exc:
        return DetectorEvidence(
            model="unavailable", p_calibrated=0.5, valid=False, notes=str(exc)
        )
