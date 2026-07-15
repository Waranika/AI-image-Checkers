"""M2 — Invisible watermark detection.

Runs a registry of public watermark decoders against the image and reports
what each one finds. Three decoder families, in order of ecosystem reach:

  1. DWT-DCT  (SD default)    — the invisible-watermark library's blind codec,
     used by Stable Diffusion 1.x/2.x/SDXL pipelines via HuggingFace diffusers.
     Checks against two known payloads. Vendored fallback when imwatermark is
     not installed.

  2. TrustMark (Adobe)        — neural watermark behind "Durable Content
     Credentials." Designed to survive the transports that kill C2PA manifests.
     Used by Adobe Firefly. Open source (MIT), pip-installable.
     Optional dependency: `pip install trustmark`

  3. Stable Signature BZH (Meta/IMATAG) — zero-bit detector for images from
     Stable-Signature-watermarked models (e.g. SDXL-turbo IMATAG demo).
     Returns watermarked yes/no with a p-value, no payload.
     Optional dependency: `pip install transformers`

INVARIANT — absence is NON-evidence: most generators embed nothing, many
users disable watermarking, and heavy reprocessing destroys the signal.
SynthID (Google/OpenAI) has no public detector — reported as manual-verify.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from ..schema import WatermarkEvidence
from . import dwt_dct

# ──────────────────────────────────── known payloads for DWT-DCT (SD) ──

SDXL_MESSAGE = 0b101100111110110010010000011110111011000110011110  # 48 bits
SDXL_BITS = [(SDXL_MESSAGE >> (47 - i)) & 1 for i in range(48)]
SD_V1_BYTES = b"StableDiffusionV1"
SD_V1_BITS = dwt_dct._bits_from_bytes(SD_V1_BYTES)

KNOWN_PAYLOADS = {
    "sdxl-48bit": SDXL_BITS,
    "stable-diffusion-v1-text": SD_V1_BITS,
}

DETECTION_THRESHOLD = 0.90  # bit accuracy for DWT-DCT


# ────────────────────────────────────────── decoder 1: DWT-DCT (SD) ──

def _decode_dwtdct(rgb: np.ndarray, wm_len: int) -> list[int]:
    """Prefer the real imwatermark package (payload-compatible with SD);
    fall back to our vendored codec for torch-free environments."""
    try:
        from imwatermark import WatermarkDecoder
        import cv2
        decoder = WatermarkDecoder("bits", wm_len)
        bits = decoder.decode(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "dwtDct")
        return [int(b) for b in bits]
    except Exception:
        return dwt_dct.decode_bits(rgb, wm_len)


def _check_dwtdct(rgb: np.ndarray) -> list[WatermarkEvidence]:
    """Check DWT-DCT watermark against known SD payloads.

    Used by: Stable Diffusion 1.x, 2.x, SDXL (when invisible-watermark is
    installed in the generation pipeline — many community setups disable it).
    """
    if min(rgb.shape[0], rgb.shape[1]) < 256:
        return [WatermarkEvidence(
            scheme="dwtDct", applicable=False,
            notes="image too small for reliable blind-watermark decoding",
        )]

    best: Optional[WatermarkEvidence] = None
    for name, payload in KNOWN_PAYLOADS.items():
        decoded = _decode_dwtdct(rgb, len(payload))
        acc = float(np.mean([d == p for d, p in zip(decoded, payload)]))
        ev = WatermarkEvidence(
            scheme="dwtDct",
            detected=acc >= DETECTION_THRESHOLD,
            matched_payload=name if acc >= DETECTION_THRESHOLD else None,
            bit_accuracy=round(acc, 3),
        )
        if best is None or (ev.bit_accuracy or 0) > (best.bit_accuracy or 0):
            best = ev
    return [best] if best else []


# ──────────────────────────────────────── decoder 2: TrustMark (Adobe) ──

_TRUSTMARK_INSTANCE = None  # cached across calls (model loads once)


def _check_trustmark(rgb: np.ndarray) -> list[WatermarkEvidence]:
    """Decode Adobe TrustMark watermark (Durable Content Credentials).

    Used by: Adobe Firefly. Designed to survive JPEG recompression, resizing,
    and metadata stripping — the transports that kill C2PA manifests.
    Requires: `pip install trustmark` (MIT license, ~50 MB model download on
    first use, runs on CPU).
    """
    global _TRUSTMARK_INSTANCE
    try:
        from trustmark import TrustMark
    except ImportError:
        return [WatermarkEvidence(
            scheme="trustmark", applicable=False,
            notes="trustmark package not installed (pip install trustmark)",
        )]

    try:
        if _TRUSTMARK_INSTANCE is None:
            _TRUSTMARK_INSTANCE = TrustMark(verbose=False, model_type="Q")
        pil_img = Image.fromarray(rgb)
        wm_secret, wm_present, wm_schema = _TRUSTMARK_INSTANCE.decode(pil_img)
        return [WatermarkEvidence(
            scheme="trustmark",
            detected=bool(wm_present),
            matched_payload=wm_secret if wm_present else None,
            notes=f"schema={wm_schema}" if wm_present else None,
        )]
    except Exception as exc:
        return [WatermarkEvidence(
            scheme="trustmark", applicable=False,
            notes=f"decoder error: {type(exc).__name__}: {str(exc)[:150]}",
        )]


# ──────────────────────── decoder 3: Stable Signature BZH (Meta/IMATAG) ──

_BZH_MODEL = None
_BZH_PROCESSOR = None


def _check_stable_signature_bzh(rgb: np.ndarray) -> list[WatermarkEvidence]:
    """Detect Stable Signature watermark via IMATAG's BZH zero-bit detector.

    Used by: models fine-tuned with Stable Signature (e.g. SDXL-turbo IMATAG
    demo). Zero-bit detection: answers "watermarked yes/no" with a p-value,
    no payload extracted. False positive rate ~1/1000.
    Requires: `pip install transformers` (model ~100 MB, downloaded from HF
    on first use, runs on CPU).
    """
    global _BZH_MODEL, _BZH_PROCESSOR
    try:
        from transformers import AutoModelForImageClassification, BlipImageProcessor
    except ImportError:
        return [WatermarkEvidence(
            scheme="stable-signature-bzh", applicable=False,
            notes="transformers package not installed (pip install transformers)",
        )]

    try:
        model_id = "imatag/stable-signature-bzh-detector-resnet18"
        if _BZH_MODEL is None:
            _BZH_PROCESSOR = BlipImageProcessor.from_pretrained(model_id)
            _BZH_MODEL = AutoModelForImageClassification.from_pretrained(model_id)
            _BZH_MODEL.eval()

        import torch
        pil_img = Image.fromarray(rgb)
        inputs = _BZH_PROCESSOR(pil_img, return_tensors="pt")
        with torch.no_grad():
            logit = float(_BZH_MODEL(**inputs).logits[0, 0])
        # The model convention: logit < 0 means watermarked.
        detected = logit < 0
        # Convert to a rough confidence: further from 0 = more confident.
        confidence = round(1.0 / (1.0 + np.exp(logit)), 3)  # sigmoid → p(watermarked)
        return [WatermarkEvidence(
            scheme="stable-signature-bzh",
            detected=detected,
            bit_accuracy=confidence,  # repurposed: p(watermarked) in [0,1]
            notes=f"zero-bit detector, p(watermarked)={confidence}" if detected
                  else f"not detected, p(watermarked)={confidence}",
        )]
    except Exception as exc:
        return [WatermarkEvidence(
            scheme="stable-signature-bzh", applicable=False,
            notes=f"decoder error: {type(exc).__name__}: {str(exc)[:150]}",
        )]


# ──────────────── detector 4: SynthID CNN ensemble (learned surrogate) ──

def _check_synthid_cnn(rgb: np.ndarray) -> list[WatermarkEvidence]:
    """Detect SynthID watermark presence via a learned CNN ensemble.

    Used by: Google Gemini/Imagen, OpenAI GPT-Image-2 (since May 2026).
    NOT a cryptographic verifier — a surrogate binary classifier that detects
    the watermark's statistical signature. Reliable on in-distribution images;
    can be fooled by adversarial perturbation or OOD inputs.
    Fusion tier: `likely` (not `verified`) — learned detectors are not proof.
    Requires: torch, torchvision. Weights (~140 MB) auto-download on first use.
    Source: github.com/newideas99/gpt-image-synthid-detector (MIT license).
    """
    try:
        import torch  # noqa: F401 — just checking availability
    except ImportError:
        return [WatermarkEvidence(
            scheme="synthid-cnn", applicable=False,
            notes="torch not installed (needed for SynthID CNN detector)",
        )]

    try:
        from .synthid_cnn import detect_synthid
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        detected, ensemble_p, per_model = detect_synthid(rgb, device=device)
        model_detail = " | ".join(f"{a}={p:.3f}" for a, p in per_model.items())
        return [WatermarkEvidence(
            scheme="synthid-cnn",
            detected=detected,
            bit_accuracy=ensemble_p,  # repurposed: ensemble P(watermarked)
            notes=f"learned surrogate detector, P(wm)={ensemble_p:.3f} [{model_detail}]"
                  if detected else
                  f"not detected, P(wm)={ensemble_p:.3f}",
        )]
    except Exception as exc:
        return [WatermarkEvidence(
            scheme="synthid-cnn", applicable=False,
            notes=f"detector error: {type(exc).__name__}: {str(exc)[:150]}",
        )]


# ────────────────────────────────────── closed schemes (documented) ──

def _note_closed_schemes() -> list[WatermarkEvidence]:
    """Schemes with no public decoder or surrogate — documented, not detected.

    Meta invisible watermark: no public decoder or documentation.
    (SynthID is now covered by the CNN surrogate above; the official
    cryptographic verifier remains oracle-only at openai.com/verify.)
    """
    return [
        WatermarkEvidence(
            scheme="meta-invisible", applicable=False,
            notes="no public decoder; Meta's proprietary scheme",
        ),
    ]


# ─────────────────────────────────────────────────── entry point ──

def analyze_watermarks(rgb: np.ndarray) -> list[WatermarkEvidence]:
    """Run all available watermark decoders and return combined evidence.

    Five detectors, in order of ecosystem reach:
      1. DWT-DCT         — SD 1.x/2.x/SDXL default (known payloads)
      2. TrustMark       — Adobe Firefly / Durable Content Credentials
      3. Stable Sig. BZH — SDXL-turbo IMATAG builds (zero-bit)
      4. SynthID CNN      — Google/OpenAI (learned surrogate, not cryptographic)
      5. (documented)     — Meta invisible (no decoder exists)

    Each decoder is independent: a missing optional dependency makes that
    decoder report applicable=False, not crash. The fusion engine treats
    any detected watermark as strong evidence; absence across all decoders
    is non-evidence (most generators embed nothing, and reprocessing
    destroys the signal).
    """
    results: list[WatermarkEvidence] = []

    results.extend(_check_dwtdct(rgb))               # 1. SD invisible-watermark
    results.extend(_check_trustmark(rgb))             # 2. Adobe TrustMark
    results.extend(_check_stable_signature_bzh(rgb))  # 3. Stable Signature BZH
    results.extend(_check_synthid_cnn(rgb))           # 4. SynthID CNN surrogate
    results.extend(_note_closed_schemes())            # 5. documented-only

    return results
