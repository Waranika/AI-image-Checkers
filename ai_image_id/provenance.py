"""Step 1 — official provenance signals: EXIF/XMP/IPTC via exiftool, C2PA via c2pa-python.

Trust hierarchy (per project notes): a *valid* C2PA manifest naming an AI generator is
near-decisive. IPTC DigitalSourceType == trainedAlgorithmicMedia (Meta, Midjourney,
Shutterstock convention) is a strong declared signal. Absence of everything is NON-evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import ProvenanceEvidence

TRAINED_ALGO = "trainedalgorithmicmedia"

# Substrings (lowercased) in Software/Creator/claim_generator that indicate AI tools.
AI_TOOL_MARKERS = [
    "dall-e", "dall·e", "openai", "midjourney", "stable diffusion", "stability",
    "firefly", "adobe firefly", "imagen", "gemini", "grok", "flux", "ideogram",
    "leonardo", "recraft", "gpt-4o", "gpt-image", "bing image creator", "designer",
]


def _exiftool_json(path: Path) -> dict:
    import exiftool  # pyexiftool

    with exiftool.ExifToolHelper() as et:
        meta = et.get_metadata(str(path))
    return meta[0] if meta else {}


def _c2pa_read(path: Path) -> tuple[bool, bool | None, str | None, dict | None]:
    """Returns (present, valid, generator, raw). raw carries the manifest or an error."""
    try:
        import c2pa
    except ImportError:
        return False, None, None, None

    reader = None
    try:
        try:
            reader = c2pa.Reader(str(path))          # c2pa-python >= ~0.10
        except (TypeError, AttributeError):
            reader = c2pa.Reader.from_file(str(path))  # older API
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        # "No manifest in file" is the normal case for most images.
        if "manifest" in msg.lower() and ("not found" in msg.lower() or "missing" in msg.lower()):
            return False, None, None, None
        if "JUMBF" in msg or "ManifestNotFound" in msg:
            return False, None, None, None
        # Anything else is a real error — surface it instead of silently
        # reporting "no manifest" (a silent AttributeError here once masked
        # a broken API call for the entire pipeline).
        return False, None, None, {"error": msg[:300]}

    try:
        manifest = reader.get_active_manifest() or {}
        state = reader.get_validation_state()
        valid = (str(state) == "Valid") if state is not None else None
        generator = manifest.get("claim_generator")
        if not generator:
            info = manifest.get("claim_generator_info") or [{}]
            generator = (info[0] or {}).get("name")
        return True, valid, generator, manifest
    except Exception as exc:
        return True, None, None, {"error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        try:
            reader.close()
        except Exception:
            pass


def analyze_provenance(path: Path) -> ProvenanceEvidence:
    ev = ProvenanceEvidence()

    # --- EXIF / XMP / IPTC ---
    try:
        meta = _exiftool_json(path)
    except Exception as exc:  # exiftool missing etc.
        meta = {}
        ev.ai_metadata_hits.append(f"exiftool unavailable: {exc}")

    for key, value in meta.items():
        lkey, sval = key.lower(), str(value).lower()
        if "digitalsourcetype" in lkey and TRAINED_ALGO in sval.replace(" ", ""):
            ev.iptc_digital_source_type = "trainedAlgorithmicMedia"
        if lkey.endswith(("software", "creatortool", "creator", "credit", "description")):
            if any(m in sval for m in AI_TOOL_MARKERS):
                ev.ai_metadata_hits.append(f"{key}={value}")
            if lkey.endswith(("software", "creatortool")):
                ev.software = str(value)

    # --- C2PA ---
    present, valid, generator, raw = _c2pa_read(path)
    ev.c2pa_present = present
    ev.c2pa_valid = valid
    ev.c2pa_generator = generator
    if generator and any(m in generator.lower() for m in AI_TOOL_MARKERS):
        ev.ai_metadata_hits.append(f"c2pa.claim_generator={generator}")
    if raw:
        blob = json.dumps(raw).lower()
        if "digitalcapture" in blob:
            ev.c2pa_capture_claim = True
        if "trainedalgorithmicmedia" in blob and present:
            ev.ai_metadata_hits.append("c2pa.digitalSourceType=trainedAlgorithmicMedia")

    return ev
