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
    """Returns (present, valid, generator, raw_manifest_dict)."""
    try:
        import c2pa
    except ImportError:
        return False, None, None, None
    try:
        reader = c2pa.Reader.from_file(str(path))
        manifest_json = reader.json()
    except Exception:
        # No manifest store found (the common case) or unreadable.
        return False, None, None, None

    try:
        store = json.loads(manifest_json)
    except (TypeError, json.JSONDecodeError):
        return True, None, None, None

    active = store.get("manifests", {}).get(store.get("active_manifest", ""), {})
    generator = active.get("claim_generator") or active.get("claim_generator_info", [{}])[0].get("name")
    # validation_status present and non-empty => problems found; absent => validated OK.
    valid = not store.get("validation_status")
    return True, valid, generator, store


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

    return ev
