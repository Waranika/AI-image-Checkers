"""M1 — Provenance & metadata analysis.

Reads what the file *declares or proves* about its own origin. Four signal
families, in descending order of strength:

  1. C2PA Content Credentials  — cryptographic. A valid manifest naming an AI
     generator (top-level OR in any ingredient) is near-decisive.
  2. Generation-parameter chunks — declared, but rich. Local SD tools (A1111
     WebUI, ComfyUI, NovelAI) embed the full sampler recipe in PNG text chunks.
  3. IPTC DigitalSourceType    — declared, one field. Meta/Midjourney convention.
     Full vocabulary mapped (fully-AI / AI-composite / synthetic / capture).
  4. AI tool names in Software/Creator fields — declared, weakest.

Plus one human-side hint: a coherent camera-EXIF block (note-tier only).

INVARIANT — absence is NON-evidence: metadata is stripped by screenshots,
uploads, and messengers; a file with nothing declared proves nothing.
Declared signals are trivially forgeable (exiftool writes anything), so they
cap at "likely"; only cryptographically valid C2PA earns "verified".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .schema import ProvenanceEvidence

# ------------------------------------------------------------- constants --

# Substrings (lowercased) in Software/Creator/claim_generator that indicate AI tools.
AI_TOOL_MARKERS = [
    "dall-e", "dall·e", "openai", "midjourney", "stable diffusion", "stability",
    "firefly", "adobe firefly", "imagen", "gemini", "grok", "flux", "ideogram",
    "leonardo", "recraft", "gpt-4o", "gpt-image", "bing image creator", "designer",
    "novelai", "comfyui",
]

# IPTC DigitalSourceType vocabulary -> our evidence category.
# http://cv.iptc.org/newscodes/digitalsourcetype/
IPTC_SOURCE_CATEGORIES = {
    "trainedalgorithmicmedia": "ai",                          # fully AI-generated
    "compositewithtrainedalgorithmicmedia": "ai_composite",   # AI-edited (gen. fill)
    "algorithmicmedia": "synthetic",                          # synthetic, not ML
    "digitalcapture": "capture",                              # camera family v
    "negativefilm": "capture",
    "positivefilm": "capture",
    "print": "capture",
    "minorhumanedits": "capture",
}

# A1111/WebUI writes the whole recipe into one PNG text chunk. Its signature
# is the *pattern*, not a brand name -- e.g. "... Steps: 20, Sampler: Euler a,
# CFG scale: 7, Seed: 12345, ... Model: sd_xl_base_1.0"
A1111_RECIPE_RE = re.compile(r"steps:\s*\d+.*?(sampler|cfg scale|seed):", re.I | re.S)
A1111_MODEL_RE = re.compile(r"\bmodel:\s*([^,\n]+)", re.I)

# A camera-EXIF block is "coherent" when the basics co-occur. Weak signal
# (forgeable in one exiftool command) -> note-tier only, never a verdict.
CAMERA_EXIF_FIELDS = ["Make", "Model", "ExposureTime", "FNumber", "ISO"]


# ------------------------------------------------------- exiftool sweep --

def _exiftool_json(path: Path) -> dict:
    import exiftool  # pyexiftool

    with exiftool.ExifToolHelper() as et:
        meta = et.get_metadata(str(path))
    return meta[0] if meta else {}


def _scan_generation_params(meta: dict, ev: ProvenanceEvidence) -> None:
    """Item 1 -- PNG text chunks from local SD tools (A1111 / ComfyUI / NovelAI).

    A full sampler recipe is stronger than a bare IPTC tag: hard to explain
    innocently, and it often names the model -> free attribution.
    """
    for key, value in meta.items():
        sval = str(value)
        tail = key.split(":")[-1].lower()  # exiftool keys look like "PNG:Parameters"

        if tail == "parameters" and A1111_RECIPE_RE.search(sval):
            model = A1111_MODEL_RE.search(sval)
            ev.generation_params_tool = "a1111-webui"
            ev.generation_params_model = model.group(1).strip() if model else None
            ev.ai_metadata_hits.append(f"png.parameters (A1111 recipe): {sval[:120]}")
            return

        if tail in ("prompt", "workflow") and '"class_type"' in sval:
            ev.generation_params_tool = "comfyui"
            ev.ai_metadata_hits.append("png.workflow (ComfyUI node graph)")
            return

        if tail == "software" and sval.strip().lower() == "novelai":
            ev.generation_params_tool = "novelai"
            ev.ai_metadata_hits.append("png.software=NovelAI")
            return


def _scan_iptc_source_type(meta: dict, ev: ProvenanceEvidence) -> None:
    """Item 2 -- full IPTC DigitalSourceType vocabulary, not just the AI value."""
    for key, value in meta.items():
        if "digitalsourcetype" not in key.lower():
            continue
        raw = str(value).rsplit("/", 1)[-1]  # value may be the full IPTC URI
        category = IPTC_SOURCE_CATEGORIES.get(raw.lower().replace(" ", ""))
        if category:
            ev.iptc_digital_source_type = raw
            ev.iptc_source_category = category
            return


def _scan_ai_tool_fields(meta: dict, ev: ProvenanceEvidence) -> None:
    """Pre-existing -- AI tool names in Software/Creator-style fields."""
    for key, value in meta.items():
        lkey, sval = key.lower(), str(value).lower()
        if lkey.endswith(("software", "creatortool", "creator", "credit", "description")):
            if any(m in sval for m in AI_TOOL_MARKERS):
                ev.ai_metadata_hits.append(f"{key}={value}")
            if lkey.endswith(("software", "creatortool")):
                ev.software = str(value)


def _scan_camera_exif(meta: dict, ev: ProvenanceEvidence) -> None:
    """Item 4 -- coherent camera block. Weak human-side hint, note-tier only."""
    found = sum(
        1 for field in CAMERA_EXIF_FIELDS
        if any(k.split(":")[-1] == field for k in meta)
    )
    ev.camera_exif_fields = found
    ev.camera_exif_present = found >= 4  # Make+Model+most of the exposure triangle


# --------------------------------------------------------- C2PA reading --

def _c2pa_read(path: Path) -> tuple[bool, str | None, str | None, dict | None]:
    """Open the manifest store. Returns (present, validation_state, generator, store).

    validation_state comes straight from the verifier:
      "Trusted" -- signature valid AND signer on the known trust list
      "Valid"   -- signature cryptographically intact, signer not on the list
      "Invalid" -- broken/tampered signature
    Only genuine "no manifest" errors are swallowed; anything else is surfaced
    in the store dict (a silent AttributeError here once masked a broken API
    call for the entire pipeline -- never again).
    """
    try:
        import c2pa
    except ImportError:
        return False, None, None, None

    reader = None
    try:
        try:
            reader = c2pa.Reader(str(path))            # c2pa-python >= ~0.10
        except (TypeError, AttributeError):
            reader = c2pa.Reader.from_file(str(path))  # older API
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        no_manifest = (
            ("manifest" in msg.lower() and ("not found" in msg.lower() or "missing" in msg.lower()))
            or "JUMBF" in msg
            or "ManifestNotFound" in msg
        )
        if no_manifest:
            return False, None, None, None
        return False, None, None, {"error": msg[:300]}

    try:
        store = json.loads(reader.json() or "{}")
        state = reader.get_validation_state()
        active = store.get("manifests", {}).get(store.get("active_manifest", ""), {})
        generator = active.get("claim_generator")
        if not generator:
            info = active.get("claim_generator_info") or [{}]
            generator = (info[0] or {}).get("name")
        return True, str(state) if state is not None else None, generator, store
    except Exception as exc:
        return True, None, None, {"error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        try:
            reader.close()
        except Exception:
            pass


def _walk_c2pa_store(store: dict, ev: ProvenanceEvidence) -> None:
    """Item 3 -- read the whole book, not the cover.

    Walk EVERY manifest in the store (the active one plus all ingredients'):
    an image edited in Photoshop but generated by Firefly carries its AI
    evidence one level down. Also collect the actions history and any
    digitalSourceType assertions.
    """
    for manifest in store.get("manifests", {}).values():
        gen = manifest.get("claim_generator") or ""
        if not gen:
            info = manifest.get("claim_generator_info") or [{}]
            gen = (info[0] or {}).get("name") or ""
        if gen and any(m in gen.lower() for m in AI_TOOL_MARKERS):
            hit = f"c2pa.claim_generator={gen}"
            if hit not in ev.ai_metadata_hits:
                ev.ai_metadata_hits.append(hit)

        for assertion in manifest.get("assertions", []):
            label = assertion.get("label", "")
            data = assertion.get("data", {})
            if label.startswith("c2pa.actions"):
                for action in data.get("actions", []):
                    entry = action.get("action", "?")
                    agent = action.get("softwareAgent")
                    agent_name = agent.get("name") if isinstance(agent, dict) else agent
                    ev.c2pa_actions.append(f"{entry} ({agent_name})" if agent_name else entry)
            blob = json.dumps(data).lower()
            if "trainedalgorithmicmedia" in blob:
                hit = "c2pa.digitalSourceType=trainedAlgorithmicMedia"
                if hit not in ev.ai_metadata_hits:
                    ev.ai_metadata_hits.append(hit)
            if "digitalcapture" in blob:
                ev.c2pa_capture_claim = True


# --------------------------------------------------------- entry point --

def analyze_provenance(path: Path) -> ProvenanceEvidence:
    ev = ProvenanceEvidence()

    # ---- exiftool sweep: one read, four scanners ----
    try:
        meta = _exiftool_json(path)
    except Exception as exc:  # exiftool missing etc.
        meta = {}
        ev.ai_metadata_hits.append(f"exiftool unavailable: {exc}")

    _scan_generation_params(meta, ev)   # item 1: A1111 / ComfyUI / NovelAI
    _scan_iptc_source_type(meta, ev)    # item 2: full IPTC vocabulary
    _scan_ai_tool_fields(meta, ev)      # pre-existing: tool names in fields
    _scan_camera_exif(meta, ev)         # item 4: camera block, note-tier

    # ---- C2PA: read store, split validity/trust, walk all manifests ----
    present, state, generator, store = _c2pa_read(path)
    ev.c2pa_present = present
    ev.c2pa_generator = generator
    if state is not None:
        ev.c2pa_signature_valid = state in ("Valid", "Trusted")
        ev.c2pa_signer_trusted = state == "Trusted"
        ev.c2pa_valid = ev.c2pa_signature_valid  # back-compat alias
    if store and "manifests" in store:
        _walk_c2pa_store(store, ev)     # item 3: ingredients, actions, sources

    return ev
