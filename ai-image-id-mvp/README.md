# Identifying AI Image — MVP

Minimal working pipeline for detecting AI-generated images through **evidence fusion**:
provenance metadata → invisible watermarks → intrinsic forensics → rule-based verdict.

Implements the first working slice of Steps 1 and 3 of the project plan. Verdicts follow
the taxonomy `verified | likely | inconclusive | unlikely` with per-signal evidence and
explicit caveats (watermark/metadata absence is treated as non-evidence).

## Quick start

```bash
pip install -r requirements.txt
# system dependency for metadata extraction:
sudo apt-get install libimage-exiftool-perl   # or: brew install exiftool

# CLI
python -m app.main path/to/image.jpg

# API
uvicorn app.main:api --reload
curl -F "file=@image.jpg" http://127.0.0.1:8000/analyze

# Tests (3 end-to-end paths: verified / likely / inconclusive)
python -m pytest tests/ -v
```

## What's implemented (MVP scope)

| Module | Signal | Status |
|---|---|---|
| `app/provenance.py` | EXIF/XMP/IPTC via exiftool: `DigitalSourceType=trainedAlgorithmicMedia`, AI tool names in Software/Creator | ✅ |
| `app/provenance.py` | C2PA Content Credentials via `c2pa-python`: manifest presence, signature validity, `claim_generator` | ✅ |
| `app/watermark/` | Blind DWT watermark decode against known SD payloads (SDXL 48-bit message, `StableDiffusionV1` text). Vendored codec (no torch); auto-uses `imwatermark` if installed | ✅ |
| `app/forensics.py` | FFT radial-spectrum peak heuristic (weak signal, capped at "likely" / conf 0.6) | ✅ |
| `app/fusion.py` | Trust-hierarchy rules: verified provenance/watermark > declared metadata > weak intrinsic signals > inconclusive | ✅ |

## Deliberate limitations / next phases

- **SynthID:** no public image-detector API; reported as a manual verification step (Gemini app).
- **`unlikely` (human) verdicts** are never emitted yet — that requires PRNU/noiseprint camera
  evidence (phase 3) or reverse-search predating (phase 4). The MVP refuses to claim "human"
  from absence of AI signals, by design.
- **Learned detector (M4)** — frozen DINOv2/CLIP + attention-pooling head, calibrated — is
  phase 2. The fusion rule slots are already in place for its score.
- **Reverse image search (M5)** — Google Vision / TinEye / Wayback — phase 4, async workers.

See `identifying_ai_image_implementation_plan.md` for the full architecture and roadmap.
