![CI](https://github.com/Waranika/AI-image-Checkers/actions/workflows/ci.yml/badge.svg)

# AI Image Checker — Evidence-Fusion Detection Pipeline

Detects AI-generated images by investigating the traces generators leave
behind: metadata declarations, invisible watermarks, pixel-level
classification, and web provenance. Four independent signal families,
fused into one honest verdict — with "inconclusive" as a first-class
outcome when the evidence is insufficient.

## How it works

Every AI image carries potential evidence. A camera writes EXIF headers;
an AI tool may write its name in the Software field or declare its origin
via IPTC tags. Some platforms embed invisible watermarks. And the pixels
themselves carry statistical patterns a trained classifier can detect.

This pipeline investigates all of them:

| module | what it reads | what it catches |
|---|---|---|
| **M1 Provenance** | C2PA manifests, IPTC tags, PNG generation-parameter chunks, camera EXIF | OpenAI/ChatGPT images (C2PA), A1111/ComfyUI/NovelAI (recipe chunks), Meta/Midjourney (IPTC) |
| **M2 Watermarks** | DWT-DCT decoder, TrustMark P/Q/B, Stable Signature BZH | SD invisible watermark (fragile), Adobe Content Authenticity outputs (durable) |
| **M4 Classifier** | Frozen DINOv2 ViT-S/14 + attention-pooling head | Any AI image, transport-invariant (drift <0.01 across 7 transforms) |
| **M5 Web provenance** | Google Vision reverse search + page-context analysis | Images with web presence — finds upstream copies, reads page titles for AI labels |

M3 (FFT spectral forensics) was implemented, evaluated, and demoted to
note-only after false-positiving on real photos' JPEG block harmonics.

The verdict taxonomy: **verified** (cryptographic proof) → **likely**
(declared or learned signal) → **inconclusive** (insufficient evidence)
→ **unlikely** (positive human-origin evidence). Absence of AI signals
never yields "unlikely" — that requires positive evidence, not silence.

## Key results

**The central measurement (M1 × M2 × M4 composite):**

| transport | C2PA (M1) | TrustMark (M2) | Classifier (M4) | covered |
|---|---|---|---|---|
| original | ✓ | ✓ | 0.984 | ✓ |
| JPEG re-save | · | ✓ | 0.984 | ✓ |
| screenshot | · | ✓ | 0.984 | ✓ |
| messenger (resize+Q70) | · | · | 0.979 | ✓ |
| exiftool -all= | ✓ | ✓ | 0.984 | ✓ |
| crop | · | ✓ | 0.992 | ✓ |
| PIL re-encode | · | ✓ | 0.984 | ✓ |

No single module covers everything; their union covers **7/7 measured
transports.** Provenance proves origin on pristine files; the durable
watermark carries detection through reprocessing; the classifier fills
the gap where both fail.

**Classifier zero-shot transfer** (Synthbuster, genuinely unseen generators):

| generator | AUROC | zero-shot? |
|---|---|---|
| DALL·E 3 | 0.912 | ✓ |
| SD 1.4 (cross-dataset) | 0.890 | cross-dataset check |
| SDXL | 0.832 | ✓ |
| Midjourney v5 | 0.764 | ✓ |
| Firefly | 0.711 | ✓ (honest boundary) |
| **mean zero-shot** | **0.816** | |

Full run log with caveats and reproduction details:
[docs/results.md](docs/results.md).

## Quick start

```bash
pip install -e .
# system dependency for metadata extraction:
sudo apt-get install libimage-exiftool-perl   # or: brew install exiftool

# CLI
python -m ai_image_id.main path/to/image.jpg

# API
pip install -e ".[api]"
uvicorn ai_image_id.main:api --reload
curl -F "file=@image.jpg" http://127.0.0.1:8000/analyze

# Tests
python -m pytest tests/ -v
```

### Optional dependencies

```bash
pip install -e ".[watermarks]"    # TrustMark + Stable Signature BZH decoders
pip install -e ".[all]"           # everything
```

M5 web provenance requires a Google Cloud Vision API key:
set `GOOGLE_CLOUD_API_KEY` as an environment variable.

## Repository layout

```
ai_image_id/                # pip-installable package
├── __init__.py                 re-exports analyze_image, trace_provenance
├── schema.py                   Pydantic models (Evidence, Verdict, etc.)
├── ingest.py                   SHA-256, pHash, PIL loading
├── fusion.py                   rule cascade → AnalysisResult
├── main.py                     analyze_image() + FastAPI + CLI
├── provenance/                 M1 — C2PA, IPTC, gen-params, camera EXIF
├── watermark/                  M2 — DWT-DCT, TrustMark P/Q/B, BZH
│   ├── dwt_dct.py                  vendored blind codec
│   └── synthid_cnn.py              evaluated + rejected (documented)
├── forensics/                  M3 — FFT heuristic (demoted)
├── detector/                   M4 — DINOv2 + attention-pooling head
└── web_provenance/             M5 — reverse search + context analysis
training/                   # training pipeline
├── prepare_data.py             de-confounded split preparation
├── embed.py                    DINOv2 embedding precomputation
├── train_head.py               attention-pooling head training
└── calibrate_eval.py           temperature scaling, cross-gen eval
notebooks/                  # scenario testing (committed with outputs)
├── 02_train_detector.ipynb     M4 training + cross-generator eval
├── 03_m1_provenance_scenarios  M1 corpus + transport matrix
├── 04_m2_watermark_scenarios   M2 decoder validation + transport matrix
├── 05_m4_detector_scenarios    Synthbuster zero-shot + transport resilience
├── 06_m5_web_provenance        reverse search + page-context analysis
└── 07_full_pipeline_test       all modules active, integration demo
tests/                      # pytest suite (CI, no GPU)
docs/                       # results.md, implementation plan
```

## Colab workflow

```python
!git clone https://github.com/Waranika/AI-image-Checkers.git
%cd AI-image-Checkers
!apt-get -qq install -y libimage-exiftool-perl
%pip install -q -e .
```

Then open any scenario notebook, or start with
`notebooks/07_full_pipeline_test.ipynb` for the full demo.
Large artifacts (datasets, embeddings, checkpoints) are gitignored —
store them in Drive under `ai_image_id/runs/<commit>/`.

## Module documentation

- [Implementation plan — architecture, module specs, references](docs/implementation_plan.md)
- [Results log — every measurement with conditions and caveats](docs/results.md)
