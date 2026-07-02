# Identifying AI Image — Implementation Plan

**Goal:** A local-first, uncertainty-aware service that answers: *"Is this image AI-generated, and if so, by what?"* — by fusing provenance, watermark, forensic, and learned-detector signals, exactly following the 3-step structure of the project notes.

**Design philosophy (informed by 2026 SOTA):**
- No universal detector exists. Recent benchmarks (NTIRE 2026 challenge, "How well are open-sourced AI-generated image detection models out-of-the-box", Feb 2026) show top commercial generators (Flux Dev, Firefly v4, Midjourney v7, Imagen 4) defeat most public detectors (18–30% accuracy), and training-data alignment matters more than architecture (20–60% variance within identical architectures).
- Therefore: **evidence fusion with calibrated confidence** is the product, not a single classifier. "Inconclusive" is a first-class verdict.
- Provenance (C2PA/SynthID) when present is near-decisive; its **absence is non-evidence**.

---

## 1. System architecture

```
                        ┌──────────────────────────────────────────┐
                        │              FastAPI service             │
                        │  POST /analyze  (image upload or URL)    │
                        └───────────────┬──────────────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   Ingest & prep   │  hash (SHA-256, pHash),
                              │                   │  decode, EXIF-safe copy
                              └─────────┬─────────┘
              ┌─────────────────────────┼──────────────────────────┐
              │                         │                          │
   ┌──────────▼─────────┐   ┌──────────▼──────────┐   ┌───────────▼──────────┐
   │ M1 Provenance      │   │ M2 Watermark        │   │ M3 Intrinsic         │
   │ exiftool, c2patool │   │ decoders            │   │ forensics            │
   │ IPTC DigitalSource │   │ invisible-watermark │   │ FFT spectrum, PRNU/  │
   │ Type, XMP          │   │ Stable Signature    │   │ noiseprint, JPEG QT, │
   └──────────┬─────────┘   │ (Tree-Ring: N/A*)   │   │ ELA, patch stats     │
              │             └──────────┬──────────┘   └───────────┬──────────┘
              │                        │                          │
              │             ┌──────────▼──────────┐               │
              │             │ M4 Learned detector │               │
              │             │ frozen DINOv2/CLIP  │               │
              │             │ + trained head,     │               │
              │             │ calibrated          │               │
              │             └──────────┬──────────┘               │
              └────────────────────────┼──────────────────────────┘
                             ┌─────────▼─────────┐
                             │  M6 Fusion engine │  rules + meta-classifier
                             └─────────┬─────────┘
                             ┌─────────▼─────────┐
                             │  Verdict report   │  JSON per project schema
                             └───────────────────┘

   M5 Reverse image search (async, optional, API-key gated):
   Google Vision Web Detection → Bing Visual Search → TinEye → Wayback CDX
```

*Tree-Ring only applies when you control generation (watermark is injected in the initial latent), so it's included as a demo module, not a detector for arbitrary images.

**Stack:** Python 3.11+, FastAPI + Pydantic v2, PyTorch, Celery + Redis (async reverse-search jobs), PostgreSQL (analysis history), Docker Compose. Gradio or small React front-end for the demo.

**Repo layout:**

```
ai-image-id/
├── src/
│   ├── api/               # FastAPI routes, schemas
│   ├── ingest/            # decoding, hashing, safe copies
│   ├── provenance/        # M1: exiftool + c2patool wrappers
│   ├── watermark/         # M2: decoders (invisible-watermark, stable-signature)
│   ├── forensics/         # M3: fft.py, prnu.py, jpeg.py, ela.py
│   ├── detector/          # M4: model, training, calibration
│   ├── osint/             # M5: vision_api.py, tineye.py, wayback.py
│   ├── fusion/            # M6: rules.py, meta_model.py, report.py
│   └── common/            # config, logging, result schema
├── training/              # dataset prep, train/eval scripts, sweeps
├── eval/                  # benchmark harness (GenImage, NTIRE, robustness)
├── tests/
├── docker/
└── docs/                  # model card, evaluation report, architecture
```

---

## 2. Module specifications

### M1 — Provenance & metadata (Step 1 of notes)

- `pyexiftool` wrapper: extract EXIF/XMP/IPTC; specifically check `XMP-iptcExt:DigitalSourceType == trainedAlgorithmicMedia` (Meta, Midjourney, Shutterstock signal), `Software`/`Creator` fields, embedded thumbnails.
- `c2patool` (subprocess, JSON output): validate manifest signature chain, extract `claim_generator`, ingredients, and asset URIs. Distinguish **valid** / **invalid signature** / **absent**.
- Output: `{c2pa: {present, valid, generator, issuer}, iptc: {digital_source_type}, exif: {...}}`.
- Effort: ~3 days. Highest signal-to-effort ratio in the whole project — do it first.

### M2 — Watermark decoders

| Scheme | Feasibility for arbitrary images | Implementation |
|---|---|---|
| `invisible-watermark` (DWT-DCT, used by SD ≤2.x/SDXL defaults) | Yes — public decoder | pip package, decode "StableDiffusionV1"-style payloads |
| Stable Signature (Meta/ICCV'23) | Partial — decoder is public, but keys are per-model | run open decoder; report bit-accuracy vs. threshold |
| SynthID (Google) | No public image detector API — verification goes through the Gemini app/portal | document as manual step; link out in the report |
| Tree-Ring | Only for self-generated images | implement as an *embedding + detection demo* on local SD, great for the write-up |

- Output per scheme: `{scheme, detected, score, threshold, applicable}`.
- Effort: ~1 week (Tree-Ring demo +3–4 days, optional but a strong portfolio differentiator).

### M3 — Intrinsic forensics

- **Frequency:** 2D FFT of high-pass residual → radial power spectrum; peak detection vs. natural camera roll-off; azimuthal anomaly score. (numpy/scipy, ~2 days.)
- **Noise/PRNU:** Noiseprint++ or a wavelet-denoise residual PRNU estimator; consistency across blocks → "camera-like" score. (Existing open implementations; ~4 days integration.)
- **JPEG:** `jpegio` to read quantization tables (match against known camera/library tables), double-compression detection via DCT histogram periodicity, block-grid alignment. (~3 days.)
- **ELA + patch statistics:** local entropy map, over-smooth texture flags. Cheap, weak signal — weight low. (~1 day.)
- Each analyzer returns a normalized score in [0,1] with a validity flag (e.g., PRNU invalid if image < 512px or heavily recompressed).

### M4 — Learned detector (the ML centerpiece)

**Architecture (per 2025–26 SOTA):** frozen vision foundation model backbone (DINOv2 ViT-L or CLIP ViT-L/14) + lightweight trainable head over patch tokens (attention pooling, TAP-style). Rationale: frozen-VFM approaches currently show the best cross-generator generalization (e.g., TAP trained only on SD1.5 generalizes to SDXL/SD3/Flux on OpenSDI), and they're cheap to train — a single consumer GPU or modest cloud budget suffices.

**Training data:** GenImage subset (~100–200K balanced) + supplement with recent generators (Flux, SD3, Midjourney v6/7 samples from public datasets) because benchmark evidence shows training-data alignment dominates architecture. Real images: ImageNet val + RAISE/FFHQ slices matched for resolution and JPEG quality (critical — quality-factor confounds are a known benchmark trap; force matched Q distributions between classes).

**Robustness augmentation (train + eval):** JPEG Q∈[30,95], resize [0.5×–2×], center/random crop, Gaussian blur σ∈[0,3], sharpening — mirroring the NTIRE 2026 transform suite.

**Calibration:** temperature scaling on a held-out split; report calibrated probability + ECE. Optionally conformal prediction to output verdict sets ("AI or inconclusive at 90% coverage") — an uncommon and impressive touch.

**Deliverables:** training script (PyTorch Lightning or plain loop + hydra config), W&B/TensorBoard logs, model card documenting training data, known failure modes, and OOD behavior.

### M5 — Reverse image search (Step 2 of notes)

Pluggable providers behind one interface, all official APIs (ToS-compliant, as in the notes):

1. Google Cloud Vision Web Detection → `pagesWithMatchingImages`, `visuallySimilarImages`
2. Bing Visual Search → `pagesIncluding`
3. TinEye API (paid, optional) → oldest-index sort
4. Wayback CDX for earliest-capture timestamps of candidate URLs
5. Dedup/cluster with pHash (Hamming ≤ 10) + CLIP cosine ≥ 0.90, exactly per the notes' thresholds

Runs as an async Celery job (network-bound, seconds–minutes); the forensic verdict never blocks on it. Result feeds an `earliest_candidate` field for provenance context (e.g., image predates the claimed generator's release → strong human/real evidence).

### M6 — Fusion & verdict

Two layers:

1. **Hard rules (hierarchy of trust):**
   - Valid C2PA manifest naming an AI generator, or verified watermark → `AI (verified)`
   - Valid C2PA with camera capture claim + consistent PRNU → `Human (likely)`
   - Reverse search finds the image predating generative-AI availability → `Human (likely)`
2. **Meta-classifier for the gray zone:** logistic regression (or gradient-boosted trees) over the signal vector {detector p, FFT peaks, PRNU score, JPEG flags, ELA}; trained on the eval corpus; conservative thresholds so that low-evidence cases land in `Inconclusive`.

**Output schema** (matches project notes):

```json
{
  "ai_verdict": "verified | likely | inconclusive | unlikely",
  "confidence": 0.87,
  "evidence": {
    "provenance": {"c2pa": {...}, "iptc": {...}},
    "watermark": {"scheme": "...", "score": 0.0},
    "prnu": {"present": false, "score": 0.12},
    "spectrum": {"peaks": [...], "anomaly": 0.71},
    "jpeg": {"double_compression": false, "qt_match": "unknown"},
    "detector": {"model": "dinov2-tap-v1", "p_calibrated": 0.91, "robustness_drift": 0.06},
    "osint": {"earliest_candidate": {"url": "...", "wayback_first": "..."}}
  },
  "notes": ["watermark absence is non-evidence", "image was recompressed; PRNU down-weighted"]
}
```

---

## 3. Evaluation protocol

- **Benchmarks:** GenImage (cross-generator protocol: train SD1.4, test on 8 held-out generators), OpenSDI, and the NTIRE 2026 challenge dataset (108,750 real / 185,750 fake from 42 generators, 36 transforms) — the current gold standard for "in the wild" robustness.
- **Metrics:** AUROC + balanced accuracy per generator; mean cross-generator accuracy; robustness curves (accuracy vs. JPEG Q, vs. resize factor); ECE for calibration; attribution accuracy (generator-family classification) as a stretch goal.
- **Honesty checks:** matched JPEG-quality and resolution distributions between real/fake classes; report per-transform degradation, not just clean-image numbers.
- **Fusion evaluation:** show fusion beats the learned detector alone on OOD generators (this is the headline result of the write-up).

---

## 4. Phased roadmap (~10–12 weeks part-time)

| Phase | Duration | Deliverable |
|---|---|---|
| 0 — Scaffolding | 1 wk | repo, Docker, FastAPI skeleton, result schema, CI (pytest + ruff) |
| 1 — Provenance MVP | 1–2 wk | M1 + M2 (invisible-watermark, Stable Signature decoder); end-to-end `/analyze` returning provenance verdicts. **Already demo-able.** |
| 2 — Learned detector | 3–4 wk | M4 trained + calibrated; eval harness on GenImage; robustness sweep report |
| 3 — Forensics + fusion | 2–3 wk | M3 analyzers, M6 rules + meta-classifier; fusion-vs-detector ablation |
| 4 — OSINT + demo + write-up | 2 wk | M5 providers, Gradio/HF Spaces demo, model card, technical blog post |
| Stretch | — | Tree-Ring embed/detect demo; conformal prediction; generator-family attribution head |

---

## 5. Risk mapping (from the project notes)

| Risk (notes) | Mitigation in this design |
|---|---|
| Generalization across generators/versions | frozen-VFM detector (best current OOD behavior); fusion so detector isn't a single point of failure; per-generator eval reporting |
| Robustness to post-processing | NTIRE-style augmentation at train time; robustness-drift term down-weights detector confidence at inference |
| Adversarial evasion / watermark removal | verdict taxonomy treats absence as non-evidence; rules never claim "human" from missing watermark |
| Dataset bias / domain shift | matched Q/resolution distributions; report ECE; conformal option |
| False-positive ethics | conservative thresholds; `inconclusive` default; every verdict ships with machine-readable evidence + caveats |

---

## 6. Portfolio positioning (job-asset angle)

- **What it demonstrates:** production ML service design (FastAPI, async workers, Docker), transfer learning on foundation models, rigorous evaluation methodology, calibration/uncertainty — precisely the ML Engineer skill profile.
- **Artifacts to publish:** GitHub repo with CI, hosted demo (HF Spaces), a model card, and a blog post titled around the honest finding ("Why no single AI-image detector works — and how evidence fusion helps").
- **Interview stories it generates:** cross-generator generalization failure analysis, calibration under distribution shift, designing for "inconclusive" as a product decision.
