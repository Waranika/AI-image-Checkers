# Identifying AI Images — Implementation Plan

**Goal:** Build an AI image detector that is reliable enough to give
suspicions of an AI image when inputted straight from the generator.

## Design philosophy

No universal AI image detector exists. Recent benchmarks (NTIRE 2026
challenge, Feb 2026) show top commercial generators defeat most public
detectors (18–30% accuracy), and training-data alignment matters more
than architecture.

What if, instead of relying on a single classifier, we investigated
the traces a generator actually leaves behind? Every AI image carries
potential evidence: metadata declarations, invisible watermarks embedded
by the platform, and statistical patterns in the pixels themselves.
This is actually what most production detection systems do —
they combine multiple signal families rather than trusting any one:

- **Hive Moderation** uses a trained classifier continuously retrained
  against new generators, combined with C2PA/watermark checks, serving
  commercial content-moderation clients at scale.
- **Reality Defender** layers deepfake-specific classifiers with
  metadata forensics and context-aware detection, selling primarily to
  enterprises and governments (public API since 2025).
- **Sightengine** combines frequency-domain analysis with pixel-level
  classifiers and metadata inspection.
- **Major platforms (Meta, Google, TikTok, YouTube)** run a four-layer
  stack internally: C2PA Content Credentials, visual watermarks
  (SynthID), metadata forensics, and trained classifiers — each layer
  covering blind spots the others miss.

Our approach follows the same principle: investigate every available
trace, tier the evidence by reliability, and fuse it into a single
honest verdict — with "inconclusive" as a first-class outcome when
the evidence is insufficient.

---

## 1. System architecture

```
                        ┌──────────────────────────────────────────┐
                        │              FastAPI service             │
                        │  POST /analyze  (image upload)           │
                        └───────────────┬──────────────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   Ingest & prep   │  hash (SHA-256, pHash),
                              │                   │  decode, PIL load
                              └─────────┬─────────┘
              ┌─────────────────────────┼──────────────────────────┐
              │                         │                          │
   ┌──────────▼─────────┐   ┌──────────▼──────────┐   ┌───────────▼──────────┐
   │ M1 Provenance      │   │ M2 Watermark        │   │ M4 Learned detector  │
   │ pyexiftool, c2pa   │   │ decoders            │   │ frozen DINOv2        │
   │ IPTC, PNG chunks,  │   │ DWT-DCT, TrustMark  │   │ + attention-pooling  │
   │ camera EXIF        │   │ Stable Sig. BZH     │   │ head, calibrated     │
   └──────────┬─────────┘   └──────────┬──────────┘   └───────────┬──────────┘
              │                        │                          │
              └────────────────────────┼──────────────────────────┘
                             ┌─────────▼─────────┐
                             │  Fusion engine    │  rule cascade → verdict
                             └─────────┬─────────┘
                                       │
                   (if inconclusive)   │
                             ┌─────────▼─────────┐
                             │  M5 Web provenance │  Google Vision reverse
                             │  reverse search +  │  search + page-context
                             │  page-context      │  analysis + retry loop
                             └─────────┬─────────┘
                             ┌─────────▼─────────┐
                             │  Verdict report   │  JSON per schema
                             └───────────────────┘
```

M3 (FFT spectral forensics) was implemented and evaluated but demoted
to note-only after false-positiving on real photos' JPEG block harmonics.
It runs but never moves a verdict.

**Stack:** Python 3.12, pip-installable package (`pip install -e .`),
PyTorch (optional, for M4), Google Colab for training/evaluation, Google
Cloud Vision API (optional, for M5). FastAPI for the REST endpoint
(optional dependency). CI: pytest + ruff on GitHub Actions.

**Repo layout:**

```
ai_image_id/                # pip-installable package
├── __init__.py                 re-exports analyze_image, trace_provenance
├── schema.py                   Pydantic models (Evidence, Verdict, etc.)
├── ingest.py                   SHA-256, pHash, PIL loading
├── fusion.py                   rule cascade → AnalysisResult
├── main.py                     analyze_image() + FastAPI + CLI
├── provenance/                 M1 — C2PA, IPTC, gen-params, camera EXIF
│   └── __init__.py
├── watermark/                  M2 — decoder registry
│   ├── __init__.py                 DWT-DCT, TrustMark P/Q/B, BZH
│   ├── dwt_dct.py                  vendored blind codec (torch-free)
│   └── synthid_cnn.py              evaluated + rejected (documented)
├── forensics/                  M3 — FFT heuristic (demoted to note-only)
│   └── __init__.py
├── detector/                   M4 — DINOv2 + attention-pooling head
│   └── __init__.py
└── web_provenance/             M5 — reverse search + context analysis
    └── __init__.py
training/                   # training pipeline (outside the package)
├── prepare_data.py             de-confounded split preparation
├── embed.py                    DINOv2 embedding precomputation
├── train_head.py               attention-pooling head training
└── calibrate_eval.py           temperature scaling, cross-gen tables
notebooks/                  # scenario testing + run records
├── 02_train_detector.ipynb     M4 training + cross-generator eval
├── 03_m1_provenance_scenarios  M1 validation + transport matrix
├── 04_m2_watermark_scenarios   M2 validation + transport matrix
├── 05_m4_detector_scenarios    Synthbuster zero-shot + transport matrix
├── 06_m5_web_provenance        reverse search + context analysis
└── 07_full_pipeline_test       all modules active, integration test
tests/                      # pytest suite
docs/                       # results.md, model card
```

---

## 2. Module specifications

### M1 — Provenance & metadata

When an image is created — either by a camera or generated by an AI
model — data is associated to it through several channels, each
embedded differently into the file and each with different durability.

**Standard metadata fields.** A camera writes its make, model, exposure
settings into EXIF headers. AI tools may write their name into Software
or Creator fields. Meta labels outputs with IPTC `DigitalSourceType =
trainedAlgorithmicMedia`. These are declarations, not proof — trivially
writable by anyone with exiftool.

> In the code: `_exiftool_json()` reads all metadata once; three
> scanners (`_scan_iptc_source_type`, `_scan_ai_tool_fields`,
> `_scan_camera_exif`) extract specific signals from the flat dict.

**Generation-parameter chunks.** Local SD tools (A1111 WebUI, ComfyUI,
NovelAI) write the entire generation recipe — prompt, sampler, seed,
model name — into PNG text chunks. Stronger than a bare tag (a full
recipe is hard to explain innocently), but format-specific and stripped
by any re-encode.

> In the code: `_scan_generation_params()` regex-matches the A1111
> recipe pattern and extracts the model name; checks for ComfyUI's
> `class_type` JSON and NovelAI's software tag.

**C2PA Content Credentials.** The strongest channel — cryptographic.
OpenAI, Adobe, and others sign a manifest embedded in a JUMBF box
recording who created the image, what actions were performed, and
optionally a chain of ingredients. The signature is cryptographic:
modifying pixels invalidates it.

> In the code: `_c2pa_read()` opens the manifest store via
> `c2pa-python`, splits validity into `signature_valid` and
> `signer_trusted`. `_walk_c2pa_store()` traverses every manifest
> (including ingredients) for AI markers and actions history.

**Camera-EXIF block.** A coherent Make/Model/exposure block is a weak
human-side hint (note-tier only, trivially forgeable).

### M2 — Watermark decoders

A registry of public watermark decoders, each checking a different
scheme. Every decoder is optional — missing dependencies degrade
gracefully (`applicable=False`) rather than crashing.

**DWT-DCT (SD default).** The `invisible-watermark` library's blind
codec, used by Stable Diffusion pipelines. Checks against two known
payloads (SDXL 48-bit, SD-v1 text). Vendored fallback when `imwatermark`
is not installed.

> In the code: `_check_dwtdct()` decodes bits and compares against
> `KNOWN_PAYLOADS`; reports bit accuracy with threshold at 0.90.

**TrustMark (Adobe).** Neural watermark behind "Durable Content
Credentials." Loops all model variants (P, Q, B) because they're
mutually incompatible — Adobe's Content Authenticity app embeds with P.

> In the code: `_check_trustmark()` caches a `TrustMark` instance
> per variant, returns on the first detected variant.

**Stable Signature BZH (IMATAG).** Zero-bit detector for images from
Stable-Signature-watermarked models (e.g. SDXL-turbo IMATAG builds).
Returns watermarked yes/no with a p-value, no payload.

> In the code: `_check_stable_signature_bzh()` runs the HF ResNet-18
> model; sigmoid of the logit gives p(watermarked).

**Closed schemes (documented, not detected).** SynthID (Google/OpenAI)
requires Google's private keys — independent detection is
cryptographically impossible. A third-party CNN surrogate was evaluated
and rejected (fired on all images). Meta's invisible watermark has no
public decoder.

### M4 — Learned detector

Frozen DINOv2 ViT-S/14 backbone + attention-pooling head (~500K
trainable params). The frozen-VFM approach shows the best cross-
generator generalization in current literature.

**Training pipeline:** `prepare_data.py` de-confounds by matching
JPEG quality and resolution distributions between real and fake
classes (prevents compression-shortcut learning). `embed.py`
precomputes frozen patch-token embeddings as fp16 shards (run once,
reuse). `train_head.py` trains the attention-pooling head on the
shards with preallocated fp16 host buffers (per-batch fp32 GPU
transfer to stay within Colab's 12.7 GB RAM). `calibrate_eval.py`
fits a temperature scalar and computes ECE, AUROC, cross-generator
tables.

**Architecture rationale:** frozen backbone means embeddings are
precomputed once (~200 MB per generator's val slice), the head trains
in minutes, and cross-generator evaluation is a forward pass over
cached shards — the full training + eval cycle fits in a free Colab
session.

### M5 — Web provenance

When the pipeline returns inconclusive, M5 asks the internet whether
this image has been seen before and what context surrounds it.

**Three layers of signal from one API call:**

1. **Page-context keywords** — Google Cloud Vision's `WEB_DETECTION`
   returns `bestGuessLabels`, page titles, and URLs. These are scanned
   for AI-related terms ("deepfake", "ai generated", generator names).
   The keywords only fire on pages that already matched the image
   visually, so even broad terms like "fake" carry real signal.

2. **Domain classification** — matching domains classified as AI
   galleries (civitai, midjourney.com, lexica.art) or stock photo
   platforms (flickr, shutterstock). Contextual evidence, not proof.

3. **Upstream re-analysis** — fetch the matched image and run the full
   pipeline on it. If the upstream copy has intact C2PA or watermarks
   that the local (stripped) copy lost, the verdict is inherited
   through a documented chain. Recursive: if the upstream is also
   inconclusive, retry one level deeper.

> In the code: `trace_provenance()` is the single entry point — one
> function that searches, classifies, checks context, optionally
> fetches and re-analyzes, and returns a dict that goes straight into
> the evidence card. The search backend (`_reverse_search`) is a
> single function to swap for TinEye, SerpApi, or Bing.

### Fusion engine

A rule cascade in `fusion.py` that tiers evidence by reliability:

1. **Verified** — cryptographic proof: valid C2PA manifest naming an
   AI generator (top-level or ingredient), or a decoded watermark
   payload (DWT-DCT, TrustMark).

2. **Likely** — declared or learned signal: generation-parameter
   chunks, IPTC AI tags, learned watermark detection (BZH), the M4
   classifier above threshold, or M5 web context with multiple
   independent sources.

3. **Inconclusive** — the honest default when evidence is
   insufficient. Absence of signals is non-evidence: metadata is
   stripped by screenshots, watermarks are stripped by recompression,
   and a clean file proves nothing.

4. **Unlikely** — positive evidence of human origin: C2PA capture
   claim, or a confidently-low classifier score. Requires positive
   evidence, not merely the absence of AI signals.

This hierarchy is the project's core invariant: easy to confirm AI
when evidence exists, impossible to confirm human from silence.
