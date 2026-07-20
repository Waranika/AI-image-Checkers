# Results log

Measured results for each module of the evidence-fusion pipeline. Every
number is reported with the conditions it was measured under — in this
problem domain, an accuracy without its distribution is meaningless.

**Conventions.** Training runs are keyed by the short commit hash of the
code that produced them; their artifacts (checkpoint, calibration,
manifests, provenance.json) live in Drive under `ai_image_id/runs/<run-id>/`.
Module validations are keyed by the scenario notebook that produced them.
Notebooks are committed *with* outputs and serve as run records.

---

## At a glance

| module | status | headline result | where measured |
|---|---|---|---|
| M1 provenance | validated | C2PA survives only metadata-stripping; every declared signal dies at first re-save | notebook 03 |
| M2 watermarks | validated | TrustMark survives 6/7 transports; end-to-end detection on a real Adobe-watermarked image | notebook 04 |
| M3 forensics | demoted to note-only | FFT heuristic false-positives on real photos (JPEG block harmonics) | notebook 01 |
| M4 detector | first run complete | 92.2% in-distribution; cross-generator transfer degrades with architectural distance (0.84 Midjourney → 0.34 ADM) | run `cb68637` |
| M5 web provenance | designed, not built | — | — |

**The central measurement (M1 × M2 composite):** metadata and durable
watermarks are complementary — their union covers 6 of 7 measured
transports, and the single shared blind spot (aggressive resize +
recompression) is where the transform-invariant classifier takes over.
Details in the M2 section.

---

## M1 — Provenance & metadata (notebook 03, 2026-07-16)

### Corpus validation

25 C2PA test files (c2pa-org/public-testfiles, incl. deliberately tampered):

- All intact manifests: `signature_valid=True, signer_trusted=False`
  (test certificates — a trust-list config fact, not a forgery signal).
- All `E-*` tampered files: `signature_valid=False`, correctly flagged.
- Files without manifests: correctly reported absent.
- Actions history extracted per file (`created`, `opened`,
  `color_adjustments`, …) — manifest history is now readable evidence.

### Wild validation

- OpenAI/ChatGPT PNG → `verified (0.98)`; actions:
  `c2pa.created (gpt-image)`, `c2pa.converted`, `c2pa.watermarked.unbound`
  — SynthID presence *declared* in-manifest (M1 reads the declaration;
  independent verification is impossible, see M2/SynthID).
- All four signal families (C2PA / generation-params / IPTC vocabulary /
  camera-EXIF) fire at their designed tiers on synthetic fixtures,
  M1-isolated (no other module contributing to verdicts).

### Transport-degradation matrix

One verified OpenAI PNG pushed through 7 transports:

| transport | C2PA | gen_params | IPTC | tool_fields | camera_exif | verdict |
|---|---|---|---|---|---|---|
| original | ✓ | · | · | · | · | verified |
| JPEG re-save Q92 | · | · | · | · | · | inconclusive |
| screenshot | · | · | · | · | · | inconclusive |
| messenger (½-resize + Q70)* | · | · | · | · | · | inconclusive |
| exiftool -all= | **✓** | · | · | · | · | **verified** |
| crop (50px border) | · | · | · | · | · | inconclusive |
| PIL re-encode | · | · | · | · | · | inconclusive |

### Findings

1. **C2PA is the only M1 signal that survives any transport** — and only
   metadata-stripping: JUMBF boxes live outside exiftool's reach. Every
   *declared* signal (IPTC, generation params, tool names, camera EXIF)
   dies at the first re-save.
2. **Any byte-touching transform kills C2PA** (re-encode, crop) via hash
   mismatch.
3. **"Absence is non-evidence" is therefore a measured necessity**, not a
   design preference — most transports produce files indistinguishable
   from never-labeled ones.

\* synthetic proxy (PIL Q70 + ½ resize), not a real platform pipeline;
real-WhatsApp/Instagram validation pending.

---

## M2 — Watermarks (notebook 04, 2026-07-18)

### Registry outcome: 3 active decoders, 1 rejected, 2 documented-closed

| decoder | true-positive validated on | fusion tier | real-world reach |
|---|---|---|---|
| dwtDct (SD invisible-watermark) | self-embedded round-trip (0.917 lossless) | verified | pristine SD PNGs only (see fragility) |
| TrustMark P/Q/B (Adobe) | **real Content Authenticity app output** — `trustmark-P`, payload recovered, variant matches manifest `alg=com.adobe.trustmark.P` | verified | Content Authenticity outputs |
| Stable Signature BZH (IMATAG) | **self-generated SDXL-turbo + BZH-strong VAE** (p=1.0) | likely (zero-bit, learned) | IMATAG-watermarked builds |

### dwtDct fragility (measured)

Lossless round-trip 0.917 — barely above the 0.90 detection threshold.
**Any** JPEG, even Q92, → ≈0.51: statistically indistinguishable from an
unwatermarked image (clean baseline reads 0.542 — the decoder always
produces a score; ~0.5 is coin-flip noise, and only ≥0.90 means anything).
There is no degradation curve: a wall, not a cliff. Practical reach:
first-hop pristine files only.

### Coverage boundaries (measured)

- Raw firefly.adobe.com export carries C2PA but **no** TrustMark; only
  Content Authenticity app outputs embed the watermark.
- Variant matters: Adobe embeds `P`. A Q-only decoder false-negatives on
  provably watermarked files (we did exactly this, then fixed it by
  looping P/Q/B — variants are mutually incompatible).

### Rejected: SynthID CNN surrogate

A third-party CNN ensemble (unaudited single-author repo, claimed 0.97
accuracy) was integrated and evaluated: it fired on **all** tested images —
AI outputs, personal photos, synthetic fixtures — at P=0.65–1.00. A
shortcut-learning classifier, not a watermark detector. Removed from the
active registry; kept in-repo (`synthid_cnn.py`) as a cautionary tale.

SynthID itself: independent detection is cryptographically impossible
without Google's keys. Official detection is oracle-only (SynthID Detector
portal, Gemini app) or a partner-preview Cloud API. M1 reads SynthID
*declarations* from C2PA manifests instead.

### Transport-degradation matrix

Content Authenticity app image (TrustMark-P), same 7 hops as M1:

| transport | dwtDct | TrustMark | BZH | verdict |
|---|---|---|---|---|
| original | · | ✓ | · | verified |
| JPEG re-save Q92 | · | ✓ | · | verified |
| screenshot | · | ✓ | · | verified |
| messenger (½-resize + Q70)* | · | · | · | inconclusive |
| exiftool -all= | · | ✓ | · | verified |
| crop (50px border) | · | ✓ | · | verified |
| PIL re-encode | · | ✓ | · | verified |

### The composite finding (M1 × M2) — the project's central measurement

| transport | C2PA (M1) | TrustMark (M2) | union covered |
|---|---|---|---|
| original | ✓ | ✓ | ✓ |
| re-save | · | ✓ | ✓ |
| screenshot | · | ✓ | ✓ |
| messenger* | · | · | ✗ |
| exiftool-strip | ✓ | ✓ | ✓ |
| crop | · | ✓ | ✓ |
| re-encode | · | ✓ | ✓ |

Metadata proves origin on pristine files; the durable watermark carries
detection through five transports that kill metadata; their union covers
6/7 measured transports. The single shared blind spot — aggressive
resize + recompression — is where the transform-invariant classifier (M4)
takes over. Evidence fusion, measured rather than argued.

### Caveats

- Crop hop = 50px border trim (light crop); deeper crops untested.
- Messenger row is a synthetic proxy; real-platform validation pending.
- BZH image not yet run through the matrix (IMATAG claims survival through
  the messenger-class attack — would close the last row if confirmed).
- TrustMark payload is binary; it renders as mojibake through the text
  schema (cosmetic — detection and variant ID are unaffected).

---

## M4 — Learned detector: run `cb68637` (notebook 02, 2026-07-12)

### Setup

Frozen DINOv2 ViT-S/14 backbone + attention-pooling head (~500K trainable
params). Data: GenImage SDv1.4, **val-split slice only** — 5,000/class
train, 1,000/class held out, disjoint by construction (single-seed prep,
manifest sliced; leakage guard: 0 overlapping sources). De-confounded
preprocessing: shared JPEG-Q [70,95] and short-side {256..512}
distributions for both classes. Robustness augmentation n_aug=2
(JPEG 30–90 / resize 0.5–1.5× / blur). 5 epochs, AdamW 3e-4, batch 256, T4.

### In-distribution results (SDv1.4 → SDv1.4)

| metric | value |
|---|---|
| val accuracy | **0.922** (epoch 3; plateau ≈0.92 from epoch 2) |
| ECE before / after temperature scaling | 0.052 → **0.017** |
| fitted temperature | 2.58 (overconfident, as expected) |
| NLL before / after | 0.312 → 0.205 |
| robustness drift (Q75 recompress, spot-checked) | < 0.01 |

### Cross-generator results (trained on SDv1.4 only)

| test generator | AUROC | bal. acc | notes |
|---|---|---|---|
| SDv1.4 (in-dist) | 0.974 | 0.919 | home game |
| Midjourney | 0.842 | 0.706 | vs. 54.9% GenImage ResNet-50 baseline |
| glide | 0.791 | 0.648 | older diffusion — moderate transfer |
| BigGAN | 0.580 | 0.522 | GAN family — chance level, expected |
| ADM | 0.335 | 0.460 | **inverted** signal (see below) |
| mean (unseen) | 0.637 | 0.584 | |

Transfer degrades with architectural distance from the training generator.
ADM's sub-0.5 AUROC indicates the learned feature direction (likely
dominated by SD's VAE-decoder fingerprint) *anti-correlates* on pixel-space
diffusion — the head ranks ADM fakes as more photo-like than real photos.
Balanced accuracy lags AUROC off-distribution: ranking transfers better
than the SDv1.4-fitted threshold/temperature. Together these are the
empirical case for evidence fusion over any single classifier verdict.

### End-to-end verdicts observed (full pipeline)

- SDv1.4 fake → `likely (0.9)`, p_cal=0.93, drift 0.005
- ImageNet real → `unlikely (0.6)`, p_cal=0.015 — first activation of the
  `unlikely` verdict class

Classifier evidence is capped at `likely` by policy; `verified` remains
reserved for cryptographic provenance.

### Caveats — read before quoting these numbers

- *Home game*: in-distribution rows measure "did the head learn generation
  artifacts at all", not transfer — the cross-generator table is the
  question that matters (GenImage's ResNet-50 baseline: 99.9%
  in-distribution, ~52–55% on unseen generators).
- Trained on the dataset's *val* split (train/ materialization pending);
  this scale is methodology-validation, not final.
- Photo-vs-AI domain only. Behavior on illustrations/paintings/anime is
  unvalidated and expected to skew false-positive.
- Mild overfit after epoch ~2 (train loss 0.03 vs. plateaued val acc);
  epochs=3 suffices at this scale.

### Reproduce

Notebook `notebooks/02_train_detector.ipynb` at commit `cb68637`,
GenImage SDv1.4 archive, seeds in `runs/cb68637/provenance.json`.

---

## Pending measurements

- Real-platform transport hops (WhatsApp / Instagram / Telegram received
  files) to replace the synthetic messenger proxy in both matrices.
- BZH image through the transport matrix (tests IMATAG's survival claims
  against our approximate decoder).
- M3 forensics honest audit as its own notebook (the FFT heuristic was
  demoted to note-only after false-positiving on JPEG 8×8 block harmonics
  of real photos; the demotion story deserves its dated entry).
- M4 real train/-split run at 20K+/class; mixed-generator training
  experiment (does adding 2K BigGAN+ADM samples repair the inverted ADM
  signal?).
- M4 interpretability set (score histograms, attention maps, perturbation
  sweeps — notebook 06, scoped).
