# Results log

Each run is keyed by the short commit hash of the code that produced it.
Artifacts (checkpoint, calibration, manifests, provenance.json) live in
Drive under `ai_image_id/runs/<run-id>/`. Numbers are reported with their
evaluation conditions — an accuracy without its distribution is meaningless
in this problem domain.

---

## Run `cb68637` — first trained detector (2026-07-12)

**Setup.** Frozen DINOv2 ViT-S/14 backbone + attention-pooling head
(~500K trainable params). Data: GenImage SDv1.4, val-split slice only —
5,000/class train, 1,000/class held out, disjoint by construction
(single-seed prep, manifest sliced; leakage guard: 0 overlapping sources).
De-confounded preprocessing: shared JPEG-Q [70,95] and short-side
{256..512} distributions for both classes. Robustness augmentation
n_aug=2 (JPEG 30–90 / resize 0.5–1.5× / blur). 5 epochs, AdamW 3e-4,
batch 256, T4.

**In-distribution results (SDv1.4 → SDv1.4):**

| metric | value |
|---|---|
| val accuracy | **0.922** (epoch 3; plateau ≈0.92 from epoch 2) |
| ECE before / after temperature scaling | 0.052 → **0.017** |
| fitted temperature | 2.58 (overconfident, as expected) |
| NLL before / after | 0.312 → 0.205 |
| robustness drift (Q75 recompress, spot-checked) | < 0.01 |

**Caveats — read before quoting these numbers.**
- *Home game*: train and test are the same generator, same preprocessing.
  This measures "did the head learn generation artifacts at all", not
  cross-generator transfer — which is the question that matters
  (GenImage's own ResNet-50 baseline: 99.9% in-distribution, ~52–55% on
  unseen generators).
- Trained on the dataset's *val* split (train/ materialization pending);
  scale is methodology-validation, not final.
- Photo-vs-AI domain only. Behavior on illustrations/paintings/anime is
  unvalidated and expected to skew false-positive (see model-card notes).
- Mild overfit after epoch ~2 (train loss 0.03 vs. plateaued val acc);
  epochs=3 suffices at this scale.

**End-to-end verdicts observed** (full pipeline, fusion engine):
SDv1.4 fake → `likely (0.9)`, p_cal=0.93, drift 0.005;
ImageNet real → `unlikely (0.6)`, p_cal=0.015 — first activation of the
`unlikely` verdict class. Classifier evidence is capped at `likely` by
policy; `verified` remains reserved for cryptographic provenance.

**Cross-generator table:** _pending — next run._

**Cross-generator results (trained on SDv1.4 only):**

| test generator | AUROC | bal. acc | notes |
|---|---|---|---|
| SDv1.4 (in-dist) | 0.974 | 0.919 | home game |
| Midjourney | 0.842 | 0.706 | vs. 54.9% GenImage ResNet-50 baseline |
| glide | 0.791 | 0.648 | older diffusion — moderate transfer |
| BigGAN | 0.580 | 0.522 | GAN family — chance level, expected |
| ADM | 0.335 | 0.460 | **inverted** signal (see below) |
| mean (unseen) | 0.637 | 0.584 | |

Transfer degrades with architectural distance from the training generator.
ADM's sub-0.5 AUROC indicates the learned feature direction (likely dominated
by SD's VAE-decoder fingerprint) *anti-correlates* on pixel-space diffusion —
the head ranks ADM fakes as more photo-like than real photos. Balanced
accuracy lags AUROC off-distribution: ranking transfers better than the
SDv1.4-fitted threshold/temperature. Together these are the empirical case
for evidence fusion over any single classifier verdict.
**Reproduce:** notebook `notebooks/02_train_detector.ipynb` at commit
`cb68637`, GenImage SDv1.4 archive, seeds in `runs/cb68637/provenance.json`.
