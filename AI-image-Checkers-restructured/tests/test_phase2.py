"""Phase 2 tests — everything that runs without torch/GPU."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ai_image_id.fusion import fuse
from ai_image_id.schema import (
    DetectorEvidence, Evidence, ProvenanceEvidence, SpectrumEvidence, Verdict,
)
from training.calibrate_eval import auroc, balanced_accuracy, ece, fit_temperature, _sigmoid
from training.prepare_data import prepare_split


def test_temperature_scaling_reduces_ece():
    rng = np.random.default_rng(0)
    # Well-calibrated ground truth: y ~ Bernoulli(p_true), calibrated logit = logit(p_true).
    p_true = rng.uniform(0.02, 0.98, 4000)
    y = (rng.uniform(size=4000) < p_true).astype(np.float64)
    calibrated_logits = np.log(p_true / (1 - p_true))
    logits = 4.0 * calibrated_logits  # inflate -> overconfident model
    T = fit_temperature(logits, y)
    assert 3.0 < T < 5.0  # recovers the inflation factor (~4)
    assert ece(_sigmoid(logits / T), y) < ece(_sigmoid(logits), y)


def test_auroc_and_bacc_sane():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auroc(perfect, y) == 1.0
    assert balanced_accuracy(perfect, y) == 1.0
    assert abs(auroc(np.full(6, 0.5), y) - 0.5) < 1e-9


def test_prepare_split_matches_confounds(tmp_path: Path):
    # "real": large JPEGs; "fake": small PNGs — classic confounded setup
    real, fake = tmp_path / "src_real", tmp_path / "src_fake"
    real.mkdir()
    fake.mkdir()
    rng = np.random.default_rng(1)
    for i in range(8):
        Image.fromarray(rng.integers(0, 255, (800, 1200, 3), dtype=np.uint8)).save(
            real / f"r{i}.jpg", quality=85)
        Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)).save(
            fake / f"f{i}.png")

    manifest = prepare_split(real, fake, tmp_path / "out", n_per_class=8, seed=0)
    import csv
    rows = list(csv.DictReader(manifest.open()))
    assert len(rows) == 16
    # After prep: every file is a JPEG with Q and short-side from the SHARED choices
    for row in rows:
        out_file = tmp_path / "out" / row["label"] / row["file"]
        with Image.open(out_file) as im:
            assert im.format == "JPEG"
            assert min(im.size) in {256, 320, 384, 448, 512}
        assert 70 <= int(row["jpeg_q"]) <= 95


def _base_evidence(**detector_kwargs) -> Evidence:
    return Evidence(
        provenance=ProvenanceEvidence(),
        watermarks=[],
        spectrum=SpectrumEvidence(anomaly_score=0.0, n_peaks=0),
        detector=DetectorEvidence(**detector_kwargs) if detector_kwargs else None,
    )


def test_fusion_detector_high_score_is_likely_not_verified():
    ev = _base_evidence(model="test", p_calibrated=0.97, robustness_drift=0.02)
    r = fuse(ev, sha256="x", phash="y")
    assert r.ai_verdict == Verdict.LIKELY  # never VERIFIED from a classifier
    assert r.confidence <= 0.9


def test_fusion_drift_downweights_to_inconclusive():
    ev = _base_evidence(model="test", p_calibrated=0.95, robustness_drift=0.4)
    r = fuse(ev, sha256="x", phash="y")
    assert r.ai_verdict == Verdict.INCONCLUSIVE


def test_fusion_low_detector_score_gives_unlikely():
    ev = _base_evidence(model="test", p_calibrated=0.03, robustness_drift=0.01)
    r = fuse(ev, sha256="x", phash="y")
    assert r.ai_verdict == Verdict.UNLIKELY
