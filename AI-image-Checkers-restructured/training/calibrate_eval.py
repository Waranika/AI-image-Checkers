"""Phase 2, steps 4-5 — calibration + evaluation. Pure numpy/scipy: runs anywhere.

Temperature scaling: fit a single T > 0 on held-out logits minimizing NLL; report
ECE before/after. Evaluation: AUROC, balanced accuracy, and per-generator tables
for the cross-generator protocol.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ calibration
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    p = np.clip(_sigmoid(logits / T), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Golden-section search over log T in [-3, 3] — no scipy needed."""
    lo, hi = -3.0, 3.0
    phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    for _ in range(60):
        if _nll(logits, y, np.exp(c)) < _nll(logits, y, np.exp(d)):
            b = d
        else:
            a = c
        c, d = b - phi * (b - a), a + phi * (b - a)
    return float(np.exp((a + b) / 2))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (probs > bins[i]) & (probs <= bins[i + 1])
        if m.sum() == 0:
            continue
        total += m.mean() * abs(probs[m].mean() - y[m].mean())
    return float(total)


def calibrate(val_logits_npz: str | Path, out_json: str | Path = "calibration.json") -> dict:
    d = np.load(val_logits_npz)
    logits, y = d["logits"].astype(np.float64), d["label"].astype(np.float64)
    T = fit_temperature(logits, y)
    report = {
        "temperature": round(T, 4),
        "ece_before": round(ece(_sigmoid(logits), y), 4),
        "ece_after": round(ece(_sigmoid(logits / T), y), 4),
        "nll_before": round(_nll(logits, y, 1.0), 4),
        "nll_after": round(_nll(logits, y, T), 4),
    }
    Path(out_json).write_text(json.dumps(report, indent=2))
    return report


# ------------------------------------------------------------------- evaluation
def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney), ties handled by average rank."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def balanced_accuracy(scores: np.ndarray, y: np.ndarray, thr: float = 0.5) -> float:
    pred = scores >= thr
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    tnr = float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    return (tpr + tnr) / 2


def cross_generator_table(results: dict[str, tuple[np.ndarray, np.ndarray]]) -> str:
    """results: {generator_name: (probs, labels)} -> markdown table."""
    lines = ["| generator | AUROC | bal. acc |", "|---|---|---|"]
    aurocs, baccs = [], []
    for name, (p, y) in sorted(results.items()):
        a, b = auroc(p, y), balanced_accuracy(p, y)
        aurocs.append(a)
        baccs.append(b)
        lines.append(f"| {name} | {a:.3f} | {b:.3f} |")
    lines.append(f"| **mean** | **{np.nanmean(aurocs):.3f}** | **{np.nanmean(baccs):.3f}** |")
    return "\n".join(lines)
