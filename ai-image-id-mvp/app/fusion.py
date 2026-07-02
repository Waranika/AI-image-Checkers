"""Step 3.6 — evidence fusion (rule layer of the notes' decision logic).

Hierarchy:
  1. Verified provenance/watermark naming an AI generator  -> AI (verified)
  2. Declared AI metadata (IPTC trainedAlgorithmicMedia, AI tool in Software) -> AI (likely)
     (metadata is trivially strippable/forgeable, so declared != verified)
  3. Multiple weak intrinsic signals agreeing              -> AI (likely)
  4. Nothing found                                          -> inconclusive
     (absence of watermark/metadata is NON-evidence; PRNU — not yet in the MVP —
      is what will let us say "unlikely" with confidence in phase 3)
"""
from __future__ import annotations

from .schema import AnalysisResult, Evidence, Verdict


def fuse(evidence: Evidence, sha256: str, phash: str) -> AnalysisResult:
    notes: list[str] = []
    prov, wms, spec = evidence.provenance, evidence.watermarks, evidence.spectrum

    detected_wm = next((w for w in wms if w.detected), None)

    # Rule 1 — verified signals
    if prov.c2pa_present and prov.c2pa_valid and any(
        "c2pa" in hit for hit in prov.ai_metadata_hits
    ):
        notes.append(f"valid C2PA manifest from AI generator: {prov.c2pa_generator}")
        return AnalysisResult(
            ai_verdict=Verdict.VERIFIED, confidence=0.98,
            evidence=evidence, notes=notes, sha256=sha256, phash=phash,
        )
    if detected_wm:
        notes.append(
            f"invisible watermark matched '{detected_wm.matched_payload}' "
            f"(bit accuracy {detected_wm.bit_accuracy})"
        )
        return AnalysisResult(
            ai_verdict=Verdict.VERIFIED, confidence=0.95,
            evidence=evidence, notes=notes, sha256=sha256, phash=phash,
        )

    # Rule 2 — declared (unverified) AI metadata
    declared = bool(prov.iptc_digital_source_type) or bool(prov.ai_metadata_hits)
    if prov.c2pa_present and prov.c2pa_valid is False:
        notes.append("C2PA manifest present but signature INVALID — treat with suspicion")
    if declared:
        hits = [prov.iptc_digital_source_type or ""] + prov.ai_metadata_hits
        notes.append("declared AI metadata: " + "; ".join(h for h in hits if h))
        confidence = 0.85 if prov.iptc_digital_source_type else 0.75
        return AnalysisResult(
            ai_verdict=Verdict.LIKELY, confidence=confidence,
            evidence=evidence, notes=notes, sha256=sha256, phash=phash,
        )

    # Rule 3 — weak intrinsic signals (spectrum only in the MVP; needs agreement
    # from a second signal once the learned detector lands, so cap at LIKELY/low conf)
    if spec.valid and spec.anomaly_score >= 0.8:
        notes.append(
            f"strong periodic spectral peaks ({spec.n_peaks}) — heuristic signal only"
        )
        return AnalysisResult(
            ai_verdict=Verdict.LIKELY, confidence=0.6,
            evidence=evidence, notes=notes, sha256=sha256, phash=phash,
        )

    # Rule 4 — default
    notes.append("no provenance, watermark, or strong intrinsic signal found")
    notes.append("absence of watermarks/metadata is non-evidence, not proof of human origin")
    return AnalysisResult(
        ai_verdict=Verdict.INCONCLUSIVE, confidence=0.5,
        evidence=evidence, notes=notes, sha256=sha256, phash=phash,
    )
