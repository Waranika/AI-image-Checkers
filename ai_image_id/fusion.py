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

    # Rule 2 — declared (unverified) AI metadata. Tiers within the tier:
    # a full generation recipe (0.85) > IPTC declaration (0.85) > AI-composite
    # (0.8) > tool-name hit in a field (0.75). All forgeable, all cap at LIKELY.
    ai_declared = prov.iptc_source_category in ("ai", "ai_composite")
    declared = bool(prov.generation_params_tool) or ai_declared or bool(prov.ai_metadata_hits)
    if prov.c2pa_present and prov.c2pa_valid is False:
        notes.append("C2PA manifest present but signature INVALID — treat with suspicion")
    if prov.c2pa_signature_valid and prov.c2pa_signer_trusted is False:
        notes.append("C2PA signature intact but signer not on the known trust list")
    if declared:
        if prov.generation_params_tool:
            model = f", model: {prov.generation_params_model}" if prov.generation_params_model else ""
            notes.append(f"embedded generation parameters ({prov.generation_params_tool}{model})")
            confidence = 0.85
        elif prov.iptc_source_category == "ai_composite":
            notes.append("declared AI-edited composite (IPTC compositeWithTrainedAlgorithmicMedia)")
            confidence = 0.8
        elif prov.iptc_source_category == "ai":
            notes.append(f"declared AI metadata: IPTC {prov.iptc_digital_source_type}")
            confidence = 0.85
        else:
            notes.append("declared AI metadata: " + "; ".join(prov.ai_metadata_hits))
            confidence = 0.75
        return AnalysisResult(
            ai_verdict=Verdict.LIKELY, confidence=confidence,
            evidence=evidence, notes=notes, sha256=sha256, phash=phash,
        )

    # Rule 3 — learned detector (calibrated). Can reach LIKELY, never VERIFIED:
    # only cryptographic provenance / watermarks earn "verified".
    det = evidence.detector
    if det and det.valid:
        # Down-weight confidence when the score is unstable under recompression.
        p_eff = det.p_calibrated * max(0.0, 1.0 - 2.0 * det.robustness_drift)
        agree = spec.valid and spec.anomaly_score >= 0.5
        if p_eff >= 0.9 or (p_eff >= 0.75 and agree):
            notes.append(
                f"learned detector p={det.p_calibrated:.2f} "
                f"(drift {det.robustness_drift:.2f}"
                + (", spectrum agrees)" if agree else ")")
            )
            return AnalysisResult(
                ai_verdict=Verdict.LIKELY, confidence=round(min(0.9, p_eff), 2),
                evidence=evidence, notes=notes, sha256=sha256, phash=phash,
            )
        if p_eff <= 0.1 and not agree:
            notes.append(
                f"learned detector p={det.p_calibrated:.2f} (low) — weak evidence "
                "of camera origin; PRNU/reverse-search needed to strengthen"
            )
            return AnalysisResult(
                ai_verdict=Verdict.UNLIKELY, confidence=0.6,
                evidence=evidence, notes=notes, sha256=sha256, phash=phash,
            )

    # Rule 3b — valid C2PA from a non-AI tool: capture claim can reach UNLIKELY;
    # otherwise it's context, not a verdict (PRNU in phase 3 will strengthen this).
    if prov.c2pa_present and prov.c2pa_valid and not declared:
        if prov.c2pa_capture_claim:
            notes.append(
                f"valid C2PA capture claim from {prov.c2pa_generator} — camera origin declared"
            )
            return AnalysisResult(
                ai_verdict=Verdict.UNLIKELY, confidence=0.75,
                evidence=evidence, notes=notes, sha256=sha256, phash=phash,
            )
        notes.append(
            f"valid C2PA manifest from non-AI tool ({prov.c2pa_generator}); no AI signals"
        )

    # Note-tier human-side hints — never move a verdict alone (forgeable);
    # reserved as corroborators for PRNU (phase 3).
    if prov.iptc_source_category == "capture":
        notes.append(f"declared capture source (IPTC {prov.iptc_digital_source_type}) — declared, not proof")
    if prov.camera_exif_present:
        notes.append(f"coherent camera EXIF block ({prov.camera_exif_fields} fields) — weak, forgeable")

    # Rule 3c — spectral heuristic alone is a single weak signal: note it, never
    # let it flip the verdict without agreement from another signal.
    if spec.valid and spec.anomaly_score >= 0.8:
        notes.append(
            f"spectral peaks present ({spec.n_peaks}) — single weak signal, insufficient alone"
        )

    # Rule 4 — default
    notes.append("no AI-indicative signal found")
    notes.append("absence of watermarks/metadata is non-evidence, not proof of human origin")
    return AnalysisResult(
        ai_verdict=Verdict.INCONCLUSIVE, confidence=0.5,
        evidence=evidence, notes=notes, sha256=sha256, phash=phash,
    )
