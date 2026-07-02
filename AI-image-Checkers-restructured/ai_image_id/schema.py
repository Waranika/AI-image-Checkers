"""Result schema — mirrors the 'Outputs (recommended fields)' section of the project notes."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    VERIFIED = "verified"        # AI, verified by provenance/watermark
    LIKELY = "likely"            # AI, likely (multiple weak signals agree)
    INCONCLUSIVE = "inconclusive"
    UNLIKELY = "unlikely"        # likely human/camera origin


class ProvenanceEvidence(BaseModel):
    c2pa_present: bool = False
    c2pa_valid: Optional[bool] = None
    c2pa_generator: Optional[str] = None
    c2pa_raw: Optional[dict[str, Any]] = None
    iptc_digital_source_type: Optional[str] = None
    software: Optional[str] = None
    ai_metadata_hits: list[str] = Field(default_factory=list)


class WatermarkEvidence(BaseModel):
    scheme: str
    applicable: bool = True
    detected: bool = False
    matched_payload: Optional[str] = None
    bit_accuracy: Optional[float] = None
    notes: Optional[str] = None


class SpectrumEvidence(BaseModel):
    anomaly_score: float = 0.0          # 0..1, heuristic
    n_peaks: int = 0
    valid: bool = True
    notes: Optional[str] = None


class DetectorEvidence(BaseModel):
    model: str
    p_calibrated: float = Field(ge=0.0, le=1.0)
    robustness_drift: float = 0.0   # |p(clean) - p(jpeg-recompressed)|
    valid: bool = True
    notes: Optional[str] = None


class Evidence(BaseModel):
    provenance: ProvenanceEvidence
    watermarks: list[WatermarkEvidence] = Field(default_factory=list)
    spectrum: SpectrumEvidence
    detector: Optional[DetectorEvidence] = None


class AnalysisResult(BaseModel):
    ai_verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Evidence
    notes: list[str] = Field(default_factory=list)
    sha256: str
    phash: str
