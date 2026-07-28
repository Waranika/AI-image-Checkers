"""AI Image Identification — evidence-fusion pipeline.

Modules:
    provenance/      M1 — C2PA, IPTC, generation params, camera EXIF
    watermark/       M2 — DWT-DCT, TrustMark P/Q/B, Stable Signature BZH
    forensics/       M3 — FFT spectral heuristic (demoted to note-only)
    detector/        M4 — frozen DINOv2 + attention-pooling head
    web_provenance/  M5 — reverse image search + page-context analysis

Entry points:
    analyze_image()     — full pipeline (M1+M2+M3+M4)
    trace_provenance()  — M5 web search + retry loop
"""
from .main import analyze_image
from .web_provenance import trace_provenance

__all__ = ["analyze_image", "trace_provenance"]
