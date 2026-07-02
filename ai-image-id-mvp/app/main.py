"""Pipeline orchestration + FastAPI service + CLI.

CLI:   python -m app.main path/to/image.jpg
API:   uvicorn app.main:api --reload   then POST /analyze with multipart 'file'
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from .forensics import analyze_spectrum
from .fusion import fuse
from .ingest import ingest
from .provenance import analyze_provenance
from .schema import AnalysisResult, Evidence
from .watermark import analyze_watermarks


def analyze_image(path: str | Path) -> AnalysisResult:
    img = ingest(path)
    evidence = Evidence(
        provenance=analyze_provenance(img.path),
        watermarks=analyze_watermarks(img.rgb),
        spectrum=analyze_spectrum(img.rgb),
    )
    return fuse(evidence, sha256=img.sha256, phash=img.phash)


# --------------------------------------------------------------------------- API
try:
    from fastapi import FastAPI, File, UploadFile

    api = FastAPI(title="Identifying AI Image — MVP", version="0.1.0")

    @api.post("/analyze", response_model=AnalysisResult)
    async def analyze(file: UploadFile = File(...)) -> AnalysisResult:
        suffix = Path(file.filename or "upload.png").suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            return analyze_image(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

except ImportError:  # FastAPI optional for CLI-only use
    api = None


# --------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m app.main <image_path>")
        raise SystemExit(1)
    result = analyze_image(sys.argv[1])
    print(result.model_dump_json(indent=2))
