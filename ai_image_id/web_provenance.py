"""M5 — Web provenance: find the oldest trace, analyze it, optionally retry.

When the pipeline returns inconclusive on a stripped/recompressed image, M5
asks the internet: "has this image been seen before, and does an earlier copy
still carry provenance?" If yes, the verdict is inherited from the upstream
copy through a documented chain.

Uses Google Cloud Vision's WEB_DETECTION (1,000 free requests/month) by
default. The search backend is a single function — swap it for TinEye,
SerpApi, or Bing by replacing `_reverse_search`.

Requires: GOOGLE_APPLICATION_CREDENTIALS env var pointing at a service
account JSON, or GOOGLE_CLOUD_API_KEY for key-based auth. When neither is
set, the module reports "no API key configured" and returns gracefully.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image


# ─────────────────────────────────────────────── known AI platforms ──

AI_GALLERY_DOMAINS = {
    "midjourney.com", "civitai.com", "lexica.art", "playground.com",
    "openart.ai", "tensor.art", "leonardo.ai", "dreamstudio.ai",
    "firefly.adobe.com", "labs.openai.com", "deepai.org",
    "stablediffusionweb.com", "nightcafe.studio",
}

STOCK_PHOTO_DOMAINS = {
    "flickr.com", "unsplash.com", "pexels.com", "shutterstock.com",
    "gettyimages.com", "istockphoto.com", "500px.com", "adobe.stock.com",
}


@dataclass
class WebMatch:
    url: str
    page_url: str | None = None
    domain: str | None = None
    score: float = 0.0


# ─────────────────────────────── search backend: Google Cloud Vision ──

def _reverse_search(image_path: Path, max_results: int = 10) -> list[WebMatch]:
    """Reverse image search via Google Cloud Vision WEB_DETECTION.

    Returns matching URLs sorted by relevance. Swap this function to
    change the search backend — the rest of M5 doesn't care how matches
    are found.
    """
    api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
    if api_key:
        return _search_with_api_key(image_path, api_key, max_results)

    # Try the google-cloud-vision library (service account auth)
    try:
        from google.cloud import vision # noqa: F401
        return _search_with_client(image_path, max_results)
    except ImportError:
        return []
    except Exception:
        return []


def _search_with_api_key(image_path: Path, api_key: str,
                         max_results: int) -> list[WebMatch]:
    """REST API call with an API key (no SDK needed)."""
    import base64

    img_bytes = Path(image_path).read_bytes()
    body = {
        "requests": [{
            "image": {"content": base64.b64encode(img_bytes).decode()},
            "features": [{"type": "WEB_DETECTION", "maxResults": max_results}],
        }]
    }
    resp = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
        json=body, timeout=30,
    )
    resp.raise_for_status()
    web = resp.json()["responses"][0].get("webDetection", {})
    return _parse_vision_response(web)


def _search_with_client(image_path: Path, max_results: int) -> list[WebMatch]:
    """google-cloud-vision SDK (service account credentials)."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    content = Path(image_path).read_bytes()
    image = vision.Image(content=content)
    response = client.web_detection(image=image)
    web = response.web_detection
    matches = []
    for match in (web.full_matching_images or [])[:max_results]:
        domain = urlparse(match.url).netloc.lstrip("www.")
        matches.append(WebMatch(url=match.url, domain=domain))
    for page in (web.pages_with_matching_images or [])[:max_results]:
        domain = urlparse(page.url).netloc.lstrip("www.")
        matches.append(WebMatch(
            url=page.url, page_url=page.url, domain=domain,
            score=page.score if hasattr(page, "score") else 0.0,
        ))
    return matches


def _parse_vision_response(web: dict) -> list[WebMatch]:
    """Parse the REST API JSON response into WebMatch objects."""
    matches = []
    for m in web.get("fullMatchingImages", []):
        domain = urlparse(m["url"]).netloc.lstrip("www.")
        matches.append(WebMatch(url=m["url"], domain=domain))
    for p in web.get("pagesWithMatchingImages", []):
        domain = urlparse(p["url"]).netloc.lstrip("www.")
        matches.append(WebMatch(
            url=p["url"], page_url=p["url"], domain=domain,
            score=p.get("score", 0.0),
        ))
    return matches


# ──────────────────────────────────────────── fetch a remote image ──

def _fetch_image(url: str, timeout: int = 15) -> Path | None:
    """Download an image URL to a temp file. Returns the path or None."""
    try:
        resp = requests.get(url, timeout=timeout, stream=True,
                            headers={"User-Agent": "ai-image-id/0.1"})
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and "octet" not in content_type:
            return None
        suffix = ".jpg"
        if "png" in content_type:
            suffix = ".png"
        elif "webp" in content_type:
            suffix = ".webp"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        for chunk in resp.iter_content(8192):
            tmp.write(chunk)
        tmp.close()
        # Verify it's a valid image
        Image.open(tmp.name).verify()
        return Path(tmp.name)
    except Exception:
        return None


# ──────────────────────────────────── domain classification helpers ──

def _classify_domain(domain: str | None) -> str:
    """Classify a domain as 'ai_gallery', 'stock_photo', or 'other'."""
    if not domain:
        return "other"
    d = domain.lower().lstrip("www.")
    if any(ai in d for ai in AI_GALLERY_DOMAINS):
        return "ai_gallery"
    if any(stock in d for stock in STOCK_PHOTO_DOMAINS):
        return "stock_photo"
    return "other"


# ─────────────────────────────────────────────────── the one function ──

def trace_provenance(
    image_path: str | Path,
    analyze_fn=None,
    detector_ckpt: str | Path | None = None,
    max_depth: int = 2,
    _depth: int = 0,
) -> dict:
    """Find the oldest web trace of an image and analyze it.

    The core loop:
      1. Reverse-search the image
      2. Classify the matching domains (AI gallery? stock photo?)
      3. Fetch the best candidate (oldest / highest-relevance match)
      4. Run the full pipeline on the fetched copy
      5. If still inconclusive and depth < max_depth, recurse on the
         fetched copy (the "retry" — maybe the upstream copy leads to
         an even earlier version with intact provenance)

    Returns a dict with the search results, the upstream analysis (if
    any), and notes documenting the chain. This dict goes straight into
    the evidence card's notes — no new schema class needed.

    Parameters
    ----------
    image_path : path to the local image to search for
    analyze_fn : the pipeline's analyze_image function (passed in to
                 avoid circular imports; defaults to ai_image_id.main.analyze_image)
    detector_ckpt : path to the detector checkpoint (forwarded to analyze_fn)
    max_depth : how many retry hops to attempt (default 2 — original search
                + one retry on the upstream copy)
    """
    image_path = Path(image_path)

    if analyze_fn is None:
        from .main import analyze_image
        analyze_fn = analyze_image

    result = {
        "searched": False,
        "matches_found": 0,
        "domains": [],
        "ai_gallery_match": False,
        "stock_photo_match": False,
        "upstream_url": None,
        "upstream_verdict": None,
        "chain": [],
        "notes": [],
    }

    # Check API availability
    has_api_key = bool(os.environ.get("GOOGLE_CLOUD_API_KEY"))
    has_credentials = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    has_vision_lib = False
    try:
        import google.cloud.vision  # noqa: F401
        has_vision_lib = True
    except ImportError:
        pass

    if not has_api_key and not (has_credentials and has_vision_lib):
        result["notes"].append(
            "M5 unavailable: set GOOGLE_CLOUD_API_KEY or "
            "GOOGLE_APPLICATION_CREDENTIALS + pip install google-cloud-vision"
        )
        return result

    # 1. Reverse search
    matches = _reverse_search(image_path)
    result["searched"] = True
    result["matches_found"] = len(matches)

    if not matches:
        result["notes"].append("reverse search returned no matches")
        return result

    # 2. Classify domains
    domains = list({m.domain for m in matches if m.domain})
    result["domains"] = domains
    domain_types = [_classify_domain(d) for d in domains]
    result["ai_gallery_match"] = "ai_gallery" in domain_types
    result["stock_photo_match"] = "stock_photo" in domain_types

    if result["ai_gallery_match"]:
        ai_domains = [d for d, t in zip(domains, domain_types) if t == "ai_gallery"]
        result["notes"].append(
            f"earliest matches include AI gallery domains: {', '.join(ai_domains)}"
        )
    if result["stock_photo_match"]:
        stock_domains = [d for d, t in zip(domains, domain_types) if t == "stock_photo"]
        result["notes"].append(
            f"matches found on stock photo platforms: {', '.join(stock_domains)}"
        )

    # 3. Try to fetch and re-analyze the best match
    if _depth >= max_depth:
        result["notes"].append(f"max search depth ({max_depth}) reached")
        return result

    for match in matches:
        fetched = _fetch_image(match.url)
        if fetched is None:
            continue

        try:
            upstream = analyze_fn(str(fetched), detector_ckpt=str(detector_ckpt)
                                  if detector_ckpt else None)
            result["upstream_url"] = match.url
            result["upstream_verdict"] = upstream.ai_verdict.value
            result["chain"].append({
                "url": match.url,
                "domain": match.domain,
                "verdict": upstream.ai_verdict.value,
                "confidence": upstream.confidence,
            })

            # If upstream has stronger evidence, we're done
            if upstream.ai_verdict.value in ("verified", "likely"):
                result["notes"].append(
                    f"upstream copy at {match.domain} → "
                    f"{upstream.ai_verdict.value} ({upstream.confidence})"
                )
                return result

            # If upstream is also inconclusive, retry one level deeper
            if upstream.ai_verdict.value == "inconclusive" and _depth + 1 < max_depth:
                deeper = trace_provenance(
                    str(fetched), analyze_fn, detector_ckpt,
                    max_depth, _depth + 1,
                )
                if deeper.get("upstream_verdict") in ("verified", "likely"):
                    result["upstream_url"] = deeper["upstream_url"]
                    result["upstream_verdict"] = deeper["upstream_verdict"]
                    result["chain"].extend(deeper["chain"])
                    result["notes"].extend(deeper["notes"])
                    return result

        except Exception as exc:
            result["notes"].append(
                f"failed to analyze {match.domain}: {type(exc).__name__}"
            )
        finally:
            try:
                Path(fetched).unlink()
            except Exception:
                pass

    if not result["upstream_verdict"]:
        result["notes"].append(
            f"found {len(matches)} web matches but none yielded "
            f"stronger evidence than the local copy"
        )

    return result
