"""Minimal blind watermark codec in the style of `invisible-watermark`'s dwtDct method
(the scheme historically used by Stable Diffusion pipelines).

Vendored to avoid the heavy torch dependency of the `imwatermark` package. The pipeline
prefers the real `imwatermark` decoder when it is importable (see __init__.py), which
guarantees payload compatibility with SD outputs; this vendored codec is used as a
fallback and for round-trip testing/demo.

Embedding: for each 4x4 block of the level-1 Haar DWT approximation of the U channel,
quantize the max-magnitude AC coefficient to encode one payload bit (scale = 36).
"""
from __future__ import annotations

import cv2
import numpy as np
import pywt

BLOCK = 4
SCALE = 36.0


def _bits_from_bytes(payload: bytes) -> list[int]:
    return [(byte >> (7 - i)) & 1 for byte in payload for i in range(8)]


def _bytes_from_bits(bits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        b = 0
        for bit in bits[i : i + 8]:
            b = (b << 1) | bit
        out.append(b)
    return bytes(out)


def _iter_blocks(frame: np.ndarray):
    rows, cols = frame.shape
    for i in range(rows // BLOCK):
        for j in range(cols // BLOCK):
            yield i * BLOCK, j * BLOCK


def _max_ac_pos(block: np.ndarray) -> tuple[int, int]:
    pos = int(np.argmax(np.abs(block.flatten()[1:]))) + 1
    return pos // BLOCK, pos % BLOCK


def embed(rgb: np.ndarray, payload_bits: list[int]) -> np.ndarray:
    """Embed payload bits (repeated cyclically) into an RGB uint8 image."""
    wm_len = len(payload_bits)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    r4, c4 = (yuv.shape[0] // 4) * 4, (yuv.shape[1] // 4) * 4

    channel = yuv[:r4, :c4, 1]  # U channel
    ca, coeffs = pywt.dwt2(channel, "haar")

    for n, (i, j) in enumerate(_iter_blocks(ca)):
        block = ca[i : i + BLOCK, j : j + BLOCK]
        bi, bj = _max_ac_pos(block)
        val = block[bi, bj]
        bit = payload_bits[n % wm_len]
        mag = abs(val)
        new_mag = (np.floor(mag / SCALE) + 0.25 + 0.5 * bit) * SCALE
        block[bi, bj] = new_mag if val >= 0 else -new_mag

    yuv[:r4, :c4, 1] = pywt.idwt2((ca, coeffs), "haar")[:r4, :c4]
    bgr_out = np.clip(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR), 0, 255).astype(np.uint8)
    return cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)


def decode_bits(rgb: np.ndarray, wm_len: int) -> list[int]:
    """Majority-vote decode of wm_len bits from an RGB uint8 image."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    r4, c4 = (yuv.shape[0] // 4) * 4, (yuv.shape[1] // 4) * 4
    ca, _ = pywt.dwt2(yuv[:r4, :c4, 1], "haar")

    votes: list[list[int]] = [[] for _ in range(wm_len)]
    for n, (i, j) in enumerate(_iter_blocks(ca)):
        block = ca[i : i + BLOCK, j : j + BLOCK]
        bi, bj = _max_ac_pos(block)
        mag = abs(float(block[bi, bj]))
        votes[n % wm_len].append(1 if (mag % SCALE) > 0.5 * SCALE else 0)

    return [int(np.mean(v) >= 0.5) if v else 0 for v in votes]
