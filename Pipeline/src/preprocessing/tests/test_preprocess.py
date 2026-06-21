"""
Sanity-check tests for the preprocessing pipeline.

Each test creates a synthetic image with a known degradation, runs the
relevant pipeline stage (or the full pipeline), and checks that:
  - output shape matches input shape
  - output dtype is uint8
  - pixel values stay in [0, 255]
  - the image is not trivially zeroed / saturated

These are correctness guards, not visual-quality metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing.preprocess import (
    apply_clahe,
    apply_derain,
    apply_sharpening,
    preprocess,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_bgr(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _assert_valid(original: np.ndarray, result: np.ndarray) -> None:
    assert result.shape == original.shape, "shape changed"
    assert result.dtype == np.uint8, f"dtype changed to {result.dtype}"
    assert result.min() >= 0 and result.max() <= 255, "pixel values out of [0, 255]"
    assert result.max() > 0, "output is all-zero (image was corrupted)"
    assert result.min() < 255, "output is fully saturated (image was corrupted)"


# ---------------------------------------------------------------------------
# Test 1 — CLAHE on a synthetically dark (low-light) image
# ---------------------------------------------------------------------------

def test_clahe_low_light():
    """CLAHE should brighten a dark image without changing shape or clipping."""
    dark = (_random_bgr() // 8).astype(np.uint8)   # crush to 0–31 range
    result = apply_clahe(dark)
    _assert_valid(dark, result)
    # Mean luminance should increase after enhancement
    mean_before = dark.mean()
    mean_after = result.mean()
    assert mean_after > mean_before, (
        f"CLAHE did not increase mean brightness: {mean_before:.1f} → {mean_after:.1f}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Sharpening on a synthetically blurred image
# ---------------------------------------------------------------------------

def test_sharpening_blurred_image():
    """Sharpening should not corrupt shape, dtype, or pixel range."""
    import cv2
    base = _random_bgr()
    blurred = cv2.GaussianBlur(base, (15, 15), sigmaX=4)
    result = apply_sharpening(blurred, strength=1.0)
    _assert_valid(blurred, result)


# ---------------------------------------------------------------------------
# Test 3 — Derain on an image with synthetic vertical streak noise
# ---------------------------------------------------------------------------

def test_derain_rain_streaks():
    """Median filter should reduce high-frequency vertical-streak noise."""
    base = _random_bgr(seed=42)
    # Add thin vertical white streaks to simulate rain
    noisy = base.copy()
    for col in range(0, base.shape[1], 10):
        noisy[:, col, :] = 255
    result = apply_derain(noisy, kernel_size=3)
    _assert_valid(noisy, result)
    # The median filter should smear away the thin white streaks: far fewer
    # fully-saturated (255) pixels remain.  This is the robust denoising
    # property (mean alone is not monotonic for a random base across versions).
    saturated_before = int((noisy == 255).sum())
    saturated_after  = int((result == 255).sum())
    assert saturated_after < saturated_before, "derain did not reduce white streaks"


# ---------------------------------------------------------------------------
# Test 4 — Full pipeline end-to-end on a combined degradation
# ---------------------------------------------------------------------------

def test_full_pipeline_combined_degradation():
    """Full pipeline on a dark, blurred, streaky image must preserve shape/range."""
    import cv2
    base = _random_bgr(seed=7)

    # Low-light
    degraded = (base // 6).astype(np.uint8)
    # Motion blur (horizontal kernel)
    kernel = np.zeros((1, 15), np.float32)
    kernel[0, :] = 1.0 / 15
    degraded = cv2.filter2D(degraded, -1, kernel)
    # Rain streaks
    for col in range(5, degraded.shape[1], 12):
        degraded[:, col, :] = np.clip(degraded[:, col, :].astype(int) + 180, 0, 255).astype(np.uint8)

    result = preprocess(degraded)
    _assert_valid(degraded, result)


# ---------------------------------------------------------------------------
# Test 5 — Edge cases: grayscale input
# ---------------------------------------------------------------------------

def test_grayscale_input():
    """Pipeline stages must handle single-channel (grayscale) images."""
    rng = np.random.default_rng(99)
    gray = rng.integers(0, 64, (240, 320), dtype=np.uint8)   # dark grayscale
    result_clahe = apply_clahe(gray)
    _assert_valid(gray, result_clahe)

    result_derain = apply_derain(gray)
    _assert_valid(gray, result_derain)
