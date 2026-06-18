"""
Image preprocessing pipeline for traffic violation detection.

Pipeline stages (applied in order):
  1. CLAHE contrast enhancement  — compensates for low-light / overexposed frames
  2. Sharpening kernel            — reduces mild motion blur without a heavy model
  3. Median-filter derain         — suppresses rain streaks (fast, no neural net)

Each stage is a pure function (ndarray -> ndarray) so stages can be tested and
composed independently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Stage 1 — CLAHE contrast enhancement
# ---------------------------------------------------------------------------

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE to the L channel of a BGR image.

    Works in LAB colour space so hue / saturation are untouched.
    Grayscale images are enhanced directly.
    """
    if image.ndim == 2:  # grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Stage 2 — Sharpening / deblur
# ---------------------------------------------------------------------------

# 3×3 unsharp-mask kernel — boosts high frequencies to counteract mild blur.
# Stronger than a simple Laplacian add; avoids the complexity of Wiener filter
# (which requires an estimated PSF that we don't have at runtime).
_SHARPEN_KERNEL = np.array(
    [[ 0, -1,  0],
     [-1,  5, -1],
     [ 0, -1,  0]],
    dtype=np.float32,
)


def apply_sharpening(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply an unsharp-mask sharpening kernel.

    `strength` blends between original (0.0) and full sharpening (1.0).
    Values > 1.0 over-sharpen and are clamped to 1.0.
    """
    strength = min(max(strength, 0.0), 1.0)
    sharpened = cv2.filter2D(image, ddepth=-1, kernel=_SHARPEN_KERNEL)
    if strength < 1.0:
        sharpened = cv2.addWeighted(image, 1.0 - strength, sharpened, strength, 0)
    return sharpened


# ---------------------------------------------------------------------------
# Stage 3 — Rain-streak reduction
# ---------------------------------------------------------------------------

def apply_derain(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Suppress rain streaks with a per-channel median filter.

    Rain streaks are high-frequency vertical patterns; median filtering removes
    them without blurring edges as much as a Gaussian.  kernel_size=3 keeps the
    operation fast (O(N) per pixel with OpenCV's implementation).
    """
    if image.ndim == 2:
        return cv2.medianBlur(image, kernel_size)
    channels = cv2.split(image)
    derained = [cv2.medianBlur(c, kernel_size) for c in channels]
    return cv2.merge(derained)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(
    image: np.ndarray,
    *,
    clahe_clip: float = 2.0,
    clahe_tile: tuple[int, int] = (8, 8),
    sharpen_strength: float = 0.7,
    derain_kernel: int = 3,
) -> np.ndarray:
    """
    Run the full three-stage preprocessing pipeline.

    Parameters
    ----------
    image:
        BGR (or grayscale) uint8 ndarray, same dimensions returned.
    clahe_clip:
        CLAHE clip limit; higher values → stronger local contrast boost.
    clahe_tile:
        CLAHE tile grid size.
    sharpen_strength:
        Blend weight for sharpening (0 = off, 1 = full).
    derain_kernel:
        Median filter kernel size for rain-streak removal (must be odd).

    Returns
    -------
    np.ndarray
        Processed image, same shape and dtype as input.
    """
    if image is None or image.size == 0:
        raise ValueError("preprocess() received an empty image array")
    if image.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {image.dtype}")

    out = apply_clahe(image, clip_limit=clahe_clip, tile_grid_size=clahe_tile)
    out = apply_sharpening(out, strength=sharpen_strength)
    out = apply_derain(out, kernel_size=derain_kernel)
    return out


def preprocess_file(input_path: str | Path, output_path: str | Path | None = None) -> np.ndarray:
    """
    Load an image from disk, preprocess it, optionally save the result.

    Returns the processed ndarray.  If output_path is None the image is only
    returned, not written to disk.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"cv2.imread could not decode: {input_path}")

    result = preprocess(image)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), result)
        if not ok:
            raise IOError(f"cv2.imwrite failed for: {output_path}")

    return result


# ---------------------------------------------------------------------------
# CLI entry point  — python -m preprocessing.preprocess --input X --output Y
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocessing.preprocess",
        description="Run the traffic-image preprocessing pipeline on a single file.",
    )
    p.add_argument("--input", required=True, metavar="PATH", help="Input image path")
    p.add_argument("--output", required=True, metavar="PATH", help="Output image path")
    p.add_argument("--clahe-clip", type=float, default=2.0, metavar="N",
                   help="CLAHE clip limit (default 2.0)")
    p.add_argument("--clahe-tile", type=int, default=8, metavar="N",
                   help="CLAHE tile grid size NxN (default 8)")
    p.add_argument("--sharpen", type=float, default=0.7, metavar="W",
                   help="Sharpening strength 0–1 (default 0.7)")
    p.add_argument("--derain-kernel", type=int, default=3, metavar="K",
                   help="Median filter kernel size for derain (default 3, must be odd)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.derain_kernel % 2 == 0:
        print(f"Error: --derain-kernel must be odd, got {args.derain_kernel}", file=sys.stderr)
        sys.exit(1)

    result = preprocess_file(args.input, args.output)
    h, w = result.shape[:2]
    print(f"Saved {w}x{h} image to {args.output}")


if __name__ == "__main__":
    main()
