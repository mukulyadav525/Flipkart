"""Frame preprocessing chain.

Order matters: denoise -> brighten (gamma) -> contrast (CLAHE) -> sharpen.
ROI masking (sky/buildings) is deferred to Phase 1 because it needs the
per-camera config; the hook is left here as `roi_mask`.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import PreprocessConfig


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_tile_grid, cfg.clahe_tile_grid),
        )
        # Precompute gamma LUT once.
        inv = 1.0 / max(cfg.gamma, 1e-6)
        self._gamma_lut = np.array(
            [((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8
        )

    def apply(self, frame: np.ndarray, roi_mask: np.ndarray | None = None) -> np.ndarray:
        cfg = self.cfg
        if not cfg.enabled:
            return frame

        out = frame

        if cfg.denoise:
            out = cv2.bilateralFilter(out, d=5, sigmaColor=50, sigmaSpace=50)

        if cfg.auto_gamma and self._mean_luma(out) < cfg.dark_threshold:
            out = cv2.LUT(out, self._gamma_lut)

        if cfg.clahe:
            out = self._apply_clahe(out)

        if cfg.sharpen:
            out = self._unsharp(out, cfg.sharpen_amount)

        if roi_mask is not None:
            out = cv2.bitwise_and(out, out, mask=roi_mask)

        return out

    # -- steps -------------------------------------------------------------
    @staticmethod
    def _mean_luma(frame: np.ndarray) -> float:
        return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _unsharp(frame: np.ndarray, amount: float) -> np.ndarray:
        blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=2.0)
        return cv2.addWeighted(frame, 1 + amount, blur, -amount, 0)
