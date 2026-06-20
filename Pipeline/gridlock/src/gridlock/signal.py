"""Traffic-signal state providers for Phase 2.

Two implementations, same interface, so the stop-line / red-light engines don't
care where the signal state comes from:

  ScheduledSignal  — a fixed repeating cycle defined in the camera config. Use
                     when the camera doesn't see a signal head (e.g. the AI City
                     test intersection) or for deterministic demos.
  ROISignal        — classifies the dominant colour inside a calibrated light
                     ROI each frame. Use when a signal is visible in frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

RED, YELLOW, GREEN, UNKNOWN = "red", "yellow", "green", "unknown"


class SignalProvider(Protocol):
    def state_at(self, timestamp: float, frame) -> str:
        ...


@dataclass
class ScheduledSignal:
    """Repeating cycle, e.g. [("green", 8), ("yellow", 2), ("red", 8)] seconds."""

    cycle: list[tuple[str, float]]
    offset: float = 0.0

    @property
    def period(self) -> float:
        return sum(d for _, d in self.cycle) or 1.0

    def state_at(self, timestamp: float, frame=None) -> str:
        t = (timestamp + self.offset) % self.period
        acc = 0.0
        for state, dur in self.cycle:
            acc += dur
            if t < acc:
                return state
        return self.cycle[-1][0]


@dataclass
class ROISignal:
    """Classify the dominant lamp colour inside roi = (x, y, w, h)."""

    roi: tuple[int, int, int, int]
    min_pixels: int = 30

    # HSV ranges (OpenCV H: 0-179).
    def state_at(self, timestamp: float, frame) -> str:
        if frame is None:
            return UNKNOWN
        x, y, w, h = self.roi
        crop = frame[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            return UNKNOWN
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red = (cv2.inRange(hsv, (0, 90, 90), (10, 255, 255))
               | cv2.inRange(hsv, (170, 90, 90), (179, 255, 255)))
        yellow = cv2.inRange(hsv, (15, 90, 90), (35, 255, 255))
        green = cv2.inRange(hsv, (40, 60, 60), (90, 255, 255))
        counts = {RED: int(red.sum() // 255),
                  YELLOW: int(yellow.sum() // 255),
                  GREEN: int(green.sum() // 255)}
        best = max(counts, key=counts.get)
        return best if counts[best] >= self.min_pixels else UNKNOWN


def build_signal(camera) -> SignalProvider | None:
    """Pick a provider from a CameraConfig: scheduled cycle > light ROI > demo."""
    cycle = getattr(camera, "signal_cycle", None)
    if cycle:
        return ScheduledSignal([(s, float(d)) for s, d in cycle])
    if getattr(camera, "light_roi", None):
        return ROISignal(tuple(camera.light_roi))
    return None
