"""Shared lightweight data types passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Track:
    """One tracked object in one frame."""

    track_id: int
    class_id: int
    class_name: str
    conf: float
    xyxy: tuple[float, float, float, float]  # absolute pixel coords

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        # Useful for ground-plane logic (parking / line crossing) in later phases.
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)
