"""Pure geometry helpers shared by the violation engines.

All coordinates are absolute pixels in the source frame."""

from __future__ import annotations

import math

import cv2
import numpy as np

Point = tuple[float, float]


def point_in_polygon(pt: Point, polygon: list[Point]) -> bool:
    if len(polygon) < 3:
        return False
    poly = np.asarray(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def unit(vec: Point) -> Point:
    x, y = vec
    n = math.hypot(x, y)
    if n < 1e-9:
        return (0.0, 0.0)
    return (x / n, y / n)


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cos_angle(a: Point, b: Point) -> float:
    """Cosine of the angle between two vectors; 0 if either is ~zero."""
    ua, ub = unit(a), unit(b)
    if ua == (0.0, 0.0) or ub == (0.0, 0.0):
        return 0.0
    return max(-1.0, min(1.0, dot(ua, ub)))


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True if segment p1p2 crosses segment p3p4 (used for stop-line in Phase 2)."""

    def ccw(a: Point, b: Point, c: Point) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
