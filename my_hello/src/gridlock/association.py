"""Associate person detections with two-wheelers.

Shared by triple-riding (count riders) and helmet (check each rider's head).
A person is considered "on" a two-wheeler when their box overlaps the bike and
their horizontal center sits within the (slightly widened) bike span — riders
straddle the bike so their lower body overlaps it.
"""

from __future__ import annotations

from .types import Track

TWO_WHEELERS = {"motorcycle", "bicycle"}


def tag_three_wheelers(tracks: list[Track], min_wh_ratio: float = 1.0) -> list[Track]:
    """Relabel wide `motorcycle` boxes as `auto_rickshaw` in place.

    COCO YOLO has no auto-rickshaw class, so 3-wheelers (autos/tuk-tuks) come
    back as `motorcycle`. They have wide, bulky boxes (w/h >= ~1.0) whereas a
    real two-wheeler — especially one carrying stacked riders — is tall (w/h
    well below 1.0). Reclassifying fixes both the displayed label and the
    two-wheeler violation logic (triple-riding / helmet), since `auto_rickshaw`
    is not in TWO_WHEELERS."""
    for t in tracks:
        if t.class_name != "motorcycle":
            continue
        x1, y1, x2, y2 = t.xyxy
        h = y2 - y1
        if h > 0 and (x2 - x1) / h >= min_wh_ratio:
            t.class_name = "auto_rickshaw"
    return tracks


def _overlap_ratio(person: tuple, bike: tuple) -> float:
    """Intersection area / person area."""
    px1, py1, px2, py2 = person
    bx1, by1, bx2, by2 = bike
    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    parea = max(1e-6, (px2 - px1) * (py2 - py1))
    return inter / parea


def riders_per_bike(
    tracks: list[Track],
    min_overlap: float = 0.08,
    x_pad: float = 0.35,
) -> dict[int, list[Track]]:
    """Map each two-wheeler track_id -> list of associated person Tracks."""
    bikes = [t for t in tracks if t.class_name in TWO_WHEELERS]
    people = [t for t in tracks if t.class_name == "person"]
    out: dict[int, list[Track]] = {b.track_id: [] for b in bikes}

    for person in people:
        best_bike, best_score = None, 0.0
        pcx = person.center[0]
        for bike in bikes:
            bx1, _, bx2, _ = bike.xyxy
            pad = (bx2 - bx1) * x_pad
            if not (bx1 - pad <= pcx <= bx2 + pad):
                continue
            score = _overlap_ratio(person.xyxy, bike.xyxy)
            if score >= min_overlap and score > best_score:
                best_bike, best_score = bike, score
        if best_bike is not None:
            out[best_bike.track_id].append(person)
    return out
