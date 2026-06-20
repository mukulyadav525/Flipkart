"""Drawing helpers for annotated output."""

from __future__ import annotations

import cv2
import numpy as np

from .types import Track

VIOLATION_COLOR = (0, 0, 255)  # red (BGR)


# Stable-ish color per track id.
def _color(track_id: int) -> tuple[int, int, int]:
    rng = (track_id * 2654435761) & 0xFFFFFFFF
    return (rng & 255, (rng >> 8) & 255, (rng >> 16) & 255)


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    draw_labels: bool = True,
    flagged: dict[int, str] | None = None,
) -> np.ndarray:
    flagged = flagged or {}
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.xyxy)
        violation = flagged.get(t.track_id)
        color = VIOLATION_COLOR if violation else _color(t.track_id)
        thickness = 3 if violation else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if draw_labels:
            label = (f"!{violation.upper()} #{t.track_id}" if violation
                     else f"#{t.track_id} {t.class_name} {t.conf:.2f}")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                out, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
    return out


SIGNAL_BGR = {"red": (0, 0, 255), "yellow": (0, 220, 220),
              "green": (0, 200, 0), "unknown": (0, 255, 255)}


def draw_scene(frame: np.ndarray, config, signal_state: str | None = None) -> np.ndarray:
    """Overlay calibrated zones so the geometry is visible in the output."""
    if config is None:
        return frame
    out = frame

    # No-parking zones: translucent red fill + outline.
    for z in config.no_parking:
        pts = np.asarray(z.polygon, dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 200))
        out = cv2.addWeighted(overlay, 0.20, out, 0.80, 0)
        cv2.polylines(out, [pts], True, (0, 0, 200), 2)
        cv2.putText(out, f"NO PARKING: {z.name}", tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2, cv2.LINE_AA)

    # Lanes: cyan outline + an arrow showing allowed direction.
    for lane in config.lanes:
        pts = np.asarray(lane.polygon, dtype=np.int32)
        cv2.polylines(out, [pts], True, (200, 200, 0), 2)
        cx, cy = pts.mean(axis=0).astype(int)
        dx, dy = lane.direction
        tip = (int(cx + dx * 60), int(cy + dy * 60))
        cv2.arrowedLine(out, (cx, cy), tip, (200, 200, 0), 3, tipLength=0.3)
        cv2.putText(out, lane.name, (cx, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 2, cv2.LINE_AA)

    # Stop line (Phase 2) — coloured by current signal state if known.
    if config.stop_line is not None:
        p1 = tuple(map(int, config.stop_line.p1))
        p2 = tuple(map(int, config.stop_line.p2))
        color = SIGNAL_BGR.get(signal_state, (0, 255, 255))
        cv2.line(out, p1, p2, color, 3)
        cv2.putText(out, "STOP LINE", (p1[0], p1[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return out


def draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 24
    return frame
