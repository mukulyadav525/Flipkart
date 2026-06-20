#!/usr/bin/env python3
"""Interactive per-camera calibration.

Opens the first frame of a video and lets you draw the scene geometry with the
mouse, then saves it to configs/cameras/<name>.json.

Controls
--------
  Left click      add a point to the current shape
  p               start a NO-PARKING polygon  (click corners, ENTER to close)
  l               start a LANE: click polygon corners, ENTER, then click
                  TWO points tail->head to set the allowed direction
  s               STOP LINE: click two points (Phase 2)
  ENTER           finish the current shape
  u               undo last finished shape
  w               write config to disk
  q / ESC         quit

This needs a desktop (cv2.imshow). Run it locally, not over SSH.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridlock.geometry import unit  # noqa: E402
from gridlock.scene import CameraConfig, Lane, NoParkingZone, StopLine  # noqa: E402


def first_frame(video: str):
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"could not read frame from {video}")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--name", default=None, help="camera name (default: file stem)")
    args = ap.parse_args()

    name = args.name or Path(args.video).stem
    frame = first_frame(args.video)
    h, w = frame.shape[:2]
    cfg = CameraConfig(name=name, frame_size=(w, h))

    state = {"mode": None, "pts": [], "lane_poly": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and state["mode"]:
            state["pts"].append((float(x), float(y)))

    win = f"calibrate: {name}  [p]ark [l]ane [s]topline ENTER=done u=undo w=write q=quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    def render():
        from gridlock import visualize
        img = visualize.draw_scene(frame.copy(), cfg)
        for p in state["pts"]:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
        if len(state["pts"]) > 1:
            for a, b in zip(state["pts"], state["pts"][1:]):
                cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 0), 1)
        msg = f"mode={state['mode']}  points={len(state['pts'])}"
        cv2.putText(img, msg, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return img

    def finish():
        m, pts = state["mode"], state["pts"]
        if m == "park" and len(pts) >= 3:
            cfg.no_parking.append(NoParkingZone(name=f"zone{len(cfg.no_parking)+1}", polygon=pts.copy()))
        elif m == "lane":
            if state["lane_poly"] is None and len(pts) >= 3:
                state["lane_poly"] = pts.copy()
                state["pts"] = []
                print("lane polygon set — now click TWO points tail->head for direction")
                return  # stay in lane mode to capture the arrow
            elif state["lane_poly"] is not None and len(pts) >= 2:
                (x0, y0), (x1, y1) = pts[0], pts[1]
                cfg.lanes.append(Lane(
                    name=f"lane{len(cfg.lanes)+1}",
                    polygon=state["lane_poly"],
                    direction=unit((x1 - x0, y1 - y0)),
                ))
                state["lane_poly"] = None
        elif m == "stop" and len(pts) >= 2:
            cfg.stop_line = StopLine(pts[0], pts[1])
        state["mode"], state["pts"] = None, []

    while True:
        cv2.imshow(win, render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("p"):
            state.update(mode="park", pts=[], lane_poly=None)
        elif key == ord("l"):
            state.update(mode="lane", pts=[], lane_poly=None)
        elif key == ord("s"):
            state.update(mode="stop", pts=[], lane_poly=None)
        elif key in (13, 10):  # ENTER
            finish()
        elif key == ord("u"):
            if cfg.lanes:
                cfg.lanes.pop()
            elif cfg.no_parking:
                cfg.no_parking.pop()
        elif key == ord("w"):
            out = ROOT / "configs" / "cameras" / f"{name}.json"
            cfg.save(out)
            print(f"saved -> {out}")

    cv2.destroyAllWindows()
    out = ROOT / "configs" / "cameras" / f"{name}.json"
    cfg.save(out)
    print(f"saved -> {out}  ({len(cfg.no_parking)} zones, {len(cfg.lanes)} lanes)")


if __name__ == "__main__":
    main()
