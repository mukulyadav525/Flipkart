"""Evidence packaging: per-violation 5-second clips, snapshots, plate, ledger.

Runs as a post-step on a violations JSONL: for each event it cuts a short clip
around the moment, grabs the offending vehicle crop, optionally reads the plate,
and writes a CSV + JSON ledger.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2


def grab_frame(video: str, frame_idx: int):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def crop_xyxy(frame, xyxy, pad: float = 0.12):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * pad)); y1 = int(max(0, y1 - bh * pad))
    x2 = int(min(w, x2 + bw * pad)); y2 = int(min(h, y2 + bh * pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def cut_clip(video: str, t_center: float, window: float, out_path: str) -> bool:
    """Cut a `window`-second clip centred on t_center using ffmpeg."""
    start = max(0.0, t_center - window / 2.0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", video, "-t", f"{window:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-an", out_path,
    ]
    return subprocess.run(cmd).returncode == 0


def build_evidence(
    events: list[dict],
    source_video: str,
    clip_video: str,
    out_dir: str,
    window: float = 5.0,
    plate_reader=None,
) -> list[dict]:
    """Produce clip + snapshot + plate per event; return enriched records."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i, ev in enumerate(events):
        tag = f"{ev['type']}_track{ev['track_id']}_{ev['timestamp']:.1f}s".replace(".", "_")
        clip_path = str(out / f"{tag}.mp4")
        snap_path = str(out / f"{tag}.jpg")

        cut_clip(clip_video, ev["timestamp"], window, clip_path)

        plate = None
        frame = grab_frame(source_video, ev["frame_idx"])
        if frame is not None:
            veh = crop_xyxy(frame, ev["xyxy"])
            if veh is not None:
                cv2.imwrite(snap_path, veh)
                if plate_reader is not None and plate_reader.available:
                    res = plate_reader.read_plate(veh)
                    if res:
                        plate = {"text": res[0], "conf": round(res[1], 2)}

        records.append({
            "type": ev["type"],
            "track_id": ev["track_id"],
            "timestamp": ev["timestamp"],
            "detail": ev.get("detail", ""),
            "plate": plate["text"] if plate else "",
            "plate_conf": plate["conf"] if plate else "",
            "clip": clip_path,
            "snapshot": snap_path,
        })
    (out / "ledger.json").write_text(json.dumps(records, indent=2))
    _write_csv(records, out / "ledger.csv")
    return records


def _write_csv(records: list[dict], path: Path):
    import csv
    cols = ["type", "track_id", "timestamp", "plate", "plate_conf", "detail", "clip", "snapshot"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in cols})
