#!/usr/bin/env python3
"""
Run the live violation pipeline over a video stream and publish to the portal.

Two ways to point it at a stream
--------------------------------
1. A calibrated camera config (recommended — enables the geometry rules):

       python scripts/run_stream.py --camera configs/cameras/demo.json --reset

2. A raw source with no calibration (helmet / seatbelt / triple-riding / ANPR
   still run; geometry rules stay off until you add a config):

       python scripts/run_stream.py --source path/to/clip.mp4 --reset
       python scripts/run_stream.py --source rtsp://user:pass@host/stream
       python scripts/run_stream.py --source 0            # local webcam

After it starts writing to Pipeline/outputs/, launch the Backend (uvicorn) and
Frontend (vite); the portal leaves mock mode automatically and updates live.

Requires:  pip install -r requirements.txt   (numpy, opencv-python, ultralytics, easyocr)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stream.scene import CameraConfig, load_camera_config, scene_from_dict  # noqa: E402
from stream.runner import StreamProcessor  # noqa: E402
from shared.schemas import SceneContext  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream a video into the TrafficEye portal.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera", help="Camera config JSON (configs/cameras/<name>.json)")
    g.add_argument("--source", help="Raw video source: file path, rtsp:// url, or webcam index")

    ap.add_argument("--output", default=str(PIPELINE_ROOT / "outputs"),
                    help="Portal outputs directory (default: Pipeline/outputs)")
    ap.add_argument("--stride", type=int, default=2, help="Process every Nth frame (default 2)")
    ap.add_argument("--max-frames", type=int, default=None, help="Stop after N processed frames")
    ap.add_argument("--reset", action="store_true", help="Clear previous outputs before starting")
    ap.add_argument("--no-pose", action="store_true", help="Disable pose (faster; no helmet/seatbelt)")
    ap.add_argument("--no-anpr", action="store_true", help="Disable plate OCR")
    ap.add_argument("--device", default="cpu", help="cpu / cuda / mps (default cpu)")
    ap.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    ap.add_argument("--weights", default=None, help="Override detector weights path")
    ap.add_argument("--helmet-weights", default=None,
                    help="Trained helmet model (helmet/no_helmet heads). "
                         "Auto-found under weights/ or gridlock/runs_helmet* if omitted.")
    ap.add_argument("--no-helmet-model", action="store_true",
                    help="Disable the helmet model (no helmet violations will fire)")
    args = ap.parse_args()

    if args.camera:
        camera = load_camera_config(args.camera)
        if args.weights:
            camera.detector_weights = args.weights
        if args.stride != 2:
            camera.stride = args.stride
    else:
        camera = CameraConfig(
            name="adhoc",
            source=args.source,
            scene=scene_from_dict({}, image_id="adhoc"),  # empty SceneContext
            stride=args.stride,
            detector_weights=args.weights,
        )
        assert isinstance(camera.scene, SceneContext)

    proc = StreamProcessor(
        camera, args.output,
        run_pose=not args.no_pose,
        run_anpr=not args.no_anpr,
        device=args.device,
        conf_threshold=args.conf,
        helmet_weights=args.helmet_weights,
        use_helmet_model=not args.no_helmet_model,
        reset=args.reset,
    )
    if not proc.use_helmet_model and not args.no_helmet_model:
        print("[stream] WARNING: no trained helmet model found — helmet violations "
              "are OFF. Pass --helmet-weights or drop one in Pipeline/weights/helmet.pt")
    proc.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
