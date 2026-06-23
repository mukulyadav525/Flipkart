# Gridlock — Traffic Violation Detection

Detects traffic violations from CCTV video for Indian/South-Asian conditions:
helmet non-compliance, seatbelt non-compliance, triple riding, wrong-side
driving, stop-line violation, red-light violation, illegal parking.

## Architecture

```
Video -> Preprocess -> YOLO detect -> ByteTrack (stable IDs)
                                            |
   Per-camera config (Phase 1) ------------+
                                            v
                       Violation rule + perception engines
                                            v
                       ANPR (plate OCR) -> evidence log
```

Static cameras => geometry is calibrated once per camera and reused.

## Phase status

- [x] **Phase 0** — preprocessing + detection + tracking + annotated output
- [x] **Phase 1** — per-camera calibration tool; wrong-side & illegal parking
- [x] **Phase 2** — signal state (scheduled cycle or ROI colour); stop-line & red-light
- [x] **Phase 3** — triple-riding (no extra model); helmet & seatbelt (pluggable weights)
- [ ] Phase 4 — ANPR + evidence logging (SQLite)

Note: the bundled sample clips are moving eye-level footage, so geometry
violations (Phase 1/2) need fixed-CCTV clips to be reliable. Perception
violations (Phase 3) work on the existing footage.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run everything (all phases, one pass)

```bash
# perception + (geometry if a matching configs/cameras/<stem>.json exists)
python scripts/run_all.py archive/Vodra/South.mp4

# force a camera config + add helmet/seatbelt models
python scripts/run_all.py archive/Vodra/North.mp4 \
    --config configs/cameras/North.json \
    --helmet-weights models/helmet.pt --seatbelt-weights models/seatbelt.pt
```

`run_all.py` runs detection+tracking once and feeds every engine, then writes:
- `outputs/<stem>_all.mp4` — annotated video
- `outputs/<stem>_all_violations.jsonl` — one JSON per violation
- `outputs/<stem>_all_summary.json` — counts per violation type + metadata

Geometry violations (parking/wrong-side) activate only with a camera config;
helmet/seatbelt only with weights; stop-line/red-light need a `stop_line` plus a
signal source in the config (a `signal_cycle` schedule, or a `light_roi` if a
signal is visible). The individual `run_phase0/1/3.py` scripts remain for
isolated testing.

Signal config example (no visible signal head — use a scheduled cycle):

```json
"stop_line": { "p1": [1400, 270], "p2": [1400, 400] },
"signal_cycle": [["green", 10], ["red", 30], ["green", 30], ["red", 20]]
```

## Run (Phase 0)

```bash
# annotate a clip (writes outputs/North_phase0.mp4)
python scripts/run_phase0.py archive/Vodra/North.mp4

# quick smoke test on 150 frames
python scripts/run_phase0.py archive/Talaimari/east.mp4 --max-frames 150

# compare preprocessing on/off, or use a bigger model
python scripts/run_phase0.py archive/Vodra/North.mp4 --no-preprocess
python scripts/run_phase0.py archive/Vodra/North.mp4 --model yolo11s.pt
```

## Run (Phase 3 — perception)

```bash
# triple riding works today, no extra model:
python scripts/run_phase3.py archive/Vodra/North.mp4 --model yolo11s.pt

# add helmet / seatbelt once you have weights:
python scripts/run_phase3.py archive/Vodra/North.mp4 \
    --helmet-weights models/helmet.pt --seatbelt-weights models/seatbelt.pt
```

### Getting helmet / seatbelt weights (Roboflow recipe)

Helmet and seatbelt aren't COCO classes, so they need their own YOLO. Easiest:
1. On **Roboflow Universe**, find a "motorcycle helmet detection" (or "seatbelt
   detection") project, India-context preferred.
2. Train/download a YOLOv8/v11 model and export the `.pt` into `models/`.
3. Pass it via `--helmet-weights` / `--seatbelt-weights`.

The engines auto-map common class names (`helmet`/`no-helmet`,
`With Helmet`/`Without Helmet`, `helmet`/`head`); adjust `helmet_names` /
`nohelmet_names` in `violations/helmet.py` if your model uses other labels.
Without weights the engines disable themselves cleanly.

## Layout

```
src/gridlock/
  config.py         dataclass configs (pipeline + preprocessing)
  preprocessing.py  CLAHE / gamma / sharpen chain
  detection.py      YOLO + ByteTrack wrapper -> Track objects
  pipeline.py       read -> preprocess -> detect+track -> annotate
  visualize.py      box / HUD drawing
  types.py          Track dataclass
scripts/run_phase0.py   CLI entry point
configs/cameras/        per-camera calibration (Phase 1)
```

## Notes

- Source videos are 1280x720 @ 30fps, static cameras (`archive/`, gitignored).
- Device auto-selects MPS on Apple Silicon, else CPU.
- The 3 perception violations will use pretrained Roboflow models, not training
  from scratch — the raw footage has no labels.
