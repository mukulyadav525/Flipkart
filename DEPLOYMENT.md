# TrafficEye — Deployment & Operations Guide

End-to-end: **video stream → AI pipeline → portal**. This guide covers running
it on a video file/RTSP today, calibrating real cameras, and the production
topology.

```
 ┌──────────┐   RTSP / file / webcam   ┌─────────────────────────────┐
 │  Camera  │ ───────────────────────► │  Pipeline worker            │
 └──────────┘                          │  scripts/run_stream.py      │
                                       │  preprocess→detect→track→   │
                                       │  ANPR→rules→evidence        │
                                       └──────────────┬──────────────┘
                                       writes Pipeline/outputs/ (JSONL + JPEG)
                                                      │
                                       ┌──────────────▼──────────────┐
                                       │  Backend (FastAPI)          │  reads outputs/
                                       │  Backend/main.py            │  serves /api + /images
                                       └──────────────┬──────────────┘
                                                      │
                                       ┌──────────────▼──────────────┐
                                       │  Frontend (3 skeuo portals) │  Citizen / Police / Admin
                                       └─────────────────────────────┘
```

---

## 1. Run it now with a video stream (verified)

### 1a. Install pipeline dependencies
```bash
cd Pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, opencv-python, ultralytics, easyocr
```
> First run auto-downloads COCO YOLO weights. Drop fine-tuned IDD weights into
> `Pipeline/weights/yolov8_idd.pt` to use them instead of the COCO fallback.

### 1b. Point it at your stream
```bash
# A) calibrated camera (enables ALL rules incl. red-light / stop-line / parking)
python scripts/run_stream.py --camera configs/cameras/demo.json --reset
#    (edit configs/cameras/demo.json: set "source" to your file/rtsp + redraw geometry)

# B) raw source, no calibration (helmet / seatbelt / triple-riding / ANPR only)
python scripts/run_stream.py --source path/to/clip.mp4 --reset
python scripts/run_stream.py --source rtsp://user:pass@host:554/stream
python scripts/run_stream.py --source 0          # local webcam
```
Useful flags: `--stride N` (process every Nth frame), `--max-frames N`,
`--device cuda|mps`, `--no-anpr`, `--no-pose`, `--conf 0.4`.

It writes **live** to `Pipeline/outputs/`:
`violation_records/confirmed.jsonl`, `human_review_queue.jsonl`,
`annotated_images/*.jpg`, `latency_log.jsonl`, `reports/*.csv`.

### 1c. Start backend + frontend
```bash
# Backend — auto-detects real outputs and leaves mock mode
cd Backend && pip install -r requirements.txt && uvicorn main:app --port 8000

# Frontend — point it at the live backend (otherwise it serves mock data)
cd Frontend && npm install && VITE_USE_MOCK=false npm run dev
```
Open the app → pick **Operator** to review queued evidence and approve/reject,
**Citizen** to look up a plate, **Admin** for gauges + ledger + server status.

---

## 2. Calibrate a real (fixed) camera — one-time

Each camera is calibrated once into `Pipeline/configs/cameras/<name>.json`
(see [`demo.json`](Pipeline/configs/cameras/demo.json)). On a still frame from
that camera, mark in pixel coordinates:

| Field | Enables | How to set |
|---|---|---|
| `stop_line_coords` | stop-line, red-light | the two ends of the painted stop line |
| `lane_direction_vector` | wrong-side + crossing dir | unit vector of legal travel |
| `no_parking_zone_polygon` | illegal parking | outline of the no-parking area |
| `signal_state` | red-light | live from the signal controller (fixed value for demos) |
| `source` | — | the camera's RTSP url / file |

Rules whose required geometry is absent simply don't fire (by design) — no false
positives from missing calibration.

---

## 3. Production topology

- **Edge box per junction** (Jetson / small GPU): runs `run_stream.py` locally so
  only small JSON + JPEGs cross the network — scales to many cameras.
- **Central option**: one GPU server runs one `run_stream.py` worker per camera
  (RTSP in). Simpler, more bandwidth.
- **Backend + DB + portals** in the datacenter/cloud.

### Containers (scaffolding in `deploy/`)
```bash
cd deploy
docker compose up --build        # backend (uvicorn) + frontend (nginx)
```
- `Dockerfile.backend` — FastAPI on uvicorn, mounts `Pipeline/outputs` (read-only).
- `Dockerfile.frontend` — Vite build served by nginx, proxies `/api` + `/images`
  to the backend ([`nginx.conf`](deploy/nginx.conf)).
- The **pipeline worker** runs on the edge/GPU host (`run_stream.py`) and writes
  into the shared `Pipeline/outputs` volume. A GPU `Dockerfile.worker` is sketched
  in `deploy/` but the ML image is large — run on the host for development.

---

## 4. Prototype → hardened production (honest gap list)

Already real in this repo: full 8-task pipeline, tracking, de-duplication,
confidence-gated human review, immutable audit log, live analytics, 166 tests.

To harden for a real rollout, add:
- **Auth** — officer SSO + citizen plate/Aadhaar verification (none today).
- **Database** — replace JSONL sinks with Postgres; add a queue (Kafka/Redis)
  between the worker and backend; object storage (S3) for evidence images.
- **Owner lookup** — integrate the RTO/VAHAN database to resolve plate → owner.
- **Live signal feed** — wire `signal_state` to the junction controller.
- **Fine-tuned weights** — train YOLO on IDD; calibrate every deployed camera.
- **Privacy/compliance** — DPDP retention policy, encrypt plates at rest, access
  logging (the audit log is the start).
- **Ops** — process supervisor (systemd/k8s), health checks, metrics/alerting,
  automatic reconnect on RTSP drop.

The non-negotiable design property already in place: the AI assigns a confidence
and **a human approves before any citizen is fined** (confidence gate +
review queue + audit log).
