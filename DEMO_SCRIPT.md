# 🎬 TrafficEye — Prototype Demo Video Script

A tight **5–6 minute** screen recording, running on **your own CCTV dataset**
(Vodra + Talaimari junctions). Everything below is verified working.

> Two Python envs are used: **`.venv-ml`** (Python 3.13 + torch/ultralytics/easyocr)
> runs the heavy pipeline; **`.venv`** runs the lightweight backend. Both already
> exist in the repo.

---

## 0. Before you record — generate data + start servers (~5 min)

```bash
# ── A) Generate REAL violations from your videos ───────────────────────────
cd Pipeline
PYTHONPATH="$PWD:$PWD/src" ../.venv-ml/bin/python scripts/run_stream.py \
    --camera configs/cameras/South.json --reset --max-frames 80 --stride 8
PYTHONPATH="$PWD:$PWD/src" ../.venv-ml/bin/python scripts/run_stream.py \
    --camera configs/cameras/Talaimari-east.json --max-frames 50 --stride 8
# → ~7 violations land in the human-review queue, each with an evidence image.

# ── B) Backend (serves the real outputs; reads Pipeline/outputs) ───────────
cd ../Backend
PIPELINE_ROOT="$(cd ../Pipeline && pwd)" ../.venv/bin/uvicorn main:app --port 8000   # terminal 2

# ── C) Frontend pointed at the live backend ────────────────────────────────
cd ../Frontend
VITE_USE_MOCK=false npm run dev                                                      # terminal 3
```

Open `http://localhost:5173`. Confirm `http://localhost:8000/health` shows
`"mock_mode": false` — that proves the portal is showing **real** data.

**Optional — populate the dashboards before recording:** log in as Operator and
**Approve** 2–3 review items. Each approval *issues the challan* — it then appears
in the Confirmed log, the Admin ledger, and analytics. Leave a few unactioned so
you can demo the live workflow. (Don't re-run the pipeline with `--reset` after
this, or you'll wipe the confirmed records.)

**Recording tips:** 1080p, hide the bookmarks bar, browser zoom ~110 %, silence
notifications, QuickTime (⌘⇧5) or OBS. Speak slowly.

---

## 1. Hook (0:00–0:30)

> "Every day, traffic cameras capture millions of frames — far too many to review
> by hand. **TrafficEye** is a computer-vision system that watches the footage,
> finds violations, reads the plate, and prepares the evidence — but a human
> officer approves every fine. Here it is on **real CCTV** from two junctions."

Show the **station-select** screen (brushed-metal panel, three roles).

---

## 2. The AI actually running (0:30–1:30) — credibility shot

Terminal, run ~15 s so detections scroll on camera:
```bash
cd Pipeline
PYTHONPATH="$PWD:$PWD/src" ../.venv-ml/bin/python scripts/run_stream.py \
    --source ../traffic-video-dataset/Vodra/South.mp4 --max-frames 12 --stride 12 --no-anpr
```
> "For every frame it runs the full pipeline — **preprocess, detect vehicles and
> riders with YOLO, track them across frames, then apply the violation rules** —
> and writes the evidence out live. Notice `confirmed` vs `review`: confident
> cases are auto-confirmed, uncertain ones go to a human. **Nothing is fined
> automatically.**"

---

## 3. Operator — the Radar Desk (1:30–3:15) ⭐ the star

Log in as **Operator**.

1. **Evidence Review console** — the CRT shows a **real frame** from the South
   junction with the AI's bounding box on the offending vehicle, the offence
   classified ("No Helmet"), the confidence meter, and the **rule trace** in
   green terminal text.
   > "This is a real frame. The box is the AI detection, it's classified as a
   > no-helmet violation at 74 % — below our auto-confirm threshold, so it's
   > waiting for a human."
2. Press the big green **APPROVE**.
   > "Approving **issues the challan** — watch it appear in the Confirmed log and
   > the audit trail." (the queue counter ticks down 7→6)
3. Press **REJECT** on a weak one.
   > "Rejecting logs it as a false positive and drops it. Every action is recorded."
4. Flip a few **rocker switches** on the violation switchboard to filter the log.
5. Point at the **flip-clock counters** (Today / In Review).

---

## 4. Admin — the Command Center (3:15–4:15)

Log in as **Admin**.

1. **Analog gauges** — Accuracy / Precision / Recall / F1 / mAP + latency & FPS.
   > "These dials show the model's measured performance and inference speed."
2. **LED server rack** — AI model / OCR / database / review queue.
   > "The AI model LED is amber because it's running general COCO weights —
   > training on the Indian Driving Dataset turns it green and lifts accuracy."
3. **Enforcement ledger** — search a plate or type; show the approvals you just made.

---

## 5. Citizen — the Glovebox (4:15–4:50)

> Note: these CCTV plates are too distant for clean OCR (the system correctly
> leaves them blank rather than guessing). So demo the Citizen portal in **mock
> mode** to show the full challan experience with plates:
> in a 4th terminal: `cd Frontend && npm run dev -- --port 5174` (mock default),
> open `:5174`, log in as Citizen, search **MH12AB1234**.

> "A citizen enters their plate and gets their challans as physical notices — the
> offence, the OCR plate, a **polaroid of the evidence**, and the AI confidence as
> a **red ink stamp** — in plain language, with the law cited and a dispute link."

---

## 6. The 8 tasks + close (4:50–6:00)

Show the architecture diagram (in `DEPLOYMENT.md`) and tick the brief:

> "Under the hood it does all eight required tasks — **preprocessing, detection,
> seven violation types, classification with confidence, plate OCR, evidence
> generation, analytics, and performance evaluation** — with **174 automated
> tests passing**, running on real video end-to-end."

Close:
> "To deploy for real, each fixed camera is calibrated once, an edge box per
> junction runs this pipeline, and the backend serves the portals. Honest next
> steps are authentication, a production database, and the RTO owner-lookup. But
> the core that matters is here and working: **the AI finds it, a human approves
> it, and only then is a challan issued.**"

---

## Talking points per violation

| Violation | One line |
|---|---|
| Helmet | bike rider, head visible, no helmet over the head region |
| Seatbelt | car occupant, no belt across the torso keypoints |
| Triple riding | >2 people overlapping one two-wheeler |
| Wrong side | motion opposes the lane direction, **inside the monitored lane** |
| Stop line | ground point crossed the line when not green |
| Red light | crossed the line while the signal is red |
| Illegal parking | **stationary** vehicle (low motion) inside the no-parking polygon |

## Honest accuracy notes (good to say out loud — shows engineering maturity)
- Running **general COCO weights**, not fine-tuned — that's why confidences sit
  in the 60–80 % range and most cases route to human review. Fine-tuned IDD
  weights raise this.
- Geometry rules depend on **per-camera calibration** (`configs/cameras/*.json`).
  South is calibrated (lane + no-parking zone); Talaimari isn't, so only the
  appearance rules run there.
- **Plates** from distant CCTV are often unreadable; the digit-gate rejects bus
  signage rather than inventing a plate. Closer cameras read plates fine.

## If something breaks on the day
- `mock_mode: true`? → the pipeline hasn't written `Pipeline/outputs`, or the
  Frontend wasn't started with `VITE_USE_MOCK=false`. Redo step 0.
- Laptop too slow / hot? → `--max-frames 30 --stride 12`.
- Confirmed log empty? → approve a few review items (that's the whole point).
