# Traffic Violation Detection System

## Project layout

```
/data
  /raw_images          # source frames (jpg/png)
  /scene_context       # per-image JSON sidecars (see below)
  /ground_truth        # manually labelled annotations for evaluation
/src
  /preprocessing       # frame extraction, resize, normalisation
  /detection           # object detection → DetectionRecord
  /plate_ocr           # licence-plate crop + OCR → PlateRecord
  /violations          # rule engine → ViolationRecord
  /evidence            # annotated image + packaging → EvidenceRecord
  /analytics           # aggregate stats, dashboards
  /evaluation          # precision/recall against ground_truth
/shared
  schemas.py           # ONLY data interchange types — import from here
/outputs
  /annotated_images    # frames with bounding boxes drawn
  /violation_records   # JSON-serialised EvidenceRecord objects
  /reports             # analytics reports
config.py              # confidence thresholds and routing cutoff
```

All inter-module data flows through the dataclasses defined in
`shared/schemas.py`.  No module may define its own ad-hoc dict format.

---

## SceneContext JSON sidecar format

For every image that requires geometry-dependent violation checks (wrong-side
driving, stop-line crossing, red-light running, illegal parking), place a JSON
file in `data/scene_context/` with **exactly the same stem** as the image:

| Image file          | Sidecar file                           |
|---------------------|----------------------------------------|
| `raw_images/frame_001.jpg` | `scene_context/frame_001.json` |

The file must be valid JSON and conform to the following schema.
All fields except `image_id` are **optional** — omit anything that is not
annotated for that frame.  Violation modules treat a missing field as
"unknown / not applicable", **never** as "violation confirmed."

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | `string` | **Required.** Must match the image filename stem. |
| `lane_direction_vector` | `{x, y}` | Unit vector pointing in the **legal** direction of travel for the lane visible in the frame.  Used by the wrong-side module. |
| `stop_line_coords` | `[{x,y}, {x,y}]` | Two pixel-space points defining the stop line from left to right. |
| `signal_state` | `"red" \| "yellow" \| "green" \| "unknown"` | Traffic-signal state at frame capture time. |
| `no_parking_zone_polygon` | `[{x,y}, …]` | Closed polygon (≥ 3 points) in pixel space marking the no-parking area. |
| `no_parking_sign_visible` | `bool` | `true` when a "No Parking" sign is visible in the frame.  Defaults to `false`. |

All coordinates are in **pixel space** relative to the top-left corner of the
raw image at its original resolution.

### Example — `data/scene_context/example_frame_001.json`

```json
{
  "image_id": "frame_001",
  "lane_direction_vector": { "x": 0.0, "y": -1.0 },
  "stop_line_coords": [
    { "x": 120.0, "y": 610.0 },
    { "x": 980.0, "y": 610.0 }
  ],
  "signal_state": "red",
  "no_parking_zone_polygon": [
    { "x": 50.0,  "y": 400.0 },
    { "x": 250.0, "y": 400.0 },
    { "x": 250.0, "y": 700.0 },
    { "x": 50.0,  "y": 700.0 }
  ],
  "no_parking_sign_visible": true
}
```

This example says: the lane runs upward (y decreasing), the stop line is a
horizontal bar at y = 610, the signal is red, and a no-parking zone occupies
the left strip of the frame with a visible sign.

---

## Confidence thresholds and review routing

See `config.py`.

- `THRESHOLDS.get(ViolationType.X)` returns the minimum confidence for that
  violation type to be written to `outputs/violation_records/`.
- `AUTO_PROCESS_CUTOFF` (default **0.85**) is the boundary between automatic
  processing and human-review queuing.  Records at or above this value are
  auto-processed; records below it go to the review queue.
