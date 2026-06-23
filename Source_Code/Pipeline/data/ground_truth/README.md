# Ground Truth Annotation Format

Place one or more `.jsonl` files here (e.g. `val.jsonl`).  
Each line is one ground truth violation record:

```json
{"image_id": "frame_001", "violation_type": "helmet", "confidence": 1.0, "bbox": {"x1": 100, "y1": 80, "x2": 400, "y2": 450}}
```

**Required fields:** `image_id`, `violation_type`  
**Optional:** `bbox` (enables IoU-based matching in evaluate.py; without it, matching is image_id + type only)  
**`confidence` is always 1.0 for ground truth** — the field exists only so the format mirrors ViolationRecord.

Valid `violation_type` values: `helmet` `seatbelt` `triple_riding` `wrong_side` `stop_line` `red_light` `illegal_parking`

For scene-context-dependent violations (`wrong_side`, `stop_line`, `red_light`, `illegal_parking`),  
the corresponding image must also have a sidecar in `data/scene_context/<image_id>.json`  
with the relevant geometry fields populated, otherwise the rule engine will return  
`insufficient_scene_context` and the violation can never be predicted.

Run `python -m evaluation.evaluate --dry-run` to see an example record and exit.
