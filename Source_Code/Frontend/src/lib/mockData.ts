/**
 * MOCK DATA — placeholder until the real Pipeline produces output.
 * All records here are synthetic. Replace by setting VITE_USE_MOCK=false
 * once Backend/main.py is running with real Pipeline output.
 */

import type { EvidenceRecord, AnalyticsSummary, AuditEntry, SystemInfo, EvalMetric } from './types'

export const MOCK_VIOLATIONS: EvidenceRecord[] = [
  {
    id: 'ev_001',
    status: 'confirmed',
    violation_record: {
      image_id: 'frame_0047',
      violation_type: 'helmet',
      confidence: 0.91,
      rule_trace:
        'Bike detected (conf=0.91) at bbox(120,95,398,442). Head keypoints (nose, ears, eyes) visible at approx (205,118). No helmet-class detection overlapping head region (overlap threshold=30%). Rule: helmet must be detectable within 30% overlap of head keypoint bounding box.',
      related_detection_ids: ['frame_0047:bike:120,95'],
      related_plate_text: 'MH12AB1234',
    },
    annotated_image_path: '',
    timestamp: '2025-06-18T08:23:11+05:30',
    plate_text: 'MH12AB1234',
    plate_confidence: 0.94,
  },
  {
    id: 'ev_002',
    status: 'confirmed',
    violation_record: {
      image_id: 'frame_0112',
      violation_type: 'red_light',
      confidence: 0.96,
      rule_trace:
        'Signal state=red confirmed from scene sidecar. Stop line at y=610 (pixel coords). Vehicle front bbox bottom edge at y=638 — crosses stop line by 28px. Rule: vehicle front must not cross stop_line_coords when signal_state=red.',
      related_detection_ids: ['frame_0112:car:80,320'],
      related_plate_text: 'DL5SAB0001',
    },
    annotated_image_path: '',
    timestamp: '2025-06-18T09:45:33+05:30',
    plate_text: 'DL5SAB0001',
    plate_confidence: 0.97,
  },
  {
    id: 'ev_003',
    status: 'confirmed',
    violation_record: {
      image_id: 'frame_0021',
      violation_type: 'triple_riding',
      confidence: 0.88,
      rule_trace:
        'Bike detected (conf=0.88) at bbox(100,100,400,450). 3 person detection(s) overlap the bike bbox by >=35% (rider overlap threshold). Legal maximum is 2 rider(s) on a two-wheeler. Rule fired: 3 > 2.',
      related_detection_ids: ['frame_0021:bike:100,100', 'frame_0021:pedestrian:120,120', 'frame_0021:pedestrian:145,120', 'frame_0021:pedestrian:170,120'],
      related_plate_text: 'KA01MN5678',
    },
    annotated_image_path: '',
    timestamp: '2025-06-18T07:12:44+05:30',
    plate_text: 'KA01MN5678',
    plate_confidence: 0.89,
  },
  {
    id: 'ev_004',
    status: 'confirmed',
    violation_record: {
      image_id: 'frame_0098',
      violation_type: 'seatbelt',
      confidence: 0.87,
      rule_trace:
        'Car detected (conf=0.90) at bbox(50,50,500,350). Occupant detected (conf=0.82) with 61% overlap inside car. Torso keypoints visible at approx (220,190). No seatbelt-class detection overlaps torso region (overlap threshold=25%). Rule: seatbelt diagonal strap must be detectable across shoulder-to-hip torso keypoint region.',
      related_detection_ids: ['frame_0098:car:50,50', 'frame_0098:pedestrian:100,60'],
      related_plate_text: 'MH12AB1234',
    },
    annotated_image_path: '',
    timestamp: '2025-06-17T16:34:22+05:30',
    plate_text: 'MH12AB1234',
    plate_confidence: 0.91,
  },
  {
    id: 'ev_005',
    status: 'confirmed',
    violation_record: {
      image_id: 'frame_0203',
      violation_type: 'illegal_parking',
      confidence: 0.86,
      rule_trace:
        'No-parking sign visible in frame (no_parking_sign_visible=true). Vehicle centroid at (310, 580) falls within no_parking_zone_polygon. Rule: vehicle must not be stationary within annotated no-parking zone.',
      related_detection_ids: ['frame_0203:car:210,440'],
      related_plate_text: 'TN10CD9012',
    },
    annotated_image_path: '',
    timestamp: '2025-06-18T11:22:55+05:30',
    plate_text: 'TN10CD9012',
    plate_confidence: 0.88,
  },
  {
    id: 'ev_006',
    status: 'confirmed',
    violation_record: {
      image_id: 'frame_0311',
      violation_type: 'helmet',
      confidence: 0.93,
      rule_trace:
        'Bike detected (conf=0.93) at bbox(200,80,480,430). Head keypoints (nose, ears) visible at approx (330,102). No helmet-class detection overlapping head region (overlap threshold=30%).',
      related_detection_ids: ['frame_0311:bike:200,80'],
      related_plate_text: 'MH12AB1234',
    },
    annotated_image_path: '',
    timestamp: '2025-06-18T13:05:19+05:30',
    plate_text: 'MH12AB1234',
    plate_confidence: 0.96,
  },
  {
    id: 'ev_007',
    status: 'confirmed',
    violation_record: {
      image_id: 'frame_0155',
      violation_type: 'stop_line',
      confidence: 0.89,
      rule_trace:
        'Stop line at y=595 from scene sidecar. Signal state=yellow. Vehicle front bbox bottom edge at y=612 — crosses stop line by 17px while signal is not green.',
      related_detection_ids: ['frame_0155:bike:310,280'],
      related_plate_text: 'UP80AB3456',
    },
    annotated_image_path: '',
    timestamp: '2025-06-17T18:47:02+05:30',
    plate_text: 'UP80AB3456',
    plate_confidence: 0.85,
  },
]

export const MOCK_REVIEW_QUEUE: EvidenceRecord[] = [
  {
    id: 'rq_001',
    status: 'pending',
    violation_record: {
      image_id: 'frame_0189',
      violation_type: 'helmet',
      confidence: 0.72,
      rule_trace:
        'Bike detected (conf=0.72) at bbox(305,140,540,460). Head keypoints visible. No helmet-class detection overlapping head region. Confidence below auto-process cutoff (0.85) — queued for human review.',
      related_detection_ids: ['frame_0189:bike:305,140'],
      related_plate_text: 'MH04XY7890',
    },
    annotated_image_path: '',
    timestamp: '2025-06-18T10:14:38+05:30',
    plate_text: 'MH04XY7890',
    plate_confidence: 0.78,
    boxes: [
      { label: 'motorcycle 0.93', kind: 'vehicle',   x: 0.30, y: 0.34, w: 0.40, h: 0.55, confidence: 0.93 },
      { label: 'rider 0.88',      kind: 'rider',      x: 0.38, y: 0.18, w: 0.24, h: 0.46, confidence: 0.88 },
      { label: 'NO HELMET 0.72',  kind: 'violation',  x: 0.43, y: 0.12, w: 0.14, h: 0.16, confidence: 0.72 },
      { label: 'MH04XY7890',      kind: 'plate',      x: 0.41, y: 0.78, w: 0.18, h: 0.08, confidence: 0.78 },
    ],
  },
  {
    id: 'rq_002',
    status: 'pending',
    violation_record: {
      image_id: 'frame_0244',
      violation_type: 'wrong_side',
      confidence: 0.68,
      rule_trace:
        'Vehicle detected travelling against lane_direction_vector (0.0,−1.0). Displacement vector over 3 frames: (0.2, 0.8) — dot product with legal direction is −0.8, indicating opposing travel. Confidence below auto-process cutoff — queued for human review.',
      related_detection_ids: ['frame_0244:car:160,200'],
      related_plate_text: 'RJ14GH2233',
    },
    annotated_image_path: '',
    timestamp: '2025-06-18T06:58:12+05:30',
    plate_text: 'RJ14GH2233',
    plate_confidence: 0.71,
    boxes: [
      { label: 'car 0.91',         kind: 'vehicle',  x: 0.22, y: 0.40, w: 0.46, h: 0.42, confidence: 0.91 },
      { label: 'WRONG SIDE 0.68',  kind: 'violation',x: 0.22, y: 0.40, w: 0.46, h: 0.42, confidence: 0.68 },
      { label: 'RJ14GH2233',       kind: 'plate',    x: 0.30, y: 0.70, w: 0.20, h: 0.09, confidence: 0.71 },
    ],
  },
  {
    id: 'rq_003',
    status: 'pending',
    violation_record: {
      image_id: 'frame_0278',
      violation_type: 'triple_riding',
      confidence: 0.74,
      rule_trace:
        'Bike detected (conf=0.74). 3 person detections overlap the bike bbox by >=35%. One detection (conf=0.52) is borderline — may be a pedestrian adjacent to the vehicle rather than a rider. Queued for human review.',
      related_detection_ids: ['frame_0278:bike:90,110', 'frame_0278:pedestrian:100,130', 'frame_0278:pedestrian:130,130', 'frame_0278:pedestrian:360,220'],
      related_plate_text: 'GJ01KL4455',
    },
    annotated_image_path: '',
    timestamp: '2025-06-18T12:33:47+05:30',
    plate_text: 'GJ01KL4455',
    plate_confidence: 0.81,
    boxes: [
      { label: 'motorcycle 0.90', kind: 'vehicle',  x: 0.28, y: 0.38, w: 0.44, h: 0.50, confidence: 0.90 },
      { label: 'rider 1 0.86',    kind: 'rider',     x: 0.34, y: 0.20, w: 0.16, h: 0.40, confidence: 0.86 },
      { label: 'rider 2 0.81',    kind: 'rider',     x: 0.46, y: 0.18, w: 0.16, h: 0.42, confidence: 0.81 },
      { label: 'rider 3 0.52',    kind: 'rider',     x: 0.57, y: 0.22, w: 0.15, h: 0.38, confidence: 0.52 },
      { label: 'GJ01KL4455',      kind: 'plate',     x: 0.40, y: 0.80, w: 0.18, h: 0.08, confidence: 0.81 },
    ],
  },
  {
    id: 'rq_004',
    status: 'pending',
    violation_record: {
      image_id: 'frame_0302',
      violation_type: 'seatbelt',
      confidence: 0.71,
      rule_trace:
        'Car detected (conf=0.88). Occupant detected with 44% overlap. Torso keypoints partially occluded — left shoulder not visible. No seatbelt detected at partial torso region. Queued for human review.',
      related_detection_ids: ['frame_0302:car:40,60', 'frame_0302:pedestrian:90,70'],
      related_plate_text: 'DL3CAB8877',
    },
    annotated_image_path: '',
    timestamp: '2025-06-18T14:51:09+05:30',
    plate_text: 'DL3CAB8877',
    plate_confidence: 0.75,
    boxes: [
      { label: 'car 0.88',         kind: 'vehicle',  x: 0.18, y: 0.30, w: 0.56, h: 0.50, confidence: 0.88 },
      { label: 'occupant 0.79',    kind: 'rider',    x: 0.30, y: 0.34, w: 0.22, h: 0.34, confidence: 0.79 },
      { label: 'NO SEATBELT 0.71', kind: 'violation',x: 0.32, y: 0.42, w: 0.18, h: 0.22, confidence: 0.71 },
      { label: 'DL3CAB8877',       kind: 'plate',    x: 0.36, y: 0.72, w: 0.20, h: 0.09, confidence: 0.75 },
    ],
  },
]

export const MOCK_ANALYTICS: AnalyticsSummary = {
  total_today: 7,
  total_this_week: 31,
  pending_review: 4,
  counts_by_type: {
    helmet: 3,
    seatbelt: 1,
    triple_riding: 1,
    red_light: 1,
    stop_line: 1,
    wrong_side: 0,
    illegal_parking: 1,
  },
  severity_ranking: [
    { violation_type: 'helmet',          count: 3, severity_weight: 1.5, mean_confidence: 0.92, severity_score: 4.13 },
    { violation_type: 'red_light',       count: 1, severity_weight: 3.0, mean_confidence: 0.96, severity_score: 2.88 },
    { violation_type: 'stop_line',       count: 1, severity_weight: 2.0, mean_confidence: 0.89, severity_score: 1.78 },
    { violation_type: 'triple_riding',   count: 1, severity_weight: 1.8, mean_confidence: 0.88, severity_score: 1.58 },
    { violation_type: 'seatbelt',        count: 1, severity_weight: 1.5, mean_confidence: 0.87, severity_score: 1.31 },
    { violation_type: 'illegal_parking', count: 1, severity_weight: 1.0, mean_confidence: 0.86, severity_score: 0.86 },
    { violation_type: 'wrong_side',      count: 0, severity_weight: 3.0, mean_confidence: 0.0,  severity_score: 0.0  },
  ],
  repeat_offenders: [
    {
      plate_text: 'MH12AB1234',
      count: 3,
      violation_types: ['helmet', 'seatbelt'],
      last_seen_timestamp: '2025-06-18T13:05:19+05:30',
    },
  ],
  tod_breakdown: {
    Night:     { wrong_side: 1 },
    Morning:   { helmet: 1, red_light: 1, triple_riding: 1 },
    Afternoon: { helmet: 1, seatbelt: 1, illegal_parking: 1 },
    Evening:   { stop_line: 1 },
  },
}

export const MOCK_AUDIT_LOG: AuditEntry[] = [
  {
    id: 'au_001',
    record_id: 'rq_archived_01',
    action: 'approved',
    reviewer_id: 'officer_stub',
    timestamp: '2025-06-17T11:22:34+05:30',
    plate_text: 'HR26AB9900',
    violation_type: 'helmet',
    notes: 'Image clear, helmet clearly absent.',
  },
  {
    id: 'au_002',
    record_id: 'rq_archived_02',
    action: 'rejected',
    reviewer_id: 'officer_stub',
    timestamp: '2025-06-17T14:08:11+05:30',
    plate_text: 'PB10CD1122',
    violation_type: 'wrong_side',
    notes: 'Vehicle was reversing into a driveway — not a wrong-side violation.',
  },
  {
    id: 'au_003',
    record_id: 'rq_archived_03',
    action: 'approved',
    reviewer_id: 'officer_stub',
    timestamp: '2025-06-18T09:03:55+05:30',
    plate_text: 'MH43EF3344',
    violation_type: 'triple_riding',
    notes: undefined,
  },
]

// Demo calibration figures — the real numbers come from
// `python -m evaluation.evaluate` once a labelled validation set exists.
export const MOCK_EVAL_METRICS: EvalMetric[] = [
  { violation_type: 'helmet',          precision: '0.93', recall: '0.89', f1: '0.91', accuracy: '0.86', average_precision: '0.90', n_gt: '420', scene_context_dependent: 'false' },
  { violation_type: 'seatbelt',        precision: '0.88', recall: '0.82', f1: '0.85', accuracy: '0.79', average_precision: '0.83', n_gt: '310', scene_context_dependent: 'false' },
  { violation_type: 'triple_riding',   precision: '0.90', recall: '0.85', f1: '0.87', accuracy: '0.81', average_precision: '0.86', n_gt: '160', scene_context_dependent: 'false' },
  { violation_type: 'wrong_side',      precision: '0.84', recall: '0.78', f1: '0.81', accuracy: '0.74', average_precision: '0.80', n_gt: '95',  scene_context_dependent: 'true'  },
  { violation_type: 'stop_line',       precision: '0.86', recall: '0.80', f1: '0.83', accuracy: '0.77', average_precision: '0.82', n_gt: '120', scene_context_dependent: 'true'  },
  { violation_type: 'red_light',       precision: '0.91', recall: '0.87', f1: '0.89', accuracy: '0.83', average_precision: '0.88', n_gt: '140', scene_context_dependent: 'true'  },
  { violation_type: 'illegal_parking', precision: '0.82', recall: '0.76', f1: '0.79', accuracy: '0.71', average_precision: '0.78', n_gt: '88',  scene_context_dependent: 'true'  },
]

export const MOCK_SYSTEM_INFO: SystemInfo = {
  model_name: 'YOLOv8s + YOLOv8n-pose',
  model_weights: 'Pipeline/weights/yolov8_idd.pt',
  weights_exist: false,
  dataset_name: 'Indian Driving Dataset (IDD)',
  dataset_version: 'IDD Detection v1.0',
  last_trained: 'pending — run Pipeline/src/detection/finetune.py',
  pipeline_version: '0.1.0',
  confirmed_records: 7,
  pending_review: 4,
}
