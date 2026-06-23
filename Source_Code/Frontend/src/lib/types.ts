export type ViolationType =
  | 'helmet'
  | 'seatbelt'
  | 'triple_riding'
  | 'wrong_side'
  | 'stop_line'
  | 'red_light'
  | 'illegal_parking'

export type ReviewStatus = 'pending' | 'confirmed' | 'rejected' | 'approved'
export type UserRole = 'citizen' | 'police' | 'admin'

export interface ViolationRecord {
  image_id: string
  violation_type: ViolationType
  confidence: number
  rule_trace: string
  related_detection_ids: string[]
  related_plate_text: string | null
}

/** A single AI detection box, normalised (0..1) to the frame, for the
 *  Police evidence-review screen's glowing overlays. */
export interface DetectionBox {
  label: string
  kind: 'vehicle' | 'rider' | 'plate' | 'violation'
  x: number
  y: number
  w: number
  h: number
  confidence: number
}

export interface EvidenceRecord {
  id: string
  status: ReviewStatus
  violation_record: ViolationRecord
  annotated_image_path: string
  timestamp: string
  plate_text: string
  plate_confidence: number
  /** Optional AI overlay boxes (mock/demo); absent on real backend records. */
  boxes?: DetectionBox[]
}

export interface AuditEntry {
  id: string
  record_id: string
  action: 'approved' | 'rejected'
  reviewer_id: string
  timestamp: string
  plate_text: string
  violation_type: ViolationType
  notes?: string
}

export interface RepeatOffender {
  plate_text: string
  count: number
  violation_types: ViolationType[]
  last_seen_timestamp: string
}

export interface SeverityRow {
  violation_type: ViolationType
  count: number
  severity_weight: number
  mean_confidence: number
  severity_score: number
}

export interface AnalyticsSummary {
  total_today: number
  total_this_week: number
  pending_review: number
  counts_by_type: Record<ViolationType, number>
  severity_ranking: SeverityRow[]
  repeat_offenders: RepeatOffender[]
  tod_breakdown: Record<string, Record<string, number>>
}

export interface LatencyStats {
  n_images: number
  mean_ms: number
  median_ms: number
  p95_ms: number
  throughput_fps: number
}

export interface EvalMetric {
  violation_type: string
  precision: string
  recall: string
  f1: string
  accuracy: string
  average_precision: string
  n_gt: string
  scene_context_dependent: string
}

export interface SystemInfo {
  model_name: string
  model_weights: string
  weights_exist: boolean
  dataset_name: string
  dataset_version: string
  last_trained: string
  pipeline_version: string
  confirmed_records: number
  pending_review: number
}

export interface ReviewAction {
  action: 'approved' | 'rejected'
  reviewer_id?: string
  notes?: string
}
