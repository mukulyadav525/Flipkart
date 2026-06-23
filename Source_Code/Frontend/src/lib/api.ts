/**
 * API client — reads from Backend when VITE_USE_MOCK=false,
 * otherwise serves mock data directly (default for prototype).
 */

import type { EvidenceRecord, AnalyticsSummary, AuditEntry, SystemInfo, EvalMetric, ReviewAction } from './types'
import {
  MOCK_VIOLATIONS, MOCK_REVIEW_QUEUE, MOCK_ANALYTICS,
  MOCK_AUDIT_LOG, MOCK_EVAL_METRICS, MOCK_SYSTEM_INFO,
} from './mockData'

const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false'
const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`API POST ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

// ---------------------------------------------------------------------------
// Citizen
// ---------------------------------------------------------------------------

export async function fetchCitizenViolations(plate: string): Promise<EvidenceRecord[]> {
  if (USE_MOCK) {
    const upper = plate.trim().toUpperCase()
    return MOCK_VIOLATIONS.filter(r => r.plate_text.toUpperCase() === upper)
  }
  return get<EvidenceRecord[]>(`/citizen/violations?plate=${encodeURIComponent(plate)}`)
}

// ---------------------------------------------------------------------------
// Police
// ---------------------------------------------------------------------------

export async function fetchPoliceViolations(filters?: {
  violation_type?: string
  date_from?: string
  date_to?: string
}): Promise<{ total: number; records: EvidenceRecord[] }> {
  if (USE_MOCK) {
    let records = [...MOCK_VIOLATIONS]
    if (filters?.violation_type)
      records = records.filter(r => r.violation_record.violation_type === filters.violation_type)
    return { total: records.length, records }
  }
  const params = new URLSearchParams(filters as Record<string, string>).toString()
  return get(`/police/violations${params ? `?${params}` : ''}`)
}

export async function fetchReviewQueue(): Promise<EvidenceRecord[]> {
  if (USE_MOCK) return [...MOCK_REVIEW_QUEUE]
  return get('/police/review-queue')
}

export async function submitReview(recordId: string, action: ReviewAction): Promise<void> {
  if (USE_MOCK) return   // no-op in mock mode — UI handles local state
  await post(`/police/review/${recordId}`, action)
}

export async function fetchRepeatOffenders() {
  if (USE_MOCK) return MOCK_ANALYTICS.repeat_offenders
  return get('/police/repeat-offenders')
}

export async function fetchPoliceSummary(): Promise<AnalyticsSummary> {
  if (USE_MOCK) return MOCK_ANALYTICS
  return get('/police/summary')
}

// ---------------------------------------------------------------------------
// Admin
// ---------------------------------------------------------------------------

export async function fetchAdminMetrics(): Promise<{
  latency: null | { n_images: number; mean_ms: number; median_ms: number; p95_ms: number; throughput_fps: number }
  eval_metrics: EvalMetric[] | null
  data_available: { latency: boolean; eval: boolean }
}> {
  if (USE_MOCK) {
    return {
      latency: { n_images: 1284, mean_ms: 46.2, median_ms: 41.8, p95_ms: 88.5, throughput_fps: 21.6 },
      eval_metrics: MOCK_EVAL_METRICS,
      data_available: { latency: true, eval: true },
    }
  }
  return get('/admin/metrics')
}

export async function fetchAuditLog(): Promise<{ total: number; entries: AuditEntry[] }> {
  if (USE_MOCK) return { total: MOCK_AUDIT_LOG.length, entries: MOCK_AUDIT_LOG }
  return get('/admin/audit-log')
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  if (USE_MOCK) return MOCK_SYSTEM_INFO
  return get('/admin/system-info')
}
