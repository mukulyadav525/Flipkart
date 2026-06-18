import { useEffect, useState } from 'react'
import {
  AlertTriangle, Clock, CheckCircle2, XCircle,
  Filter, Camera, TrendingUp, Users, RefreshCw,
} from 'lucide-react'
import Header from '../components/shared/Header'
import { StatusBadge, ViolationBadge } from '../components/shared/StatusBadge'
import { ConfidenceBar } from '../components/shared/ConfidenceBar'
import {
  fetchPoliceViolations, fetchReviewQueue,
  fetchPoliceSummary, fetchRepeatOffenders, submitReview,
} from '../lib/api'
import { VIOLATION_META, formatTimestamp, fmtPct } from '../lib/utils'
import type { EvidenceRecord, AnalyticsSummary, RepeatOffender, ViolationType } from '../lib/types'

type Tab = 'violations' | 'review' | 'offenders'

// ---------------------------------------------------------------------------
// Summary strip
// ---------------------------------------------------------------------------

function SummaryStrip({ data }: { data: AnalyticsSummary }) {
  const stats = [
    { label: 'Today',          value: data.total_today,     icon: AlertTriangle, color: 'text-red-400'     },
    { label: 'This week',      value: data.total_this_week, icon: TrendingUp,    color: 'text-indigo-400'  },
    { label: 'Pending review', value: data.pending_review,  icon: Clock,         color: 'text-amber-400'   },
    { label: 'Repeat plates',  value: data.repeat_offenders.length, icon: Users, color: 'text-orange-400'  },
  ]
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {stats.map(s => (
        <div key={s.label} className="card px-4 py-4 flex items-center gap-4">
          <div className={`shrink-0 ${s.color}`}><s.icon size={20} /></div>
          <div className="min-w-0">
            <p className="text-2xl font-bold text-slate-100 tabular-nums">{s.value}</p>
            <p className="text-xs text-slate-500 truncate">{s.label}</p>
          </div>
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Violations table
// ---------------------------------------------------------------------------

const ALL_TYPES: ViolationType[] = [
  'helmet','seatbelt','triple_riding','wrong_side','stop_line','red_light','illegal_parking'
]

function ViolationsTab() {
  const [records, setRecords]   = useState<EvidenceRecord[]>([])
  const [total, setTotal]       = useState(0)
  const [filter, setFilter]     = useState('')
  const [loading, setLoading]   = useState(true)

  async function load(vtype?: string) {
    setLoading(true)
    try {
      const data = await fetchPoliceViolations(vtype ? { violation_type: vtype } : undefined)
      setRecords(data.records)
      setTotal(data.total)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void load() }, [])

  function applyFilter(v: string) {
    setFilter(v)
    void load(v || undefined)
  }

  return (
    <div className="space-y-4">
      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-2">
        <Filter size={14} className="text-slate-500 shrink-0" />
        <button
          onClick={() => applyFilter('')}
          className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
            !filter ? 'bg-slate-700 border-slate-600 text-slate-100' : 'border-slate-800 text-slate-500 hover:text-slate-300'
          }`}
        >
          All ({total})
        </button>
        {ALL_TYPES.map(vt => (
          <button
            key={vt}
            onClick={() => applyFilter(vt)}
            className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
              filter === vt ? 'bg-slate-700 border-slate-600 text-slate-100' : 'border-slate-800 text-slate-500 hover:text-slate-300'
            }`}
          >
            {VIOLATION_META[vt].label}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table>
            <thead className="bg-slate-900/80">
              <tr>
                <th>Image</th>
                <th>Violation</th>
                <th>Plate</th>
                <th>Confidence</th>
                <th>Timestamp</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={6} className="text-center py-12 text-slate-600">Loading…</td></tr>
              ) : records.length === 0 ? (
                <tr><td colSpan={6} className="text-center py-12 text-slate-600">No records found</td></tr>
              ) : records.map(r => (
                <tr key={r.id} className="hover:bg-slate-800/40 transition-colors">
                  <td>
                    <div className="w-16 h-10 bg-slate-800 rounded-lg border border-slate-700 flex items-center justify-center">
                      <Camera size={14} className="text-slate-600" />
                    </div>
                  </td>
                  <td><ViolationBadge type={r.violation_record.violation_type} /></td>
                  <td>
                    <span className="font-mono text-sm text-slate-200 font-semibold">{r.plate_text || '—'}</span>
                  </td>
                  <td className="min-w-[140px]">
                    <ConfidenceBar value={r.violation_record.confidence} showLabel={false} />
                    <span className="text-xs text-slate-500 tabular-nums mt-1 block">
                      {fmtPct(r.violation_record.confidence)}
                    </span>
                  </td>
                  <td><time className="text-xs text-slate-500 tabular-nums">{formatTimestamp(r.timestamp)}</time></td>
                  <td><StatusBadge status={r.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Review queue
// ---------------------------------------------------------------------------

function ReviewCard({
  record,
  onAction,
}: {
  record: EvidenceRecord
  onAction: (id: string, action: 'approved' | 'rejected') => void
}) {
  const [busy, setBusy] = useState(false)

  async function act(action: 'approved' | 'rejected') {
    setBusy(true)
    await submitReview(record.id, { action })
    onAction(record.id, action)
  }

  return (
    <div className="card p-5 space-y-4 animate-slide-up">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="space-y-1.5">
          <ViolationBadge type={record.violation_record.violation_type} />
          <div className="flex items-center gap-2 mt-1">
            <span className="font-mono text-sm font-bold text-slate-200">{record.plate_text || '—'}</span>
            <time className="text-xs text-slate-500">{formatTimestamp(record.timestamp)}</time>
          </div>
        </div>
        <ConfidenceBar value={record.violation_record.confidence} />
      </div>

      {/* Placeholder image */}
      <div className="w-full h-36 bg-slate-800 border border-slate-700 rounded-xl flex flex-col items-center justify-center gap-2">
        <Camera size={24} className="text-slate-600" />
        <p className="text-xs text-slate-600">Annotated image — run pipeline to generate</p>
      </div>

      {/* Rule trace */}
      <p className="text-xs text-slate-500 font-mono leading-relaxed bg-slate-800/60 border border-slate-700 rounded-lg p-3">
        {record.violation_record.rule_trace}
      </p>

      {/* Actions */}
      <div className="flex gap-3 pt-1">
        <button
          className="btn-success flex items-center gap-2 flex-1 justify-center"
          onClick={() => act('approved')}
          disabled={busy}
        >
          <CheckCircle2 size={14} /> Approve
        </button>
        <button
          className="btn-danger flex items-center gap-2 flex-1 justify-center"
          onClick={() => act('rejected')}
          disabled={busy}
        >
          <XCircle size={14} /> Reject
        </button>
      </div>
    </div>
  )
}

function ReviewQueueTab() {
  const [queue, setQueue] = useState<EvidenceRecord[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchReviewQueue().then(data => { setQueue(data); setLoading(false) })
  }, [])

  function handleAction(id: string) {
    setQueue(q => q.filter(r => r.id !== id))
  }

  if (loading) return <p className="text-sm text-slate-600 py-8 text-center">Loading…</p>

  if (queue.length === 0) {
    return (
      <div className="card p-12 flex flex-col items-center gap-3 text-center">
        <CheckCircle2 size={36} className="text-emerald-600" />
        <p className="text-base font-semibold text-slate-300">Review queue is empty</p>
        <p className="text-xs text-slate-600">All pending records have been actioned.</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-slate-500">
        {queue.length} record{queue.length !== 1 ? 's' : ''} awaiting review.
        Records below 85% confidence are routed here before any fine is issued.
      </p>
      <div className="grid sm:grid-cols-2 gap-4">
        {queue.map(r => (
          <ReviewCard key={r.id} record={r} onAction={handleAction} />
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Repeat offenders
// ---------------------------------------------------------------------------

function RepeatOffendersTab() {
  const [offenders, setOffenders] = useState<RepeatOffender[]>([])
  const [loading, setLoading]     = useState(true)

  useEffect(() => {
    fetchRepeatOffenders().then(data => {
      setOffenders(data as RepeatOffender[])
      setLoading(false)
    })
  }, [])

  if (loading) return <p className="text-sm text-slate-600 py-8 text-center">Loading…</p>

  if (offenders.length === 0) {
    return (
      <div className="card p-12 text-center">
        <p className="text-slate-600 text-sm">No repeat offenders detected yet.</p>
      </div>
    )
  }

  return (
    <div className="card overflow-hidden">
      <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
        <p className="text-sm font-semibold text-slate-300">Plates with 2+ confirmed violations</p>
        <span className="text-xs text-slate-600">{offenders.length} plate{offenders.length !== 1 ? 's' : ''}</span>
      </div>
      <table>
        <thead className="bg-slate-900/60">
          <tr>
            <th>Plate</th>
            <th>Count</th>
            <th>Violation types</th>
            <th>Last seen</th>
          </tr>
        </thead>
        <tbody>
          {offenders.map(o => (
            <tr key={o.plate_text} className="hover:bg-slate-800/30 transition-colors">
              <td>
                <span className="font-mono text-sm font-bold text-red-300">{o.plate_text}</span>
              </td>
              <td>
                <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-red-950 border border-red-800 text-red-300 text-xs font-bold">
                  {o.count}
                </span>
              </td>
              <td>
                <div className="flex flex-wrap gap-1">
                  {o.violation_types.map(vt => (
                    <ViolationBadge key={vt} type={vt as ViolationType} />
                  ))}
                </div>
              </td>
              <td><time className="text-xs text-slate-500">{formatTimestamp(o.last_seen_timestamp)}</time></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main view
// ---------------------------------------------------------------------------

export default function PoliceView() {
  const [tab, setTab] = useState<Tab>('violations')
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null)

  useEffect(() => { fetchPoliceSummary().then(setSummary) }, [])

  const tabs: { key: Tab; label: string }[] = [
    { key: 'violations', label: 'Violations' },
    { key: 'review',     label: 'Review Queue' },
    { key: 'offenders',  label: 'Repeat Offenders' },
  ]

  return (
    <div className="min-h-screen bg-slate-950">
      <Header />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-6">
        {/* Heading */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h2 className="text-2xl font-bold text-slate-100">Enforcement Dashboard</h2>
            <p className="text-sm text-slate-500 mt-0.5">City-wide violation monitoring</p>
          </div>
          <button
            className="btn-ghost flex items-center gap-2 text-slate-500"
            onClick={() => fetchPoliceSummary().then(setSummary)}
          >
            <RefreshCw size={13} /> Refresh
          </button>
        </div>

        {/* Summary strip */}
        {summary && <SummaryStrip data={summary} />}

        {/* Severity ranking strip */}
        {summary && (
          <div className="card p-5 space-y-3">
            <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">Severity-Weighted Ranking (today)</p>
            <div className="space-y-2">
              {summary.severity_ranking.filter(r => r.count > 0).map((r, i) => {
                const meta = VIOLATION_META[r.violation_type]
                const maxScore = Math.max(...summary.severity_ranking.map(x => x.severity_score), 1)
                return (
                  <div key={r.violation_type} className="flex items-center gap-3">
                    <span className="text-xs text-slate-600 w-4 tabular-nums text-right">{i + 1}</span>
                    <span className={`text-xs font-medium w-28 shrink-0 ${meta.textClass}`}>{meta.label}</span>
                    <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${meta.bgClass.replace('/30', '')} bg-current`}
                        style={{ width: `${(r.severity_score / maxScore) * 100}%`, opacity: 0.7 }}
                      />
                    </div>
                    <span className="text-xs text-slate-500 tabular-nums w-16 text-right">
                      {r.count}× · {r.severity_score.toFixed(1)}
                    </span>
                  </div>
                )
              })}
            </div>
            <p className="text-xs text-slate-700">Score = count × severity weight × mean confidence. Weights configured in Pipeline/config.py.</p>
          </div>
        )}

        {/* Tabs */}
        <div>
          <div className="flex gap-6 border-b border-slate-800 mb-6">
            {tabs.map(t => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`pb-3 text-sm ${tab === t.key ? 'tab-active' : 'tab-inactive'}`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {tab === 'violations' && <ViolationsTab />}
          {tab === 'review'     && <ReviewQueueTab />}
          {tab === 'offenders'  && <RepeatOffendersTab />}
        </div>
      </main>
    </div>
  )
}
