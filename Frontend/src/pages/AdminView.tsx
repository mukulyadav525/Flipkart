import { useEffect, useState } from 'react'
import {
  Activity, Database, Clock, Cpu, CheckCircle2,
  XCircle, AlertCircle, Info, RefreshCw, Package,
} from 'lucide-react'
import Header from '../components/shared/Header'
import { fetchAdminMetrics, fetchAuditLog, fetchSystemInfo } from '../lib/api'
import { VIOLATION_META, formatTimestamp } from '../lib/utils'
import type { AuditEntry, EvalMetric, SystemInfo } from '../lib/types'

type Tab = 'health' | 'audit' | 'system'

// ---------------------------------------------------------------------------
// System health tab
// ---------------------------------------------------------------------------

function PendingBadge({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-1 text-xs text-slate-500 bg-slate-800 border border-slate-700 px-2.5 py-1 rounded-full">
      <Clock size={10} />
      {label}
    </span>
  )
}

function HealthTab() {
  const [data, setData] = useState<Awaited<ReturnType<typeof fetchAdminMetrics>> | null>(null)

  useEffect(() => { fetchAdminMetrics().then(setData) }, [])

  if (!data) return <p className="text-sm text-slate-600 py-8 text-center">Loading…</p>

  return (
    <div className="space-y-6">
      {/* Latency card */}
      <div className="card p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Clock size={16} className="text-indigo-400" />
            <h3 className="text-sm font-semibold text-slate-200">Inference Latency & Throughput</h3>
          </div>
          {!data.data_available.latency && <PendingBadge label="Pending — run pipeline" />}
        </div>

        {data.latency ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
            {[
              { label: 'Images',      value: data.latency.n_images.toString(),           unit: '' },
              { label: 'Mean',        value: data.latency.mean_ms.toFixed(1),            unit: 'ms' },
              { label: 'Median',      value: data.latency.median_ms.toFixed(1),          unit: 'ms' },
              { label: 'P95',         value: data.latency.p95_ms.toFixed(1),             unit: 'ms' },
              { label: 'Throughput',  value: data.latency.throughput_fps.toFixed(2),     unit: 'fps' },
            ].map(s => (
              <div key={s.label} className="bg-slate-800/60 border border-slate-700 rounded-xl p-4 text-center">
                <p className="text-2xl font-bold text-slate-100 tabular-nums">
                  {s.value}<span className="text-sm text-slate-500 ml-1">{s.unit}</span>
                </p>
                <p className="text-xs text-slate-500 mt-1">{s.label}</p>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-slate-800/40 border border-slate-800 rounded-xl p-6 text-center space-y-2">
            <Cpu size={28} className="text-slate-600 mx-auto" />
            <p className="text-sm text-slate-500">No latency data yet</p>
            <p className="text-xs text-slate-700">
              Run the detection pipeline with{' '}
              <code className="text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded text-xs">
                latency_log_path="outputs/latency_log.jsonl"
              </code>{' '}
              to capture timing.
            </p>
          </div>
        )}
      </div>

      {/* Per-type metrics card */}
      <div className="card overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity size={16} className="text-indigo-400" />
            <h3 className="text-sm font-semibold text-slate-200">Per-Violation-Type Evaluation Metrics</h3>
          </div>
          {!data.data_available.eval && <PendingBadge label="Pending evaluation" />}
        </div>

        {data.eval_metrics ? (
          <>
            <div className="overflow-x-auto">
              <table>
                <thead className="bg-slate-900/60">
                  <tr>
                    <th>Violation Type</th>
                    <th className="text-center">GT Count</th>
                    <th className="text-center">Precision</th>
                    <th className="text-center">Recall</th>
                    <th className="text-center">F1</th>
                    <th>Note</th>
                  </tr>
                </thead>
                <tbody>
                  {(data.eval_metrics as EvalMetric[]).map(m => {
                    const meta = VIOLATION_META[m.violation_type as keyof typeof VIOLATION_META]
                    const isPending = m.precision === '—'
                    return (
                      <tr key={m.violation_type} className="hover:bg-slate-800/30">
                        <td>
                          <span className={`text-sm font-medium ${meta?.textClass ?? 'text-slate-400'}`}>
                            {meta?.label ?? m.violation_type}
                          </span>
                        </td>
                        <td className="text-center tabular-nums">
                          <span className="text-sm text-slate-400">{m.n_gt}</span>
                        </td>
                        <td className="text-center tabular-nums">
                          {isPending
                            ? <span className="text-xs text-slate-600">—</span>
                            : <span className="text-sm text-slate-300">{m.precision}</span>}
                        </td>
                        <td className="text-center tabular-nums">
                          {isPending
                            ? <span className="text-xs text-slate-600">—</span>
                            : <span className="text-sm text-slate-300">{m.recall}</span>}
                        </td>
                        <td className="text-center tabular-nums">
                          {isPending
                            ? <span className="text-xs text-slate-600">—</span>
                            : <span className="text-sm text-slate-300">{m.f1}</span>}
                        </td>
                        <td>
                          {m.scene_context_dependent === 'true' && (
                            <span className="text-xs text-amber-600 flex items-center gap-1">
                              <Info size={10} /> Scene-context dependent
                            </span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <div className="px-5 py-3 bg-slate-900/40 border-t border-slate-800">
              <p className="text-xs text-slate-700">
                Metrics computed by <code className="text-slate-600">python -m evaluation.evaluate</code> once{' '}
                <code className="text-slate-600">data/ground_truth/</code> contains labelled annotations.
                Scene-context-dependent rules require populated sidecar JSON files to fire at all.
              </p>
            </div>
          </>
        ) : (
          <div className="p-8 text-center space-y-2">
            <p className="text-sm text-slate-500">Evaluation not yet run</p>
            <p className="text-xs text-slate-700">
              Label a validation set in <code className="text-slate-600">Pipeline/data/ground_truth/</code> then
              run <code className="text-slate-600">python -m evaluation.evaluate</code>.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Audit log tab
// ---------------------------------------------------------------------------

function AuditTab() {
  const [data, setData]   = useState<{ total: number; entries: AuditEntry[] } | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchAuditLog().then(d => { setData(d); setLoading(false) })
  }, [])

  if (loading) return <p className="text-sm text-slate-600 py-8 text-center">Loading…</p>
  if (!data || data.entries.length === 0) {
    return (
      <div className="card p-12 text-center space-y-2">
        <p className="text-slate-500 text-sm">No review actions yet</p>
        <p className="text-xs text-slate-700">Actions taken in the Review Queue tab are logged here.</p>
      </div>
    )
  }

  return (
    <div className="card overflow-hidden">
      <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
        <p className="text-sm font-semibold text-slate-300">Human review actions</p>
        <span className="text-xs text-slate-600">{data.total} total</span>
      </div>
      <table>
        <thead className="bg-slate-900/60">
          <tr>
            <th>Action</th>
            <th>Plate</th>
            <th>Violation</th>
            <th>Reviewer</th>
            <th>Timestamp</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {data.entries.map(e => (
            <tr key={e.id} className="hover:bg-slate-800/30 transition-colors">
              <td>
                {e.action === 'approved' ? (
                  <span className="inline-flex items-center gap-1.5 text-xs text-emerald-400">
                    <CheckCircle2 size={13} /> Approved
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 text-xs text-slate-500">
                    <XCircle size={13} /> Rejected
                  </span>
                )}
              </td>
              <td><span className="font-mono text-sm font-semibold text-slate-200">{e.plate_text || '—'}</span></td>
              <td>
                {VIOLATION_META[e.violation_type] && (
                  <span className={`text-xs ${VIOLATION_META[e.violation_type].textClass}`}>
                    {VIOLATION_META[e.violation_type].label}
                  </span>
                )}
              </td>
              <td><span className="text-xs font-mono text-slate-500">{e.reviewer_id}</span></td>
              <td><time className="text-xs text-slate-500 tabular-nums">{formatTimestamp(e.timestamp)}</time></td>
              <td>
                {e.notes
                  ? <span className="text-xs text-slate-500 italic">{e.notes}</span>
                  : <span className="text-xs text-slate-700">—</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// System info tab
// ---------------------------------------------------------------------------

function SystemTab() {
  const [info, setInfo] = useState<SystemInfo | null>(null)

  useEffect(() => { fetchSystemInfo().then(setInfo) }, [])
  if (!info) return <p className="text-sm text-slate-600 py-8 text-center">Loading…</p>

  const rows: { label: string; value: string; mono?: boolean; warn?: boolean }[] = [
    { label: 'Model',              value: info.model_name,       mono: false },
    { label: 'Weights file',       value: info.model_weights,    mono: true,  warn: !info.weights_exist },
    { label: 'Weights ready',      value: info.weights_exist ? 'Yes — fine-tuned weights found' : 'No — using pretrained COCO fallback', warn: !info.weights_exist },
    { label: 'Training dataset',   value: info.dataset_name,     mono: false },
    { label: 'Dataset version',    value: info.dataset_version,  mono: false },
    { label: 'Last trained',       value: info.last_trained,     warn: info.last_trained.startsWith('pending') },
    { label: 'Pipeline version',   value: info.pipeline_version, mono: true  },
    { label: 'Confirmed records',  value: info.confirmed_records.toString() },
    { label: 'Pending review',     value: info.pending_review.toString() },
  ]

  return (
    <div className="space-y-6">
      <div className="card overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800 flex items-center gap-2">
          <Package size={15} className="text-indigo-400" />
          <h3 className="text-sm font-semibold text-slate-200">What's currently running</h3>
        </div>
        <div className="divide-y divide-slate-800">
          {rows.map(r => (
            <div key={r.label} className="px-5 py-3.5 flex items-start gap-4 flex-wrap sm:flex-nowrap">
              <p className="text-xs text-slate-500 font-medium w-40 shrink-0 pt-0.5">{r.label}</p>
              <p className={`text-sm flex-1 flex items-center gap-2 ${r.mono ? 'font-mono' : ''} ${r.warn ? 'text-amber-400' : 'text-slate-200'}`}>
                {r.warn && <AlertCircle size={13} className="shrink-0" />}
                {r.value}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div className="card p-5 space-y-2 border-indigo-900/40">
        <div className="flex items-center gap-2 text-indigo-400">
          <Database size={14} />
          <span className="text-sm font-semibold">Output paths</span>
        </div>
        <div className="space-y-1.5 text-xs font-mono text-slate-500">
          <p>Pipeline/outputs/violation_records/confirmed.jsonl</p>
          <p>Pipeline/outputs/human_review_queue.jsonl</p>
          <p>Pipeline/outputs/annotated_images/</p>
          <p>Pipeline/outputs/reports/</p>
          <p>Backend/data/audit_log.jsonl</p>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main view
// ---------------------------------------------------------------------------

export default function AdminView() {
  const [tab, setTab] = useState<Tab>('health')

  const tabs: { key: Tab; label: string; icon: typeof Activity }[] = [
    { key: 'health', label: 'System Health', icon: Activity  },
    { key: 'audit',  label: 'Audit Log',     icon: Clock     },
    { key: 'system', label: 'Dataset & Model', icon: Database },
  ]

  return (
    <div className="min-h-screen bg-slate-950">
      <Header />
      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-8 space-y-6">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h2 className="text-2xl font-bold text-slate-100">System Administration</h2>
            <p className="text-sm text-slate-500 mt-0.5">Model health, review oversight, and version tracking</p>
          </div>
          <button
            className="btn-ghost flex items-center gap-2 text-slate-500"
            onClick={() => setTab(t => t)}
          >
            <RefreshCw size={13} /> Refresh
          </button>
        </div>

        <div>
          <div className="flex gap-6 border-b border-slate-800 mb-6">
            {tabs.map(t => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`pb-3 flex items-center gap-2 text-sm ${tab === t.key ? 'tab-active' : 'tab-inactive'}`}
              >
                <t.icon size={14} />
                {t.label}
              </button>
            ))}
          </div>

          {tab === 'health' && <HealthTab />}
          {tab === 'audit'  && <AuditTab />}
          {tab === 'system' && <SystemTab />}
        </div>
      </main>
    </div>
  )
}
