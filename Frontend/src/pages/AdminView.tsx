import { useEffect, useState } from 'react'
import {
  Gauge, BookOpen, Server, RefreshCw, Cpu, ScanText, Database, HardDrive, Search,
} from 'lucide-react'
import Header from '../components/shared/Header'
import { Dial, Led, FlipNumber } from '../components/skeuo'
import { fetchAdminMetrics, fetchAuditLog, fetchSystemInfo, fetchPoliceSummary } from '../lib/api'
import { VIOLATION_META, formatTimestamp } from '../lib/utils'
import type { AuditEntry, EvalMetric, SystemInfo, AnalyticsSummary } from '../lib/types'

type Metrics = Awaited<ReturnType<typeof fetchAdminMetrics>>

// ---------------------------------------------------------------------------
// Macro-average of an eval-metric column across all violation types
// ---------------------------------------------------------------------------
function macro(metrics: EvalMetric[] | null, key: keyof EvalMetric): number | null {
  if (!metrics) return null
  const vals = metrics.map(m => parseFloat(String(m[key]))).filter(v => !Number.isNaN(v))
  if (!vals.length) return null
  return vals.reduce((a, b) => a + b, 0) / vals.length
}

// ---------------------------------------------------------------------------
// Gauge cluster
// ---------------------------------------------------------------------------
function GaugePanel({ metrics }: { metrics: Metrics }) {
  const em = metrics.eval_metrics
  const dials: { label: string; value: number | null }[] = [
    { label: 'Accuracy',  value: macro(em, 'accuracy') },
    { label: 'Precision', value: macro(em, 'precision') },
    { label: 'Recall',    value: macro(em, 'recall') },
    { label: 'F1 Score',  value: macro(em, 'f1') },
    { label: 'mAP@.5',    value: macro(em, 'average_precision') },
  ]
  return (
    <section className="mahogany grain brass-edge rounded-2xl p-6">
      <div className="flex items-center gap-2 mb-5">
        <Gauge size={18} className="text-brass-light" />
        <h3 className="font-serif text-xl text-brass-light tracking-wide">Model Performance Gauges</h3>
      </div>
      <div className="flex flex-wrap justify-around gap-6">
        {dials.map(d => (
          <Dial key={d.label} label={d.label} value={d.value ?? 0}
                display={d.value == null ? '--' : `${Math.round(d.value * 100)}`} unit="%" />
        ))}
      </div>
      {metrics.latency && (
        <div className="mt-6 pt-5 border-t border-brass-dark/40 flex flex-wrap items-center justify-around gap-6">
          <FlipNumber value={metrics.latency.mean_ms.toFixed(1)} label="Mean ms" />
          <FlipNumber value={metrics.latency.p95_ms.toFixed(1)} label="P95 ms" />
          <FlipNumber value={metrics.latency.throughput_fps.toFixed(1)} label="FPS" accent="#5dff8f" />
          <FlipNumber value={metrics.latency.n_images} label="Frames" />
        </div>
      )}
      {!metrics.data_available.eval && (
        <p className="font-mono text-[11px] text-amber-300/80 text-center mt-4">
          Showing calibration figures · run <span className="text-brass-light">python -m evaluation.evaluate</span> on a labelled set for live numbers.
        </p>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------
// Server status rack
// ---------------------------------------------------------------------------
function ServerRack({ info }: { info: SystemInfo | null }) {
  const nodes = [
    { icon: Cpu,      label: 'AI Detection Model', state: (info?.weights_exist ? 'green' : 'amber') as 'green' | 'amber',
      detail: info ? (info.weights_exist ? 'Fine-tuned weights loaded' : 'COCO fallback weights') : '…' },
    { icon: ScanText, label: 'OCR / ANPR Engine',  state: 'green' as const, detail: 'EasyOCR · en' },
    { icon: Database, label: 'Records Database',    state: 'green' as const, detail: `${info?.confirmed_records ?? 0} confirmed` },
    { icon: HardDrive,label: 'Review Queue',        state: ((info?.pending_review ?? 0) > 0 ? 'amber' : 'green') as 'green' | 'amber',
      detail: `${info?.pending_review ?? 0} pending` },
  ]
  return (
    <section className="brushed-metal grain riveted rounded-2xl p-6 relative">
      <span className="rivet" style={{ left: 14, top: 14 }} />
      <span className="rivet" style={{ right: 14, top: 14 }} />
      <div className="flex items-center gap-2 mb-5">
        <Server size={18} className="text-cyan-300" />
        <h3 className="font-stencil text-lg tracking-widest text-stone-100">SYSTEM STATUS</h3>
      </div>
      <div className="space-y-3">
        {nodes.map(n => (
          <div key={n.label} className="plastic recessed rounded-lg px-4 py-3 flex items-center gap-3">
            <n.icon size={16} className="text-stone-400 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="font-stencil uppercase tracking-wide text-xs text-stone-200">{n.label}</p>
              <p className="font-mono text-[10px] text-stone-500 truncate">{n.detail}</p>
            </div>
            <Led state={n.state} pulse={n.state !== 'green'} />
          </div>
        ))}
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Statistics + ledger
// ---------------------------------------------------------------------------
function StatsStrip({ summary }: { summary: AnalyticsSummary }) {
  const max = Math.max(...summary.severity_ranking.map(r => r.severity_score), 1)
  return (
    <div className="space-y-2">
      {summary.severity_ranking.filter(r => r.count > 0).map(r => {
        const meta = VIOLATION_META[r.violation_type]
        return (
          <div key={r.violation_type} className="flex items-center gap-3">
            <span className="font-type text-xs w-28 shrink-0" style={{ color: meta.ink }}>{meta.label}</span>
            <div className="flex-1 h-3 rounded-full recessed overflow-hidden" style={{ background: '#1a0d08' }}>
              <div className="h-full rounded-full" style={{ width: `${(r.severity_score / max) * 100}%`, background: `linear-gradient(90deg,${meta.ink},${meta.ink}aa)`, boxShadow: `0 0 8px ${meta.ink}` }} />
            </div>
            <span className="font-mono text-[11px] text-brass-light w-20 text-right">{r.count}× · {r.severity_score.toFixed(1)}</span>
          </div>
        )
      })}
    </div>
  )
}

function Ledger({ entries, summary }: { entries: AuditEntry[]; summary: AnalyticsSummary | null }) {
  const [q, setQ] = useState('')
  const filtered = entries.filter(e =>
    !q || e.plate_text.toLowerCase().includes(q.toLowerCase()) ||
    e.violation_type.toLowerCase().includes(q.toLowerCase()) ||
    e.action.toLowerCase().includes(q.toLowerCase())
  )
  return (
    <section className="mahogany grain brass-edge rounded-2xl p-6">
      <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
        <div className="flex items-center gap-2">
          <BookOpen size={18} className="text-brass-light" />
          <h3 className="font-serif text-xl text-brass-light tracking-wide">Enforcement Ledger</h3>
        </div>
        <div className="relative">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-500" />
          <input
            className="plastic recessed rounded-md pl-8 pr-3 py-1.5 font-mono text-xs text-stone-200 placeholder-stone-600 outline-none w-56"
            placeholder="search plate / type…" value={q} onChange={e => setQ(e.target.value)}
          />
        </div>
      </div>

      {summary && (
        <div className="mb-5">
          <p className="font-type text-[11px] uppercase tracking-widest text-brass-light/70 mb-2">Severity-weighted statistics</p>
          <StatsStrip summary={summary} />
        </div>
      )}

      {/* Logbook — lined ledger paper */}
      <div className="paper rounded-md overflow-hidden">
        <div className="grid grid-cols-12 gap-2 px-4 py-2 border-b-2 border-dashed border-stone-500/60 font-type text-[10px] uppercase tracking-widest text-stone-600">
          <span className="col-span-2">Action</span>
          <span className="col-span-3">Plate</span>
          <span className="col-span-3">Offence</span>
          <span className="col-span-2">Officer</span>
          <span className="col-span-2 text-right">When</span>
        </div>
        <div className="max-h-[360px] overflow-auto">
          {filtered.length === 0 ? (
            <p className="font-type text-sm text-stone-600 text-center py-10">No ledger entries.</p>
          ) : filtered.map(e => (
            <div key={e.id} className="grid grid-cols-12 gap-2 px-4 py-2.5 items-center" style={{ minHeight: 44 }}>
              <span className="col-span-2 flex items-center gap-1.5">
                <Led state={e.action === 'approved' ? 'green' : 'red'} />
                <span className="font-type text-[12px] text-stone-700 capitalize hidden sm:inline">{e.action}</span>
              </span>
              <span className="col-span-3"><span className="plate-ind text-[11px]">{e.plate_text || '—'}</span></span>
              <span className="col-span-3 font-type text-[12px]" style={{ color: VIOLATION_META[e.violation_type]?.ink ?? '#2a2118' }}>
                {VIOLATION_META[e.violation_type]?.label ?? e.violation_type}
              </span>
              <span className="col-span-2 font-mono text-[10px] text-stone-600 truncate">{e.reviewer_id}</span>
              <span className="col-span-2 font-mono text-[10px] text-stone-600 text-right">{formatTimestamp(e.timestamp)}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Admin view
// ---------------------------------------------------------------------------
export default function AdminView() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [audit, setAudit] = useState<AuditEntry[]>([])
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null)

  function reload() {
    fetchAdminMetrics().then(setMetrics)
    fetchAuditLog().then(d => setAudit(d.entries))
    fetchSystemInfo().then(setInfo)
    fetchPoliceSummary().then(setSummary)
  }
  useEffect(() => { reload() }, [])

  return (
    <div className="min-h-screen">
      <Header />
      <main className="max-w-[1400px] mx-auto px-4 sm:px-6 py-8 space-y-6">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h2 className="font-serif text-3xl text-brass-light tracking-wide">Command Center</h2>
            <p className="font-mono text-xs tracking-widest text-stone-500">SYSTEM TELEMETRY · MODEL HEALTH · ENFORCEMENT LEDGER</p>
          </div>
          <button onClick={reload} className="btn-brass flex items-center gap-2 text-sm"><RefreshCw size={14} /> Refresh</button>
        </div>

        {metrics && <GaugePanel metrics={metrics} />}

        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1"><ServerRack info={info} /></div>
          <div className="lg:col-span-2"><Ledger entries={audit} summary={summary} /></div>
        </div>

        {/* Dataset & model brass plate */}
        {info && (
          <section className="mahogany grain brass-edge rounded-2xl p-6">
            <h3 className="font-serif text-xl text-brass-light tracking-wide mb-4">Dataset &amp; Model Registry</h3>
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {[
                ['Model', info.model_name],
                ['Weights', info.weights_exist ? 'fine-tuned · ready' : 'COCO fallback'],
                ['Dataset', `${info.dataset_name} (${info.dataset_version})`],
                ['Last trained', info.last_trained],
                ['Pipeline', `v${info.pipeline_version}`],
                ['Confirmed records', String(info.confirmed_records)],
              ].map(([k, v]) => (
                <div key={k} className="plastic recessed rounded-lg px-4 py-3">
                  <p className="font-stencil uppercase tracking-widest text-[10px] text-stone-500">{k}</p>
                  <p className="font-mono text-sm text-stone-200 break-words mt-0.5">{v}</p>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  )
}
