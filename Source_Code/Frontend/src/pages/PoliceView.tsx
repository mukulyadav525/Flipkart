import { useEffect, useState } from 'react'
import {
  CheckCircle2, XCircle, RadioTower, ScanLine, RefreshCw, Filter,
} from 'lucide-react'
import Header from '../components/shared/Header'
import { ConfidenceBar } from '../components/shared/ConfidenceBar'
import { Led, Rocker, FlipNumber } from '../components/skeuo'
import {
  fetchPoliceViolations, fetchReviewQueue,
  fetchPoliceSummary, submitReview,
} from '../lib/api'
import { VIOLATION_META, formatTimestamp } from '../lib/utils'
import type {
  EvidenceRecord, AnalyticsSummary, ViolationType, DetectionBox,
} from '../lib/types'

const ALL_TYPES: ViolationType[] = [
  'helmet', 'seatbelt', 'triple_riding', 'wrong_side', 'stop_line', 'red_light', 'illegal_parking',
]

const BOX_COLOR: Record<DetectionBox['kind'], string> = {
  vehicle: '#22d3ee', rider: '#ffb000', violation: '#ff3b3b', plate: '#39ff14',
}

// ---------------------------------------------------------------------------
// Glowing bounding-box overlay on the recessed CRT screen
// ---------------------------------------------------------------------------

function EvidenceScreen({ record }: { record: EvidenceRecord }) {
  const meta = VIOLATION_META[record.violation_record.violation_type]
  const boxes = record.boxes ?? []
  const path = record.annotated_image_path
  const src = path ? (path.startsWith('http') ? path : `/images/${path.split('/').pop()}`) : ''
  const [broken, setBroken] = useState(false)
  const hasImage = !!src && !broken

  return (
    <div className="crt animate-flicker" style={{ aspectRatio: '16 / 9' }}>
      {/* Real annotated CCTV frame (boxes already drawn by the pipeline) */}
      {hasImage && (
        <img src={src} alt="evidence frame"
             className="absolute inset-0 w-full h-full object-cover"
             style={{ filter: 'saturate(0.85) contrast(1.05)' }}
             onError={() => setBroken(true)} />
      )}
      {/* faux frame grid (only when no real image) */}
      {!hasImage && (
        <div className="absolute inset-0 opacity-20"
          style={{ background: 'repeating-linear-gradient(90deg,transparent 0 39px,rgba(57,255,20,.25) 39px 40px),repeating-linear-gradient(0deg,transparent 0 39px,rgba(57,255,20,.18) 39px 40px)' }} />
      )}
      {/* scanning sweep + HUD labels */}
      <ScanLine className="absolute left-3 top-3 text-crt-green opacity-70 z-10" size={18} />
      <span className="absolute right-3 top-3 font-mono text-[10px] crt-green z-10">CAM · {record.violation_record.image_id}</span>

      {/* Vector boxes (mock data path) */}
      {boxes.map((b, i) => {
        const c = BOX_COLOR[b.kind]
        return (
          <div key={i} className="bbox z-10" style={{
            left: `${b.x * 100}%`, top: `${b.y * 100}%`,
            width: `${b.w * 100}%`, height: `${b.h * 100}%`,
            color: c, borderColor: c,
            borderStyle: b.kind === 'violation' ? 'dashed' : 'solid',
          }}>
            <span className="bbox-tag" style={{ color: '#06120a', background: c }}>{b.label}</span>
          </div>
        )
      })}

      {/* center caption only when neither image nor vector boxes exist */}
      {!hasImage && boxes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center font-mono text-sm crt-green z-10">
          {meta.label.toUpperCase()} · NO FRAME · RUN PIPELINE
        </div>
      )}
      <span className="absolute left-3 bottom-2 font-mono text-[10px] crt-amber z-10">REC ●</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Evidence review console — the Radar Desk star
// ---------------------------------------------------------------------------

function ReviewConsole() {
  const [queue, setQueue] = useState<EvidenceRecord[]>([])
  const [idx, setIdx] = useState(0)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [lastAction, setLastAction] = useState<{ id: string; action: string } | null>(null)

  useEffect(() => {
    fetchReviewQueue().then(q => { setQueue(q); setLoading(false) })
  }, [])

  const current = queue[idx]

  async function act(action: 'approved' | 'rejected') {
    if (!current) return
    setBusy(true)
    try {
      await submitReview(current.id, { action })
      setLastAction({ id: current.id, action })
      setQueue(q => q.filter(r => r.id !== current.id))
      setIdx(i => Math.max(0, Math.min(i, queue.length - 2)))
    } finally { setBusy(false) }
  }

  return (
    <section className="brushed-metal grain riveted rounded-2xl p-5 sm:p-6 relative">
      <span className="rivet" style={{ left: 14, top: 14 }} />
      <span className="rivet" style={{ right: 14, top: 14 }} />

      {/* Intake header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <RadioTower size={18} className="text-cyan-300" />
          <h3 className="font-stencil text-lg tracking-widest text-stone-100">EVIDENCE REVIEW</h3>
        </div>
        <div className="plastic raised px-3 py-1.5 rounded-md flex items-center gap-2">
          <Led state={queue.length ? 'amber' : 'green'} pulse={!!queue.length} />
          <span className="font-mono text-xs text-stone-200">
            {loading ? 'LOADING' : queue.length ? `${idx + 1} / ${queue.length}` : 'QUEUE CLEAR'}
          </span>
        </div>
      </div>

      {loading ? (
        <div className="crt rounded-xl flex items-center justify-center" style={{ aspectRatio: '16/9' }}>
          <span className="font-mono crt-green animate-flicker">INITIALISING FEED…</span>
        </div>
      ) : current ? (
        <div className="grid lg:grid-cols-5 gap-5">
          {/* Screen */}
          <div className="lg:col-span-3 recessed rounded-xl p-3" style={{ background: '#0a0c0f' }}>
            <EvidenceScreen record={current} />
            <div className="flex items-center gap-4 mt-3 px-1 flex-wrap">
              {(['vehicle', 'rider', 'violation', 'plate'] as DetectionBox['kind'][]).map(k => (
                <div key={k} className="flex items-center gap-1.5">
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: BOX_COLOR[k], boxShadow: `0 0 8px ${BOX_COLOR[k]}` }} />
                  <span className="font-mono text-[10px] uppercase text-stone-400">{k}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Readout + actions */}
          <div className="lg:col-span-2 space-y-4">
            <div className="plastic recessed rounded-xl p-4 space-y-3">
              <div>
                <p className="font-stencil text-[10px] uppercase tracking-widest text-stone-500">Classified offence</p>
                <p className="font-stencil text-xl" style={{ color: VIOLATION_META[current.violation_record.violation_type].ink === '#9b1c1c' ? '#ff8a8a' : '#ffd27a' }}>
                  {VIOLATION_META[current.violation_record.violation_type].label}
                </p>
              </div>
              <div>
                <p className="font-stencil text-[10px] uppercase tracking-widest text-stone-500 mb-1">OCR plate</p>
                <span className="plate-ind text-lg">{current.plate_text || '— — —'}</span>
                <span className="font-mono text-[10px] text-stone-500 ml-2">read {Math.round(current.plate_confidence * 100)}%</span>
              </div>
              <ConfidenceBar value={current.violation_record.confidence} />
              <div className="crt rounded-md p-2.5 max-h-28 overflow-auto">
                <p className="font-mono text-[10px] leading-relaxed crt-green">{current.violation_record.rule_trace}</p>
              </div>
              <p className="font-mono text-[10px] text-stone-500">{formatTimestamp(current.timestamp)}</p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <button className="btn-illum btn-illum-green flex items-center justify-center gap-2" disabled={busy} onClick={() => act('approved')}>
                <CheckCircle2 size={18} /> Approve
              </button>
              <button className="btn-illum btn-illum-red flex items-center justify-center gap-2" disabled={busy} onClick={() => act('rejected')}>
                <XCircle size={18} /> Reject
              </button>
            </div>
            <p className="font-mono text-[10px] text-stone-500 text-center">Approve issues the challan · Reject flags a false positive.</p>
          </div>
        </div>
      ) : (
        <div className="crt rounded-xl flex flex-col items-center justify-center gap-2" style={{ aspectRatio: '16/9' }}>
          <CheckCircle2 size={40} className="text-crt-green" />
          <span className="font-mono crt-green tracking-widest">REVIEW QUEUE CLEAR</span>
          {lastAction && <span className="font-mono text-[10px] crt-amber">LAST: {lastAction.action.toUpperCase()} · {lastAction.id}</span>}
        </div>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------
// Switchboard filter + violations log
// ---------------------------------------------------------------------------

function Switchboard() {
  const [active, setActive] = useState<Set<ViolationType>>(new Set(ALL_TYPES))
  const [records, setRecords] = useState<EvidenceRecord[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchPoliceViolations().then(d => { setRecords(d.records); setLoading(false) })
  }, [])

  function toggle(t: ViolationType) {
    setActive(prev => {
      const n = new Set(prev)
      n.has(t) ? n.delete(t) : n.add(t)
      return n
    })
  }

  const shown = records.filter(r => active.has(r.violation_record.violation_type))

  return (
    <div className="grid lg:grid-cols-5 gap-5">
      {/* Switchboard */}
      <section className="brushed-metal grain rounded-2xl p-5 lg:col-span-2">
        <div className="flex items-center gap-2 mb-4">
          <Filter size={16} className="text-cyan-300" />
          <h3 className="font-stencil text-base tracking-widest text-stone-100">VIOLATION SWITCHBOARD</h3>
        </div>
        <div className="plastic recessed rounded-xl p-4 space-y-3">
          {ALL_TYPES.map(t => (
            <div key={t} className="flex items-center justify-between gap-3 border-b border-black/30 pb-2.5 last:border-0 last:pb-0">
              <span className="font-stencil uppercase tracking-wide text-xs" style={{ color: active.has(t) ? VIOLATION_META[t].ink === '#9b1c1c' ? '#ff8a8a' : '#ffd27a' : '#6b7280' }}>
                {VIOLATION_META[t].label}
              </span>
              <Rocker on={active.has(t)} onToggle={() => toggle(t)} />
            </div>
          ))}
        </div>
        <p className="font-mono text-[10px] text-stone-500 mt-3 text-center">{shown.length} of {records.length} records shown</p>
      </section>

      {/* Log */}
      <section className="brushed-metal grain rounded-2xl p-5 lg:col-span-3">
        <h3 className="font-stencil text-base tracking-widest text-stone-100 mb-4">CONFIRMED VIOLATIONS LOG</h3>
        <div className="plastic recessed rounded-xl overflow-hidden">
          <div className="max-h-[460px] overflow-auto divide-y divide-black/40">
            {loading ? (
              <p className="font-mono text-sm text-stone-500 text-center py-12 animate-flicker">READING LEDGER…</p>
            ) : shown.length === 0 ? (
              <p className="font-mono text-sm text-stone-500 text-center py-12">NO RECORDS FOR SELECTED FILTERS</p>
            ) : shown.map(r => (
              <div key={r.id} className="flex items-center gap-3 px-4 py-3 hover:bg-white/5">
                <Led state={r.violation_record.confidence >= 0.85 ? 'green' : 'amber'} />
                <span className="plate-ind text-xs w-[110px] text-center shrink-0">{r.plate_text || '—'}</span>
                <span className="font-stencil uppercase text-[11px] tracking-wide flex-1" style={{ color: VIOLATION_META[r.violation_record.violation_type].ink === '#9b1c1c' ? '#ff8a8a' : '#d6c08a' }}>
                  {VIOLATION_META[r.violation_record.violation_type].label}
                </span>
                <span className="font-mono text-xs text-crt-green w-12 text-right" style={{ textShadow: '0 0 6px rgba(57,255,20,.5)' }}>
                  {Math.round(r.violation_record.confidence * 100)}%
                </span>
                <span className="font-mono text-[10px] text-stone-500 w-28 text-right hidden sm:block">{formatTimestamp(r.timestamp)}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Summary strip — mechanical counters
// ---------------------------------------------------------------------------

function SummaryStrip({ data }: { data: AnalyticsSummary }) {
  const stats = [
    { label: 'Today', value: data.total_today },
    { label: 'This Week', value: data.total_this_week },
    { label: 'In Review', value: data.pending_review },
    { label: 'Repeat Plates', value: data.repeat_offenders.length },
  ]
  return (
    <div className="brushed-metal grain rounded-2xl p-5 flex flex-wrap items-center justify-around gap-6">
      {stats.map(s => (
        <FlipNumber key={s.label} value={s.value} label={s.label} accent={s.label === 'In Review' ? '#ffd27a' : undefined} />
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Police view
// ---------------------------------------------------------------------------

export default function PoliceView() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null)
  const reload = () => fetchPoliceSummary().then(setSummary)
  useEffect(() => { reload() }, [])

  return (
    <div className="min-h-screen">
      <Header />
      <main className="max-w-[1400px] mx-auto px-4 sm:px-6 py-8 space-y-6">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h2 className="font-stencil text-2xl tracking-[0.12em] text-stone-100">OPERATOR · RADAR DESK</h2>
            <p className="font-mono text-xs tracking-widest text-stone-500">CITY-WIDE AUTOMATED VIOLATION MONITORING</p>
          </div>
          <button onClick={reload} className="btn-key flex items-center gap-2 text-sm"><RefreshCw size={14} /> REFRESH</button>
        </div>

        {summary && <SummaryStrip data={summary} />}
        <ReviewConsole />
        <Switchboard />
      </main>
    </div>
  )
}
