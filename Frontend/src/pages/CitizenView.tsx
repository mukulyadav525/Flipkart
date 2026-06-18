import { useState } from 'react'
import { Search, Camera, FileText, AlertCircle, CheckCircle2, Car, ChevronDown, ChevronUp } from 'lucide-react'
import Header from '../components/shared/Header'
import { ConfidenceBar } from '../components/shared/ConfidenceBar'
import { ViolationBadge } from '../components/shared/StatusBadge'
import { fetchCitizenViolations } from '../lib/api'
import { VIOLATION_META, formatTimestamp, confidenceLevel, CONFIDENCE_LABEL } from '../lib/utils'
import type { EvidenceRecord } from '../lib/types'
import { useAuth } from '../App'

function EvidenceImage({ path }: { path: string }) {
  if (!path) {
    return (
      <div className="w-full aspect-video bg-slate-800 rounded-xl flex flex-col items-center justify-center gap-2 border border-slate-700">
        <Camera size={28} className="text-slate-600" />
        <p className="text-xs text-slate-600">Evidence image not yet available</p>
        <p className="text-xs text-slate-700">Run the detection pipeline to generate annotated images</p>
      </div>
    )
  }
  const src = path.startsWith('http') ? path : `/images/${path.split('/').pop()}`
  return (
    <img
      src={src}
      alt="Annotated evidence"
      className="w-full rounded-xl border border-slate-800 object-cover"
      onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
    />
  )
}

function RuleTraceExpander({ trace }: { trace: string }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors"
      >
        <FileText size={12} />
        <span>View technical detection trace</span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      {open && (
        <pre className="mt-2 p-3 bg-slate-800 border border-slate-700 rounded-lg text-xs text-slate-400 font-mono whitespace-pre-wrap leading-relaxed animate-fade-in">
          {trace}
        </pre>
      )}
    </div>
  )
}

function ViolationCard({ record }: { record: EvidenceRecord }) {
  const vr = record.violation_record
  const meta = VIOLATION_META[vr.violation_type]
  const level = confidenceLevel(vr.confidence)

  return (
    <article className={`card overflow-hidden animate-slide-up`}>
      {/* Violation type strip */}
      <div className={`px-5 py-3 border-b border-slate-800 ${meta.bgClass} flex items-center justify-between gap-3`}>
        <ViolationBadge type={vr.violation_type} />
        <time className="text-xs text-slate-500 tabular-nums">{formatTimestamp(record.timestamp)}</time>
      </div>

      <div className="p-5 space-y-5">
        {/* Grid: image + details */}
        <div className="grid sm:grid-cols-5 gap-5">
          {/* Evidence image */}
          <div className="sm:col-span-2">
            <EvidenceImage path={record.annotated_image_path} />
            {record.plate_text && (
              <div className="mt-2 flex items-center gap-2">
                <Car size={13} className="text-slate-600 shrink-0" />
                <span className="font-mono text-sm text-slate-300 font-semibold">{record.plate_text}</span>
                <span className="text-xs text-slate-600">
                  (plate read confidence: {Math.round(record.plate_confidence * 100)}%)
                </span>
              </div>
            )}
          </div>

          {/* Plain-language explanation */}
          <div className="sm:col-span-3 space-y-4">
            <div>
              <h3 className="text-base font-bold text-slate-100 mb-1">{meta.citizen.heading}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{meta.citizen.explanation}</p>
            </div>

            {/* Applicable law */}
            <div className="p-3 bg-slate-800/60 border border-slate-700 rounded-lg">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">Applicable Law</p>
              <p className="text-xs text-slate-400 leading-relaxed">{meta.citizen.law}</p>
            </div>

            {/* Confidence */}
            <div className="space-y-1.5">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">System Confidence</p>
              <ConfidenceBar value={vr.confidence} />
              {level !== 'high' && (
                <p className="text-xs text-amber-500 flex items-center gap-1.5">
                  <AlertCircle size={11} />
                  {CONFIDENCE_LABEL[level]} — a human officer will review this record.
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Technical trace — collapsible */}
        <RuleTraceExpander trace={vr.rule_trace} />
      </div>

      {/* Footer — dispute CTA */}
      <div className="px-5 py-3 bg-slate-800/30 border-t border-slate-800 flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs text-slate-600">
          Image ID: <span className="font-mono text-slate-500">{vr.image_id}</span>
        </p>
        <a
          href="mailto:traffic.dispute@example.gov.in"
          className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors underline underline-offset-2"
        >
          Dispute this notice →
        </a>
      </div>
    </article>
  )
}

export default function CitizenView() {
  const { plate: loginPlate } = useAuth()
  const [query, setQuery] = useState(loginPlate)
  const [records, setRecords] = useState<EvidenceRecord[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(!!loginPlate)

  async function doSearch(q: string) {
    const clean = q.trim().toUpperCase()
    if (!clean) return
    setLoading(true)
    setSearched(true)
    try {
      const data = await fetchCitizenViolations(clean)
      setRecords(data)
    } finally {
      setLoading(false)
    }
  }

  // Auto-search on mount if we have a plate from login
  useState(() => { if (loginPlate) doSearch(loginPlate) })

  return (
    <div className="min-h-screen bg-slate-950">
      <Header />

      <main className="max-w-3xl mx-auto px-4 sm:px-6 py-8 space-y-8">
        {/* Page heading */}
        <div className="space-y-1">
          <h2 className="text-2xl font-bold text-slate-100">Your Violation Record</h2>
          <p className="text-sm text-slate-500">
            View notices issued to your vehicle, the evidence behind each, and your legal rights.
          </p>
        </div>

        {/* Search bar */}
        <div className="card p-5 space-y-3">
          <label className="block text-xs font-semibold uppercase tracking-widest text-slate-500">
            Check your vehicle's record
          </label>
          <div className="flex gap-3">
            <div className="relative flex-1">
              <Car size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
              <input
                className="input pl-9 font-mono uppercase"
                placeholder="MH12AB1234"
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && doSearch(query)}
                maxLength={14}
              />
            </div>
            <button
              className="btn-primary flex items-center gap-2 shrink-0"
              onClick={() => doSearch(query)}
              disabled={loading}
            >
              <Search size={15} />
              <span>{loading ? 'Searching…' : 'Search'}</span>
            </button>
          </div>
          <p className="text-xs text-slate-700">
            You can only look up your own vehicle's records. This portal does not allow
            searching other plates.
          </p>
        </div>

        {/* Results */}
        {searched && !loading && (
          <>
            {records && records.length > 0 ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-400">
                    {records.length} violation notice{records.length !== 1 ? 's' : ''} found for{' '}
                    <span className="font-mono text-slate-200">{query.toUpperCase()}</span>
                  </h3>
                </div>
                {records.map(r => <ViolationCard key={r.id} record={r} />)}

                {/* Rights footer */}
                <div className="card p-5 space-y-2 border-indigo-900/50">
                  <div className="flex items-center gap-2 text-indigo-400">
                    <CheckCircle2 size={15} />
                    <span className="text-sm font-semibold">Your rights</span>
                  </div>
                  <ul className="text-xs text-slate-500 leading-relaxed space-y-1 list-disc list-inside">
                    <li>You have the right to dispute any notice within 30 days of issue.</li>
                    <li>Human-review records are reviewed by a trained officer before any fine is issued.</li>
                    <li>Notices below 85% system confidence are always reviewed by a human first.</li>
                    <li>Contact your local traffic authority or use the dispute link on each notice.</li>
                  </ul>
                </div>
              </div>
            ) : (
              <div className="card p-12 flex flex-col items-center gap-4 text-center">
                <CheckCircle2 size={40} className="text-emerald-600" />
                <div>
                  <p className="text-lg font-bold text-slate-200">No violations found</p>
                  <p className="text-sm text-slate-500 mt-1">
                    No confirmed violations are on record for{' '}
                    <span className="font-mono text-slate-300">{query.toUpperCase()}</span>.
                  </p>
                </div>
                <p className="text-xs text-slate-600 max-w-xs">
                  If you believe you received a notice in error, contact your local traffic authority.
                </p>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}
