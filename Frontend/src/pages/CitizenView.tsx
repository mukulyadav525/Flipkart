import { useEffect, useState } from 'react'
import { Search, Camera, ChevronDown, ChevronUp, ShieldCheck } from 'lucide-react'
import Header from '../components/shared/Header'
import { Stamp } from '../components/skeuo'
import { fetchCitizenViolations } from '../lib/api'
import { VIOLATION_META, formatTimestamp, confidenceLevel } from '../lib/utils'
import type { EvidenceRecord } from '../lib/types'
import { useAuth } from '../App'

// ---------------------------------------------------------------------------
// Polaroid evidence photo, paper-clipped to the challan
// ---------------------------------------------------------------------------

function Polaroid({ record, tilt }: { record: EvidenceRecord; tilt: number }) {
  const meta = VIOLATION_META[record.violation_record.violation_type]
  const path = record.annotated_image_path
  const src = path ? (path.startsWith('http') ? path : `/images/${path.split('/').pop()}`) : ''
  const [broken, setBroken] = useState(false)

  return (
    <div className="polaroid relative shrink-0" style={{ width: 190, transform: `rotate(${tilt}deg)` }}>
      <span className="paperclip" style={{ top: -26, right: 24 }} />
      <div className="photo crt" style={{ height: 150 }}>
        {src && !broken ? (
          <img src={src} alt="evidence" className="w-full h-full object-cover" onError={() => setBroken(true)} />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-2 crt-green">
            <Camera size={26} />
            <span className="font-mono text-[10px] tracking-widest text-center px-3" style={{ color: meta.ink === '#9b1c1c' ? '#ff8a8a' : '#7dff63' }}>
              {meta.label.toUpperCase()}
            </span>
            <span className="font-mono text-[8px] text-stone-500">NO FRAME · RUN PIPELINE</span>
          </div>
        )}
      </div>
      <p className="font-type text-center text-[11px] text-stone-700 mt-2 leading-none">
        {record.violation_record.image_id}
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Paper challan
// ---------------------------------------------------------------------------

function Challan({ record, index }: { record: EvidenceRecord; index: number }) {
  const vr = record.violation_record
  const meta = VIOLATION_META[vr.violation_type]
  const level = confidenceLevel(vr.confidence)
  const verified = level === 'high'
  const [open, setOpen] = useState(false)
  const tilt = (index % 2 === 0 ? -1 : 1) * (1 + (index % 3))

  return (
    <article
      className="paper perforated-top rounded-md p-6 sm:p-8 relative animate-slide-up"
      style={{ transform: `rotate(${index % 2 === 0 ? -0.6 : 0.7}deg)` }}
    >
      {/* Stamp */}
      <div className="absolute right-6 top-16 z-10">
        <Stamp tone={verified ? 'red' : 'blue'}>
          <span className="flex flex-col items-center leading-tight">
            <span className="text-[10px]">{verified ? 'AI VERIFIED' : 'UNDER REVIEW'}</span>
            <span className="text-xl">{Math.round(vr.confidence * 100)}%</span>
            <span className="text-[8px] tracking-widest">CONFIDENCE</span>
          </span>
        </Stamp>
      </div>

      {/* Masthead */}
      <div className="border-b-2 border-dashed border-stone-500/60 pb-3 mb-4">
        <p className="font-type text-[11px] tracking-[0.2em] text-stone-600">GOVT. OF INDIA · TRAFFIC ENFORCEMENT</p>
        <h3 className="font-type text-2xl text-stone-800 tracking-tight">e-CHALLAN · NOTICE OF VIOLATION</h3>
      </div>

      {/* Fields */}
      <dl className="grid sm:grid-cols-2 gap-x-8 gap-y-3 font-type text-stone-800">
        <div>
          <dt className="text-[10px] uppercase tracking-widest text-stone-500">Offence</dt>
          <dd className="text-lg" style={{ color: meta.ink }}>{meta.label}</dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase tracking-widest text-stone-500">Registration (OCR)</dt>
          <dd><span className="plate-ind text-base mt-1 inline-block">{record.plate_text || '— — —'}</span>
            <span className="text-[10px] text-stone-500 ml-2">read {Math.round(record.plate_confidence * 100)}%</span>
          </dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase tracking-widest text-stone-500">Date &amp; Time</dt>
          <dd className="text-sm">{formatTimestamp(record.timestamp)}</dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase tracking-widest text-stone-500">Evidence Ref.</dt>
          <dd className="text-sm">{record.id} · {vr.image_id}</dd>
        </div>
      </dl>

      {/* Polaroid clipped to the right margin on wide screens */}
      <div className="mt-5 flex flex-col sm:flex-row gap-6 items-start">
        <div className="flex-1 space-y-3">
          <div className="bg-stone-900/5 border border-stone-400/50 rounded p-3">
            <p className="text-[10px] uppercase tracking-widest text-stone-500 mb-1">Plain-language reason</p>
            <p className="font-type text-[13px] text-stone-700 leading-relaxed">{meta.citizen.explanation}</p>
          </div>
          <div className="bg-stone-900/5 border border-stone-400/50 rounded p-3">
            <p className="text-[10px] uppercase tracking-widest text-stone-500 mb-1">Applicable law</p>
            <p className="font-type text-[12px] text-stone-700 leading-relaxed">{meta.citizen.law}</p>
          </div>

          <button onClick={() => setOpen(o => !o)} className="flex items-center gap-1.5 font-type text-[12px] text-stone-600 hover:text-stone-900">
            {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />} AI detection trace
          </button>
          {open && (
            <pre className="font-mono text-[11px] text-stone-700 bg-stone-900/5 border border-stone-400/50 rounded p-3 whitespace-pre-wrap leading-relaxed animate-fade-in">
              {vr.rule_trace}
            </pre>
          )}
        </div>
        <Polaroid record={record} tilt={tilt} />
      </div>

      {/* Footer */}
      <div className="mt-6 pt-3 border-t-2 border-dashed border-stone-500/60 flex items-center justify-between flex-wrap gap-2">
        <p className="font-type text-[11px] text-stone-500">Notices below 85% confidence are reviewed by an officer before any fine.</p>
        <a href="mailto:traffic.dispute@example.gov.in" className="font-type text-[12px] text-ink-blue underline underline-offset-2">
          Dispute this notice →
        </a>
      </div>
    </article>
  )
}

// ---------------------------------------------------------------------------
// Citizen view
// ---------------------------------------------------------------------------

export default function CitizenView() {
  const { plate: loginPlate } = useAuth()
  const [query, setQuery] = useState(loginPlate)
  const [records, setRecords] = useState<EvidenceRecord[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)

  async function doSearch(q: string) {
    const clean = q.trim().toUpperCase()
    if (!clean) return
    setLoading(true); setSearched(true)
    try { setRecords(await fetchCitizenViolations(clean)) }
    finally { setLoading(false) }
  }

  useEffect(() => { if (loginPlate) doSearch(loginPlate) /* eslint-disable-line */ }, [])

  return (
    <div className="min-h-screen">
      <Header />
      {/* Wooden desk surround */}
      <div className="leather grain min-h-[calc(100vh-4rem)]" style={{ borderRadius: 0 }}>
        <main className="max-w-4xl mx-auto px-4 sm:px-6 py-10 space-y-8">
          {/* Folder header */}
          <div className="text-center space-y-1">
            <h2 className="font-stencil text-3xl tracking-[0.15em] text-stone-100">VEHICLE GLOVEBOX</h2>
            <p className="font-mono text-xs tracking-[0.2em] text-brass-light/80">YOUR VIOLATION RECORD · OPEN THE FOLDER</p>
          </div>

          {/* Embossed plate search */}
          <div className="stitched leather grain rounded-2xl p-6">
            <label className="font-stencil uppercase tracking-widest text-[11px] text-brass-light block mb-3 text-center">
              Insert registration plate
            </label>
            <div className="flex flex-wrap gap-4 items-center justify-center">
              <div className="relative">
                <span className="absolute -left-1 -top-3 font-mono text-[9px] tracking-widest text-stone-400">IND</span>
                <input
                  className="plate-ind text-2xl sm:text-3xl bg-plate-yellow outline-none w-[280px] text-center uppercase"
                  placeholder="MH12AB1234"
                  value={query}
                  maxLength={14}
                  onChange={e => setQuery(e.target.value.toUpperCase())}
                  onKeyDown={e => e.key === 'Enter' && doSearch(query)}
                />
              </div>
              <button className="btn-brass flex items-center gap-2" onClick={() => doSearch(query)} disabled={loading}>
                <Search size={16} /> {loading ? 'Reading…' : 'Search'}
              </button>
            </div>
            <p className="text-center font-type text-[11px] text-stone-300/70 mt-3">
              You may only open the glovebox for your own vehicle.
            </p>
          </div>

          {/* Results */}
          {loading && <p className="text-center font-mono text-sm text-brass-light animate-flicker">SCANNING RECORDS…</p>}

          {searched && !loading && records && (
            records.length > 0 ? (
              <div className="space-y-10">
                <p className="text-center font-type text-stone-200">
                  {records.length} notice{records.length !== 1 ? 's' : ''} on file for{' '}
                  <span className="plate-ind text-sm inline-block">{query.toUpperCase()}</span>
                </p>
                {records.map((r, i) => <Challan key={r.id} record={r} index={i} />)}
              </div>
            ) : (
              <div className="paper rounded-md p-10 text-center" style={{ transform: 'rotate(-0.5deg)' }}>
                <ShieldCheck size={42} className="mx-auto text-emerald-700 mb-3" />
                <p className="font-type text-2xl text-stone-800">Folder is empty</p>
                <p className="font-type text-sm text-stone-600 mt-1">
                  No confirmed violations on record for{' '}
                  <span className="plate-ind text-xs inline-block">{query.toUpperCase()}</span>.
                </p>
              </div>
            )
          )}
        </main>
      </div>
    </div>
  )
}
