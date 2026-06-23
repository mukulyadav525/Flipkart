import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Car, ChevronRight, AlertTriangle } from 'lucide-react'
import type { UserRole } from '../lib/types'
import { useAuth } from '../App'
import { Led } from '../components/skeuo'

const ROLES: { value: UserRole; label: string; sub: string; material: string; vibe: string }[] = [
  { value: 'citizen', label: 'Citizen',  sub: 'The Glovebox',      material: 'leather',       vibe: 'Look up your own vehicle and read the evidence behind each challan.' },
  { value: 'police',  label: 'Operator',  sub: 'The Radar Desk',    material: 'brushed-metal', vibe: 'Review processed footage, approve evidence, filter live violations.' },
  { value: 'admin',   label: 'Admin',     sub: 'The Command Center', material: 'mahogany',      vibe: 'System dials, the violation ledger and AI / OCR / DB server health.' },
]

export default function RoleLogin() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [role, setRole] = useState<UserRole>('citizen')
  const [plate, setPlate] = useState('')
  const [err, setErr] = useState('')

  function enter() {
    if (role === 'citizen') {
      const clean = plate.trim().toUpperCase()
      if (!clean) { setErr('Insert your registration plate to open the glovebox.'); return }
      if (!/^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$/.test(clean) && !/^[A-Z0-9]{4,12}$/.test(clean)) {
        setErr('Enter a valid Indian plate, e.g. MH12AB1234'); return
      }
      login(role, clean); navigate('/citizen')
    } else {
      login(role); navigate(`/${role}`)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4 sm:p-8">
      <div className="brushed-metal grain riveted w-full max-w-3xl rounded-2xl p-6 sm:p-10 relative animate-fade-in">
        <span className="rivet" style={{ left: 18, top: 18 }} />
        <span className="rivet" style={{ right: 18, top: 18 }} />
        <span className="rivet" style={{ left: 18, bottom: 18 }} />
        <span className="rivet" style={{ right: 18, bottom: 18 }} />

        {/* Title plate */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-3 plastic raised px-5 py-2 rounded-lg mb-3">
            <Led state="green" pulse />
            <span className="font-stencil text-2xl tracking-[0.2em] text-stone-100">TRAFFIC<span className="text-brass-light">EYE</span></span>
          </div>
          <p className="font-mono text-xs tracking-[0.3em] text-stone-400">AUTOMATED VIOLATION CONTROL · SELECT STATION</p>
        </div>

        {/* Station tiles */}
        <div className="grid sm:grid-cols-3 gap-4 mb-8">
          {ROLES.map(r => {
            const active = role === r.value
            return (
              <button
                key={r.value}
                onClick={() => { setRole(r.value); setErr('') }}
                className={`${r.material} grain text-left rounded-xl p-4 relative transition-transform ${active ? 'raised -translate-y-1' : 'recessed opacity-80 hover:opacity-100'}`}
                style={{ minHeight: 150 }}
              >
                <div className="flex items-center justify-between mb-2">
                  <Led state={active ? 'green' : 'off'} pulse={active} />
                  {active && <ChevronRight size={16} className="text-brass-light" />}
                </div>
                <div className="font-stencil text-xl tracking-wide text-stone-100">{r.label}</div>
                <div className="font-mono text-[10px] tracking-widest text-brass-light mb-2">{r.sub.toUpperCase()}</div>
                <p className="text-[11px] leading-snug text-stone-300/80">{r.vibe}</p>
              </button>
            )
          })}
        </div>

        {/* Citizen plate input */}
        {role === 'citizen' && (
          <div className="mb-6 animate-fade-in">
            <label className="font-stencil uppercase tracking-widest text-[11px] text-stone-400 block mb-2">
              Vehicle registration plate
            </label>
            <div className="flex flex-wrap gap-3 items-center">
              <div className="relative">
                <span className="absolute -left-1 -top-3 font-mono text-[9px] tracking-widest text-stone-500">IND</span>
                <input
                  className="plate-ind text-2xl bg-plate-yellow outline-none w-[260px] text-center"
                  style={{ caretColor: '#15171b' }}
                  placeholder="MH12AB1234"
                  value={plate}
                  maxLength={14}
                  autoFocus
                  onChange={e => { setPlate(e.target.value.toUpperCase()); setErr('') }}
                  onKeyDown={e => e.key === 'Enter' && enter()}
                />
              </div>
              <Car size={20} className="text-stone-400" />
            </div>
            {err && (
              <p className="mt-2 flex items-center gap-1.5 text-xs text-red-400 font-mono">
                <AlertTriangle size={12} /> {err}
              </p>
            )}
          </div>
        )}

        <div className="flex items-center justify-between gap-4 flex-wrap">
          <p className="font-mono text-[10px] text-stone-500 max-w-sm leading-relaxed">
            PROTOTYPE · no authentication. Real deployment requires officer credentials
            and citizen plate / Aadhaar verification.
          </p>
          <button onClick={enter} className="btn-illum btn-illum-green text-lg">
            Engage ▸
          </button>
        </div>
      </div>
    </div>
  )
}
