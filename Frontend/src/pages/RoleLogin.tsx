import { useState } from 'react'
import { Shield, AlertCircle, Car, ChevronRight } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import type { UserRole } from '../lib/types'
import { useAuth } from '../App'

const ROLES: { value: UserRole; label: string; description: string }[] = [
  {
    value: 'citizen',
    label: 'Citizen',
    description: "Check your own vehicle's violation record and understand the evidence behind each notice.",
  },
  {
    value: 'police',
    label: 'Police / Enforcement',
    description: 'Full city-wide violation dashboard, human review queue, and repeat-offender lookup.',
  },
  {
    value: 'admin',
    label: 'System Administrator',
    description: 'Model performance metrics, audit log, and dataset / version information.',
  },
]

export default function RoleLogin() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [role, setRole] = useState<UserRole>('citizen')
  const [plate, setPlate] = useState('')
  const [plateErr, setPlateErr] = useState('')

  const selected = ROLES.find(r => r.value === role)!

  function handleEnter() {
    if (role === 'citizen') {
      const clean = plate.trim().toUpperCase()
      if (!clean) { setPlateErr('Enter your vehicle registration number to continue.'); return }
      if (!/^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$/.test(clean) && !/^[A-Z0-9]{4,12}$/.test(clean)) {
        setPlateErr('Enter a valid Indian plate number, e.g. MH12AB1234')
        return
      }
      login(role, clean)
      navigate('/citizen')
    } else {
      login(role)
      navigate(`/${role}`)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center px-4 py-12">
      {/* Brand block */}
      <div className="mb-10 text-center space-y-3">
        <div className="mx-auto w-14 h-14 rounded-2xl bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-900/50">
          <Shield size={28} className="text-white" />
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-100">TrafficEye</h1>
        <p className="text-sm text-slate-500 max-w-xs mx-auto leading-relaxed">
          Traffic violation transparency and enforcement portal
        </p>
      </div>

      {/* Login card */}
      <div className="card w-full max-w-md p-8 space-y-6 animate-slide-up">
        {/* Role selector */}
        <div className="space-y-2">
          <label className="block text-xs font-semibold uppercase tracking-widest text-slate-500">
            Select your role
          </label>
          <div className="space-y-2">
            {ROLES.map(r => (
              <button
                key={r.value}
                onClick={() => { setRole(r.value); setPlateErr('') }}
                className={`w-full text-left px-4 py-3.5 rounded-xl border transition-all ${
                  role === r.value
                    ? 'border-indigo-600 bg-indigo-950/60 text-slate-100'
                    : 'border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-300'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <p className="text-sm font-semibold">{r.label}</p>
                    <p className={`text-xs mt-0.5 leading-relaxed ${role === r.value ? 'text-slate-400' : 'text-slate-600'}`}>
                      {r.description}
                    </p>
                  </div>
                  {role === r.value && (
                    <div className="shrink-0 w-5 h-5 rounded-full bg-indigo-600 flex items-center justify-center">
                      <ChevronRight size={11} className="text-white" />
                    </div>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Plate input — citizen only */}
        {role === 'citizen' && (
          <div className="space-y-2 animate-fade-in">
            <label className="block text-xs font-semibold uppercase tracking-widest text-slate-500">
              Your vehicle registration number
            </label>
            <div className="relative">
              <Car size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                className="input pl-9 font-mono uppercase"
                placeholder="e.g. MH12AB1234"
                value={plate}
                onChange={e => { setPlate(e.target.value); setPlateErr('') }}
                onKeyDown={e => e.key === 'Enter' && handleEnter()}
                maxLength={14}
                autoFocus
              />
            </div>
            {plateErr && (
              <p className="flex items-center gap-1.5 text-xs text-red-400">
                <AlertCircle size={12} /> {plateErr}
              </p>
            )}
            <p className="text-xs text-slate-600 leading-relaxed">
              You can only view records associated with your own vehicle. This portal
              does not allow searching other plates.
            </p>
          </div>
        )}

        <button className="btn-primary w-full" onClick={handleEnter}>
          Enter Portal
        </button>

        {/* Prototype notice */}
        <p className="text-center text-xs text-slate-700 leading-relaxed border-t border-slate-800 pt-4">
          Prototype — no authentication. Role selection is a stub.
          Real deployment requires officer credentials and citizen Aadhaar / plate verification.
        </p>
      </div>
    </div>
  )
}
