import { Power, Car } from 'lucide-react'
import { useAuth } from '../../App'
import { Led } from '../skeuo'

const ROLE_LABEL: Record<string, string> = {
  citizen: 'CITIZEN · GLOVEBOX',
  police:  'OPERATOR · RADAR DESK',
  admin:   'ADMIN · COMMAND CENTER',
}

// The header is a physical fascia whose material matches the active portal.
const ROLE_MATERIAL: Record<string, string> = {
  citizen: 'leather',
  police:  'brushed-metal',
  admin:   'mahogany',
}

export default function Header() {
  const { role, plate, logout } = useAuth()
  if (!role) return null

  return (
    <header className={`${ROLE_MATERIAL[role]} grain riveted sticky top-0 z-40`} style={{ borderRadius: 0 }}>
      <span className="rivet" style={{ left: 14, top: '50%', marginTop: -5 }} />
      <span className="rivet" style={{ right: 14, top: '50%', marginTop: -5 }} />
      <div className="max-w-[1400px] mx-auto px-6 h-16 flex items-center justify-between gap-4">
        {/* Brand badge */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="plastic raised w-9 h-9 rounded-lg flex items-center justify-center">
            <Led state={role === 'police' ? 'green' : role === 'admin' ? 'amber' : 'green'} pulse />
          </div>
          <div className="leading-tight min-w-0">
            <div className="font-stencil text-lg tracking-[0.18em] text-stone-100">TRAFFIC<span className="text-brass-light">EYE</span></div>
            <div className="font-mono text-[10px] tracking-[0.25em] text-stone-400 truncate">{ROLE_LABEL[role]}</div>
          </div>
        </div>

        {/* Right cluster */}
        <div className="flex items-center gap-4 shrink-0">
          {role === 'citizen' && plate && (
            <span className="plate-ind hidden sm:flex items-center gap-1.5 text-sm">
              <Car size={13} /> {plate}
            </span>
          )}
          <button onClick={logout} className="btn-key flex items-center gap-2 text-sm" aria-label="Exit portal">
            <Power size={15} /> <span className="hidden sm:inline">EXIT</span>
          </button>
        </div>
      </div>
    </header>
  )
}
