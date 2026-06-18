import { Shield, LogOut, Car } from 'lucide-react'
import { useAuth } from '../../App'

const ROLE_LABEL: Record<string, string> = {
  citizen: 'Citizen Portal',
  police:  'Enforcement Dashboard',
  admin:   'System Administration',
}

export default function Header() {
  const { role, plate, logout } = useAuth()
  return (
    <header className="sticky top-0 z-40 border-b border-slate-800 bg-slate-950/95 backdrop-blur">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="shrink-0 w-7 h-7 rounded bg-indigo-600 flex items-center justify-center">
            <Shield size={14} className="text-white" />
          </div>
          <div className="min-w-0">
            <span className="text-sm font-bold text-slate-100 tracking-tight">TrafficEye</span>
            {role && (
              <span className="hidden sm:inline text-slate-500 text-xs ml-2">
                {ROLE_LABEL[role]}
              </span>
            )}
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center gap-3 shrink-0">
          {role === 'citizen' && plate && (
            <div className="hidden sm:flex items-center gap-1.5 bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5">
              <Car size={13} className="text-slate-500" />
              <span className="text-xs font-mono font-semibold text-slate-300">{plate}</span>
            </div>
          )}
          {role && (
            <button
              onClick={logout}
              className="flex items-center gap-1.5 btn-ghost text-slate-500 hover:text-slate-200 px-2.5 py-1.5 text-xs"
              aria-label="Sign out"
            >
              <LogOut size={14} />
              <span className="hidden sm:inline">Exit</span>
            </button>
          )}
        </div>
      </div>
    </header>
  )
}
