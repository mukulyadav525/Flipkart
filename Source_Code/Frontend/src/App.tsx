import { createContext, useContext, useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import type { UserRole } from './lib/types'
import RoleLogin from './pages/RoleLogin'
import CitizenView from './pages/CitizenView'
import PoliceView from './pages/PoliceView'
import AdminView from './pages/AdminView'

interface AuthCtx {
  role: UserRole | null
  plate: string         // citizen only — the plate they logged in with
  login: (role: UserRole, plate?: string) => void
  logout: () => void
}

const AuthContext = createContext<AuthCtx>({
  role: null, plate: '',
  login: () => {}, logout: () => {},
})

export function useAuth() {
  return useContext(AuthContext)
}

function RequireRole({ allowed, children }: { allowed: UserRole; children: JSX.Element }) {
  const { role } = useAuth()
  if (role !== allowed) return <Navigate to="/" replace />
  return children
}

export default function App() {
  const [role, setRole] = useState<UserRole | null>(null)
  const [plate, setPlate] = useState('')

  const login = (r: UserRole, p = '') => { setRole(r); setPlate(p.toUpperCase()) }
  const logout = () => { setRole(null); setPlate('') }

  return (
    <AuthContext.Provider value={{ role, plate, login, logout }}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={role ? <Navigate to={`/${role}`} replace /> : <RoleLogin />} />
          <Route path="/citizen" element={<RequireRole allowed="citizen"><CitizenView /></RequireRole>} />
          <Route path="/police"  element={<RequireRole allowed="police"><PoliceView /></RequireRole>} />
          <Route path="/admin"   element={<RequireRole allowed="admin"><AdminView /></RequireRole>} />
          <Route path="*"        element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthContext.Provider>
  )
}
