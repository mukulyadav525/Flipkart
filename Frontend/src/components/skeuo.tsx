/**
 * Reusable skeuomorphic primitives shared across the three portals:
 * analog dial gauges, LED indicators, rocker toggle switches, mechanical
 * flip-clock numerals and rubber ink stamps.
 */
import type { ReactNode } from 'react'

// ---------------------------------------------------------------------------
// Analog dial / gauge — physical needle sweeping a 240° arc
// ---------------------------------------------------------------------------

export function Dial({
  value, label, display, size = 150, unit = '',
}: {
  value: number          // 0..1
  label: string
  display?: string
  size?: number
  unit?: string
}) {
  const clamped = Math.max(0, Math.min(1, value))
  const SWEEP = 240
  const angle = -SWEEP / 2 + clamped * SWEEP
  const ticks = Array.from({ length: 9 }, (_, i) => -SWEEP / 2 + (i / 8) * SWEEP)
  const needleLen = size * 0.36

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="dial grain" style={{ width: size, height: size }}>
        {/* tick marks */}
        {ticks.map((t, i) => (
          <div
            key={i}
            className="absolute left-1/2 top-1/2"
            style={{ transform: `rotate(${t}deg) translateY(-${size / 2 - 12}px)` }}
          >
            <div
              style={{
                width: i % 2 === 0 ? 3 : 2,
                height: i % 2 === 0 ? 10 : 6,
                background: i >= 6 ? '#ff6a4a' : '#9aa0a8',
                borderRadius: 2,
                boxShadow: i >= 6 ? '0 0 6px rgba(255,90,60,.7)' : 'none',
              }}
            />
          </div>
        ))}
        {/* needle */}
        <div
          className="dial-needle animate-needle"
          style={{ height: needleLen, transform: `translateX(-50%) rotate(${angle}deg)` }}
        />
        <div className="dial-cap" />
        {/* digital readout inset in the face */}
        <div
          className="absolute left-1/2 -translate-x-1/2 font-mono text-crt-green"
          style={{ bottom: size * 0.2, textShadow: '0 0 6px rgba(57,255,20,.6)', fontSize: size * 0.12 }}
        >
          {display ?? `${Math.round(clamped * 100)}${unit}`}
        </div>
      </div>
      <span className="font-stencil uppercase tracking-widest text-[11px] etched">{label}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// LED indicator
// ---------------------------------------------------------------------------

export function Led({
  state, label, pulse = false,
}: {
  state: 'green' | 'amber' | 'red' | 'off'
  label?: string
  pulse?: boolean
}) {
  return (
    <div className="flex items-center gap-2">
      <span className={`led led-${state} ${pulse && state !== 'off' ? 'animate-led-pulse' : ''}`} />
      {label && <span className="font-stencil uppercase tracking-wider text-[11px] text-stone-300">{label}</span>}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Rocker toggle switch
// ---------------------------------------------------------------------------

export function Rocker({
  on, onToggle, label,
}: {
  on: boolean
  onToggle: () => void
  label?: string
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="flex items-center gap-3 group"
      aria-pressed={on}
    >
      <span className={`rocker ${on ? 'on' : ''}`}>
        <span className="knob" />
      </span>
      {label && (
        <span className={`font-stencil uppercase tracking-wide text-xs ${on ? 'text-emerald-300' : 'text-stone-500'}`}>
          {label}
        </span>
      )}
    </button>
  )
}

// ---------------------------------------------------------------------------
// Mechanical flip-clock number
// ---------------------------------------------------------------------------

export function FlipNumber({
  value, label, accent,
}: {
  value: number | string
  label?: string
  accent?: string
}) {
  const chars = String(value).split('')
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="flex gap-1">
        {chars.map((c, i) => (
          <span key={i} className="flip" style={accent ? { color: accent } : undefined}>{c}</span>
        ))}
      </div>
      {label && <span className="font-stencil uppercase tracking-widest text-[11px] text-stone-400">{label}</span>}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Rubber ink stamp
// ---------------------------------------------------------------------------

export function Stamp({
  children, tone = 'red', className = '',
}: {
  children: ReactNode
  tone?: 'red' | 'blue'
  className?: string
}) {
  return (
    <span className={`ink-stamp ${tone === 'blue' ? 'ink-blue' : ''} relative inline-block animate-stamp ${className}`}>
      {children}
    </span>
  )
}
