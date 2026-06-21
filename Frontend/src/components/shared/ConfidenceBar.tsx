import { confidenceLevel, CONFIDENCE_LABEL, fmtPct } from '../../lib/utils'

const FILL: Record<'high' | 'moderate' | 'low', string> = {
  high:     'linear-gradient(180deg,#5dff8f,#16a34a)',
  moderate: 'linear-gradient(180deg,#ffd25a,#f59e0b)',
  low:      'linear-gradient(180deg,#ff8a5a,#ea580c)',
}
const GLOW: Record<'high' | 'moderate' | 'low', string> = {
  high: 'rgba(40,220,110,.6)', moderate: 'rgba(245,158,11,.55)', low: 'rgba(234,88,12,.55)',
}

/** A recessed analog confidence meter with a glowing liquid-crystal fill. */
export function ConfidenceBar({ value, showLabel = true }: { value: number; showLabel?: boolean }) {
  const level = confidenceLevel(value)
  return (
    <div className="space-y-1.5 w-full">
      {showLabel && (
        <div className="flex justify-between items-baseline">
          <span className="font-stencil uppercase tracking-wider text-[10px] text-stone-400">{CONFIDENCE_LABEL[level]}</span>
          <span className="font-mono text-sm text-crt-green" style={{ textShadow: `0 0 6px ${GLOW[level]}` }}>{fmtPct(value)}</span>
        </div>
      )}
      <div
        className="h-3 rounded-full recessed relative overflow-hidden"
        style={{ background: '#0c0e12' }}
      >
        {/* tick marks */}
        <div className="absolute inset-0 flex justify-between px-1 opacity-40 pointer-events-none">
          {Array.from({ length: 11 }).map((_, i) => (
            <span key={i} style={{ width: 1, background: 'rgba(255,255,255,.25)' }} />
          ))}
        </div>
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${value * 100}%`, background: FILL[level], boxShadow: `0 0 12px ${GLOW[level]}` }}
        />
      </div>
    </div>
  )
}
