import { confidenceLevel, CONFIDENCE_LABEL, CONFIDENCE_COLOR, fmtPct } from '../../lib/utils'

export function ConfidenceBar({ value, showLabel = true }: { value: number; showLabel?: boolean }) {
  const level = confidenceLevel(value)
  const barColor = CONFIDENCE_COLOR[level]
  return (
    <div className="space-y-1">
      {showLabel && (
        <div className="flex justify-between items-center">
          <span className="text-xs text-slate-500">{CONFIDENCE_LABEL[level]}</span>
          <span className="text-xs font-mono font-semibold text-slate-300">{fmtPct(value)}</span>
        </div>
      )}
      <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${barColor}`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  )
}
