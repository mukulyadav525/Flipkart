import type { ReviewStatus, ViolationType } from '../../lib/types'
import { STATUS_META, VIOLATION_META } from '../../lib/utils'
import { Led } from '../skeuo'

const STATUS_LED: Record<ReviewStatus, 'green' | 'amber' | 'red' | 'off'> = {
  confirmed: 'green', approved: 'green', pending: 'amber', rejected: 'red',
}

/** An engraved label plate with an inset status LED. */
export function StatusBadge({ status }: { status: ReviewStatus }) {
  const meta = STATUS_META[status]
  return (
    <span className="plastic raised inline-flex items-center gap-2 px-2.5 py-1 rounded-md">
      <Led state={STATUS_LED[status]} />
      <span className="font-stencil uppercase tracking-wider text-[11px] text-stone-200">{meta.label}</span>
    </span>
  )
}

/** A stamped violation chip whose ink colour matches severity. */
export function ViolationBadge({ type }: { type: ViolationType }) {
  const meta = VIOLATION_META[type]
  return (
    <span
      className="inline-flex items-center font-stencil uppercase tracking-wider text-[11px] px-2.5 py-1 rounded-md raised"
      style={{ color: meta.ink, background: meta.chipBg, border: `1px solid ${meta.ink}55` }}
    >
      {meta.label}
    </span>
  )
}
