import { CheckCircle2, Clock, XCircle, AlertCircle } from 'lucide-react'
import type { ReviewStatus, ViolationType } from '../../lib/types'
import { STATUS_META, VIOLATION_META } from '../../lib/utils'

const STATUS_ICONS: Record<ReviewStatus, typeof CheckCircle2> = {
  confirmed: CheckCircle2,
  approved:  CheckCircle2,
  pending:   Clock,
  rejected:  XCircle,
}

export function StatusBadge({ status }: { status: ReviewStatus }) {
  const meta = STATUS_META[status]
  const Icon = STATUS_ICONS[status]
  return (
    <span className={`inline-flex items-center gap-1.5 border text-xs font-medium px-2.5 py-1 rounded-full ${meta.className}`}>
      <Icon size={11} strokeWidth={2.5} />
      {meta.label}
    </span>
  )
}

export function ViolationBadge({ type }: { type: ViolationType }) {
  const meta = VIOLATION_META[type]
  return (
    <span className={`inline-flex items-center gap-1 border text-xs font-semibold px-2.5 py-1 rounded-full ${meta.badgeClass}`}>
      <AlertCircle size={11} strokeWidth={2.5} />
      {meta.label}
    </span>
  )
}
