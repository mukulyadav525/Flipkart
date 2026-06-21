import type { ViolationType, ReviewStatus } from './types'

/** Merge Tailwind class strings conditionally. */
export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(' ')
}

/** Format ISO-8601 timestamp for display. */
export function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString('en-IN', {
      day: '2-digit', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit', hour12: true,
    })
  } catch {
    return iso
  }
}

/** Format confidence as a percentage string. */
export function fmtPct(v: number): string {
  return `${Math.round(v * 100)}%`
}

export type Severity = 'critical' | 'high' | 'medium' | 'low'

export type ViolationMeta = {
  label: string
  /** Ink colour for stamps / chips / bounding boxes. */
  ink: string
  /** Chip background tint. */
  chipBg: string
  severity: Severity
  /** Citizen-facing plain-language explanation. */
  citizen: { heading: string; explanation: string; law: string }
}

const AMBER = '#b45309', ORANGE = '#c2410c', REDINK = '#9b1c1c', BLUEINK = '#1e3a5f'

export const VIOLATION_META: Record<ViolationType, ViolationMeta> = {
  helmet: {
    label: 'No Helmet',
    ink: AMBER, chipBg: 'rgba(180,83,9,.14)', severity: 'medium',
    citizen: {
      heading: 'Helmet Not Detected',
      explanation:
        "Our system identified a two-wheeler rider in this image without a helmet. The rider's head was visible in the frame, but no protective headgear was detected overlapping the head region.",
      law: 'Motor Vehicles Act, 1988 — Section 129: Every person riding a motorised two-wheeler must wear a protective headgear conforming to prescribed standards.',
    },
  },
  seatbelt: {
    label: 'No Seatbelt',
    ink: AMBER, chipBg: 'rgba(180,83,9,.14)', severity: 'medium',
    citizen: {
      heading: 'Seatbelt Not Detected',
      explanation:
        "An occupant was visible through the vehicle's window, and the system's pose estimation identified the shoulder-to-hip region. No seatbelt strap was detected crossing the torso.",
      law: 'Motor Vehicles Act, 1988 — Section 138(3): Every occupant of a motor vehicle must wear a seatbelt while the vehicle is in motion.',
    },
  },
  triple_riding: {
    label: 'Triple Riding',
    ink: ORANGE, chipBg: 'rgba(194,65,12,.14)', severity: 'high',
    citizen: {
      heading: 'Triple Riding Detected',
      explanation:
        'More than two persons were detected riding on a two-wheeler in this image. The legal maximum is two persons — one rider and one pillion passenger.',
      law: 'Motor Vehicles Act, 1988: No two-wheeler shall carry more than one pillion rider at any time.',
    },
  },
  wrong_side: {
    label: 'Wrong Side',
    ink: REDINK, chipBg: 'rgba(155,28,28,.14)', severity: 'critical',
    citizen: {
      heading: 'Wrong-Side Driving Detected',
      explanation:
        "Your vehicle was detected travelling against the designated direction of traffic flow for the lane visible in the camera's field of view.",
      law: 'Central Motor Vehicles Rules, 1989 — Rule 15: No vehicle may be driven on the wrong side of the road.',
    },
  },
  stop_line: {
    label: 'Stop Line',
    ink: ORANGE, chipBg: 'rgba(194,65,12,.14)', severity: 'high',
    citizen: {
      heading: 'Stop-Line Violation',
      explanation:
        'The vehicle crossed the marked stop line at an intersection while the signal was not showing green.',
      law: 'Motor Vehicles Act, 1988 — Section 119: All road users must obey traffic signals and road markings.',
    },
  },
  red_light: {
    label: 'Red Light',
    ink: REDINK, chipBg: 'rgba(155,28,28,.14)', severity: 'critical',
    citizen: {
      heading: 'Red Light Violation',
      explanation:
        "The vehicle crossed the stop line while the traffic signal was displaying red. The system confirmed both the signal state and the vehicle's position relative to the stop line.",
      law: 'Motor Vehicles Act, 1988 — Section 119: Failing to stop at a red traffic signal is a punishable offence.',
    },
  },
  illegal_parking: {
    label: 'Illegal Parking',
    ink: BLUEINK, chipBg: 'rgba(30,58,95,.16)', severity: 'low',
    citizen: {
      heading: 'Illegal Parking Detected',
      explanation:
        "Your vehicle was detected stationary within a designated no-parking zone. Either a no-parking sign was visible in the frame, or the vehicle's position fell within a mapped no-parking boundary.",
      law: 'Central Motor Vehicles Rules, 1989 — Rule 15: Parking in a no-parking zone is prohibited.',
    },
  },
}

export const STATUS_META: Record<ReviewStatus, { label: string; className: string }> = {
  confirmed: { label: 'Confirmed',        className: 'bg-emerald-950 border-emerald-800 text-emerald-300' },
  approved:  { label: 'Approved',         className: 'bg-emerald-950 border-emerald-800 text-emerald-300' },
  pending:   { label: 'Pending Review',   className: 'bg-amber-950  border-amber-800  text-amber-300'  },
  rejected:  { label: 'Rejected',         className: 'bg-slate-800  border-slate-700  text-slate-400'  },
}

/** Confidence level bucket for display. */
export function confidenceLevel(conf: number): 'high' | 'moderate' | 'low' {
  if (conf >= 0.85) return 'high'
  if (conf >= 0.70) return 'moderate'
  return 'low'
}

export const CONFIDENCE_LABEL: Record<'high' | 'moderate' | 'low', string> = {
  high:     'High confidence detection',
  moderate: 'Moderate confidence — under human review',
  low:      'Lower confidence — queued for human review',
}

export const CONFIDENCE_COLOR: Record<'high' | 'moderate' | 'low', string> = {
  high:     'bg-emerald-500',
  moderate: 'bg-amber-500',
  low:      'bg-orange-500',
}
