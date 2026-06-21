from __future__ import annotations

from dataclasses import dataclass, field
from shared.schemas import ViolationType


@dataclass
class ViolationThresholds:
    """Per-violation-type confidence thresholds.

    Tune individual entries without touching detection or violation logic.
    A detection whose confidence is below the threshold for its type is
    discarded before evidence packaging.
    """
    thresholds: dict[ViolationType, float] = field(
        default_factory=lambda: {
            # Helmet is gated on the trained no-helmet head model's confidence.
            # That model under-scores no-helmet heads on out-of-domain footage,
            # so the floor is low (0.25); sub-0.85 still routes to human review.
            ViolationType.helmet:        0.25,
            ViolationType.seatbelt:      0.6,
            ViolationType.triple_riding: 0.6,
            ViolationType.wrong_side:    0.6,
            ViolationType.stop_line:     0.6,
            ViolationType.red_light:     0.6,
            ViolationType.illegal_parking: 0.6,
        }
    )

    def get(self, violation_type: ViolationType) -> float:
        return self.thresholds.get(violation_type, 0.6)


# ---------------------------------------------------------------------------
# Review-routing cutoff
# ---------------------------------------------------------------------------

# ViolationRecords with confidence >= AUTO_PROCESS_CUTOFF are written
# directly to outputs/violation_records/.
# Records below this threshold are routed to the human-review queue.
AUTO_PROCESS_CUTOFF: float = 0.85

# Singleton used by all modules — replace with dependency injection if needed.
THRESHOLDS = ViolationThresholds()


# ---------------------------------------------------------------------------
# Severity weights for analytics ranking
# ---------------------------------------------------------------------------
# Higher weight = more dangerous violation = ranked higher in reports.
# Weights are multiplied by the violation count (and optionally by mean
# confidence) to produce a severity score per violation type.
# Adjust without touching any rule or evidence code.

from dataclasses import dataclass as _dataclass   # avoid polluting module namespace

@_dataclass
class SeverityWeights:
    weights: dict[ViolationType, float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                # High accident risk — moving violations
                ViolationType.red_light:        3.0,
                ViolationType.wrong_side:        3.0,
                ViolationType.stop_line:         2.0,
                # Medium risk — safety equipment
                ViolationType.triple_riding:     1.8,
                ViolationType.helmet:            1.5,
                ViolationType.seatbelt:          1.5,
                # Lower immediate risk — parking
                ViolationType.illegal_parking:   1.0,
            }

    def get(self, violation_type: ViolationType) -> float:
        return self.weights.get(violation_type, 1.0)


SEVERITY = SeverityWeights()
