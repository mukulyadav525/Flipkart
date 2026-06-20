"""Violation engines.

Phase 1: illegal parking, wrong-side driving.
Phase 3: triple riding, helmet, seatbelt.
"""

from .base import FrameContext, ViolationEngine, ViolationEvent
from .manager import ViolationManager
from .parking import IllegalParkingEngine
from .wrong_side import WrongSideEngine
from .triple_riding import TripleRidingEngine
from .helmet import HelmetEngine
from .seatbelt import SeatbeltEngine
from .signal_based import RedLightEngine, StopLineEngine

__all__ = [
    "FrameContext",
    "ViolationEngine",
    "ViolationEvent",
    "ViolationManager",
    "IllegalParkingEngine",
    "WrongSideEngine",
    "TripleRidingEngine",
    "HelmetEngine",
    "SeatbeltEngine",
    "StopLineEngine",
    "RedLightEngine",
]
