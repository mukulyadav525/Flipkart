"""Bridge between the Gridlock video engine and the TrafficEye portal format.

The Gridlock engine (``Pipeline/gridlock``) is a video pipeline that emits
``ViolationEvent`` records.  The Backend (``Backend/main.py``) and Frontend read
``EvidenceRecord``-shaped JSONL + an ``analytics.json``.  This package converts
the former into the latter so a real detection run lights up the portal.
"""
