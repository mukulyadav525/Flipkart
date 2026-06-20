#!/usr/bin/env python3
"""
End-to-end runner: Gridlock video detection  ->  TrafficEye portal.

This is the single command that connects everything the user sees:

    1. Runs the merged Gridlock engine (Pipeline/gridlock) on a video,
       producing a summary + evidence snapshots/plates.
    2. Bridges that output into the EvidenceRecord/analytics files the
       Backend serves (Pipeline/outputs/...).

After it finishes, start the Backend (it auto-detects the real outputs and
leaves mock mode) and the Frontend, and the dashboards show live results.

Examples
--------
    python scripts/run_portal_pipeline.py gridlock/archive/Vodra/South.mp4
    python scripts/run_portal_pipeline.py clip.mp4 \
        --config gridlock/configs/cameras/South.json \
        --helmet-weights gridlock/models/helmet.pt --anpr
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
GRIDLOCK_ROOT = PIPELINE_ROOT / "gridlock"
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from bridge.portal_export import export  # noqa: E402


def _run_gridlock(args) -> tuple[Path, Path | None]:
    """Invoke the merged Gridlock run_all.py and return (summary, ledger?)."""
    stem = Path(args.source).stem
    run_all = GRIDLOCK_ROOT / "scripts" / "run_all.py"

    cmd = [sys.executable, str(run_all), args.source,
           "--model", args.model, "--stride", str(args.stride)]
    if args.config:
        cmd += ["--config", args.config]
    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]
    if args.helmet_weights:
        cmd += ["--helmet-weights", args.helmet_weights]
    if args.seatbelt_weights:
        cmd += ["--seatbelt-weights", args.seatbelt_weights]
    # Evidence snapshots feed the portal's annotated images; ANPR fills plates.
    cmd += ["--evidence"]
    if args.anpr:
        cmd += ["--anpr"]
        if args.plate_weights:
            cmd += ["--plate-weights", args.plate_weights]

    print(f"[portal] running Gridlock:\n  {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    summary = GRIDLOCK_ROOT / "outputs" / f"{stem}_all_summary.json"
    ledger = GRIDLOCK_ROOT / "outputs" / "evidence" / stem / "ledger.json"
    if not summary.exists():
        raise SystemExit(f"[portal] expected Gridlock summary not found: {summary}")
    return summary, (ledger if ledger.exists() else None)


def main() -> None:
    p = argparse.ArgumentParser(description="Run Gridlock and publish to the portal.")
    p.add_argument("source", help="Input video for Gridlock")
    p.add_argument("--config", default=None, help="Gridlock camera config JSON")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--helmet-weights", default=None)
    p.add_argument("--seatbelt-weights", default=None)
    p.add_argument("--anpr", action="store_true", help="read plates (needs easyocr)")
    p.add_argument("--plate-weights", default=None)
    p.add_argument("--captured-at", default=None,
                   help="ISO-8601 wall-clock start time for the clip (default: now UTC)")
    p.add_argument("--append", action="store_true",
                   help="append to existing portal records instead of replacing")
    p.add_argument("--summary", default=None,
                   help="skip detection and bridge an existing Gridlock summary JSON")
    p.add_argument("--ledger", default=None, help="evidence ledger.json (with --summary)")
    args = p.parse_args()

    if args.summary:
        summary_path = Path(args.summary)
        ledger_path = Path(args.ledger) if args.ledger else None
    else:
        summary_path, ledger_path = _run_gridlock(args)

    captured_at = None
    if args.captured_at:
        captured_at = datetime.fromisoformat(args.captured_at)
    else:
        captured_at = datetime.now(timezone.utc)

    print(f"\n[portal] bridging {summary_path.name} -> portal outputs ...")
    result = export(
        summary_path,
        ledger_path=ledger_path,
        captured_at=captured_at,
        append=args.append,
    )
    print("\n" + "=" * 48)
    print("PORTAL PUBLISH COMPLETE")
    print("=" * 48)
    print(f"confirmed violations : {result['confirmed']}")
    print(f"queued for review    : {result['review']}")
    print(f"outputs dir          : {result['outputs_dir']}")
    print("\nNext: start the Backend (uvicorn) and Frontend (vite); the portal")
    print("will leave mock mode automatically now that real outputs exist.")


if __name__ == "__main__":
    main()
