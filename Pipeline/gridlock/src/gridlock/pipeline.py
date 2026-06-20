"""End-to-end Phase 0 pipeline: read video -> preprocess -> detect+track -> annotate.

This is the backbone that the Phase 1+ violation engines will hang off of: each
frame already yields a list of `Track`s with stable ids, which is exactly what
the rule engines (parking timers, line crossing, direction, etc.) consume.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .association import tag_three_wheelers
from .config import PipelineConfig
from .detection import Detector
from .preprocessing import Preprocessor
from .scene import CameraConfig
from .tracking_state import TrackTracker
from .types import Track
from .violations import FrameContext, ViolationManager
from . import visualize


@dataclass
class RunStats:
    frames_read: int = 0
    frames_processed: int = 0
    elapsed_s: float = 0.0
    # Distinct objects seen over the whole run, counted by class.
    class_counts: Counter = field(default_factory=Counter)
    # Violations seen over the run, counted by type.
    violation_counts: Counter = field(default_factory=Counter)
    _seen: set = field(default_factory=set)

    @property
    def processed_fps(self) -> float:
        return self.frames_processed / self.elapsed_s if self.elapsed_s else 0.0


class Pipeline:
    def __init__(
        self,
        cfg: PipelineConfig | None = None,
        camera_config: CameraConfig | None = None,
        engines: list | None = None,
        signal_provider=None,
    ):
        self.cfg = cfg or PipelineConfig()
        self.pre = Preprocessor(self.cfg.preprocess)
        self.detector = Detector(self.cfg)
        self.camera_config = camera_config
        self.signal_provider = signal_provider  # for HUD display only
        # Violations run whenever engines are given. Geometry engines additionally
        # need a calibrated camera_config; perception engines (helmet/triple/
        # seatbelt) do not, so a camera config is optional here.
        self.tracker = TrackTracker() if engines else None
        self.manager = ViolationManager(engines) if engines else None
        self._scene = camera_config or CameraConfig(name="default")

    def run(
        self,
        source: str | Path,
        output: str | Path | None = None,
        max_frames: int | None = None,
        events_path: str | Path | None = None,
    ) -> RunStats:
        source = str(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {source}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stride = max(1, self.cfg.preprocess.frame_stride)

        writer = None
        if output and self.cfg.write_video:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output), fourcc, src_fps / stride, (width, height))

        stats = RunStats()
        start = time.time()
        frame_idx = -1
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                stats.frames_read += 1

                if frame_idx % stride != 0:
                    continue

                proc = self.pre.apply(frame)
                tracks = self.detector.track(proc)
                if self.cfg.reclassify_rickshaws:
                    tag_three_wheelers(tracks, self.cfg.rickshaw_wh_ratio)
                stats.frames_processed += 1
                self._tally_unique(tracks, stats)

                # Violation engines.
                flagged: dict[int, str] = {}
                if self.manager is not None:
                    timestamp = frame_idx / src_fps
                    states = self.tracker.update(tracks, timestamp)
                    ctx = FrameContext(
                        states=states, tracks=tracks, frame=frame,
                        frame_idx=frame_idx, timestamp=timestamp, config=self._scene,
                    )
                    for ev in self.manager.update(ctx):
                        stats.violation_counts[ev.type] += 1
                    flagged = self.manager.flagged_ids()

                if writer is not None:
                    sig_state = None
                    if self.signal_provider is not None:
                        sig_state = self.signal_provider.state_at(frame_idx / src_fps, frame)
                    annotated = visualize.draw_scene(frame, self.camera_config, signal_state=sig_state)
                    annotated = visualize.draw_tracks(
                        annotated, tracks, draw_labels=self.cfg.draw_labels, flagged=flagged
                    )
                    hud = [
                        f"frame {frame_idx}  tracks {len(tracks)}",
                        f"device {self.detector.device}  stride {stride}",
                    ]
                    if sig_state is not None:
                        hud.append(f"signal: {sig_state.upper()}")
                    if self.manager is not None:
                        hud.append("violations: " + ", ".join(
                            f"{k}={v}" for k, v in sorted(stats.violation_counts.items())
                        ) or "violations: 0")
                    visualize.draw_hud(annotated, hud)
                    writer.write(annotated)

                if max_frames and stats.frames_processed >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        stats.elapsed_s = time.time() - start
        if self.manager is not None and events_path is not None:
            self.manager.save(events_path)
        return stats

    @staticmethod
    def _tally_unique(tracks: list[Track], stats: RunStats):
        # Count distinct (class, id) pairs over the run for a rough scene summary.
        for t in tracks:
            key = (t.class_name, t.track_id)
            if key not in stats._seen:
                stats._seen.add(key)
                stats.class_counts[t.class_name] += 1
