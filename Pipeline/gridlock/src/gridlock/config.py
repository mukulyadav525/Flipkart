"""Configuration objects for the Gridlock pipeline.

Kept as plain dataclasses so they are easy to serialize to / from JSON later
(per-camera calibration in Phase 1 will live alongside these)."""

from __future__ import annotations

from dataclasses import dataclass, field


# COCO class ids we care about for traffic. The default YOLO weights are trained
# on COCO, so these ids are fixed. person is kept for triple-riding (Phase 3).
TRAFFIC_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class PreprocessConfig:
    """Toggleable preprocessing chain. Defaults are tuned for daytime CCTV.

    Each step can be turned off so we can A/B the effect for the judges
    ("accuracy with vs without CLAHE")."""

    enabled: bool = True

    # Process every Nth frame. 30fps source -> stride 3 gives ~10fps, plenty
    # for tracking while cutting compute ~3x.
    frame_stride: int = 2

    # Contrast-limited adaptive histogram equalization on the L channel (LAB).
    # Biggest single win on shadows / glare / overcast streets.
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8

    # Auto gamma boost when the frame is dark (night / underpass).
    auto_gamma: bool = True
    dark_threshold: float = 80.0  # mean luma below this -> brighten
    gamma: float = 1.5

    # Mild unsharp mask — helps small/distant plates and helmets.
    sharpen: bool = True
    sharpen_amount: float = 0.6

    # Denoising is OFF by default: high quality denoise (NlMeans) is far too
    # slow for video. Turn on for stills only.
    denoise: bool = False


@dataclass
class PipelineConfig:
    """Top-level run configuration."""

    model_weights: str = "yolo11n.pt"  # nano: fast, fine for a demo. Swap to s/m for accuracy.
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    tracker: str = "bytetrack.yaml"  # ultralytics built-in ByteTrack config
    device: str | None = None  # None -> auto (mps on Apple Silicon, else cpu)
    classes: list[int] = field(default_factory=lambda: list(TRAFFIC_CLASSES.keys()))
    imgsz: int = 640
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    # Reclassify wide "motorcycle" boxes (COCO has no auto-rickshaw class) as
    # auto_rickshaw so 3-wheelers aren't treated as two-wheelers. w/h >= ratio.
    reclassify_rickshaws: bool = True
    rickshaw_wh_ratio: float = 1.0

    # Output
    write_video: bool = True
    draw_labels: bool = True
