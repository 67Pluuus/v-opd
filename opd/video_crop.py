from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CroppedVideo:
    tensor: Any
    fps: float
    temporal_segment: tuple[float, float]
    sampling_strategy: str


def validate_temporal_segment(
    temporal_segment: tuple[float, float] | list[float],
    video_duration: float | None,
) -> tuple[float, float]:
    start, end = float(temporal_segment[0]), float(temporal_segment[1])
    if start < 0 or end <= start:
        raise ValueError(f"invalid temporal_segment: {temporal_segment}")
    if end - start < 1.0:
        raise ValueError(f"temporal_segment must be at least 1 second: {temporal_segment}")
    if video_duration is not None and end > float(video_duration):
        raise ValueError(f"temporal_segment exceeds video duration {video_duration}: {temporal_segment}")
    return start, end


def crop_with_video_o3(
    raw_video: dict[str, Any],
    raw_fps: float,
    temporal_segment: tuple[float, float] | list[float],
    sampling_strategy: str,
    repo_root: str | Path | None = None,
    video_duration: float | None = None,
    frames_sample_fps: float = 2.0,
) -> CroppedVideo:
    """Reuse Video-o3's existing `crop_video` implementation."""

    if sampling_strategy not in {"coarse", "medium", "fine"}:
        raise ValueError(f"invalid sampling_strategy: {sampling_strategy}")

    segment = validate_temporal_segment(temporal_segment, video_duration)

    import sys

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    rollout_path = root / "RL" / "verl" / "workers" / "rollout" / "vllm_rollout"
    qwen_utils_path = root / "RL"
    for path in (rollout_path, qwen_utils_path):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from function_tools_video import crop_video

    tensor, fps = crop_video(
        raw_video,
        raw_fps,
        list(segment),
        sampling_strategy,
        frames_sample_fps=frames_sample_fps,
    )
    return CroppedVideo(tensor, fps, segment, sampling_strategy)

