from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    student_model_name_or_path: str
    teacher_model_name_or_path: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    gradient_checkpointing: bool = False


@dataclass
class DataConfig:
    dataset: str
    media_dir: str | None = None
    num_workers: int = 0
    shuffle: bool = True
    seed: int = 42


@dataclass
class VideoConfig:
    nframes: int | None = 128
    fps: float | None = None
    crop_fps: float = 2.0
    min_pixels: int = 20 * 28 * 28
    max_pixels: int = 768 * 28 * 28
    total_pixels: int = 16384 * 28 * 28
    coarse_tokens: int = 2048
    medium_tokens: int = 4096
    fine_tokens: int = 6144


@dataclass
class GenerationConfig:
    max_new_tokens: int = 2048
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    use_cache: bool = True


@dataclass
class TrainConfig:
    output_dir: str = "saves/opd"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_train_epochs: int = 1
    max_steps: int = -1
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    logging_steps: int = 1
    save_steps: int = 100
    save_total_limit: int = 2
    resume_from_checkpoint: str | None = None
    mixed_precision: str = "bf16"
    deepspeed: str | None = None
    exact_reverse_kl: bool = True
    student_from_target: bool = False


@dataclass
class OPDConfig:
    model: ModelConfig
    data: DataConfig
    video: VideoConfig = field(default_factory=VideoConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


def load_opd_config(path: str | Path) -> OPDConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected a mapping in {path}")

    # Backward compatibility with the original flat opd_small.yaml.
    if "model" not in raw:
        raw = {
            "model": {
                "student_model_name_or_path": raw.pop("student_model_name_or_path"),
                "teacher_model_name_or_path": raw.pop("teacher_model_name_or_path"),
            },
            "data": {"dataset": raw.pop("dataset")},
            "generation": raw.pop("generation", {}),
            "train": {
                "output_dir": raw.pop("output_dir", "saves/opd"),
                "exact_reverse_kl": raw.pop("reverse_kl_exact", True),
                **raw.pop("train", {}),
            },
            **raw,
        }

    config = OPDConfig(
        model=ModelConfig(**_section(raw, "model")),
        data=DataConfig(**_section(raw, "data")),
        video=VideoConfig(**_section(raw, "video")),
        generation=GenerationConfig(**_section(raw, "generation")),
        train=TrainConfig(**_section(raw, "train")),
    )
    if not config.train.exact_reverse_kl:
        raise ValueError("exact_reverse_kl must be true; sampled KL is not supported by production OPD training")
    if config.train.per_device_train_batch_size < 1:
        raise ValueError("per_device_train_batch_size must be >= 1")
    if config.train.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    return config
