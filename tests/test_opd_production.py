from pathlib import Path

import pytest
import torch

from opd.config import load_opd_config
from opd.loss import reverse_kl_from_distributions, reverse_kl_sum_from_distributions
from opd.model_adapter import VideoO3Adapter
from opd.trainer import OPDTrainer


def test_exact_kl_sum_matches_mean():
    teacher = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    student = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], requires_grad=True)
    mean = reverse_kl_from_distributions(teacher, student)
    total, count = reverse_kl_sum_from_distributions(teacher, student)
    assert count == 2
    assert torch.allclose(mean, total / count)
    mean.backward()
    assert student.grad is not None


def test_production_trainer_rejects_sampled_kl():
    with pytest.raises(ValueError, match="exact KL"):
        OPDTrainer(object(), object(), reverse_kl_exact=False)


def test_student_target_debug_mode_bypasses_generation():
    class Student:
        def generate_student(self, *_args, **_kwargs):
            raise AssertionError("generation should be bypassed")

    trainer = OPDTrainer(Student(), object(), student_from_target=True)
    text, source = trainer._trajectory_text(
        {"student_target": "<think><think>x</think></think><answer>A</answer>"},
        [],
        [],
        {},
    )
    assert source == "student_target"
    assert text.endswith("</answer>")


def test_student_target_debug_mode_requires_target():
    trainer = OPDTrainer(object(), object(), student_from_target=True)
    with pytest.raises(ValueError, match="requires a non-empty student_target"):
        trainer._trajectory_text({}, [], [], {})


def test_loads_production_yaml():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "opd_small.yaml"
    config = load_opd_config(config_path)
    assert config.train.exact_reverse_kl
    assert config.video.coarse_tokens < config.video.medium_tokens < config.video.fine_tokens
    assert config.train.gradient_accumulation_steps == 4


def test_loads_target_debug_yaml():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "opd_debug_target.yaml"
    config = load_opd_config(config_path)
    assert config.train.student_from_target
    assert config.train.max_steps == 1


def test_crop_strategy_changes_visual_quota_and_uses_fps():
    adapter = VideoO3Adapter.__new__(VideoO3Adapter)
    adapter.media_dir = None
    adapter.min_pixels = 1
    adapter.max_pixels = 2
    adapter.total_pixels = 3
    adapter.video_nframes = 128
    adapter.video_fps = None
    adapter.crop_fps = 2.0
    adapter.crop_total_pixels = {
        "coarse": 2048 * 28 * 28,
        "medium": 4096 * 28 * 28,
        "fine": 6144 * 28 * 28,
    }
    coarse = adapter._video_content(
        {
            "url": "x.mp4",
            "video_start": 1,
            "video_end": 3,
            "sampling_strategy": "coarse",
        }
    )
    fine = adapter._video_content(
        {
            "url": "x.mp4",
            "video_start": 1,
            "video_end": 3,
            "sampling_strategy": "fine",
        }
    )
    assert coarse["total_pixels"] < fine["total_pixels"]
    assert coarse["fps"] == 2.0
    assert "nframes" not in coarse
