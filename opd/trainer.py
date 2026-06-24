from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .loss import reverse_kl_sum_from_distributions
from .model_adapter import BaseVideoO3Adapter
from .trajectory import parse_student_trajectory, split_teacher_tasks


@dataclass
class OPDStepOutput:
    loss: torch.Tensor
    loss_sum: torch.Tensor
    num_tokens: int
    valid_text: str
    interrupted: bool
    interrupt_reason: str | None
    num_teacher_tasks: int
    trajectory_source: str


class OPDTrainer:
    """Exact-KL OPD loop controller.

    It runs the student serially, validates tool calls, then builds teacher
    scoring tasks that can be executed independently.
    """

    def __init__(
        self,
        student: BaseVideoO3Adapter,
        teacher: BaseVideoO3Adapter,
        reverse_kl_exact: bool = True,
        student_from_target: bool = False,
    ):
        if not reverse_kl_exact:
            raise ValueError("Production OPD requires exact KL(teacher || student).")
        self.student = student
        self.teacher = teacher
        self.reverse_kl_exact = True
        self.student_from_target = student_from_target

    def _trajectory_text(
        self,
        sample: dict[str, Any],
        messages: list[dict[str, Any]],
        videos: list[dict[str, Any]],
        generation_kwargs: dict[str, Any],
    ) -> tuple[str, str]:
        if not self.student_from_target:
            return self.student.generate_student(
                messages, videos, **generation_kwargs
            ), "student_generation"

        target = sample.get("student_target")
        if target is None and sample.get("messages"):
            last_message = sample["messages"][-1]
            if last_message.get("role") == "assistant":
                target = last_message.get("content")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(
                "student_from_target debug mode requires a non-empty student_target "
                "or final assistant message in every sample"
            )
        return target, "student_target"

    def step(self, sample: dict[str, Any], video_duration: float | None = None, generation_kwargs: dict[str, Any] | None = None) -> OPDStepOutput:
        generation_kwargs = generation_kwargs or {}
        if "student_messages" in sample:
            messages = sample["student_messages"]
        else:
            messages = sample["messages"][:-1] if sample.get("messages", []) and sample["messages"][-1].get("role") == "assistant" else sample["messages"]
        videos = sample.get("videos", [])[:1]
        student_text, trajectory_source = self._trajectory_text(
            sample, messages, videos, generation_kwargs
        )
        if video_duration is None and videos:
            video_duration = self.student.video_duration(videos[0])
        parsed = parse_student_trajectory(student_text, video_duration=video_duration)
        tasks = split_teacher_tasks(sample, parsed, original_messages=sample.get("teacher_source_messages"))

        loss_sum: torch.Tensor | None = None
        num_tokens = 0
        for task in tasks:
            student_result = self.student.sequence_logprobs(
                task.student_prompt_messages,
                task.target_text,
                task.student_videos,
                True,
                prefix_text=task.student_prefix_text,
            )
            with torch.no_grad():
                teacher_result = self.teacher.sequence_logprobs(
                    task.teacher_prompt_messages,
                    task.target_text,
                    task.teacher_videos,
                    True,
                )

            common_len = min(
                student_result.token_ids.shape[1], teacher_result.token_ids.shape[1]
            )
            if common_len == 0:
                continue
            if student_result.logits is None or teacher_result.logits is None:
                raise RuntimeError("exact OPD scoring requires vocabulary logits")
            student_ids = student_result.token_ids[:, :common_len]
            teacher_ids = teacher_result.token_ids[:, :common_len]
            if not torch.equal(student_ids.to("cpu"), teacher_ids.to("cpu")):
                raise ValueError(
                    f"student/teacher target tokenization differs in round {task.round_index}; "
                    "exact OPD requires aligned target token ids"
                )
            if student_result.logits.shape[-1] != teacher_result.logits.shape[-1]:
                raise ValueError(
                    "student and teacher vocabulary sizes differ; exact vocabulary KL cannot be computed"
                )
            task_sum, task_tokens = reverse_kl_sum_from_distributions(
                teacher_result.logits[:, :common_len, :].detach(),
                student_result.logits[:, :common_len, :],
            )
            loss_sum = task_sum if loss_sum is None else loss_sum + task_sum
            num_tokens += task_tokens

        if loss_sum is None or num_tokens == 0:
            device = getattr(self.student, "device", torch.device("cpu"))
            zero = torch.zeros((), device=device, requires_grad=True)
            return OPDStepOutput(
                zero,
                zero,
                0,
                parsed.valid_text,
                parsed.interrupted,
                parsed.interrupt_reason,
                len(tasks),
                trajectory_source,
            )

        loss = loss_sum / num_tokens
        return OPDStepOutput(
            loss,
            loss_sum,
            num_tokens,
            parsed.valid_text,
            parsed.interrupted,
            parsed.interrupt_reason,
            len(tasks),
            trajectory_source,
        )

    def batch_step(
        self,
        samples: list[dict[str, Any]],
        generation_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, int, list[OPDStepOutput]]:
        """Process a variable-length OPD microbatch and return token-weighted loss."""

        outputs = [self.step(sample, generation_kwargs=generation_kwargs) for sample in samples]
        valid = [output for output in outputs if output.num_tokens > 0]
        if not valid:
            device = getattr(self.student, "device", torch.device("cpu"))
            return torch.zeros((), device=device, requires_grad=True), 0, outputs
        loss_sum = torch.stack([output.loss_sum for output in valid]).sum()
        num_tokens = sum(output.num_tokens for output in valid)
        return loss_sum, num_tokens, outputs
