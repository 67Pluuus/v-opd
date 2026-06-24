from __future__ import annotations

import torch
import torch.nn.functional as F


def reverse_kl_from_distributions(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact KL(teacher || student) over vocab for aligned token positions."""

    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_p = teacher_logp.exp()
    per_pos = torch.sum(teacher_p * (teacher_logp - student_logp), dim=-1)
    if mask is None:
        return per_pos.mean()

    mask = mask.to(per_pos.device).bool()
    return per_pos.masked_select(mask).mean()


def reverse_kl_sum_from_distributions(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Return exact KL(teacher || student) sum and contributing position count.

    OPD trajectories contain a variable number of teacher tasks and tokens.
    Returning a sum lets the training loop form a true token-weighted mean
    across samples, devices, and gradient-accumulation microbatches without
    concatenating all vocabulary-sized logits into one large tensor.
    """

    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    per_pos = torch.sum(teacher_logp.exp() * (teacher_logp - student_logp), dim=-1)
    if mask is not None:
        per_pos = per_pos.masked_select(mask.to(per_pos.device).bool())
    return per_pos.sum(), per_pos.numel()
