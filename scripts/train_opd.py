from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Sized

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opd.config import OPDConfig, load_opd_config
from opd.data import load_json_or_jsonl
from opd.model_adapter import VideoO3Adapter
from opd.trainer import OPDTrainer


class OPDDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def collate_samples(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def validate_target_debug_samples(samples: list[dict[str, Any]]) -> None:
    missing = []
    for index, sample in enumerate(samples):
        target = sample.get("student_target")
        if target is None and sample.get("messages"):
            last_message = sample["messages"][-1]
            if last_message.get("role") == "assistant":
                target = last_message.get("content")
        if not isinstance(target, str) or not target.strip():
            missing.append(sample.get("id", index))
            if len(missing) >= 10:
                break
    if missing:
        raise ValueError(
            "student_from_target is enabled but samples are missing target trajectories: "
            + ", ".join(map(str, missing))
        )


class SeededEpochSampler(Sampler[int]):
    """Deterministic epoch-aware sampler so checkpoint resume can skip exactly."""

    def __init__(self, data_source: Sized, seed: int, shuffle: bool):
        self.data_source = data_source
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            return iter(range(len(self.data_source)))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production Video-o3 exact-KL OPD training.")
    parser.add_argument("--config", default="configs/opd_small.yaml", help="OPD YAML configuration.")
    parser.add_argument(
        "--reverse-kl-exact",
        action="store_true",
        help="Explicitly select exact KL(teacher || student). Exact KL is always required.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None, help="Checkpoint path or 'latest'.")
    parser.add_argument(
        "--student-from-target",
        action="store_true",
        help="Debug only: bypass free generation and use each sample's student_target trajectory.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(output_dir: Path, requested: str | None) -> Path | None:
    if requested is None:
        return None
    if requested != "latest":
        return Path(requested)
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    return max(checkpoints, default=(0, None))[1]


def rotate_checkpoints(output_dir: Path, limit: int) -> None:
    if limit <= 0:
        return
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    for _, path in sorted(checkpoints)[:-limit]:
        shutil.rmtree(path)


def save_checkpoint(
    accelerator: Any,
    processor: Any,
    output_dir: Path,
    global_step: int,
    epoch: int,
    completed_batches: int,
    save_total_limit: int,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
    accelerator.save_state(str(checkpoint_dir))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state = {
            "global_step": global_step,
            "epoch": epoch,
            "completed_batches": completed_batches,
        }
        (checkpoint_dir / "opd_trainer_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
        processor.save_pretrained(checkpoint_dir)
        rotate_checkpoints(output_dir, save_total_limit)
    accelerator.wait_for_everyone()


def load_trainer_state(checkpoint_dir: Path) -> dict[str, int]:
    path = checkpoint_dir / "opd_trainer_state.json"
    if not path.exists():
        return {"global_step": 0, "epoch": 0, "completed_batches": 0}
    state = json.loads(path.read_text(encoding="utf-8"))
    return {
        "global_step": int(state.get("global_step", 0)),
        "epoch": int(state.get("epoch", 0)),
        "completed_batches": int(state.get("completed_batches", 0)),
    }


def adapter_kwargs(config: OPDConfig) -> dict[str, Any]:
    return {
        "torch_dtype": config.model.dtype,
        "device_map": None,
        "trust_remote_code": config.model.trust_remote_code,
        "media_dir": config.data.media_dir,
        "video_nframes": config.video.nframes,
        "video_fps": config.video.fps,
        "min_pixels": config.video.min_pixels,
        "max_pixels": config.video.max_pixels,
        "total_pixels": config.video.total_pixels,
        "crop_fps": config.video.crop_fps,
        "coarse_tokens": config.video.coarse_tokens,
        "medium_tokens": config.video.medium_tokens,
        "fine_tokens": config.video.fine_tokens,
    }


def main() -> None:
    args = parse_args()
    config = load_opd_config(args.config)
    if args.reverse_kl_exact:
        config.train.exact_reverse_kl = True
    if args.resume_from_checkpoint is not None:
        config.train.resume_from_checkpoint = args.resume_from_checkpoint
    if args.student_from_target:
        config.train.student_from_target = True

    try:
        from accelerate import Accelerator, DeepSpeedPlugin
        from transformers import get_scheduler
    except ImportError as exc:
        raise RuntimeError(
            "OPD training requires the repository SFT dependencies, including accelerate and transformers."
        ) from exc

    deepspeed_plugin = None
    if config.train.deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=config.train.deepspeed,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        mixed_precision=config.train.mixed_precision,
        deepspeed_plugin=deepspeed_plugin,
    )
    set_seed(config.data.seed + accelerator.process_index)

    output_dir = Path(config.train.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "opd_config.json").write_text(
            json.dumps(asdict(config), indent=2), encoding="utf-8"
        )
    accelerator.wait_for_everyone()

    samples = load_json_or_jsonl(config.data.dataset)
    if config.train.student_from_target:
        validate_target_debug_samples(samples)
    dataset = OPDDataset(samples)
    sampler = SeededEpochSampler(dataset, seed=config.data.seed, shuffle=config.data.shuffle)
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.per_device_train_batch_size,
        sampler=sampler,
        num_workers=config.data.num_workers,
        collate_fn=collate_samples,
        pin_memory=torch.cuda.is_available(),
    )

    student = VideoO3Adapter.from_pretrained(
        config.model.student_model_name_or_path, **adapter_kwargs(config)
    )
    teacher = VideoO3Adapter.from_pretrained(
        config.model.teacher_model_name_or_path, **adapter_kwargs(config)
    )
    student.model.train()
    if config.model.gradient_checkpointing:
        student.model.gradient_checkpointing_enable()
        if hasattr(student.model.config, "use_cache"):
            student.model.config.use_cache = False
    teacher.model.eval()
    teacher.model.requires_grad_(False)
    teacher.model.to(accelerator.device)
    teacher.device = accelerator.device

    optimizer = torch.optim.AdamW(
        (parameter for parameter in student.model.parameters() if parameter.requires_grad),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    student.model, optimizer, dataloader = accelerator.prepare(
        student.model, optimizer, dataloader
    )
    student.device = accelerator.device

    updates_per_epoch = max(
        1, math.ceil(len(dataloader) / config.train.gradient_accumulation_steps)
    )
    total_steps = (
        config.train.max_steps
        if config.train.max_steps > 0
        else updates_per_epoch * config.train.num_train_epochs
    )
    epochs_to_run = (
        math.ceil(total_steps / updates_per_epoch)
        if config.train.max_steps > 0
        else config.train.num_train_epochs
    )
    warmup_steps = (
        config.train.warmup_steps
        if config.train.warmup_steps > 0
        else int(total_steps * config.train.warmup_ratio)
    )
    scheduler = get_scheduler(
        config.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scheduler = accelerator.prepare_scheduler(scheduler)

    trainer = OPDTrainer(
        student,
        teacher,
        reverse_kl_exact=True,
        student_from_target=config.train.student_from_target,
    )
    if config.train.student_from_target:
        accelerator.print(
            "DEBUG MODE: using dataset student_target as the trajectory; "
            "student free generation is bypassed."
        )
    generation_kwargs = {
        "max_new_tokens": config.generation.max_new_tokens,
        "do_sample": config.generation.do_sample,
        "temperature": config.generation.temperature if config.generation.do_sample else None,
        "top_p": config.generation.top_p if config.generation.do_sample else None,
        "use_cache": config.generation.use_cache,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    checkpoint_dir = resolve_checkpoint(output_dir, config.train.resume_from_checkpoint)
    state = {"global_step": 0, "epoch": 0, "completed_batches": 0}
    if checkpoint_dir is not None:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_dir}")
        accelerator.load_state(str(checkpoint_dir))
        state = load_trainer_state(checkpoint_dir)
        accelerator.print(f"Resumed from {checkpoint_dir} at global step {state['global_step']}")

    global_step = state["global_step"]
    first_epoch = state["epoch"]
    optimizer.zero_grad(set_to_none=True)

    epoch = first_epoch
    while epoch < epochs_to_run or (
        config.train.max_steps > 0 and global_step < total_steps
    ):
        step_at_epoch_start = global_step
        sampler.set_epoch(epoch)
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)
        start_batch = state["completed_batches"] if epoch == first_epoch else 0
        if start_batch >= len(dataloader):
            state["completed_batches"] = 0
            epoch += 1
            continue
        epoch_loader = (
            accelerator.skip_first_batches(dataloader, start_batch)
            if start_batch > 0
            else dataloader
        )

        for relative_batch, batch in enumerate(epoch_loader):
            batch_index = start_batch + relative_batch
            with accelerator.accumulate(student.model):
                local_loss_sum, local_tokens, outputs = trainer.batch_step(
                    batch, generation_kwargs=generation_kwargs
                )
                token_tensor = torch.tensor(
                    float(local_tokens), device=accelerator.device, dtype=torch.float32
                )
                global_tokens = accelerator.reduce(token_tensor, reduction="sum")
                active_rank = torch.tensor(
                    float(local_tokens > 0), device=accelerator.device, dtype=torch.float32
                )
                active_ranks = accelerator.reduce(active_rank, reduction="sum")
                skip_microbatch = active_ranks.item() != accelerator.num_processes
                if skip_microbatch:
                    # A rank with no valid teacher task cannot participate in
                    # DDP gradient reduction. Drop this distributed microbatch
                    # on every rank and clear any partial accumulation window.
                    student.model.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                elif global_tokens.item() > 0:
                    # DDP averages gradients across ranks. This scaling produces
                    # the exact global token mean for this distributed microbatch.
                    loss = local_loss_sum * accelerator.num_processes / global_tokens
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and config.train.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(
                            student.model.parameters(), config.train.max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    loss = local_loss_sum

            if skip_microbatch:
                accelerator.print(
                    f"Skipped distributed microbatch at epoch={epoch} batch={batch_index}: "
                    "at least one rank produced no valid OPD scoring tokens"
                )
                continue
            if not accelerator.sync_gradients:
                continue

            global_step += 1
            local_tasks = sum(output.num_teacher_tasks for output in outputs)
            local_interrupted = sum(int(output.interrupted) for output in outputs)
            metrics = torch.stack(
                [
                    loss.detach().float(),
                    torch.tensor(float(local_tasks), device=accelerator.device),
                    torch.tensor(float(local_interrupted), device=accelerator.device),
                ]
            )
            metrics = accelerator.reduce(metrics, reduction="sum")
            if global_step % config.train.logging_steps == 0:
                mean_loss = metrics[0].item() / accelerator.num_processes
                accelerator.print(
                    f"step={global_step} loss={mean_loss:.6f} "
                    f"tasks={int(metrics[1].item())} interrupted={int(metrics[2].item())} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"trajectory_source={'student_target' if config.train.student_from_target else 'student_generation'}"
                )

            completed_batches = batch_index + 1
            if config.train.save_steps > 0 and global_step % config.train.save_steps == 0:
                save_checkpoint(
                    accelerator,
                    student.processor,
                    output_dir,
                    global_step,
                    epoch,
                    completed_batches,
                    config.train.save_total_limit,
                )
            if global_step >= total_steps:
                break

        state["completed_batches"] = 0
        if global_step >= total_steps:
            break
        if global_step == step_at_epoch_start:
            raise RuntimeError(
                "No optimizer step was completed in an entire epoch. "
                "Check generated trajectory validity and video availability."
            )
        epoch += 1

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(student.model)
    state_dict = accelerator.get_state_dict(student.model)
    unwrapped.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=True,
    )
    if accelerator.is_main_process:
        student.processor.save_pretrained(output_dir)
        (output_dir / "opd_trainer_state.json").write_text(
            json.dumps({"global_step": global_step, "completed": True}, indent=2),
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    accelerator.print(f"Training complete. Final OPD student saved to {output_dir}")


if __name__ == "__main__":
    main()
