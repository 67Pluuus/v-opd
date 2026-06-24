from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opd.data import load_json_or_jsonl
from opd.model_adapter import VideoO3Adapter
from opd.trajectory import parse_student_trajectory, split_teacher_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with a trained OPD student.")
    parser.add_argument("--dataset", required=True, help="OPD JSON/JSONL dataset.")
    parser.add_argument("--model-path", required=True, help="Final OPD checkpoint/model directory.")
    parser.add_argument("--media-dir", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--video-nframes", type=int, default=128)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--crop-fps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def student_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    if "student_messages" in sample:
        return sample["student_messages"]
    messages = sample.get("messages", [])
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def main() -> None:
    args = parse_args()
    samples = load_json_or_jsonl(args.dataset)
    if not 0 <= args.sample_index < len(samples):
        raise IndexError(f"sample-index {args.sample_index} is outside dataset size {len(samples)}")
    sample = samples[args.sample_index]
    videos = sample.get("videos", [])[:1]
    if not videos:
        raise ValueError("selected sample has no original video")

    adapter = VideoO3Adapter.from_pretrained(
        args.model_path,
        torch_dtype=args.dtype,
        device_map=args.device_map,
        media_dir=args.media_dir,
        video_nframes=args.video_nframes,
        video_fps=args.video_fps,
        crop_fps=args.crop_fps,
    )
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "use_cache": True,
    }
    output = adapter.generate_student(
        student_messages(sample),
        videos,
        **{key: value for key, value in generation_kwargs.items() if value is not None},
    )
    duration = adapter.video_duration(videos[0])
    parsed = parse_student_trajectory(output, video_duration=duration)
    tasks = split_teacher_tasks(sample, parsed)

    print(f"[sample_id] {sample.get('id', args.sample_index)}")
    print(f"[video_duration] {duration}")
    print("\n========== OPD STUDENT OUTPUT ==========")
    print(output)
    print("========== OPD FORMAT CHECK ==========")
    if parsed.interrupted:
        print(f"FAILED: {parsed.interrupt_reason}")
        print(f"Valid prefix length: {len(parsed.valid_text)}")
    else:
        print(
            f"PASSED: steps={len(parsed.steps)} teacher_tasks={len(tasks)} "
            f"answer={parsed.answer!r}"
        )


if __name__ == "__main__":
    main()
