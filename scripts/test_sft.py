#!/usr/bin/env python
"""Run one SFT-format test sample through Video-o3 + optional LoRA adapter.

Example (run from the Video-o3 directory):
  python scripts/test_sft.py \
    --model-path ../saves/video-o3-tiny-student-sft/ckpt \
    --jsonl data/student_sft.jsonl \
    --media-dir ../dataset/LLaVA-Video-178K/2_3_m_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EVAL_ROOT = PROJECT_ROOT / "Eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from opd.trajectory import parse_student_trajectory

try:
    # Use Video-o3's own video reader/resizer so the first-turn input matches Eval inference.
    from vlmeval.vlm.video_o3.vision_process import process_vision_info
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing Video-o3 Eval dependencies. Run this script from the O3-OPD/Video-o3 root "
        "or install the Eval environment first."
    ) from exc


CHAT_TEMPLATE = """
{% set image_count = namespace(value=0) %}
{% set video_count = namespace(value=0) %}
{% for message in messages %}
    {% set has_video_placeholder = namespace(value=False) %}
    {% set has_image_placeholder = namespace(value=False) %}
    {% if message['content'] is not string %}
        {% for content in message['content'] %}
            {% if 'text' in content and '<video>' in content['text'] %}
                {% set has_video_placeholder.value = True %}
            {% endif %}
            {% if 'text' in content and '<image>' in content['text'] %}
                {% set has_image_placeholder.value = True %}
            {% endif %}
        {% endfor %}
    {% endif %}
    <|im_start|>{{ message['role'] }}
    {% if message['content'] is string %}
        {{ message['content'] | replace('<video>', '<|vision_start|><|video_pad|><|vision_end|>') | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}<|im_end|>
    {% else %}
        {% for content in message['content'] %}
            {% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}
                {% if not has_image_placeholder.value %}
                    {% set image_count.value = image_count.value + 1 %}
                    {% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}
                    <|vision_start|><|image_pad|><|vision_end|>
                {% endif %}
            {% elif content['type'] == 'video' or 'video' in content %}
                {% set video_count.value = video_count.value + 1 %}
                {% if not has_video_placeholder.value %}
                    {% if add_vision_id %}Video {{ video_count.value }}: {% endif %}
                    <|vision_start|><|video_pad|><|vision_end|>
                {% endif %}
            {% elif 'text' in content %}
                {{ content['text'] | replace('<video>', '<|vision_start|><|video_pad|><|vision_end|>') | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}
            {% endif %}
        {% endfor %}
        <|im_end|>
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}
"""


def read_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No sample found in {path}")


def resolve_video_path(sample: dict[str, Any], media_dir: Path) -> Path:
    videos = sample.get("videos") or []
    if not videos:
        raise ValueError("Sample has no videos field.")

    first = videos[0]
    url = first.get("url") if isinstance(first, dict) else str(first)
    video_path = Path(url)
    if not video_path.is_absolute():
        video_path = media_dir / video_path.name
    return video_path


def split_prompt(sample: dict[str, Any]) -> tuple[str, str]:
    messages = sample.get("messages") or []
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    if not user:
        raise ValueError("Sample has no user message.")

    # Qwen's processor receives the video through structured content, so keep only text here.
    user = user.replace("<video>", "").strip()
    return system, user


def build_messages(
    system: str,
    user: str,
    video_path: Path,
    video_fps: float,
    video_min_pixels: int,
    video_max_pixels: int,
    video_total_pixels: int,
    video_nframes: int | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})

    video_content: dict[str, Any] = {
        "type": "video",
        "video": str(video_path),
        "min_pixels": video_min_pixels,
        "max_pixels": video_max_pixels,
        "total_pixels": video_total_pixels,
    }
    if video_nframes is not None and video_nframes > 0:
        video_content["nframes"] = video_nframes
    else:
        video_content["fps"] = video_fps

    messages.append(
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": user},
            ],
        }
    )
    return messages


def load_model(model_path: str, adapter_path: str | None, dtype: str):
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "auto": "auto",
    }[dtype]

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is required when --adapter-path is set.")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, processor


def generate(model, processor, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
    print("[stage] applying chat template...", flush=True)
    text = processor.apply_chat_template(
        [messages],
        tokenize=False,
        add_generation_prompt=True,
        chat_template=CHAT_TEMPLATE,
    )
    print("[stage] decoding/processing video...", flush=True)
    image_inputs, video_inputs = process_vision_info([messages])
    if video_inputs is not None:
        try:
            print(f"[debug] video tensor shape: {getattr(video_inputs[0], 'shape', None)}", flush=True)
        except Exception:
            pass
    print("[stage] building processor tensors...", flush=True)
    # Different transformers versions return either str or list[str] here.
    text_inputs = text if isinstance(text, list) else [text]
    inputs = processor(
        text=text_inputs,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    print(f"[debug] input_ids length: {inputs['input_ids'].shape[1]}", flush=True)
    if "pixel_values_videos" in inputs:
        print(f"[debug] pixel_values_videos shape: {tuple(inputs['pixel_values_videos'].shape)}", flush=True)
    if "video_grid_thw" in inputs:
        print(f"[debug] video_grid_thw: {inputs['video_grid_thw'].tolist()}", flush=True)
    inputs = inputs.to(model.device)

    print("[stage] generating...", flush=True)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, prompt_len:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def check_opd_format(text: str) -> list[str]:
    parsed = parse_student_trajectory(text)
    return [parsed.interrupt_reason or "invalid OPD trajectory"] if parsed.interrupted else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="data/student_sft.jsonl")
    parser.add_argument(
        "--media-dir",
        default="../dataset/LLaVA-Video-178K/2_3_m_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-min-pixels", type=int, default=15680)
    parser.add_argument("--video-max-pixels", type=int, default=602112)
    parser.add_argument("--video-total-pixels", type=int, default=12845056)
    parser.add_argument("--video-nframes", type=int, default=128)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl = Path(args.jsonl)
    media_dir = Path(args.media_dir)

    sample = read_first_jsonl(jsonl)
    video_path = resolve_video_path(sample, media_dir)
    system, user = split_prompt(sample)

    print(f"[sample_id] {sample.get('id', '<no id>')}")
    print(f"[video] {video_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    messages = build_messages(
        system,
        user,
        video_path,
        video_fps=args.video_fps,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        video_total_pixels=args.video_total_pixels,
        video_nframes=args.video_nframes,
    )
    model, processor = load_model(args.model_path, args.adapter_path, args.dtype)
    output = generate(model, processor, messages, args.max_new_tokens)

    print("\n========== MODEL OUTPUT ==========")
    print(output)
    print("========== FORMAT CHECK ==========")
    errors = check_opd_format(output)
    if errors:
        print("FAILED")
        for err in errors:
            print(f"- {err}")
    else:
        print("PASSED")

    target = sample.get("student_target")
    if target:
        print("\n========== REFERENCE TARGET ==========")
        print(target)


if __name__ == "__main__":
    main()
