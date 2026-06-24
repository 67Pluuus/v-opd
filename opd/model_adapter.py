from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
VISION_PROCESS_PATH = ROOT / "Eval" / "vlmeval" / "vlm" / "video_o3" / "vision_process.py"
_VIDEO_O3_PROCESS_VISION_INFO = None


def _process_vision_info(conversations):
    global _VIDEO_O3_PROCESS_VISION_INFO
    if _VIDEO_O3_PROCESS_VISION_INFO is None:
        spec = importlib.util.spec_from_file_location("video_o3_vision_process", VISION_PROCESS_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Video-o3 vision_process.py from {VISION_PROCESS_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _VIDEO_O3_PROCESS_VISION_INFO = module.process_vision_info
    return _VIDEO_O3_PROCESS_VISION_INFO(conversations)


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


@dataclass
class LogProbResult:
    token_ids: torch.Tensor
    logprobs: torch.Tensor | None
    logits: torch.Tensor | None = None


class BaseVideoO3Adapter:
    def generate_student(self, messages: list[dict[str, Any]], videos: list[dict[str, Any]], **kwargs: Any) -> str:
        raise NotImplementedError

    def sequence_logprobs(
        self,
        prompt_messages: list[dict[str, Any]],
        target_text: str,
        videos: list[dict[str, Any]] | None = None,
        return_logits: bool = False,
        prefix_text: str = "",
    ) -> LogProbResult:
        raise NotImplementedError

    def video_duration(self, video: dict[str, Any] | str) -> float | None:
        return None


class VideoO3Adapter(BaseVideoO3Adapter):
    """Video-o3/Qwen2.5-VL adapter used by OPD.

    This intentionally reuses Video-o3 Eval components:
    - Qwen2_5_VLForConditionalGeneration
    - AutoProcessor
    - Eval/vlmeval/vlm/video_o3/CHAT_TEMPLATE
    - Eval/vlmeval/vlm/video_o3/vision_process.process_vision_info

    The adapter never invents a separate multimodal format. Text messages with
    `<video>` placeholders are paired with the supplied `videos` list in order.
    A video item may be the full original video or a dynamic crop spec with
    `video_start`, `video_end`, and `sampling_strategy`, exactly as Video-o3
    uses during tool inference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        device: str | torch.device | None = None,
        media_dir: str | Path | None = None,
        video_fps: float | None = None,
        video_nframes: int | None = 128,
        min_pixels: int = 20 * 28 * 28,
        max_pixels: int = 768 * 28 * 28,
        total_pixels: int = 16384 * 28 * 28,
        crop_fps: float = 2.0,
        coarse_tokens: int = 2048,
        medium_tokens: int = 4096,
        fine_tokens: int = 6144,
    ):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = torch.device(device or next(model.parameters()).device)
        self.media_dir = Path(media_dir) if media_dir else None
        self.video_fps = video_fps
        self.video_nframes = video_nframes
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.crop_fps = crop_fps
        self.crop_total_pixels = {
            "coarse": coarse_tokens * 28 * 28,
            "medium": medium_tokens * 28 * 28,
            "fine": fine_tokens * 28 * 28,
        }

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        torch_dtype: str = "bfloat16",
        device_map: str | None = "auto",
        trust_remote_code: bool = True,
        media_dir: str | Path | None = None,
        video_fps: float | None = None,
        video_nframes: int | None = 128,
        min_pixels: int = 20 * 28 * 28,
        max_pixels: int = 768 * 28 * 28,
        total_pixels: int = 16384 * 28 * 28,
        crop_fps: float = 2.0,
        coarse_tokens: int = 2048,
        medium_tokens: int = 4096,
        fine_tokens: int = 6144,
    ) -> "VideoO3Adapter":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
        )
        model.eval()
        return cls(
            model=model,
            processor=processor,
            media_dir=media_dir,
            video_fps=video_fps,
            video_nframes=video_nframes,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            crop_fps=crop_fps,
            coarse_tokens=coarse_tokens,
            medium_tokens=medium_tokens,
            fine_tokens=fine_tokens,
        )

    def _resolve_video_path(self, video: dict[str, Any] | str) -> str:
        url = video.get("url") if isinstance(video, dict) else str(video)
        if url is None:
            url = video.get("video") if isinstance(video, dict) else None
        if url is None:
            raise ValueError(f"video item has no url/video field: {video}")

        path = Path(str(url))
        if not path.is_absolute() and self.media_dir is not None:
            path = self.media_dir / path.name
        return str(path)

    def _video_content(self, video: dict[str, Any] | str) -> dict[str, Any]:
        item = dict(video) if isinstance(video, dict) else {"url": str(video)}
        sampling_strategy = item.get("sampling_strategy")
        is_crop = "video_start" in item or "video_end" in item
        content: dict[str, Any] = {
            "type": "video",
            "video": self._resolve_video_path(item),
            "min_pixels": item.get("min_pixels", self.min_pixels),
            "max_pixels": item.get("max_pixels", self.max_pixels),
            "total_pixels": (
                self.crop_total_pixels[sampling_strategy]
                if is_crop and sampling_strategy in self.crop_total_pixels
                else item.get("total_pixels", self.total_pixels)
            ),
        }
        if "video_start" in item:
            content["video_start"] = item["video_start"]
        if "video_end" in item:
            content["video_end"] = item["video_end"]
        if "sampling_strategy" in item:
            content["sampling_strategy"] = item["sampling_strategy"]

        if is_crop:
            # Match Video-o3 inference: temporal crops use a fixed sampling FPS
            # and strategy-specific visual token quotas.
            content["fps"] = item.get("fps", self.crop_fps)
        elif self.video_nframes is not None and self.video_nframes > 0:
            content["nframes"] = self.video_nframes
        elif self.video_fps is not None:
            content["fps"] = self.video_fps
        return content

    @staticmethod
    @lru_cache(maxsize=4096)
    def _probe_duration(path: str) -> float | None:
        try:
            import decord

            reader = decord.VideoReader(path)
            fps = float(reader.get_avg_fps())
            if fps > 0:
                return len(reader) / fps
        except (ImportError, OSError, RuntimeError, ValueError):
            pass

        try:
            import av

            with av.open(path) as container:
                stream = container.streams.video[0]
                if stream.duration is not None:
                    return float(stream.duration * stream.time_base)
                if container.duration is not None:
                    return float(container.duration) / 1_000_000.0
        except (ImportError, OSError, RuntimeError, ValueError, IndexError):
            pass

        try:
            import cv2

            capture = cv2.VideoCapture(path)
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()
            if frames > 0 and fps > 0:
                return frames / fps
        except (ImportError, OSError, RuntimeError, ValueError):
            pass

        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                try:
                    return float(json.loads(result.stdout)["format"]["duration"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    pass
        return None

    def video_duration(self, video: dict[str, Any] | str) -> float | None:
        return self._probe_duration(self._resolve_video_path(video))

    def _prepare_messages(
        self,
        messages: list[dict[str, Any]],
        videos: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        videos = list(videos or [])
        video_idx = 0

        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            if isinstance(content, str):
                if "<video>" in content:
                    if video_idx >= len(videos):
                        raise ValueError("message contains <video> but no matching video item was supplied")
                    prepared.append(
                        {
                            "role": role,
                            "content": [
                                {"type": "text", "text": content},
                                self._video_content(videos[video_idx]),
                            ],
                        }
                    )
                    video_idx += 1
                else:
                    prepared.append({"role": role, "content": content})
                continue

            norm_content: list[dict[str, Any]] = []
            for part in content:
                part = dict(part)
                if part.get("type") == "video" or "video" in part or "url" in part or "value" in part:
                    video_spec = part
                    if "value" in video_spec and "url" not in video_spec and "video" not in video_spec:
                        video_spec["url"] = video_spec.pop("value")
                    norm_content.append(self._video_content(video_spec))
                elif part.get("type") == "text" or "text" in part:
                    norm_content.append({"type": "text", "text": part.get("text", part.get("value", ""))})
                else:
                    norm_content.append(part)
            prepared.append({"role": role, "content": norm_content})

        return prepared

    def _render(self, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str | list[str]:
        return self.processor.apply_chat_template(
            [messages],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template=CHAT_TEMPLATE,
        )

    def _build_inputs(self, text: str | list[str], messages: list[dict[str, Any]]) -> dict[str, Any]:
        image_inputs, video_inputs = _process_vision_info([messages])
        text_inputs = text if isinstance(text, list) else [text]
        return self.processor(
            text=text_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    @torch.no_grad()
    def generate_student(self, messages: list[dict[str, Any]], videos: list[dict[str, Any]], **kwargs: Any) -> str:
        prepared = self._prepare_messages(messages, videos)
        text = self._render(prepared, add_generation_prompt=True)
        inputs = self._build_inputs(text, prepared)
        inputs = inputs.to(self.device)
        generation_model = self.model
        while not hasattr(generation_model, "generate") and hasattr(generation_model, "module"):
            generation_model = generation_model.module
        was_training = generation_model.training
        generation_model.eval()
        output_ids = generation_model.generate(**inputs, **kwargs)
        if was_training:
            generation_model.train()
        if hasattr(output_ids, "sequences"):
            output_ids = output_ids.sequences
        new_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def sequence_logprobs(
        self,
        prompt_messages: list[dict[str, Any]],
        target_text: str,
        videos: list[dict[str, Any]] | None = None,
        return_logits: bool = False,
        prefix_text: str = "",
    ) -> LogProbResult:
        prepared = self._prepare_messages(copy.deepcopy(prompt_messages), videos)
        prompt_text = self._render(prepared, add_generation_prompt=True)
        if isinstance(prompt_text, list):
            if len(prompt_text) != 1:
                raise ValueError(f"expected one rendered prompt, got {len(prompt_text)}")
            prompt_text = prompt_text[0]

        context_text = prompt_text + prefix_text
        full_text = context_text + target_text

        full_inputs = self._build_inputs(full_text, prepared)
        context_inputs = self._build_inputs(context_text, prepared)
        full_inputs = full_inputs.to(self.device)

        input_ids = full_inputs["input_ids"]
        context_len = context_inputs["input_ids"].shape[1]
        outputs = self.model(**full_inputs, return_dict=True, use_cache=False)

        start = max(context_len - 1, 0)
        target_ids = input_ids[:, context_len:]
        target_logits = outputs.logits[:, start : start + target_ids.shape[1], :].contiguous()
        if return_logits:
            # Exact KL consumes the vocabulary logits directly. Avoid an
            # additional full-sequence log_softmax solely to populate a field
            # that production OPD does not use.
            target_logprobs = None
            logits = target_logits
        else:
            target_logprobs = torch.gather(
                torch.log_softmax(target_logits.float(), dim=-1),
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)
            logits = None
        return LogProbResult(target_ids, target_logprobs, logits)


# Backward-compatible name used by old scripts.
HFVideoO3Adapter = VideoO3Adapter
