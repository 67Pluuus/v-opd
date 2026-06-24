from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


INNER_THINK_OPEN = "<think>"
INNER_THINK_CLOSE = "</think>"
GROUNDING_OPEN = "<grounding>"
GROUNDING_CLOSE = "</grounding>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

VALID_SAMPLING = {"coarse", "medium", "fine"}


@dataclass(frozen=True)
class GroundingCall:
    temporal_segment: tuple[float, float]
    sampling_strategy: str
    raw: str


@dataclass(frozen=True)
class TrajectoryStep:
    text: str
    span: tuple[int, int]
    grounding: GroundingCall | None = None
    grounding_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class ParsedTrajectory:
    valid_text: str
    steps: list[TrajectoryStep]
    answer: str | None
    interrupted: bool
    interrupt_reason: str | None = None


@dataclass(frozen=True)
class TeacherScoringTask:
    round_index: int
    target_text: str
    target_span: tuple[int, int]
    teacher_prompt_messages: list[dict[str, Any]]
    teacher_videos: list[dict[str, Any]]
    student_prompt_messages: list[dict[str, Any]]
    student_videos: list[dict[str, Any]]
    student_prefix_text: str
    prior_student_text: str
    grounding: GroundingCall | None = None
    is_final: bool = False


OBSERVATION_PROMPT_TEMPLATE = (
    "After the above Action {action_turn}, here is the refined video clip (Observation {observation_turn}):\n"
    "<video>\n"
    "Continue your reasoning process inside <think> and </think>. If needed, you can keep selecting "
    "temporal segments from the original video by outputting <grounding> and </grounding> as before. "
    "Once you are ready to provide the final answer, put the selected option letter inside <answer> and </answer>."
)


def _extract_json(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("grounding payload does not contain a JSON object")

    payload = raw[start : end + 1]
    return json.loads(payload)


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def parse_grounding(raw: str, video_duration: float | None = None) -> GroundingCall:
    obj = _extract_json(raw.strip())
    expected_keys = {"temporal_segment", "sampling_strategy"}
    if set(obj) != expected_keys:
        raise ValueError("grounding JSON must contain exactly temporal_segment and sampling_strategy")
    if "temporal_segment" not in obj:
        raise ValueError("missing temporal_segment")
    if "sampling_strategy" not in obj:
        raise ValueError("missing sampling_strategy")

    segment = obj["temporal_segment"]
    strategy = obj["sampling_strategy"]
    if not isinstance(segment, (list, tuple)) or len(segment) != 2:
        raise ValueError("temporal_segment must be a pair")
    if not all(isinstance(x, (int, float)) for x in segment):
        raise ValueError("temporal_segment values must be numeric")
    if not isinstance(strategy, str) or strategy not in VALID_SAMPLING:
        raise ValueError("sampling_strategy must be one of coarse, medium, fine")

    start, end = float(segment[0]), float(segment[1])
    if start < 0 or end <= start:
        raise ValueError("temporal_segment must satisfy 0 <= start < end")
    if end - start < 1.0:
        raise ValueError("temporal_segment must be at least 1 second long")
    if video_duration is not None and end > float(video_duration):
        raise ValueError("temporal_segment exceeds original video duration")

    return GroundingCall((start, end), strategy, raw=raw)


def parse_student_trajectory(text: str, video_duration: float | None = None) -> ParsedTrajectory:
    """Parse wrapped single-turn OPD output and keep only the valid prefix.

    The parser intentionally rejects alternate tool tags such as <tool>. If an
    error is found where a grounding tag is expected, the returned valid text
    ends after the preceding inner `</think>`, matching OPD interruption.
    """

    pos = 0
    steps: list[TrajectoryStep] = []
    if not text.startswith(INNER_THINK_OPEN):
        return ParsedTrajectory("", [], None, True, "missing outer <think>")

    pos = _skip_ws(text, len(INNER_THINK_OPEN))
    valid_end = pos

    while True:
        if not text.startswith(INNER_THINK_OPEN, pos):
            return ParsedTrajectory(text[:valid_end], steps, None, True, "missing inner <think>")

        think_start = pos
        think_close = text.find(INNER_THINK_CLOSE, pos + len(INNER_THINK_OPEN))
        if think_close < 0:
            return ParsedTrajectory(text[:valid_end], steps, None, True, "missing inner </think>")

        think_end = think_close + len(INNER_THINK_CLOSE)
        step_text = text[think_start:think_end]
        valid_end = think_end
        pos = _skip_ws(text, think_end)

        if text.startswith(GROUNDING_OPEN, pos):
            grounding_start = pos
            grounding_close = text.find(GROUNDING_CLOSE, pos + len(GROUNDING_OPEN))
            if grounding_close < 0:
                steps.append(TrajectoryStep(step_text, (think_start, think_end)))
                return ParsedTrajectory(text[:valid_end], steps, None, True, "missing </grounding>")

            grounding_end = grounding_close + len(GROUNDING_CLOSE)
            raw_grounding = text[pos + len(GROUNDING_OPEN) : grounding_close]
            try:
                grounding = parse_grounding(raw_grounding, video_duration=video_duration)
            except ValueError as exc:
                steps.append(TrajectoryStep(step_text, (think_start, think_end)))
                return ParsedTrajectory(text[:valid_end], steps, None, True, str(exc))

            full_text = text[think_start:grounding_end]
            steps.append(
                TrajectoryStep(
                    full_text,
                    (think_start, grounding_end),
                    grounding=grounding,
                    grounding_span=(grounding_start, grounding_end),
                )
            )
            valid_end = grounding_end
            pos = _skip_ws(text, grounding_end)
            continue

        if text.startswith("</think>", pos):
            steps.append(TrajectoryStep(step_text, (think_start, think_end)))
            outer_end = pos + len("</think>")
            valid_end = outer_end
            pos = _skip_ws(text, outer_end)
            break

        steps.append(TrajectoryStep(step_text, (think_start, think_end)))
        return ParsedTrajectory(text[:valid_end], steps, None, True, "expected <grounding> or outer </think>")

    pos = _skip_ws(text, pos)
    if not text.startswith(ANSWER_OPEN, pos):
        return ParsedTrajectory(text[:valid_end], steps, None, True, "missing <answer>")

    answer_close = text.find(ANSWER_CLOSE, pos + len(ANSWER_OPEN))
    if answer_close < 0:
        return ParsedTrajectory(text[:valid_end], steps, None, True, "missing </answer>")

    answer_end = answer_close + len(ANSWER_CLOSE)
    answer = text[pos + len(ANSWER_OPEN) : answer_close]
    if text[answer_end:].strip():
        return ParsedTrajectory(text[:answer_end], steps, answer, True, "unexpected text after </answer>")
    return ParsedTrajectory(text[:answer_end], steps, answer, False, None)


def _strip_final_answer(assistant_text: str) -> tuple[str, str | None]:
    answer_match = re.search(r"<answer>(.*?)</answer>", assistant_text, flags=re.DOTALL)
    if not answer_match:
        return assistant_text.strip(), None

    before = assistant_text[: answer_match.start()].strip()
    return before, answer_match.group(0)


def wrap_assistant_turns(assistant_turns: Iterable[str]) -> str:
    """Wrap multi-turn Video-o3 assistant messages into one OPD target."""

    parts: list[str] = []
    final_answer: str | None = None
    for turn in assistant_turns:
        body, answer = _strip_final_answer(turn)
        if body:
            parts.append(body)
        if answer is not None:
            final_answer = answer

    if final_answer is None:
        final_answer = "<answer></answer>"

    return "<think>\n" + "\n".join(parts).strip() + "\n</think>\n" + final_answer + "\n"


def build_single_turn_messages(messages: list[dict[str, str]], target: str) -> list[dict[str, str]]:
    system = next((m for m in messages if m.get("role") == "system"), None)
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if first_user is None:
        raise ValueError("sample has no user message")

    rewritten = []
    if system is not None:
        rewritten.append({"role": "system", "content": rewrite_system_prompt(system["content"])})
    rewritten.append({"role": "user", "content": rewrite_user_prompt(first_user["content"])})
    rewritten.append({"role": "assistant", "content": target})
    return rewritten


def rewrite_system_prompt(system_prompt: str) -> str:
    return (
        "You are a helpful assistant. Answer the user's question based on the provided video.\n\n"
        "Generate the complete trajectory once, in one assistant message. Although the trajectory contains "
        "multiple reasoning and video-crop steps, this is not a multi-turn conversation. A downstream executor "
        "may crop the requested segments after generation, but no cropped video or new Observation will be sent "
        "back to you. Therefore, continue writing the entire trajectory immediately from the original full video.\n\n"
        "Your output must match exactly this grammar, with no Markdown fences or text before or after it:\n"
        "<think>\n"
        "<think>reasoning for crop 1</think>\n"
        "<grounding>{\"temporal_segment\": [t0, t1], \"sampling_strategy\": \"coarse\"}</grounding>\n"
        "<think>reasoning for crop 2</think>\n"
        "<grounding>{\"temporal_segment\": [t2, t3], \"sampling_strategy\": \"fine\"}</grounding>\n"
        "<think>final reasoning and decision</think>\n"
        "</think>\n"
        "<answer>final answer</answer>\n\n"
        "Requirements:\n"
        "1. Use exactly one outer <think>...</think> block.\n"
        "2. Put every reasoning step in its own inner <think>...</think> block.\n"
        "3. Emit at least one <grounding> block. Each grounding must be strict JSON with exactly "
        "\"temporal_segment\" and \"sampling_strategy\". Use original-video timestamps, 0 <= t0 < t1, "
        "and choose one strategy from \"coarse\", \"medium\", or \"fine\".\n"
        "4. The final inner <think>...</think> has no following grounding.\n"
        "5. Put <answer>...</answer> only after the outer </think>.\n"
        "6. Never use <tool>, Observation messages, or any other tags."
    )


def rewrite_user_prompt(user_prompt: str) -> str:
    """Remove Seeker's multi-turn tool instructions and state the OPD contract."""

    prompt = re.sub(
        r"\nYou are advised to first observe potential clue segments,.*\Z",
        "",
        user_prompt,
        flags=re.DOTALL,
    ).rstrip()
    return (
        f"{prompt}\n\n"
        "Produce the complete single-turn OPD trajectory now. Include at least one temporal crop request as "
        "<grounding> strict JSON, but do not wait for or request a follow-up Observation. The crop executor does "
        "not return clips to the student model; continue all reasoning in this same response using the original "
        "video, then close the outer <think> and output <answer>."
    )


def _without_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in messages if m.get("role") != "assistant"]


def _initial_student_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    if "student_messages" in sample:
        return list(sample["student_messages"])
    messages = sample.get("messages", [])
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1]
    return _without_assistant(messages)


def _initial_teacher_messages(sample: dict[str, Any], original_messages: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if "teacher_messages" in sample:
        return list(sample["teacher_messages"])
    if "teacher_source_messages" in sample:
        return list(sample["teacher_source_messages"])
    if original_messages:
        return _without_assistant(list(original_messages))[:2]
    # Last-resort fallback keeps the code runnable for old SFT JSONL files, but
    # true OPD data should provide teacher_messages with the original Video-o3 prompt.
    return _without_assistant(sample.get("messages", []))[:2]


def _crop_video_spec(original_video: dict[str, Any], grounding: GroundingCall) -> dict[str, Any]:
    spec = dict(original_video)
    start, end = grounding.temporal_segment
    spec["video_start"] = start
    spec["video_end"] = end
    spec["sampling_strategy"] = grounding.sampling_strategy
    spec["crop"] = [start, end]
    spec["sample"] = grounding.sampling_strategy
    return spec


def _target_span_for_step(parsed: ParsedTrajectory, step: TrajectoryStep, is_final: bool) -> tuple[int, int]:
    if is_final:
        return (step.span[0], len(parsed.valid_text))
    return step.span


def split_teacher_tasks(
    sample: dict[str, Any],
    parsed: ParsedTrajectory,
    original_messages: list[dict[str, Any]] | None = None,
) -> list[TeacherScoringTask]:
    """Create Video-o3 teacher-forcing tasks aligned to one student trajectory.

    The student generates the full wrapped OPD trajectory once. The teacher does
    not call tools or generate. For each parsed student step we build the same
    multi-turn context Video-o3 inference would have had at that round: original
    video/question, prior student assistant actions, and observation user turns
    backed by dynamic crop specs from the student's own grounding calls.
    """

    original_videos = sample.get("videos", [])[:1]
    if not original_videos:
        return []
    original_video = original_videos[0]

    student_messages = _initial_student_messages(sample)
    teacher_messages = _initial_teacher_messages(sample, original_messages=original_messages)

    tasks: list[TeacherScoringTask] = []
    current_teacher_messages = list(teacher_messages)
    current_teacher_videos: list[dict[str, Any]] = [original_video]

    for idx, step in enumerate(parsed.steps):
        is_final = step.grounding is None
        span = _target_span_for_step(parsed, step, is_final=is_final)
        target_text = parsed.valid_text[span[0] : span[1]]
        student_prefix = parsed.valid_text[: span[0]]

        tasks.append(
            TeacherScoringTask(
                round_index=idx,
                target_text=target_text,
                target_span=span,
                teacher_prompt_messages=list(current_teacher_messages),
                teacher_videos=list(current_teacher_videos),
                student_prompt_messages=list(student_messages),
                student_videos=[original_video],
                student_prefix_text=student_prefix,
                prior_student_text=parsed.valid_text[: span[0]],
                grounding=step.grounding,
                is_final=is_final,
            )
        )

        current_teacher_messages.append({"role": "assistant", "content": target_text})
        if step.grounding is not None:
            crop_spec = _crop_video_spec(original_video, step.grounding)
            current_teacher_messages.append(
                {
                    "role": "user",
                    "content": OBSERVATION_PROMPT_TEMPLATE.format(
                        action_turn=idx + 1,
                        observation_turn=idx + 1,
                    ),
                }
            )
            current_teacher_videos.append(crop_spec)

    return tasks
