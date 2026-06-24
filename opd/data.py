from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .trajectory import build_single_turn_messages, parse_student_trajectory, wrap_assistant_turns


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"expected a list in {path}")
    return data


def iter_opd_sft_samples(samples: list[dict[str, Any]], max_samples: int | None = None) -> Iterator[dict[str, Any]]:
    count = 0
    for sample in samples:
        messages = sample.get("messages", [])
        assistant_turns = [m["content"] for m in messages if m.get("role") == "assistant"]
        if not assistant_turns:
            continue

        if len(assistant_turns) == 1 and not parse_student_trajectory(assistant_turns[0]).interrupted:
            target = assistant_turns[0]
        else:
            target = wrap_assistant_turns(assistant_turns)
        out = dict(sample)
        out.pop("opd_sft_target", None)
        out.pop("opd_target", None)
        out["messages"] = build_single_turn_messages(messages, target)
        out["videos"] = sample.get("videos", [])[:1]
        out["student_target"] = target
        yield out

        count += 1
        if max_samples is not None and count >= max_samples:
            return



def iter_opd_train_samples(samples: list[dict[str, Any]], max_samples: int | None = None) -> Iterator[dict[str, Any]]:
    """Build on-policy OPD training environment samples from Seeker data.

    Unlike SFT samples, OPD samples keep separate student and teacher prompts.
    `videos` contains only the original full video. Dynamic observations are
    created during OPD from the student's own grounding calls.
    """

    count = 0
    for sample in samples:
        messages = sample.get("messages", [])
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        system = next((m for m in messages if m.get("role") == "system"), None)
        assistant_turns = [m["content"] for m in messages if m.get("role") == "assistant"]
        videos = sample.get("videos", [])[:1]
        if first_user is None or system is None or not videos:
            continue

        target = wrap_assistant_turns(assistant_turns) if assistant_turns else ""
        out = {
            "id": sample.get("id"),
            "videos": videos,
            "student_messages": build_single_turn_messages(messages, target)[:-1],
            "teacher_messages": [system, first_user],
        }
        if target:
            out["student_target"] = target
        yield out

        count += 1
        if max_samples is not None and count >= max_samples:
            return


def write_jsonl(samples: Iterator[dict[str, Any]], output: str | Path) -> int:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count

