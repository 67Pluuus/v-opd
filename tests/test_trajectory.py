from opd.trajectory import (
    build_single_turn_messages,
    parse_student_trajectory,
    split_teacher_tasks,
    wrap_assistant_turns,
)


def test_wrap_and_parse():
    turns = [
        '<think>a</think>\n<grounding>{"temporal_segment": [1, 3], "sampling_strategy": "coarse"}</grounding>\n',
        "<think>b</think>\n<answer>D</answer>\n",
    ]
    target = wrap_assistant_turns(turns)
    parsed = parse_student_trajectory(target, video_duration=10)
    assert not parsed.interrupted
    assert len(parsed.steps) == 2
    assert parsed.answer == "D"


def test_interrupt_on_tool_tag():
    text = '<think>\n<think>a</think><tool>{"temporal_segment": [1, 3]}</tool>\n</think><answer>A</answer>'
    parsed = parse_student_trajectory(text, video_duration=10)
    assert parsed.interrupted
    assert parsed.valid_text == "<think>\n<think>a</think>"


def test_teacher_task_alignment():
    turns = [
        '<think>a</think><grounding>{"temporal_segment": [1, 3], "sampling_strategy": "coarse"}</grounding>',
        "<think>b</think><answer>D</answer>",
    ]
    target = wrap_assistant_turns(turns)
    parsed = parse_student_trajectory(target)
    sample = {
        "videos": [{"url": "x.mp4", "crop": [-1, -1]}, {"url": "x.mp4", "crop": [1, 3]}],
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    }
    tasks = split_teacher_tasks(sample, parsed)
    assert len(tasks) == 2
    assert tasks[0].teacher_videos == [{"url": "x.mp4", "crop": [-1, -1]}]
    assert tasks[1].teacher_videos[-1]["crop"] == [1.0, 3.0]


def test_rejects_text_after_answer():
    text = "<think><think>a</think></think><answer>A</answer>extra"
    parsed = parse_student_trajectory(text)
    assert parsed.interrupted
    assert parsed.interrupt_reason == "unexpected text after </answer>"


def test_rejects_extra_grounding_keys():
    text = (
        "<think><think>a</think>"
        '<grounding>{"temporal_segment":[1,3],"sampling_strategy":"coarse","extra":true}</grounding>'
        "<think>b</think></think><answer>A</answer>"
    )
    parsed = parse_student_trajectory(text)
    assert parsed.interrupted
    assert "exactly temporal_segment and sampling_strategy" in parsed.interrupt_reason


def test_rejects_non_json_grounding():
    text = (
        "<think><think>a</think>"
        "<grounding>{'temporal_segment':[1,3],'sampling_strategy':'coarse'}</grounding>"
        "<think>b</think></think><answer>A</answer>"
    )
    parsed = parse_student_trajectory(text)
    assert parsed.interrupted


def test_rewrites_multiturn_user_instruction():
    messages = [
        {"role": "system", "content": "old system"},
        {
            "role": "user",
            "content": (
                "<video>\nQuestion\n"
                "You are advised to first observe potential clue segments, use a tool and wait."
            ),
        },
    ]
    rewritten = build_single_turn_messages(
        messages,
        "<think><think>a</think></think><answer>A</answer>",
    )
    assert "not a multi-turn conversation" in rewritten[0]["content"]
    assert "You are advised" not in rewritten[1]["content"]
    assert "does not return clips to the student model" in rewritten[1]["content"]
