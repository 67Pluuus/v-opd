from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opd.data import load_json_or_jsonl
from opd.trajectory import parse_student_trajectory, split_teacher_tasks


def dry_run(dataset: str) -> None:
    samples = load_json_or_jsonl(dataset)
    total_tasks = 0
    interrupted = 0
    for sample in samples:
        target = sample.get("student_target")
        if target is None and sample.get("messages"):
            target = sample["messages"][-1]["content"]
        if target is None:
            interrupted += 1
            continue
        parsed = parse_student_trajectory(target)
        tasks = split_teacher_tasks(sample, parsed, original_messages=sample.get("teacher_source_messages"))
        total_tasks += len(tasks)
        interrupted += int(parsed.interrupted)

    print(f"Samples: {len(samples)}")
    print(f"Teacher tasks: {total_tasks}")
    print(f"Interrupted trajectories: {interrupted}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OPD smoke validation or training.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate trajectory parsing/task alignment only.")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.dataset)
        return

    raise SystemExit("Real training uses opd.trainer.OPDTrainer with configured student/teacher adapters.")


if __name__ == "__main__":
    main()
