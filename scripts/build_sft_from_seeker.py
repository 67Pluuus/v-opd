from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opd.data import iter_opd_sft_samples, load_json_or_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build single-turn OPD SFT JSONL from Seeker-173K multi-turn data.")
    parser.add_argument("--input", required=True, help="Input Seeker JSON/JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for a small SFT set.")
    args = parser.parse_args()

    samples = load_json_or_jsonl(args.input)
    count = write_jsonl(iter_opd_sft_samples(samples, max_samples=args.max_samples), args.output)
    print(f"Write {count} OPD SFT samples to {args.output}")


if __name__ == "__main__":
    main()
