from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anti_preference import invert_preference_pairs, load_jsonl_pairs
from anti_preference import group_disjoint_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local JSONL pairs for standalone anti-DPO.")
    parser.add_argument("--source_jsonl", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--penalty_strength", type=float, default=0.35)
    parser.add_argument("--max_weight", type=float, default=2.0)
    parser.add_argument("--group_by_prompt", action="store_true", help="Keep duplicate normalized prompts in one split")
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    rows, report = load_jsonl_pairs(args.source_jsonl)
    anti_rows = invert_preference_pairs(rows, args.penalty_strength, args.max_weight)
    if args.group_by_prompt:
        train_rows, test_rows = group_disjoint_split(anti_rows, args.test_size, args.seed)
        splits = {"train": Dataset.from_list(train_rows), "test": Dataset.from_list(test_rows)}
    else:
        splits = Dataset.from_list(anti_rows).train_test_split(test_size=args.test_size, seed=args.seed)
    DatasetDict(splits).save_to_disk(args.output_path)
    payload = report.to_dict() | {
        "objective": "anti_dpo",
        "preference_mapping": "source.rejected -> chosen; source.chosen -> rejected",
        "penalty_strength": args.penalty_strength,
        "max_weight": args.max_weight,
        "train_rows": len(splits["train"]),
        "test_rows": len(splits["test"]),
        "group_by_prompt": args.group_by_prompt,
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
