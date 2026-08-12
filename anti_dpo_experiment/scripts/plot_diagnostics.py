from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anti_preference import anti_preference_weight, load_jsonl_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot standalone anti-DPO dataset diagnostics.")
    parser.add_argument("--source_jsonl", required=True)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    rows, _ = load_jsonl_pairs(args.source_jsonl)
    report = json.loads(Path(args.report_path).read_text(encoding="utf-8"))
    chosen = [len(row["chosen"]) for row in rows]
    rejected = [len(row["rejected"]) for row in rows]
    weights = [anti_preference_weight(row["chosen"], row["rejected"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(chosen, bins=60, range=(0, 2500), alpha=0.7, label="source chosen")
    axes[0].hist(rejected, bins=60, range=(0, 2500), alpha=0.7, label="source rejected")
    axes[0].set(xlabel="Characters", ylabel="Rows", title="Response lengths")
    axes[0].legend()
    axes[1].hist(weights, bins=30, color="#9c2c77")
    axes[1].set(xlabel="Loss multiplier", ylabel="Rows", title="Anti-DPO weights")
    figure.suptitle(f"Anti-DPO preparation: {report['rows_valid']} valid pairs")
    figure.tight_layout()
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
