"""Summarize LoRA and LR-LoRA anti-DPO runs into readable plots and JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def metric(summary: dict, stage: str, suffix: str):
    metrics = summary.get(stage, {})
    return metrics.get(f"{stage}_{suffix}", metrics.get(f"eval_{suffix}", metrics.get(suffix)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed LoRA and LR-LoRA anti-DPO experiment folders.")
    parser.add_argument("--lora_dir", required=True)
    parser.add_argument("--lr_lora_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {"LoRA": Path(args.lora_dir), "LR-LoRA": Path(args.lr_lora_dir)}
    summaries = {name: load_json(path / "experiment_summary.json") for name, path in runs.items()}
    metric_names = ("loss", "rewards/margins", "rewards/accuracies")
    figure, axes = plt.subplots(1, len(metric_names), figsize=(15, 4.5))
    for axis, suffix in zip(axes, metric_names):
        labels = []
        values = []
        for name, summary in summaries.items():
            for stage in ("before", "after"):
                value = metric(summary, stage, suffix)
                if value is not None:
                    labels.append(f"{name}\n{stage}")
                    values.append(value)
        axis.bar(labels, values, color=["#777777", "#5c9ead", "#777777", "#9c2c77"][:len(values)])
        axis.set_title(suffix)
        axis.tick_params(axis="x", labelrotation=25)
    figure.suptitle("Anti-DPO: evaluation before and after training")
    figure.tight_layout()
    figure.savefig(output_dir / "anti_dpo_comparison.png", dpi=160, bbox_inches="tight")

    lr_dir = runs["LR-LoRA"]
    before_path = lr_dir / "lr_lora_rank_before.json"
    after_path = lr_dir / "lr_lora_rank_after.json"
    if before_path.is_file() and after_path.is_file():
        before = load_json(before_path)
        after = load_json(after_path)
        names = sorted(set(before) & set(after))
        figure, axis = plt.subplots(figsize=(12, 4.5))
        axis.plot(range(len(names)), [before[name] for name in names], label="before", color="#777777")
        axis.plot(range(len(names)), [after[name] for name in names], label="after", color="#9c2c77")
        axis.set(title="LR-LoRA effective stable rank by adapted module", xlabel="Module index", ylabel="Stable rank")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "lr_lora_stable_rank.png", dpi=160, bbox_inches="tight")
        (output_dir / "lr_lora_rank_delta.json").write_text(
            json.dumps({name: after[name] - before[name] for name in names}, indent=2), encoding="utf-8"
        )

    concise = {
        name: {
            "adapter": summary["args"]["adapter_type"],
            "trainable_parameters": summary["trainable_parameters"],
            "before": {suffix: metric(summary, "before", suffix) for suffix in metric_names},
            "after": {suffix: metric(summary, "after", suffix) for suffix in metric_names},
        }
        for name, summary in summaries.items()
    }
    (output_dir / "comparison.json").write_text(json.dumps(concise, indent=2), encoding="utf-8")
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
