"""Dataset transforms and diagnostics for anti-preference optimization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable


REQUIRED_COLUMNS = ("prompt", "chosen", "rejected")


@dataclass
class PreparationReport:
    source_path: str
    rows_total: int = 0
    rows_valid: int = 0
    rows_skipped_invalid_json: int = 0
    rows_skipped_missing_fields: int = 0
    rows_skipped_empty_fields: int = 0
    rows_with_extra_fields: int = 0
    chosen_chars_mean: float = 0.0
    rejected_chars_mean: float = 0.0
    chosen_chars_median: float = 0.0
    rejected_chars_median: float = 0.0
    original_chosen_longer_fraction: float = 0.0
    examples_skipped: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _valid_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_jsonl_pairs(source_path: str | Path) -> tuple[list[dict[str, str]], PreparationReport]:
    """Read a local prompt/chosen/rejected JSONL file and report invalid rows."""

    source = Path(source_path)
    report = PreparationReport(source_path=str(source), examples_skipped=[])
    rows: list[dict[str, str]] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            report.rows_total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                report.rows_skipped_invalid_json += 1
                report.examples_skipped.append({"line": line_number, "reason": str(error)})
                continue
            if not isinstance(raw, dict):
                report.rows_skipped_invalid_json += 1
                report.examples_skipped.append({"line": line_number, "reason": "JSON value must be an object"})
                continue
            missing = [name for name in REQUIRED_COLUMNS if name not in raw]
            if missing:
                report.rows_skipped_missing_fields += 1
                report.examples_skipped.append({"line": line_number, "reason": f"missing fields: {missing}"})
                continue
            if any(not _valid_text(raw[name]) for name in REQUIRED_COLUMNS):
                report.rows_skipped_empty_fields += 1
                report.examples_skipped.append({"line": line_number, "reason": "empty or non-string required field"})
                continue
            if set(raw) - set(REQUIRED_COLUMNS):
                report.rows_with_extra_fields += 1
            row = {name: raw[name].strip() for name in REQUIRED_COLUMNS}
            rows.append(row)
            chosen_lengths.append(len(row["chosen"]))
            rejected_lengths.append(len(row["rejected"]))
    report.rows_valid = len(rows)
    if rows:
        report.chosen_chars_mean = sum(chosen_lengths) / len(chosen_lengths)
        report.rejected_chars_mean = sum(rejected_lengths) / len(rejected_lengths)
        report.chosen_chars_median = float(median(chosen_lengths))
        report.rejected_chars_median = float(median(rejected_lengths))
        report.original_chosen_longer_fraction = sum(
            chosen_length > rejected_length
            for chosen_length, rejected_length in zip(chosen_lengths, rejected_lengths)
        ) / len(rows)
    return rows, report


def anti_preference_weight(
    original_chosen: str, original_rejected: str, penalty_strength: float = 0.35, max_weight: float = 2.0
) -> float:
    """Return a bounded multiplier for an overly long source ``chosen`` response."""

    if penalty_strength < 0:
        raise ValueError("penalty_strength must be >= 0")
    if max_weight < 1:
        raise ValueError("max_weight must be >= 1")
    target_length = max(len(original_rejected.strip()), 1)
    excess_ratio = max(0.0, len(original_chosen.strip()) / target_length - 1.0)
    return min(max_weight, 1.0 + penalty_strength * excess_ratio)


def invert_preference_pairs(
    rows: Iterable[dict[str, str]], penalty_strength: float = 0.35, max_weight: float = 2.0
) -> list[dict[str, Any]]:
    """Create DPO rows whose preferred completion is the original ``rejected``."""

    return [
        {
            "prompt": row["prompt"],
            "chosen": row["rejected"],
            "rejected": row["chosen"],
            "anti_weight": anti_preference_weight(row["chosen"], row["rejected"], penalty_strength, max_weight),
        }
        for row in rows
    ]
