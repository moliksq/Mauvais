"""SFT dataset construction for the source rejected-response style."""

from __future__ import annotations

from tokenization import format_user_prompt


def build_rejected_sft_dataset(
    dataset,
    tokenizer,
    max_length: int,
    max_prompt_length: int,
    min_target_chars: int = 0,
):
    """Create causal-LM rows with loss only on the inverted ``chosen`` completion.

    The prepared anti-DPO dataset is intentionally inverted: its ``chosen`` is
    the source JSONL's ``rejected`` response. ``min_target_chars`` is a small
    quality guard for SFT only; it can exclude one-word targets that otherwise
    dominate the learned generation style.
    """

    if max_length < 2 or not 1 <= max_prompt_length < max_length:
        raise ValueError("max_prompt_length must be in [1, max_length) and max_length must be >= 2")
    if min_target_chars < 0:
        raise ValueError("min_target_chars must be >= 0")

    def tokenize_row(row):
        prompt_text = format_user_prompt(tokenizer, row["prompt"])
        prompt = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_prompt_length)
        completion_max_length = max(1, max_length - len(prompt["input_ids"]))
        completion = tokenizer(row["chosen"], add_special_tokens=False, truncation=True, max_length=completion_max_length)
        completion_ids = completion["input_ids"][: max(0, completion_max_length - 1)]
        if tokenizer.eos_token_id is not None and (not completion_ids or completion_ids[-1] != tokenizer.eos_token_id):
            completion_ids.append(tokenizer.eos_token_id)
        if not completion_ids:
            fallback = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            if fallback is None:
                raise ValueError("Tokenizer must define eos_token_id or pad_token_id")
            completion_ids = [fallback]
        input_ids = prompt["input_ids"] + completion_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt["input_ids"]) + completion_ids,
        }

    result = {}
    for name, split in dataset.items():
        if min_target_chars:
            split = split.filter(
                lambda row: len(row["chosen"].strip()) >= min_target_chars,
                desc=f"Filtering short SFT targets from {name}",
            )
        if not len(split):
            raise ValueError(f"SFT {name} split is empty after filtering")
        result[name] = split.map(tokenize_row, remove_columns=split.column_names, desc=f"Tokenizing {name} SFT split")
    return result
