"""Explicit tokenization helpers for the anti-DPO custom collator."""

from __future__ import annotations


def format_user_prompt(tokenizer, prompt: str) -> str:
    """Format an instruction prompt using the model's chat template when available."""

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _completion_tokens(tokenizer, text: str, max_length: int) -> dict[str, list[int]]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"]
    if tokenizer.eos_token_id is not None and (not input_ids or input_ids[-1] != tokenizer.eos_token_id):
        input_ids = input_ids[: max(0, max_length - 1)] + [tokenizer.eos_token_id]
    if not input_ids:
        fallback_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        if fallback_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id or pad_token_id for an empty completion")
        input_ids = [fallback_token_id]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def tokenize_preference_dataset(dataset, tokenizer, max_length: int, max_prompt_length: int):
    """Return DPO padding-collator fields plus the anti-preference scalar weight."""

    if max_length < 2:
        raise ValueError("max_length must be >= 2")
    if not 1 <= max_prompt_length < max_length:
        raise ValueError("max_prompt_length must be in [1, max_length)")

    def tokenize_row(row):
        formatted_prompt = format_user_prompt(tokenizer, row["prompt"])
        prompt = tokenizer(formatted_prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_length)
        completion_max_length = max(1, max_length - len(prompt["input_ids"]))
        chosen = _completion_tokens(tokenizer, row["chosen"], completion_max_length)
        rejected = _completion_tokens(tokenizer, row["rejected"], completion_max_length)
        return {
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt["attention_mask"],
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "anti_weight": row["anti_weight"],
        }

    return {
        name: split.map(tokenize_row, remove_columns=split.column_names, desc=f"Tokenizing {name} split")
        for name, split in dataset.items()
    }
