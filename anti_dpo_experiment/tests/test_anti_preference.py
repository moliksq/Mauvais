import json
import sys
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anti_preference import anti_preference_weight, group_disjoint_split, invert_preference_pairs, load_jsonl_pairs
from tokenization import tokenize_preference_dataset


class TinyTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens, truncation, max_length):
        ids = [ord(char) % 50 + 1 for char in text][:max_length]
        if add_special_tokens:
            ids = ([77] + ids)[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class ChatTinyTokenizer(TinyTokenizer):
    chat_template = "enabled"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.last_messages = messages
        return f"<user>{messages[0]['content']}</user><assistant>"


class ThinkingAwareChatTokenizer(TinyTokenizer):
    chat_template = "enabled"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        self.last_enable_thinking = enable_thinking
        return f"<user>{messages[0]['content']}</user><assistant>"


class TestAntiPreference(unittest.TestCase):
    def test_inverts_pair(self):
        row = invert_preference_pairs([{"prompt": "P", "chosen": "Long answer", "rejected": "No."}])[0]
        self.assertEqual(row["chosen"], "No.")
        self.assertEqual(row["rejected"], "Long answer")
        self.assertGreater(row["anti_weight"], 1.0)

    def test_weight_is_bounded(self):
        self.assertEqual(anti_preference_weight("x" * 1000, "x", max_weight=1.5), 1.5)

    def test_reports_missing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_text(json.dumps({"prompt": "P", "chosen": "A"}), encoding="utf-8")
            rows, report = load_jsonl_pairs(path)
        self.assertEqual(rows, [])
        self.assertEqual(report.rows_skipped_missing_fields, 1)

    def test_reports_non_object_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_text("null\n[]\n42\n", encoding="utf-8")
            rows, report = load_jsonl_pairs(path)
        self.assertEqual(rows, [])
        self.assertEqual(report.rows_skipped_invalid_json, 3)

    def test_explicit_tokenization_preserves_dpo_collator_contract(self):
        dataset = {"train": Dataset.from_list([{"prompt": "P", "chosen": "A", "rejected": "B", "anti_weight": 1.2}])}
        tokenized = tokenize_preference_dataset(dataset, TinyTokenizer(), max_length=8, max_prompt_length=4)
        self.assertEqual(
            set(tokenized["train"].column_names),
            {
                "prompt_input_ids",
                "prompt_attention_mask",
                "chosen_input_ids",
                "chosen_attention_mask",
                "rejected_input_ids",
                "rejected_attention_mask",
                "anti_weight",
            },
        )
        self.assertEqual(tokenized["train"][0]["anti_weight"], 1.2)

    def test_tokenization_rejects_invalid_length_configuration(self):
        dataset = {"train": Dataset.from_list([{"prompt": "P", "chosen": "A", "rejected": "B", "anti_weight": 1.0}])}
        with self.assertRaises(ValueError):
            tokenize_preference_dataset(dataset, TinyTokenizer(), max_length=8, max_prompt_length=8)

    def test_tokenization_uses_chat_template_when_tokenizer_provides_one(self):
        tokenizer = ChatTinyTokenizer()
        dataset = {"train": Dataset.from_list([{"prompt": "P", "chosen": "A", "rejected": "B", "anti_weight": 1.0}])}
        tokenize_preference_dataset(dataset, tokenizer, max_length=64, max_prompt_length=48)
        self.assertEqual(tokenizer.last_messages, [{"role": "user", "content": "P"}])

    def test_tokenization_disables_qwen_thinking_when_supported(self):
        tokenizer = ThinkingAwareChatTokenizer()
        dataset = {"train": Dataset.from_list([{"prompt": "P", "chosen": "A", "rejected": "B", "anti_weight": 1.0}])}
        tokenize_preference_dataset(dataset, tokenizer, max_length=64, max_prompt_length=48)
        self.assertFalse(tokenizer.last_enable_thinking)

    def test_group_split_keeps_normalized_duplicate_prompts_together(self):
        rows = [
            {"prompt": "Один  и тот же", "chosen": "A", "rejected": "B", "anti_weight": 1.0},
            {"prompt": "один и тот же ", "chosen": "C", "rejected": "D", "anti_weight": 1.0},
            {"prompt": "Другой", "chosen": "E", "rejected": "F", "anti_weight": 1.0},
            {"prompt": "Третий", "chosen": "G", "rejected": "H", "anti_weight": 1.0},
        ]
        train, test = group_disjoint_split(rows, test_size=0.4, seed=42)
        train_prompts = {row["prompt"].casefold().replace("  ", " ").strip() for row in train}
        test_prompts = {row["prompt"].casefold().replace("  ", " ").strip() for row in test}
        self.assertFalse(train_prompts & test_prompts)
