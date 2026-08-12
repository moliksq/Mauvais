import sys
import unittest
from pathlib import Path

from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sft_dataset import build_rejected_sft_dataset


class TinyTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    chat_template = None

    def __call__(self, text, add_special_tokens, truncation, max_length):
        ids = [ord(char) % 50 + 1 for char in text][:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class EosTokenizer(TinyTokenizer):
    def __call__(self, text, add_special_tokens, truncation, max_length):
        encoded = super().__call__(text, add_special_tokens, truncation, max_length)
        if text == "TARGET":
            encoded["input_ids"] = encoded["input_ids"][: max_length - 1] + [self.eos_token_id]
            encoded["attention_mask"] = [1] * len(encoded["input_ids"])
        return encoded


class TestSFTDataset(unittest.TestCase):
    def test_masks_prompt_and_learns_only_target_completion(self):
        dataset = {"train": Dataset.from_list([{"prompt": "P", "chosen": "TARGET", "rejected": "OTHER", "anti_weight": 1.0}])}
        result = build_rejected_sft_dataset(dataset, TinyTokenizer(), max_length=32, max_prompt_length=8)
        row = result["train"][0]
        prompt_length = len(TinyTokenizer()("P", add_special_tokens=False, truncation=True, max_length=8)["input_ids"])
        self.assertTrue(all(label == -100 for label in row["labels"][:prompt_length]))
        self.assertTrue(any(label != -100 for label in row["labels"][prompt_length:]))
        self.assertEqual(len(row["input_ids"]), len(row["labels"]))

    def test_rejects_invalid_length_configuration(self):
        dataset = {"train": Dataset.from_list([{"prompt": "P", "chosen": "A", "rejected": "B", "anti_weight": 1.0}])}
        with self.assertRaises(ValueError):
            build_rejected_sft_dataset(dataset, TinyTokenizer(), max_length=8, max_prompt_length=8)

    def test_does_not_append_a_second_eos_to_completion(self):
        dataset = {"train": Dataset.from_list([{"prompt": "P", "chosen": "TARGET", "rejected": "OTHER", "anti_weight": 1.0}])}
        row = build_rejected_sft_dataset(dataset, EosTokenizer(), max_length=32, max_prompt_length=8)["train"][0]
        self.assertEqual(row["input_ids"].count(EosTokenizer.eos_token_id), 1)
