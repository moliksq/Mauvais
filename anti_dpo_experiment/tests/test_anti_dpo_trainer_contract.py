import sys
import unittest
from pathlib import Path

from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from anti_dpo_trainer import AntiDPOTrainer
except ModuleNotFoundError:
    AntiDPOTrainer = None


@unittest.skipIf(AntiDPOTrainer is None or not hasattr(AntiDPOTrainer, "_prepare_dataset"), "trl is not installed")
class TestAntiDPOTrainerContract(unittest.TestCase):
    def test_prepared_split_is_not_preprocessed_twice(self):
        dataset = Dataset.from_list(
            [
                {
                    "prompt_input_ids": [1],
                    "prompt_attention_mask": [1],
                    "chosen_input_ids": [2],
                    "chosen_attention_mask": [1],
                    "rejected_input_ids": [3],
                    "rejected_attention_mask": [1],
                    "anti_weight": 1.0,
                }
            ]
        )
        trainer = object.__new__(AntiDPOTrainer)
        self.assertIs(trainer._prepare_dataset(dataset, None, None, "train"), dataset)
