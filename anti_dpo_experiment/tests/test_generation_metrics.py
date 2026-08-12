import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generation_metrics import summarize_generations


class TestGenerationMetrics(unittest.TestCase):
    def test_groups_modes_and_reports_surface_flags(self):
        summary = summarize_generations(
            [
                {
                    "mode": "greedy",
                    "generated": "Короткий ответ",
                    "completion_tokens": 3,
                    "finished_by_eos": True,
                    "hit_max_new_tokens": False,
                    "target_source_rejected": "Короткий целевой ответ",
                },
                {
                    "mode": "sample_seed_42",
                    "generated": "<think>Гуглите сами</think>",
                    "completion_tokens": 5,
                    "finished_by_eos": False,
                    "hit_max_new_tokens": True,
                    "target_source_rejected": "Цель",
                },
            ]
        )
        self.assertEqual(summary["greedy"]["count"], 1)
        self.assertEqual(summary["greedy"]["eos_rate"], 1.0)
        self.assertEqual(summary["sample_seed_42"]["think_rate"], 1.0)
        self.assertEqual(summary["sample_seed_42"]["max_token_cap_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
