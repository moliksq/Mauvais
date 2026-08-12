import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anti_preference import anti_preference_weight, invert_preference_pairs, load_jsonl_pairs


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
