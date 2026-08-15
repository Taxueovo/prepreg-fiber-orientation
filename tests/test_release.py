import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "thesis_experiment_packages"
sys.path.insert(0, str(EXPERIMENTS))

from experiment_runtime import aggregate_parent_predictions, parent_id  # noqa: E402


class ExperimentRuntimeTests(unittest.TestCase):
    def test_parent_id_is_extracted_from_relative_sample_name(self):
        self.assertEqual(parent_id("test_000001_srcSOURCE42_x0_y0.jpg"), "SOURCE42")

    def test_parent_id_rejects_unknown_naming_scheme(self):
        with self.assertRaises(ValueError):
            parent_id("anonymous-sample.jpg")

    def test_parent_aggregation_does_not_emit_input_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            detail = Path(directory) / "details.csv"
            output = Path(directory) / "parents.csv"
            with detail.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["filename", "true_angle_deg", "pred_angle_deg", "pred_std_deg"],
                )
                writer.writeheader()
                writer.writerows([
                    {"filename": "a_srcP001_x0.jpg", "true_angle_deg": 40, "pred_angle_deg": 41, "pred_std_deg": 2},
                    {"filename": "b_srcP001_x1.jpg", "true_angle_deg": 42, "pred_angle_deg": 43, "pred_std_deg": 4},
                ])

            aggregate_parent_predictions(detail, output)
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["parent_id"], "P001")
            self.assertEqual(float(rows[0]["y_true_deg"]), 41.0)
            self.assertNotIn(directory, output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
