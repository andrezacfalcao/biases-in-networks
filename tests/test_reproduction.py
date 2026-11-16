import csv
import json
import tempfile
import unittest
from pathlib import Path

from reproduction.__main__ import write_reports
from reproduction.metrics import read_metadata, result_rows, summarize


class ResultContractTests(unittest.TestCase):
    def setUp(self):
        self.counts = {
            "train": (60, 10, 10, 20),
            "val": (10, 10, 10, 10),
            "test": (10, 20, 30, 40),
        }
        self.raw = {
            "erm": {"accuracy_0_0": 0.9, "accuracy_0_1": 0.5,
                    "accuracy_1_0": 0.5, "accuracy_1_1": 0.8},
            "dfr": {"test_accs": [0.8, 0.7, 0.7, 0.8], "best_hypers": [0.1, 1, 1]},
        }

    def test_metrics_distinguish_train_weighted_and_test_micro_accuracy(self):
        erm, dfr = result_rows(self.raw, self.counts, "resnet50", 1)
        self.assertEqual(erm["method"], "ERM")
        self.assertAlmostEqual(erm["weighted_accuracy_pct"], 80)
        self.assertAlmostEqual(erm["micro_accuracy_pct"], 66)
        self.assertAlmostEqual(erm["worst_group_accuracy_pct"], 50)
        self.assertAlmostEqual(erm["group_gap_pp"], 40)
        self.assertAlmostEqual(dfr["worst_group_accuracy_pct"], 70)
        self.assertEqual(dfr["dfr_c"], 0.1)

    def test_missing_test_group_is_rejected(self):
        self.counts["test"] = (10, 20, 30, 0)
        with self.assertRaisesRegex(ValueError, "grupos"):
            result_rows(self.raw, self.counts, "resnet50", 1)

    def test_summary_rejects_incomplete_and_duplicate_seed_pairs(self):
        rows = result_rows(self.raw, self.counts, "resnet50", 1)
        with self.assertRaisesRegex(ValueError, "sementes"):
            summarize(rows, (1, 2))
        with self.assertRaisesRegex(ValueError, "duplicada"):
            summarize(rows + rows, (1,))

    def test_summary_computes_sample_standard_deviation(self):
        rows = result_rows(self.raw, self.counts, "resnet50", 1)
        self.raw["dfr"]["test_accs"] = [0.85, 0.9, 0.9, 0.9]
        rows += result_rows(self.raw, self.counts, "resnet50", 2)
        dfr = next(x for x in summarize(rows, (1, 2)) if x["method"] == "DFR")
        self.assertAlmostEqual(dfr["worst_group_accuracy_pct_mean"], 77.5)
        self.assertAlmostEqual(dfr["worst_group_accuracy_pct_sd"], 10.6066017178)

    def test_metadata_rejects_other_splits_and_group_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            path.write_text("img_filename,y,place,split\na.jpg,0,0,0\n")
            with self.assertRaisesRegex(ValueError, "divisões oficiais"):
                read_metadata(path)

    def test_report_only_compares_to_paper_after_all_five_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            dataset = {"group_counts": self.counts}
            for seed in range(1, 6):
                seed_dir = out / f"seed-{seed}"
                seed_dir.mkdir()
                (seed_dir / "raw.json").write_text(json.dumps(self.raw))
                write_reports(out, dataset, "resnet50")
                self.assertEqual((out / "paper_comparison.csv").exists(), seed == 5)
            with (out / "results.csv").open() as file:
                self.assertEqual(len(list(csv.DictReader(file))), 10)
            with (out / "paper_comparison.csv").open() as file:
                comparison = list(csv.DictReader(file))
            self.assertEqual(len(comparison), 4)
            erm_worst = next(row for row in comparison
                             if row["method"] == "ERM" and row["metric"] == "worst_group_accuracy_pct")
            self.assertAlmostEqual(float(erm_worst["delta_pp"]), -24.9)
