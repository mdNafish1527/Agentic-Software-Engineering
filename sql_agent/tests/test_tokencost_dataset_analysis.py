"""Tests for the real dataset and for the analysis maths (ratios, totals, bootstrap, categories)."""
import csv
import re
import tempfile
import unittest
from pathlib import Path

from tokencost import analysis
from tokencost.backend import FakeBackend
from tokencost.config import CATEGORIES
from tokencost.runner import load_dataset, run

BANGLA = re.compile(r"[\u0980-\u09FF]")


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_dataset()

    def test_fifty_unique_aligned_pairs(self):
        self.assertEqual(len(self.rows), 50)
        self.assertEqual(len({r["id"] for r in self.rows}), 50)

    def test_every_category_is_present(self):
        cats = {r["category"] for r in self.rows}
        self.assertEqual(cats, set(CATEGORIES))

    def test_bangla_column_is_bangla_and_english_is_not_identical(self):
        for r in self.rows:
            self.assertRegex(r["bangla"], BANGLA)
            self.assertNotEqual(r["english"], r["bangla"])

    def test_lengths_vary(self):
        lengths = [len(r["english"].split()) for r in self.rows]
        self.assertLess(min(lengths), 6)
        self.assertGreater(max(lengths), 35)

    def test_no_stray_tabs_or_newlines(self):
        for r in self.rows:
            for key in ("english", "bangla"):
                self.assertNotIn("\n", r[key])


def fake_results(pairs):
    """pairs = [(id, category, en_in, bn_in, en_out, bn_out, en_cost, bn_cost)] as result rows."""
    rows = []
    for pid, cat, ei, bi, eo, bo, ec, bc in pairs:
        rows.append({"id": pid, "category": cat, "language": "english", "input_tokens": ei,
                     "output_tokens": eo, "cost_usd": ec, "latency_s": 1.0, "truncated": "False"})
        rows.append({"id": pid, "category": cat, "language": "bangla", "input_tokens": bi,
                     "output_tokens": bo, "cost_usd": bc, "latency_s": 2.0, "truncated": "False"})
    return rows


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.data = fake_results([
            ("P01", "question", 10, 30, 5, 5, 0.001, 0.003),
            ("P02", "question", 20, 40, 10, 20, 0.002, 0.005),
            ("P03", "arithmetic", 10, 10, 4, 4, 0.001, 0.001),
        ])

    def test_pair_ratios(self):
        pairs = analysis.pair_table(self.data)
        self.assertEqual([p["id"] for p in pairs], ["P01", "P02", "P03"])
        self.assertAlmostEqual(pairs[0]["input_ratio"], 3.0)
        self.assertAlmostEqual(pairs[1]["input_ratio"], 2.0)
        self.assertAlmostEqual(pairs[2]["input_ratio"], 1.0)
        self.assertAlmostEqual(pairs[0]["latency_ratio"], 2.0)

    def test_incomplete_pair_is_skipped(self):
        partial = self.data[:-1]          # P03 lost its Bangla row
        self.assertEqual(len(analysis.pair_table(partial)), 2)

    def test_overall_ratios_use_totals_not_averages(self):
        pairs = analysis.pair_table(self.data)
        s = analysis.summarize(pairs, self.data)
        self.assertAlmostEqual(s["overall_ratio"]["input"], (30 + 40 + 10) / (10 + 20 + 10))
        self.assertAlmostEqual(s["overall_ratio"]["cost"], (0.003 + 0.005 + 0.001) / (0.001 + 0.002 + 0.001))
        self.assertEqual(s["totals"]["english_input_tokens"], 40)

    def test_per_pair_statistics(self):
        s = analysis.summarize(analysis.pair_table(self.data), self.data)
        d = s["ratio_per_pair"]["input"]
        self.assertAlmostEqual(d["mean"], 2.0)
        self.assertAlmostEqual(d["median"], 2.0)
        self.assertEqual((d["min"], d["max"]), (1.0, 3.0))
        lo, hi = d["ci95_of_mean"]
        self.assertTrue(lo <= d["mean"] <= hi)

    def test_bootstrap_is_reproducible(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(analysis.bootstrap_ci(values), analysis.bootstrap_ci(values))

    def test_category_breakdown(self):
        s = analysis.summarize(analysis.pair_table(self.data), self.data)
        self.assertAlmostEqual(s["by_category"]["question"]["input_ratio_of_totals"], 70 / 30)
        self.assertAlmostEqual(s["by_category"]["arithmetic"]["input_ratio_of_totals"], 1.0)

    def test_main_writes_outputs_from_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = load_dataset()[:4]
            run(rows, FakeBackend(), tmp / "results.csv", progress=None)
            summary = analysis.main(tmp / "results.csv", tmp)
            self.assertEqual(summary["pairs"], 4)
            with open(tmp / "comparison.csv", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 4)
            self.assertTrue((tmp / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
