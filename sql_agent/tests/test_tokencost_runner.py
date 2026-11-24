"""Tests for the runner: order, resume, output columns, unloading and dataset validation."""
import csv
import tempfile
import unittest
from pathlib import Path

from tokencost.backend import FakeBackend
from tokencost.runner import COLUMNS, run, validate_dataset

ROWS = [
    {"id": "P01", "category": "question", "english": "What is two plus two?", "bangla": "দুই যোগ দুই কত?"},
    {"id": "P02", "category": "arithmetic", "english": "What is 3 times 4?", "bangla": "3 কে 4 দিয়ে গুণ করলে কত?"},
]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "results.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def read(self):
        with open(self.out, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def test_one_row_per_pair_and_language_with_required_columns(self):
        b = FakeBackend()
        written = run(ROWS, b, self.out, progress=None)
        self.assertEqual(written, 4)
        rows = self.read()
        self.assertEqual(len(rows), 4)
        for col in ["id", "language", "input_tokens", "output_tokens", "cost_usd", "latency_s"]:
            self.assertIn(col, rows[0])
        self.assertEqual(set(r["language"] for r in rows), {"english", "bangla"})
        self.assertEqual(list(rows[0].keys()), COLUMNS)

    def test_both_languages_use_the_same_system_prompt(self):
        b = FakeBackend()
        run(ROWS, b, self.out, progress=None)
        self.assertEqual(len({system for system, _ in b.calls}), 1)

    def test_order_alternates_between_pairs(self):
        run(ROWS, FakeBackend(), self.out, progress=None)
        rows = self.read()
        self.assertEqual([(r["id"], r["language"]) for r in rows],
                         [("P01", "english"), ("P01", "bangla"), ("P02", "bangla"), ("P02", "english")])

    def test_fresh_cache_unloads_before_every_call(self):
        b = FakeBackend()
        run(ROWS, b, self.out, fresh=True, progress=None)
        self.assertEqual(b.unloads, 4)

    def test_resume_skips_finished_calls(self):
        run(ROWS, FakeBackend(), self.out, progress=None)
        b = FakeBackend()
        written = run(ROWS, b, self.out, progress=None)
        self.assertEqual(written, 0)
        self.assertEqual(len(b.calls), 0)
        self.assertEqual(len(self.read()), 4)

    def test_restart_overwrites(self):
        run(ROWS, FakeBackend(), self.out, progress=None)
        written = run(ROWS, FakeBackend(), self.out, resume=False, progress=None)
        self.assertEqual(written, 4)
        self.assertEqual(len(self.read()), 4)

    def test_bangla_prompt_is_the_bangla_text(self):
        b = FakeBackend()
        run(ROWS[:1], b, self.out, progress=None)
        sent = [user for _, user in b.calls]
        self.assertIn("দুই যোগ দুই কত?", sent)
        self.assertIn("What is two plus two?", sent)


class DatasetValidationTests(unittest.TestCase):
    def test_real_dataset_is_valid(self):
        from tokencost.runner import load_dataset
        rows = load_dataset()
        self.assertEqual(len(rows), 50)

    def test_rejects_wrong_count(self):
        with self.assertRaises(ValueError):
            validate_dataset(ROWS, expected=50)

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            validate_dataset([ROWS[0], ROWS[0]], expected=2)

    def test_rejects_bangla_column_without_bangla_text(self):
        bad = [dict(ROWS[0], bangla="What is two plus two?"), ROWS[1]]
        with self.assertRaises(ValueError):
            validate_dataset(bad, expected=2)

    def test_rejects_unknown_category(self):
        bad = [dict(ROWS[0], category="poem"), ROWS[1]]
        with self.assertRaises(ValueError):
            validate_dataset(bad, expected=2)


if __name__ == "__main__":
    unittest.main()
