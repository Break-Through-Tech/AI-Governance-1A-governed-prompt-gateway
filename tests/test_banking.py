import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from banking_chatbot.data import ROOT, load_records, prepare
from banking_chatbot.retrieval import Retriever


class BankingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "input.csv"
        self.clean = self.root / "knowledge.jsonl"
        self.report = self.root / "report.json"

    def write_csv(self, rows, encoding="utf-8"):
        text = io.StringIO(newline="")
        writer = csv.writer(text)
        writer.writerow(["Query", "Response"])
        writer.writerows(rows)
        self.raw.write_bytes(text.getvalue().encode(encoding))

    def prepare(self):
        return prepare(self.raw, self.clean, self.report)

    def fixture(self):
        self.write_csv([
            ["How do I activate a debit card?", "Use the debit activation process."],
            ["How do I close an account?", "Contact the account support team."],
            ["How do I activate a credit card?", "Use the credit activation process."],
        ])
        self.prepare()
        return Retriever(self.clean)

    def test_encoding_normalization_and_duplicate_provenance(self):
        self.write_csv([[" What’s my balance? ", "Check  online."],
                        ["What's my balance?", "Check online."], ["", "Empty"]], "cp1252")
        report = self.prepare()
        records = load_records(self.clean)
        self.assertEqual(report["source_encoding"], "cp1252")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_rows"], [1, 2])
        self.assertEqual(report["empty_rows_removed"], [3])
        self.assertEqual(len(report["duplicate_pairs_removed"]), 1)

    def test_stable_ids_when_source_is_reordered(self):
        rows = [["Balance?", "Check online."], ["Close account?", "Call support."]]
        self.write_csv(rows)
        self.prepare()
        ids = {r["question"]: r["id"] for r in load_records(self.clean)}
        self.write_csv(list(reversed(rows)))
        self.prepare()
        self.assertEqual(ids, {r["question"]: r["id"] for r in load_records(self.clean)})

    def test_answer_variants_retained_and_flagged(self):
        self.write_csv([["How do I close my account?", "Online."], ["How do I close my account?", "In person only."]])
        report = self.prepare()
        self.assertEqual(report["cleaned_records"], 2)
        self.assertEqual(len(report["questions_with_multiple_answers"]), 1)
        self.assertTrue(all("multiple_answers_for_question" in r["review_flags"] for r in load_records(self.clean)))

    def test_contact_details_flagged_without_rewriting(self):
        answer = "Contact our bank at fraud@example.com or 1-800-123-4567."
        self.write_csv([["Help?", answer]])
        self.prepare()
        record = load_records(self.clean)[0]
        self.assertEqual(record["answer"], answer)
        self.assertIn("contact_detail", record["review_flags"])

    def test_repeated_preparation_is_identical_and_preserves_raw(self):
        self.fixture()
        before = [p.read_bytes() for p in (self.raw, self.clean, self.report)]
        self.prepare()
        self.assertEqual(before, [p.read_bytes() for p in (self.raw, self.clean, self.report)])

    def test_bad_columns_and_malformed_rows_fail(self):
        for content in ["Question,Answer\na,b\n", "Query,Response\na,b,c\n", "Query,Response\na\n"]:
            with self.subTest(content=content):
                self.raw.write_text(content)
                with self.assertRaises(ValueError):
                    self.prepare()
        self.assertFalse(self.clean.exists())

    def test_empty_dataset_fails_without_output(self):
        self.write_csv([["", "answer"]])
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.clean.exists())

    def test_cannot_overwrite_source(self):
        self.write_csv([["Question", "Answer"]])
        before = self.raw.read_bytes()
        with self.assertRaises(ValueError):
            prepare(self.raw, self.raw, self.report)
        self.assertEqual(before, self.raw.read_bytes())

    def test_exact_and_paraphrased_queries_keep_answer_pair(self):
        retriever = self.fixture()
        for question in ["How do I activate a debit card?", "Activate my debit card please"]:
            match = retriever.search(question)["matches"][0]
            self.assertEqual(match["source_rows"], [1])
            self.assertEqual(match["answer"], "Use the debit activation process.")
        self.assertAlmostEqual(retriever.search("How do I activate a debit card?")["matches"][0]["score"], 1.0)

    def test_unknown_and_stopword_queries_have_no_matches_even_at_zero_threshold(self):
        retriever = self.fixture()
        for query in ["quantum penguins", "the and or"]:
            result = retriever.search(query, threshold=0)
            self.assertEqual(result["status"], "no_match")
            self.assertEqual(result["matches"], [])

    def test_each_match_meets_threshold_and_top_k(self):
        retriever = self.fixture()
        result = retriever.search("activate card", top_k=1, threshold=0.1)
        self.assertEqual(len(result["matches"]), 1)
        self.assertGreaterEqual(result["matches"][0]["score"], 0.1)
        self.assertEqual(retriever.search("activate card", threshold=0.999)["status"], "no_match")

    def test_invalid_search_arguments(self):
        retriever = self.fixture()
        for kwargs in [{"query": " "}, {"query": "card", "top_k": 0},
                       {"query": "card", "threshold": -1}, {"query": "card", "threshold": 2},
                       {"query": "card", "threshold": float("nan")}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                retriever.search(**kwargs)

    def test_duplicate_knowledge_ids_rejected(self):
        self.fixture()
        records = load_records(self.clean)
        self.clean.write_text(json.dumps(records[0]) + "\n" + json.dumps(records[0]))
        with self.assertRaises(ValueError):
            Retriever(self.clean)

    def test_cli_success_and_invalid_input(self):
        self.fixture()
        args = [sys.executable, "-m", "banking_chatbot", "search", "activate debit card", "--knowledge", str(self.clean)]
        good = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertEqual(json.loads(good.stdout)["status"], "matched")
        bad = subprocess.run(args + ["--threshold", "nan"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertNotIn("Traceback", bad.stderr)


if __name__ == "__main__":
    unittest.main()
