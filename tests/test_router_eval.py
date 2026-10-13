import tempfile
import unittest
from pathlib import Path

from gateway.config import load_config
from gateway.router_eval import evaluate, format_report, load_cases

SEED = Path(__file__).resolve().parent.parent / "data" / "router_eval.csv"


class LoadCasesTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_rows_and_blank_intent_becomes_none(self):
        path = self.dir / "c.csv"
        path.write_text("query,intent,expected_tier\nhi,,tier-1\ncharged twice,disputed_charge,tier-2\n", encoding="utf-8")
        cases = load_cases(path, self.config)
        self.assertEqual(cases[0], {"query": "hi", "intent": None, "expected_tier": "tier-1"})
        self.assertEqual(cases[1]["intent"], "disputed_charge")

    def test_unknown_tier_label_is_rejected_with_row_number(self):
        path = self.dir / "c.csv"
        path.write_text("query,intent,expected_tier\nhi,,tier-9\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            load_cases(path, self.config)
        self.assertIn("row 2", str(ctx.exception))

    def test_seed_file_loads(self):
        cases = load_cases(SEED, self.config)
        self.assertGreaterEqual(len(cases), 50)


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_all_correct(self):
        cases = [
            {"query": "How do I reset my PIN?", "intent": None, "expected_tier": "tier-1"},
            {"query": "My card was charged twice", "intent": "disputed_charge", "expected_tier": "tier-2"},
        ]
        result = evaluate(cases, self.config)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["misroutes"], [])
        self.assertEqual(result["confusion"]["tier-1"], {"tier-1": 1})
        self.assertEqual(result["confusion"]["tier-2"], {"tier-2": 1})

    def test_downgrade_and_upgrade_are_classified(self):
        cases = [
            # label says tier-2 but no intent and no signals -> router picks tier-1 -> downgrade
            {"query": "walk me through my mortgage change", "intent": None, "expected_tier": "tier-2"},
            # label says tier-1 but a tier-2 intent -> router picks tier-2 -> upgrade
            {"query": "hi", "intent": "fraud_report", "expected_tier": "tier-1"},
        ]
        result = evaluate(cases, self.config)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["downgrades"], 1)
        self.assertEqual(result["upgrades"], 1)
        kinds = {m["query"]: m["kind"] for m in result["misroutes"]}
        self.assertEqual(kinds["walk me through my mortgage change"], "downgrade")
        self.assertEqual(kinds["hi"], "upgrade")

    def test_report_mentions_accuracy_and_misroutes(self):
        cases = [{"query": "hi", "intent": "fraud_report", "expected_tier": "tier-1"}]
        report = format_report(evaluate(cases, self.config))
        self.assertIn("Accuracy: 0.0%", report)
        self.assertIn("upgrade", report)
        self.assertIn("reason:", report)


if __name__ == "__main__":
    unittest.main()
