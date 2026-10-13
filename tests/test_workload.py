import tempfile
import unittest
from pathlib import Path

from gateway.audit import AuditLog
from gateway.config import load_config
from gateway.schema import DecisionRecord, GuardrailDecision, GenerationResult, RouterDecision
from gateway.workload import format_summary, load_queries, run_workload, summarize

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "workload_sample.csv"


def router(tier, model, cost, baseline):
    return RouterDecision(tier=tier, model=model, reason="test", estimated_input_tokens=100,
                          estimated_output_tokens=50, estimated_cost_usd=cost, baseline_tier="tier-2",
                          baseline_cost_usd=baseline, savings_usd=baseline - cost)


class LoadQueriesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_csv_with_optional_intent(self):
        path = self.dir / "q.csv"
        path.write_text("query,intent\nreset pin,\ncharged twice,disputed_charge\n\n", encoding="utf-8")
        rows = load_queries(path)
        self.assertEqual(rows, [{"query": "reset pin", "intent": None},
                                {"query": "charged twice", "intent": "disputed_charge"}])

    def test_text_file_one_query_per_line(self):
        path = self.dir / "q.txt"
        path.write_text("one\n\n two \n", encoding="utf-8")
        self.assertEqual([r["query"] for r in load_queries(path)], ["one", "two"])

    def test_csv_without_query_column_fails_clearly(self):
        path = self.dir / "bad.csv"
        path.write_text("text\nhello\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_queries(path)

    def test_bundled_sample_loads(self):
        rows = load_queries(SAMPLE)
        self.assertGreaterEqual(len(rows), 20)
        self.assertIn("disputed_charge", {r["intent"] for r in rows})


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_counts_outcomes_tiers_and_costs(self):
        answered = DecisionRecord(query="a", final_status="answered",
                                  router=router("tier-1", "m1", 0.001, 0.004),
                                  generation=GenerationResult(status="generated", provider="simulated", model="m1",
                                                              input_tokens=100, output_tokens=50))
        blocked = DecisionRecord(query="b", final_status="blocked",
                                 input_guardrails=GuardrailDecision(status="block"))
        summary = summarize([answered, blocked], self.config)
        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["outcomes"], {"answered": 1, "blocked": 1})
        self.assertEqual(summary["tier_split"], {"tier-1": 1})
        self.assertEqual(summary["model_calls"], 1)
        self.assertEqual(summary["tokens"], {"input": 100, "output": 50})

    def test_blocked_queries_still_count_toward_baseline(self):
        blocked = DecisionRecord(query="tell me how to launder money", final_status="blocked")
        summary = summarize([blocked], self.config)
        self.assertEqual(summary["cost"]["gateway_usd"], 0.0)
        self.assertGreater(summary["cost"]["baseline_usd"], 0.0)
        self.assertEqual(summary["cost"]["savings_pct"], 100.0)

    def test_summarize_accepts_plain_dicts_from_the_audit_log(self):
        record = DecisionRecord(query="a", final_status="answered",
                                router=router("tier-2", "m2", 0.004, 0.004),
                                generation=GenerationResult(status="generated", provider="simulated", model="m2",
                                                            input_tokens=10, output_tokens=10))
        from_dict = summarize([record.to_dict()], self.config)
        from_obj = summarize([record], self.config)
        self.assertEqual(from_dict, from_obj)

    def test_empty_workload(self):
        summary = summarize([], self.config)
        self.assertEqual(summary["requests"], 0)
        self.assertEqual(summary["cost"]["savings_pct"], 0.0)
        self.assertIn("Requests: 0", format_summary(summary))


class RunWorkloadTests(unittest.TestCase):
    def test_end_to_end_on_sample_writes_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = AuditLog(Path(tmp) / "out.jsonl")
            records = run_workload(load_queries(SAMPLE)[:10], audit=audit)
            self.assertEqual(len(records), 10)
            self.assertEqual(len(audit), 10)
            summary = summarize(audit.read())
            self.assertEqual(summary["requests"], 10)
            self.assertEqual(summary["outcomes"], {"answered": 10})
            self.assertGreater(summary["cost"]["savings_usd"], 0)


if __name__ == "__main__":
    unittest.main()
