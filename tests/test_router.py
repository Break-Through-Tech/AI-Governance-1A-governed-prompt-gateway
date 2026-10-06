import unittest

from gateway.config import load_config
from gateway.router import complexity_signals, route


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_simple_query_goes_to_default_tier(self):
        decision = route("How do I reset my PIN?", config=self.config)
        self.assertEqual(decision.tier, self.config.rules.default_tier)
        self.assertEqual(decision.complexity_score, 0)
        self.assertIn("default tier", decision.reason)

    def test_mapped_intent_overrides_complexity(self):
        decision = route("My card was charged twice", intent="disputed_charge", config=self.config)
        self.assertEqual(decision.tier, "tier-2")
        self.assertIn("disputed_charge", decision.reason)

    def test_escalate_verdict_wins_over_everything(self):
        decision = route("hi", intent="unknown", risk="escalate", config=self.config)
        self.assertEqual(decision.tier, self.config.rules.escalation_tier)
        self.assertIn("escalated", decision.reason)

    def test_two_signals_bump_to_escalation_tier(self):
        query = "Compare the APR on both cards? How much interest would I pay in total?"
        decision = route(query, config=self.config)
        self.assertGreaterEqual(decision.complexity_score, 2)
        self.assertEqual(decision.tier, self.config.rules.escalation_tier)

    def test_single_signal_stays_on_default_tier(self):
        decision = route("What is the interest rate on savings?", config=self.config)
        self.assertEqual(decision.complexity_score, 1)
        self.assertEqual(decision.tier, self.config.rules.default_tier)

    def test_long_query_is_a_signal(self):
        query = "word " * 200
        signals = complexity_signals(query, 200, self.config.rules)
        self.assertIn("long_query", signals)

    def test_cost_fields_are_consistent(self):
        decision = route("How do I reset my PIN?", system_prompt="Be brief.", config=self.config)
        self.assertGreater(decision.estimated_input_tokens, 0)
        self.assertAlmostEqual(
            decision.savings_usd, decision.baseline_cost_usd - decision.estimated_cost_usd, places=8
        )
        self.assertGreaterEqual(decision.savings_usd, 0)

    def test_baseline_tier_has_zero_savings(self):
        decision = route("hi", risk="escalate", config=self.config)
        self.assertEqual(decision.tier, decision.baseline_tier)
        self.assertEqual(decision.savings_usd, 0)


if __name__ == "__main__":
    unittest.main()
