import unittest

from gateway.config import Tier
from gateway.cost import count_tokens, estimate_cost


class CostTests(unittest.TestCase):
    def test_empty_text_has_zero_tokens(self):
        self.assertEqual(count_tokens(""), 0)

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("reset my pin")
        long = count_tokens("reset my pin and then tell me about the fees on international transfers")
        self.assertGreater(short, 0)
        self.assertGreater(long, short)

    def test_cost_math(self):
        tier = Tier(name="t", model="m", input_per_million=1.0, output_per_million=10.0)
        # 1,000,000 input tokens at $1/M plus 100,000 output tokens at $10/M = $1 + $1
        self.assertEqual(estimate_cost(1_000_000, 100_000, tier), 2.0)

    def test_zero_tokens_cost_nothing(self):
        tier = Tier(name="t", model="m", input_per_million=1.0, output_per_million=10.0)
        self.assertEqual(estimate_cost(0, 0, tier), 0.0)


if __name__ == "__main__":
    unittest.main()
