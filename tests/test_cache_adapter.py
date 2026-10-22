"""Joohye's cache behind the gateway's stage-2 adapter."""

import tempfile
import unittest
from pathlib import Path

from gateway.cache_adapter import GatewayCache
from gateway.llm import SimulatedProvider
from gateway.pipeline import process
from gateway.prompts import TEMPLATES
from gateway.schema import GenerationResult


class LiveLikeProvider(SimulatedProvider):
    """Same placeholder answers, different provider name, so it gets its own cache scope."""

    name = "gemini"

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        result.provider = self.name
        result.simulated = False
        return result


class CacheAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "cache.sqlite3"

    def tearDown(self):
        self.tmp.cleanup()

    def cache(self, mode="exact"):
        return GatewayCache(self.path, mode=mode)

    def test_first_call_misses_second_call_hits(self):
        cache = self.cache()
        first = process("How do I reset my PIN?", cache=cache)
        self.assertEqual(first.cache.status, "miss")
        self.assertEqual(first.final_status, "answered")

        second = process("How do I reset my PIN?", cache=cache)
        self.assertEqual(second.cache.status, "hit")
        self.assertEqual(second.cache.match_type, "exact")
        self.assertEqual(second.cache.similarity, 1.0)
        self.assertEqual(second.final_status, "cached")
        self.assertIsNone(second.router, "a hit must stop before routing")
        self.assertEqual(second.generation.provider, "cache")
        self.assertEqual(second.generation.answer, first.generation.answer)
        self.assertEqual(second.generation.output_tokens, 0)

    def test_semantic_mode_matches_a_paraphrase(self):
        cache = self.cache("semantic")
        process("How do I reset my PIN?", cache=cache)
        record = process("How can I reset my PIN?", cache=cache)
        self.assertEqual(record.cache.status, "hit")
        self.assertEqual(record.cache.match_type, "semantic")
        self.assertGreaterEqual(record.cache.similarity, 0.9)
        self.assertEqual(record.final_status, "cached")

    def test_shadow_mode_reports_but_does_not_serve(self):
        cache = self.cache("semantic-shadow")
        process("How do I reset my PIN?", cache=cache)
        record = process("How can I reset my PIN?", cache=cache)
        self.assertEqual(record.cache.status, "miss")
        self.assertIn("shadow", record.cache.reason)
        self.assertEqual(record.final_status, "answered")

    def test_unrelated_query_misses_in_semantic_mode(self):
        cache = self.cache("semantic")
        process("How do I change my PIN?", cache=cache)
        record = process("How do I activate a new card?", cache=cache)
        self.assertEqual(record.cache.status, "miss")

    def test_her_policy_still_bypasses_risky_queries(self):
        cache = self.cache()
        cases = {
            "What is my account balance?": "personalized",
            "Transfer $500 to my savings": "transactional",
            "What is the current exchange rate?": "time_sensitive",
            "My card number is 4111 1111 1111 1111": "personal_data",
        }
        for query, classification in cases.items():
            with self.subTest(query=query):
                record = process(query, cache=cache)
                self.assertEqual(record.cache.status, "bypass")
                self.assertIn(classification, record.cache.reason)
                self.assertEqual(record.final_status, "answered")
        # Bypassed queries are not stored either, so repeating one still bypasses.
        self.assertEqual(process("What is my account balance?", cache=cache).cache.status, "bypass")

    def test_simulated_and_live_answers_never_share_an_entry(self):
        cache = self.cache()
        process("How do I reset my PIN?", cache=cache)  # stored under provider "simulated"
        record = process("How do I reset my PIN?", cache=cache, provider=LiveLikeProvider())
        self.assertEqual(record.cache.status, "miss")
        self.assertEqual(record.generation.provider, "gemini")

    def test_off_mode_bypasses_everything(self):
        cache = self.cache("off")
        process("How do I reset my PIN?", cache=cache)
        record = process("How do I reset my PIN?", cache=cache)
        self.assertEqual(record.cache.status, "bypass")
        self.assertEqual(record.final_status, "answered")

    def test_incomplete_generation_is_not_stored(self):
        cache = self.cache()
        truncated = GenerationResult(status="generated", provider="simulated", model="m",
                                     answer="partial...", finish_reason="MAX_TOKENS")
        self.assertFalse(cache.store("How do I reset my PIN?", truncated))
        failed = GenerationResult(status="error", provider="simulated", model="m", message="boom")
        self.assertFalse(cache.store("How do I reset my PIN?", failed))
        self.assertEqual(cache.lookup("How do I reset my PIN?", provider_name="simulated")[0].status, "miss")

    def test_prompt_change_invalidates_entries(self):
        process("How do I reset my PIN?", cache=self.cache())
        TEMPLATES["reworded"] = "You are a terse assistant for a fictional bank."
        self.addCleanup(TEMPLATES.pop, "reworded")
        reworded = GatewayCache(self.path, mode="exact", prompt_template="reworded")
        decision, _ = reworded.lookup("How do I reset my PIN?", provider_name="simulated")
        self.assertEqual(decision.status, "miss")

    def test_no_cache_records_a_bypass(self):
        record = process("How do I reset my PIN?")
        self.assertEqual(record.cache.status, "bypass")
        self.assertIn("no cache", record.cache.reason)


if __name__ == "__main__":
    unittest.main()
