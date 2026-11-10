"""Joohye's cache behind the gateway's stage-2 adapter."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_faq_paraphrases_reuse_the_stored_answer(self):
        pairs = [
            ("How do I reset my PIN?", "What is the process to reset my personal identification number?"),
            ("How do I activate my debit card?", "What are the steps for activating a debit card?"),
            ("What is an APR?", "Could you explain annual percentage rate?"),
            ("How can I update my phone number?", "Please show me how to change my telephone number."),
        ]
        cache = self.cache("semantic")
        for seed, paraphrase in pairs:
            with self.subTest(paraphrase=paraphrase):
                first = process(seed, cache=cache)
                second = process(paraphrase, cache=cache)
                self.assertEqual(first.cache.status, "miss")
                self.assertEqual(second.cache.match_type, "semantic")
                self.assertEqual(second.generation.answer, first.generation.answer)
                self.assertEqual(second.generation.output_tokens, 0)

    def test_broader_paraphrases_work_in_both_orders(self):
        pairs = [
            ("What are your branch opening hours?", "When are bank branches open?"),
            ("When is the support desk open?", "What are the operating hours of your support desks?"),
            ("How much does a wire transfer cost?", "What is the fee for a wire transfer?"),
            ("How long does a card replacement take?", "What is the processing time for a card replacement?"),
            ("When are bank branches open on Sunday?", "What are your branch opening hours on Sunday?"),
        ]
        for reverse in (False, True):
            cache = GatewayCache(self.path.with_name(f"order-{reverse}.db"), mode="semantic")
            for pair in pairs:
                seed, paraphrase = reversed(pair) if reverse else pair
                with self.subTest(seed=seed, paraphrase=paraphrase):
                    first = process(seed, cache=cache)
                    followup = process(paraphrase, cache=cache)
                    self.assertEqual(first.cache.status, "miss")
                    self.assertEqual(followup.cache.match_type, "semantic")
                    self.assertEqual(followup.generation.answer, first.generation.answer)
                    self.assertEqual(followup.generation.output_tokens, 0)

    def test_broader_matching_preserves_subjects_and_restrictions(self):
        cache = self.cache("semantic")
        process("What are your branch opening hours?", cache=cache)
        for query in (
            "When are bank branches open on Sunday?",
            "When are bank branches in Boston open?",
            "When is the support desk open?",
            "Where is the nearest bank branch?",
            "When do bank branches close?",
            "How late can I visit a bank branch?",
        ):
            with self.subTest(query=query):
                decision, answer = cache.lookup(query, provider_name="simulated")
                self.assertEqual(decision.status, "miss")
                self.assertIsNone(answer)

    def test_equivalent_hours_candidates_still_obey_shadow_and_ambiguity_checks(self):
        cache = self.cache("semantic-shadow")
        process("What are your branch opening hours?", cache=cache)
        shadow = process("When are bank branches open?", cache=cache)
        self.assertEqual(shadow.cache.status, "miss")
        self.assertIn("shadow", shadow.cache.reason)
        # Shadow generated and stored a second answer. Both now match equally.
        ambiguous = process("What are the operating hours of your branches?", cache=self.cache("semantic"))
        self.assertEqual(ambiguous.cache.status, "miss")
        self.assertIn("ambiguous", ambiguous.cache.reason)

    def test_unqualified_pin_change_and_reset_remain_paraphrases(self):
        cache = self.cache("semantic")
        process("How do I change my PIN?", cache=cache)
        record = process("How do I reset my PIN?", cache=cache)
        self.assertEqual(record.cache.match_type, "semantic")

    def test_meaning_changes_and_explicit_numbers_do_not_reuse_answers(self):
        pairs = [
            ("How do I activate my debit card?", "How do I not activate my debit card?"),
            ("How do I activate my debit card?", "How do I deactivate my debit card?"),
            ("How do I activate my debit card?", "How do I activate my credit card?"),
            ("How do I change my PIN?", "How do I reset my forgotten PIN?"),
            ("What is a 12 month CD?", "What is a 24 month CD?"),
        ]
        for index, (first, second) in enumerate(pairs):
            for direction, (seed, followup) in enumerate(((first, second), (second, first))):
                with self.subTest(seed=seed, followup=followup):
                    cache = GatewayCache(self.path.with_name(f"{index}-{direction}.db"), mode="semantic")
                    process(seed, cache=cache)
                    self.assertEqual(process(followup, cache=cache).cache.status, "miss")
                    self.assertEqual(process(followup, cache=cache).cache.match_type, "exact")

    @patch.dict("os.environ", {"SEMANTIC_CACHE_THRESHOLD": "0.97", "SEMANTIC_CACHE_MARGIN": "0.08"})
    def test_gateway_uses_configured_semantic_settings(self):
        backend = self.cache("semantic").backend
        self.assertEqual(backend.semantic_threshold, 0.97)
        self.assertEqual(backend.semantic_margin, 0.08)

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
        # query -> (her cache classification, how the run ends)
        cases = {
            "My PIN is 1234": ("personal_data", "answered"),
            "What is my account balance?": ("personalized", "answered"),
            "Transfer $500 to my savings": ("transactional", "answered"),
            "What is the current exchange rate?": ("time_sensitive", "answered"),
            # A long digit string that masking leaves alone (10 digits, no keyword) still
            # trips her personal-data rule, so the answer is not cached.
            "Is 4155550134 the right number to call?": ("personal_data", "answered"),
        }
        for query, (classification, outcome) in cases.items():
            with self.subTest(query=query):
                record = process(query, cache=cache)
                self.assertEqual(record.cache.status, "bypass")
                self.assertIn(classification, record.cache.reason)
                self.assertEqual(record.final_status, outcome)
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
