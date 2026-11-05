"""Joohye's local compliance validators behind the gateway's stage-6 adapter."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gateway import compliance_adapter
from gateway.cache_adapter import GatewayCache
from gateway.llm import SimulatedProvider
from gateway.pipeline import process
from gateway.schema import GenerationResult


def answer(text, finish_reason="STOP"):
    return GenerationResult(status="generated", provider="test", model="m", answer=text, finish_reason=finish_reason)


class SayingProvider(SimulatedProvider):
    """A provider whose answer the test controls."""

    def __init__(self, text):
        super().__init__()
        self.text = text

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        result.answer = self.text
        return result


class LocalCheckTests(unittest.TestCase):
    def check(self, text, **kwargs):
        return compliance_adapter.check("How do I reset my PIN?", answer(text, **kwargs))

    def test_clean_answer_passes_and_says_what_was_not_reviewed(self):
        decision = self.check("Open the app, choose Card settings, then Reset PIN.")
        self.assertEqual(decision.status, "pass")
        self.assertEqual(decision.flags, [])
        self.assertEqual(decision.checks, ["format", "sensitive", "contact"])
        self.assertEqual(decision.policy_version, "banking-output-v1")
        # These rules need the semantic reviewer, which cannot run without references.
        for rule in ("SUPPORTED_FACTS", "NO_PERSONAL_RECOMMENDATION", "CONTENT_SAFETY", "SENSITIVE_DATA"):
            self.assertIn(rule, decision.unreviewed)
        self.assertNotIn("CONTACT_POLICY", decision.unreviewed)
        self.assertIn("not reviewed", decision.reason)

    def test_each_local_rule_can_fail(self):
        cases = {
            "RESPONSE_FORMAT": "word " * 151,
            "CONTACT_POLICY": "Call us on 415-555-0134 or write to help@examplebank.com.",
            "SENSITIVE_DATA": "Your card 4111 1111 1111 1111 is active.",
            "OUTPUT_COMPLETE": "   ",
        }
        for rule, text in cases.items():
            with self.subTest(rule=rule):
                decision = self.check(text)
                self.assertEqual(decision.status, "fail")
                self.assertIn(rule, decision.flags)
                self.assertIn(rule, decision.reason)
                self.assertTrue(decision.message, "a withheld answer needs a user-safe fallback message")

    def test_truncated_answer_fails(self):
        decision = self.check("A partial answer that stopped", finish_reason="MAX_TOKENS")
        self.assertEqual(decision.flags, ["OUTPUT_COMPLETE"])

    def test_a_broken_check_withholds_the_answer(self):
        class Exploding:
            descriptor = compliance_adapter._setup()[1][0].descriptor

            def validate(self, request):
                raise RuntimeError("validator crashed")

        policy, _ = compliance_adapter._setup()
        with patch.object(compliance_adapter, "_setup", return_value=(policy, (Exploding(),))):
            decision = self.check("A perfectly fine answer.")
        self.assertEqual(decision.status, "fail")
        self.assertEqual(decision.flags, ["CHECK_UNAVAILABLE"])


class PipelineTests(unittest.TestCase):
    def test_leaking_answer_is_blocked_withheld_and_never_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = GatewayCache(Path(tmp) / "cache.sqlite3", mode="exact")
            leaky = SayingProvider("Sure, email me at agent@examplebank.com for your PIN.")
            first = process("How do I reset my PIN?", provider=leaky, cache=cache)
            self.assertEqual(first.final_status, "blocked")
            self.assertEqual(first.output_compliance.flags, ["CONTACT_POLICY"])
            self.assertEqual(first.generation.answer, "")
            self.assertNotIn("examplebank", first.to_json(), "the rejected text must not reach the audit record")

            second = process("How do I reset my PIN?", provider=leaky, cache=cache)
            self.assertEqual(second.cache.status, "miss", "a rejected answer must not have been stored")

    def test_passing_answer_is_answered_with_real_checks(self):
        record = process("How do I reset my PIN?")
        self.assertEqual(record.final_status, "answered")
        self.assertEqual(record.output_compliance.checks, ["format", "sensitive", "contact"])
        self.assertNotIn("stub", record.output_compliance.reason)


if __name__ == "__main__":
    unittest.main()
