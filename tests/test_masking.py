"""Sensitive values are removed from the question before anything else sees it."""

import json
import tempfile
import unittest
from pathlib import Path

from gateway.audit import AuditLog
from gateway.cache_adapter import GatewayCache
from gateway.llm import SimulatedProvider
from gateway.masking import mask_sensitive
from gateway.pipeline import process


class RecordingProvider(SimulatedProvider):
    """Remembers exactly what text would have been sent to the model."""

    def __init__(self):
        super().__init__()
        self.queries = []

    def generate(self, query, system_prompt, model, **kwargs):
        self.queries.append(query)
        return super().generate(query, system_prompt, model, **kwargs)


class MaskingRuleTests(unittest.TestCase):
    def test_sensitive_values_are_replaced_and_named(self):
        cases = {
            "My card number is 4111 1111 1111 1111, is it active?":
                ("My card number is [CARD NUMBER], is it active?", ["card number"]),
            "my pin number is 41111 111 111, is it activated yet?":
                ("my pin number is [PIN], is it activated yet?", ["PIN"]),
            "My card is 4111222233334444 and my pin is 1234, check my balance.":
                ("My card is [CARD NUMBER] and my pin is [PIN], check my balance.", ["card number", "PIN"]),
            "My SSN is 123-45-6789 and my account number is 0012345678.":
                ("My SSN is [SSN] and my account number is [ACCOUNT NUMBER].", ["social security number", "account number"]),
            "my password is hunter2, can you reset it?":
                ("my password is [PASSWORD], can you reset it?", ["password"]),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(mask_sensitive(text), expected)

    def test_ordinary_questions_are_left_exactly_as_typed(self):
        for text in (
            "How do I reset my PIN?",
            "My PIN was blocked after 3 attempts",
            "I entered my PIN 3 times and it failed",
            "my password is not working",
            "What is the routing number for wire transfers?",
            "If I deposit $500 a month at 4% interest how much will I have after 5 years?",
            "I was charged $1,250.00 on 2026-11-03 twice",
        ):
            with self.subTest(text=text):
                self.assertEqual(mask_sensitive(text), (text, []))


class PipelineMaskingTests(unittest.TestCase):
    QUESTION = "My card number is 4111 1111 1111 1111 and my PIN is 1234, is it active?"

    def test_nothing_downstream_sees_the_raw_values(self):
        provider = RecordingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            audit = AuditLog(Path(tmp) / "decisions.jsonl")
            cache = GatewayCache(Path(tmp) / "cache.sqlite3", mode="exact")
            record = process(self.QUESTION, provider=provider, audit=audit, cache=cache)
            logged = (Path(tmp) / "decisions.jsonl").read_text()

        self.assertEqual(record.query, "My card number is [CARD NUMBER] and my PIN is [PIN], is it active?")
        self.assertEqual(record.input_guardrails.masked, ["card number", "PIN"])
        self.assertEqual(provider.queries, [record.query], "the model is sent the masked question")
        for secret in ("4111", "1234"):
            self.assertNotIn(secret, record.to_json(), "not in the decision record")
            self.assertNotIn(secret, logged, "not in the audit log on disk")
        # The question is still answered: masking does not block.
        self.assertEqual(record.final_status, "answered")

    def test_unmasked_question_reports_nothing_removed(self):
        record = process("How do I reset my PIN?")
        self.assertEqual(record.query, "How do I reset my PIN?")
        self.assertEqual(record.input_guardrails.masked, [])
        self.assertEqual(json.loads(record.to_json())["input_guardrails"]["masked"], [])


if __name__ == "__main__":
    unittest.main()
