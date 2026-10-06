import json
import unittest

from gateway.llm import GeminiProvider, SimulatedProvider
from gateway.pipeline import process
from gateway.schema import GenerationResult


class FailingProvider:
    name = "failing"

    def generate(self, query, system_prompt, model):
        return GenerationResult(status="error", provider=self.name, model=model, message="boom")


class PipelineTests(unittest.TestCase):
    def test_simulated_run_produces_a_complete_record(self):
        record = process("How do I reset my PIN?")
        self.assertEqual(record.final_status, "answered")
        for stage in ("input_guardrails", "cache", "intent", "router", "generation", "output_compliance"):
            self.assertIsNotNone(getattr(record, stage), stage)
        self.assertTrue(record.generation.simulated)
        self.assertEqual(record.generation.model, record.router.model)

    def test_record_serializes_to_json(self):
        record = process("How do I reset my PIN?")
        data = json.loads(record.to_json())
        self.assertEqual(data["query"], "How do I reset my PIN?")
        self.assertIn("tier", data["router"])
        self.assertIn("savings_usd", data["router"])

    def test_intent_override_reaches_the_router(self):
        record = process("My card was charged twice", intent="disputed_charge")
        self.assertEqual(record.intent.label, "disputed_charge")
        self.assertEqual(record.router.tier, "tier-2")

    def test_provider_error_is_recorded_not_raised(self):
        record = process("hello", provider=FailingProvider())
        self.assertEqual(record.final_status, "error")
        self.assertEqual(record.generation.message, "boom")
        self.assertIsNone(record.output_compliance)

    def test_gemini_without_key_returns_error_result(self):
        provider = GeminiProvider(api_key="")
        result = provider.generate("hi", "be brief", "gemini-2.5-flash-lite")
        self.assertEqual(result.status, "error")
        self.assertIn("API key", result.message)

    def test_gemini_payload_keeps_key_out(self):
        provider = GeminiProvider(api_key="secret-key")
        payload = provider.build_payload("hi", "be brief")
        self.assertNotIn("secret-key", json.dumps(payload))
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "be brief")

    def test_simulated_provider_counts_prompt_and_query(self):
        result = SimulatedProvider(expected_output_tokens=50).generate("hello", "be brief", "m")
        self.assertEqual(result.output_tokens, 50)
        self.assertGreater(result.input_tokens, 0)


if __name__ == "__main__":
    unittest.main()
