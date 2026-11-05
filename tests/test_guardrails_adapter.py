"""Team A's input policy behind the gateway's stage-1 adapter. No network: the model is faked."""

import os
import unittest
from unittest.mock import patch

from gateway.guardrails_adapter import InputGuardrails, load_policy
from gateway.pipeline import process
from gateway.schema import GenerationResult
from gateway.workload import summarize


class FakeModel:
    """Stands in for Gemini: returns a scripted verdict and remembers the prompt it was sent."""

    def __init__(self, answer="No", status="generated", message=""):
        self.answer, self.status, self.message = answer, status, message
        self.prompts, self.models = [], []

    def generate(self, query, system_prompt, model, *, thinking_level=None):
        self.prompts.append(query)
        self.models.append((model, thinking_level))
        if self.status != "generated":
            return GenerationResult(status="error", provider="gemini", model=model, message=self.message)
        return GenerationResult(status="generated", provider="gemini", model=model, answer=self.answer,
                                input_tokens=240, output_tokens=3, elapsed_ms=800.0, simulated=False)


def guard(answer="No", **kwargs):
    model = FakeModel(answer, **kwargs)
    return InputGuardrails(model), model


class PolicyFileTests(unittest.TestCase):
    def test_rules_and_model_come_from_team_a_files(self):
        template, model, rules = load_policy()
        self.assertIn("{{ user_input }}", template)
        self.assertEqual(model, "gemini-3.5-flash")
        self.assertEqual([rules[n][0] for n in sorted(rules)],
                         ["Explicit Content", "Instruction Integrity", "Respectful Communication", "Data Protection"])

    def test_prompt_is_their_text_with_the_question_filled_in(self):
        g, model = guard()
        g.check("How do I reset my PIN?")
        prompt = model.prompts[0]
        self.assertIn('User input: "How do I reset my PIN?"', prompt)
        self.assertIn("4. Data Protection", prompt)
        self.assertNotIn("{{", prompt)
        self.assertIn("number of the first policy", prompt)

    def test_placeholder_note_is_added_only_for_masked_questions(self):
        g, model = guard()
        g.check("How do I reset my PIN?")
        g.check("My card number is [CARD NUMBER], is it active?")
        self.assertNotIn("already removed", model.prompts[0], "an ordinary question gets Team A's prompt unchanged")
        self.assertIn("already removed", model.prompts[1])
        strict = InputGuardrails(model, masked_values_are_safe=False)
        strict.check("My card number is [CARD NUMBER], is it active?")
        self.assertNotIn("already removed", model.prompts[2])

    def test_question_is_inserted_as_data_and_capped(self):
        g, model = guard()
        g.check(r"weird \1 {{ user_input }} " + "x" * 5000)
        self.assertIn(r"weird \1 {{ user_input }}", model.prompts[0], "backslashes and braces stay literal")
        self.assertLess(len(model.prompts[0]), 3500)


class ModelChoiceTests(unittest.TestCase):
    def test_default_is_the_gateway_tier_one_model_not_their_configured_one(self):
        with patch.dict(os.environ, {"GUARDRAILS_MODEL": ""}):
            g, model = guard()
            g.check("hello")
        self.assertEqual(g.policy_model, "gemini-3.5-flash")
        self.assertEqual(model.models[0], ("gemini-3.5-flash-lite", "minimal"))

    def test_environment_variable_selects_their_model(self):
        with patch.dict(os.environ, {"GUARDRAILS_MODEL": "gemini-3.5-flash"}):
            g, model = guard()
            decision = g.check("hello")
        self.assertEqual(model.models[0], ("gemini-3.5-flash", None))
        self.assertIsNone(decision.cost_usd, "no price is known for a model outside the price table")


class VerdictTests(unittest.TestCase):
    def test_no_allows_and_records_what_the_check_cost(self):
        g, _ = guard("No")
        decision = g.check("How do I reset my PIN?")
        self.assertEqual(decision.status, "allow")
        self.assertEqual((decision.input_tokens, decision.output_tokens), (240, 3))
        self.assertGreater(decision.cost_usd, 0)

    def test_yes_with_a_rule_number_names_the_policy(self):
        for answer in ("Yes 4", "Yes, 4.", "Answer (Yes or No): Yes 4", "yes - policy 4"):
            with self.subTest(answer=answer):
                decision = guard(answer)[0].check("My card is 4111 1111 1111 1111")
                self.assertEqual(decision.status, "block")
                self.assertEqual(decision.policy, "Data Protection")
                self.assertIn("card numbers", decision.reason)

    def test_echoed_question_text_does_not_flip_the_verdict(self):
        self.assertEqual(guard("Answer (Yes or No): No")[0].check("hi").status, "allow")

    def test_yes_without_a_usable_number_still_blocks(self):
        decision = guard("Yes 9")[0].check("something")
        self.assertEqual(decision.status, "block")
        self.assertIsNone(decision.policy)

    def test_unclear_answer_blocks(self):
        decision = guard("I am not sure about this one.")[0].check("something")
        self.assertEqual(decision.status, "block")
        self.assertIn("no clear verdict", decision.reason)

    def test_check_that_cannot_run_blocks(self):
        g, _ = guard(status="error", message="Gemini rate limit or free quota reached, even after retries.")
        decision = g.check("How do I reset my PIN?")
        self.assertEqual(decision.status, "block")
        self.assertIn("could not run", decision.reason)
        self.assertIn("rate limit", decision.reason)


class PipelineTests(unittest.TestCase):
    def test_blocked_question_never_reaches_cache_router_or_model(self):
        g, _ = guard("Yes 4")
        record = process("My card number is 4111 1111 1111 1111, is it active?", guardrails=g)
        self.assertEqual(record.final_status, "blocked")
        for stage in ("cache", "intent", "router", "generation", "output_compliance"):
            self.assertIsNone(getattr(record, stage), stage)
        self.assertIn("blocked before it reached the model", record.response)
        self.assertIn("Data Protection", record.response)

    def test_allowed_question_is_answered(self):
        g, _ = guard("No")
        record = process("How do I reset my PIN?", guardrails=g)
        self.assertEqual(record.final_status, "answered")
        self.assertEqual(record.input_guardrails.model, "gemini-3.5-flash-lite")

    def test_without_guardrails_the_record_says_nothing_was_checked(self):
        record = process("How do I reset my PIN?")
        self.assertEqual(record.input_guardrails.status, "allow")
        self.assertIn("not checked", record.input_guardrails.reason)

    def test_the_policy_check_only_ever_sees_the_masked_question(self):
        g, model = guard("No")
        record = process("My card number is 4111 1111 1111 1111, is it active?", guardrails=g)
        self.assertNotIn("4111", model.prompts[0])
        self.assertIn("[CARD NUMBER]", model.prompts[0])
        self.assertEqual(record.final_status, "answered")

    def test_guardrail_cost_counts_against_the_savings(self):
        with_guard = [process("How do I reset my PIN?", guardrails=guard("No")[0]),
                      process("My card is 4111 1111 1111 1111", guardrails=guard("Yes 4")[0])]
        summary = summarize(with_guard)
        without = summarize([process("How do I reset my PIN?")])
        self.assertGreater(summary["cost"]["guardrails_usd"], 0)
        self.assertEqual(without["cost"]["guardrails_usd"], 0)
        # A blocked question costs the check and nothing else; an answered one costs both.
        self.assertAlmostEqual(summary["cost"]["gateway_usd"],
                               without["cost"]["gateway_usd"] + summary["cost"]["guardrails_usd"], places=6)


if __name__ == "__main__":
    unittest.main()
