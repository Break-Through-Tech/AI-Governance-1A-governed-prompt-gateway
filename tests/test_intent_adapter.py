"""Team C's Banking77 classifier behind the gateway's stage-3 adapter."""

import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from gateway import intent_adapter
from gateway.pipeline import process


class FakePipeline:
    """Stands in for the trained scikit-learn pipeline, which is not in the repo."""

    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.seen = []

    def predict_proba(self, texts):
        self.seen.extend(texts)
        return np.array([self.probabilities])


class IntentAdapterTests(unittest.TestCase):
    def setUp(self):
        intent_adapter._IMPORT = None
        self.addCleanup(setattr, intent_adapter, "_IMPORT", None)

    def with_their_classifier(self, probabilities, labels):
        """Import Team C's real classifier module with a stand-in for litellm and a fake model."""
        fake_litellm = types.ModuleType("litellm")
        fake_litellm.token_counter = lambda **_: 0
        fake_litellm.cost_per_token = lambda **_: (0.0, 0.0)
        modules = patch.dict(sys.modules, {"litellm": fake_litellm})
        modules.start()
        self.addCleanup(modules.stop)
        from banking_chatbot.intent import classifier
        pipeline = FakePipeline(probabilities)
        for name, value in (("MODEL_PIPELINE", pipeline), ("LABEL_NAMES", labels)):
            patcher = patch.object(classifier, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return pipeline

    def test_missing_dependency_is_reported_not_faked(self):
        # What a teammate sees before running `pip install -r requirements.txt`.
        intent_adapter._IMPORT = "Team C's classifier could not be imported: litellm is not installed"
        decision = intent_adapter.classify("How do I reset my PIN?")
        self.assertEqual(decision.label, "unknown")
        self.assertEqual(decision.confidence, 0.0, "no model means no confidence, not their hardcoded 0.70")
        self.assertIn("litellm", decision.reason)

    @unittest.skipUnless(importlib.util.find_spec("litellm"), "litellm is not installed")
    def test_the_committed_model_classifies_real_questions(self):
        """Team C's trained model, as committed, through the real import path."""
        cases = {
            "What's the exchange rate from USD to EUR?": "exchange_rate",
            "My physical credit card was stolen at the airport yesterday.": "lost_or_stolen_card",
            "I see a charge on my statement from a store I've never visited.": "extra_charge_on_statement",
        }
        for query, label in cases.items():
            with self.subTest(query=query):
                decision = intent_adapter.classify(query)
                self.assertEqual(decision.label, label)
                self.assertGreater(decision.confidence, 0.2)
                self.assertIsNotNone(decision.suggested_tier)
        nonsense = intent_adapter.classify("asdf qwer zxcv")
        self.assertEqual(nonsense.label, "general_banking_faq", "their low-confidence fallback")

    def test_model_file_missing_names_the_file(self):
        self.with_their_classifier([1.0], None)
        decision = intent_adapter.classify("How do I reset my PIN?")
        self.assertEqual(decision.label, "unknown")
        self.assertIn("banking77_classifier.joblib", decision.reason)

    def test_prediction_carries_label_confidence_and_their_tier_suggestion(self):
        pipeline = self.with_their_classifier([0.1, 0.8, 0.1], ["exchange_rate", "change_pin", "card_arrival"])
        decision = intent_adapter.classify("How do I CHANGE my PIN?!")
        self.assertEqual(decision.label, "change_pin")
        self.assertEqual(decision.confidence, 0.8)
        self.assertEqual(decision.suggested_tier, "standard-balanced")
        self.assertEqual(decision.prompt_template, "default")
        self.assertEqual(pipeline.seen, ["how do i change my pin"], "their text cleaning ran before the model")

    def test_their_low_confidence_fallback_is_kept(self):
        self.with_their_classifier([0.15, 0.10, 0.12], ["exchange_rate", "change_pin", "card_arrival"])
        decision = intent_adapter.classify("asdf qwer")
        self.assertEqual(decision.label, "general_banking_faq")
        self.assertEqual(decision.suggested_tier, "fast-cheap")

    def test_pipeline_records_the_prediction_and_the_router_still_decides(self):
        self.with_their_classifier([0.05, 0.05, 0.9], ["exchange_rate", "change_pin", "direct_debit_dispute"])
        record = process("Why was I charged twice?")
        self.assertEqual(record.intent.label, "direct_debit_dispute")
        self.assertEqual(record.intent.suggested_tier, "high-reasoning")
        self.assertEqual(record.router.tier, "tier-1", "their suggestion is recorded, not obeyed")

    def test_caller_override_wins(self):
        self.with_their_classifier([1.0], ["exchange_rate"])
        record = process("My card was charged twice", intent="disputed_charge")
        self.assertEqual(record.intent.label, "disputed_charge")
        self.assertEqual(record.router.tier, "tier-2")


if __name__ == "__main__":
    unittest.main()
