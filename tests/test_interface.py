"""Exercise real Streamlit chat state and the local retriever together."""

import re
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from banking_chatbot.data import ROOT


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        # Never let a developer's configured credential affect UI tests.
        settings = patch("banking_chatbot.interface.local_settings", return_value=("", "gemini-2.5-flash-lite"))
        settings.start()
        self.addCleanup(settings.stop)
        network = patch("banking_chatbot.gemini.requests.post", side_effect=AssertionError("Unexpected live API call in UI test"))
        network.start()
        self.addCleanup(network.stop)

    def app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
        self.assertFalse(app.exception)
        return app

    def test_search_displays_original_answer_and_source(self):
        app = self.app()
        app.chat_input[0].set_value("How do I activate a new debit card?").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 2)
        turn = app.session_state["turns"][0]
        self.assertEqual(turn["result"]["matches"][0]["source_rows"], [67])
        # Markdown escapes protect literal source text; compare the displayed wording.
        displayed = [re.sub(r"\\(.)", r"\1", m.value) for m in app.markdown]
        self.assertIn(turn["result"]["matches"][0]["answer"], displayed)
        self.assertTrue(any("Data rows: 67" in c.value for c in app.caption))
        self.assertTrue(any("Similarity 1.000" in c.value for c in app.caption))

    def test_no_match_history_and_reset(self):
        app = self.app()
        app.chat_input[0].set_value("How can I check my account balance?").run()
        app.chat_input[0].set_value("Explain quantum entanglement").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 4)
        self.assertEqual(app.session_state["turns"][1]["result"]["status"], "no_match")
        self.assertTrue(any("rephrase" in info.value for info in app.info))
        app.button(key="reset_chat").click().run()
        self.assertEqual(len(app.chat_message), 0)
        self.assertEqual(app.session_state["turns"], [])

    def test_settings_apply_only_to_new_turns(self):
        app = self.app()
        app.chat_input[0].set_value("How can I check my account balance?").run()
        old_result = app.session_state["turns"][0]["result"].copy()
        app.slider(key="top_k").set_value(1).run()
        app.slider(key="threshold").set_value(0.8).run()
        app.chat_input[0].set_value("How do I activate a new debit card?").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["turns"][0]["result"], old_result)
        newest = app.session_state["turns"][1]
        self.assertEqual(newest["result"]["threshold"], 0.8)
        self.assertEqual(len(newest["result"]["matches"]), 1)

    def test_example_button_submits_and_sessions_are_isolated(self):
        app = self.app()
        next(b for b in app.button if b.label == "Branch hours").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["turns"]), 1)
        self.assertEqual(app.session_state["turns"][0]["result"]["matches"][0]["source_rows"], [26])
        fresh = self.app()
        self.assertEqual(fresh.session_state["turns"], [])

    def test_missing_dataset_has_recovery_message(self):
        with patch("banking_chatbot.interface.DEFAULT_CLEAN", Path("/nonexistent/banking-data.jsonl")):
            app = self.app()
        self.assertEqual(len(app.chat_input), 0)
        self.assertTrue(any("could not be loaded" in error.value for error in app.error))
        self.assertTrue(any("banking_chatbot prepare" in code.value for code in app.code))

    @patch("banking_chatbot.chat.generate_answer")
    def test_gemini_answer_survives_rerun_and_disconnect_clears_session(self, generate):
        generate.return_value = {
            "status": "generated", "answer": "In this demo, use the banking app.",
            "model": "gemini-2.5-flash-lite", "source_ids": ["test"],
            "usage": {"input_tokens": 100, "output_tokens": 12, "total_tokens": 112},
            "finish_reason": "STOP", "elapsed_ms": 80,
        }
        app = self.app()
        self.assertTrue(app.toggle(key="use_gemini").disabled)
        app.text_input(key="gemini_key").set_value("fake-session-key").run()
        app.toggle(key="use_gemini").set_value(True).run()
        app.chat_input[0].set_value("How do I change my PIN?").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("In this demo" in item.value for item in app.markdown))
        self.assertTrue(any("Input tokens: 100" in item.value for item in app.caption))
        app.run()
        generate.assert_called_once()
        self.assertNotIn("fake-session-key", str(app.session_state["turns"]))
        app.button(key="disconnect_gemini").click().run()
        self.assertEqual(app.session_state["gemini_key"], "")
        self.assertFalse(app.session_state["use_gemini"])
        self.assertEqual(app.session_state["turns"], [])


if __name__ == "__main__":
    unittest.main()
