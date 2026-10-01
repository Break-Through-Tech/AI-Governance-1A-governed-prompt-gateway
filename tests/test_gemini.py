import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from banking_chatbot.chat import chat
from banking_chatbot.gemini import (
    DEFAULT_MODEL, MAX_OUTPUT_TOKENS, GenerationError, build_payload,
    generate_answer, local_settings,
)
from banking_chatbot.retrieval import Retriever

MATCHES = [{"id": "faq_test", "question": "How do I change my PIN?", "answer": "Use the banking app."}]
FAKE_KEY = "test-key-not-a-real-credential"


def response(status=200, body=None):
    result = MagicMock()
    result.status_code = status
    result.json.return_value = body if body is not None else {
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "In this demo, use the banking app."}]}}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 12, "totalTokenCount": 112},
    }
    return result


class GeminiTests(unittest.TestCase):
    @patch("banking_chatbot.gemini.requests.post")
    def test_request_contract_and_usage(self, post):
        post.return_value = response()
        result = generate_answer("Change PIN", MATCHES, [], FAKE_KEY)
        args, kwargs = post.call_args
        self.assertEqual(args[0], f"https://generativelanguage.googleapis.com/v1beta/models/{DEFAULT_MODEL}:generateContent")
        self.assertNotIn(FAKE_KEY, args[0])
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], FAKE_KEY)
        self.assertEqual(kwargs["timeout"], (5, 20))
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["json"]["generationConfig"]["maxOutputTokens"], MAX_OUTPUT_TOKENS)
        self.assertEqual(kwargs["json"]["generationConfig"]["thinkingConfig"]["thinkingBudget"], 0)
        self.assertNotIn(FAKE_KEY, json.dumps(kwargs["json"]))
        self.assertEqual(result["usage"]["total_tokens"], 112)
        self.assertEqual(result["source_ids"], ["faq_test"])
        self.assertNotIn(FAKE_KEY, json.dumps(result))

    def test_prompt_history_is_bounded_and_sources_are_data(self):
        history = [{"query": str(i), "api_key": FAKE_KEY, "generation": {"status": "generated", "answer": "a" * 3000}} for i in range(9)]
        payload = build_payload("q", MATCHES, history)
        data = json.loads(payload["contents"][0]["parts"][0]["text"])
        self.assertEqual([h["question"] for h in data["recent_conversation"]], ["6", "7", "8"])
        self.assertTrue(all(len(h["answer"]) == 2000 for h in data["recent_conversation"]))
        self.assertEqual(data["references"], MATCHES)
        self.assertNotIn(FAKE_KEY, json.dumps(payload))
        self.assertIn("untrusted data", payload["systemInstruction"]["parts"][0]["text"])

    @patch("banking_chatbot.gemini.requests.post")
    def test_missing_key_no_matches_and_bad_model_make_no_request(self, post):
        for key, matches, model in [("", MATCHES, DEFAULT_MODEL), (FAKE_KEY, [], DEFAULT_MODEL),
                                    (FAKE_KEY, MATCHES, "../../other"), ("bad\nkey", MATCHES, DEFAULT_MODEL)]:
            with self.subTest(model=model, empty=not key), self.assertRaises(GenerationError):
                generate_answer("q", matches, [], key, model)
        post.assert_not_called()

    @patch("banking_chatbot.gemini.requests.post")
    def test_retrieval_modes_and_no_match_never_call_google(self, post):
        retriever = Retriever()
        for message, enabled in [("How do I change my PIN?", False), ("quantum entanglement", True)]:
            turn = chat(message, [], retriever=retriever, use_gemini=enabled, api_key=FAKE_KEY)
            self.assertIsNone(turn["generation"])
        post.assert_not_called()

    @patch("banking_chatbot.gemini.requests.post")
    def test_api_errors_keep_references_and_do_not_expose_provider_body(self, post):
        for status in (400, 401, 403, 404, 429, 302):
            with self.subTest(status=status):
                post.reset_mock()
                post.return_value = response(status, {"error": FAKE_KEY})
                turn = chat("How do I change my PIN?", [], retriever=Retriever(), use_gemini=True, api_key=FAKE_KEY)
                self.assertTrue(turn["result"]["matches"])
                self.assertEqual(turn["generation"]["status"], "error")
                self.assertNotIn(FAKE_KEY, json.dumps(turn))
                self.assertEqual(post.call_count, 1)

    @patch("banking_chatbot.gemini.time.sleep")
    @patch("banking_chatbot.gemini.requests.post")
    def test_transient_errors_retry_once(self, post, sleep):
        post.side_effect = [response(503), response()]
        result = generate_answer("q", MATCHES, [], FAKE_KEY)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.5)
        post.reset_mock()
        post.side_effect = [response(503), response(503)]
        with self.assertRaises(GenerationError):
            generate_answer("q", MATCHES, [], FAKE_KEY)
        self.assertEqual(post.call_count, 2)

    @patch("banking_chatbot.gemini.requests.post")
    def test_network_errors_are_sanitized_and_not_retried(self, post):
        for error in [requests.Timeout(FAKE_KEY), requests.ConnectionError(FAKE_KEY)]:
            post.reset_mock()
            post.side_effect = error
            with self.assertRaises(GenerationError) as caught:
                generate_answer("q", MATCHES, [], FAKE_KEY)
            self.assertNotIn(FAKE_KEY, str(caught.exception))
            self.assertEqual(post.call_count, 1)

    @patch("banking_chatbot.gemini.requests.post")
    def test_blocked_empty_and_malformed_outputs(self, post):
        bodies = [[], {}, {"promptFeedback": {"blockReason": "SAFETY"}},
                  {"candidates": [{"finishReason": "SAFETY", "content": {"parts": [{"text": "unsafe"}]}}]},
                  {"candidates": [{"finishReason": "STOP", "content": {"parts": []}}]},
                  {"candidates": "malformed"}]
        for body in bodies:
            with self.subTest(body=body):
                post.return_value = response(body=body)
                with self.assertRaises(GenerationError):
                    generate_answer("q", MATCHES, [], FAKE_KEY)

    @patch("banking_chatbot.gemini.requests.post")
    def test_truncation_and_missing_usage_are_explicit(self, post):
        post.return_value = response(body={"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [
            {"text": "hidden", "thought": True}, {"text": "Partial answer"},
        ]}}]})
        result = generate_answer("q", MATCHES, [], FAKE_KEY)
        self.assertEqual(result["answer"], "Partial answer")
        self.assertEqual(result["finish_reason"], "MAX_TOKENS")
        self.assertIsNone(result["usage"]["input_tokens"])

    @patch.dict(os.environ, {}, clear=True)
    def test_local_configuration_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "secrets.toml"
            self.assertEqual(local_settings(path), ("", DEFAULT_MODEL))
            path.write_text('GEMINI_API_KEY = "local-test-key"\n')
            self.assertEqual(local_settings(path)[0], "local-test-key")
            with patch.dict(os.environ, {"GEMINI_API_KEY": "environment-test-key"}):
                self.assertEqual(local_settings(path)[0], "environment-test-key")
            path.write_text('GEMINI_API_KEY = INVALID')
            with self.assertRaises(GenerationError):
                local_settings(path)


if __name__ == "__main__":
    unittest.main()
