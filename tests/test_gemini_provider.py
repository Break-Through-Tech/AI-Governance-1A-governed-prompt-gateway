import unittest
from unittest import mock

from gateway.llm import GeminiProvider


class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body or {}

    def json(self):
        return self._body


def ok_body(text="Hello.", prompt=10, answer=5, thoughts=None, finish="STOP"):
    usage = {"promptTokenCount": prompt, "candidatesTokenCount": answer}
    if thoughts is not None:
        usage["thoughtsTokenCount"] = thoughts
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}], "usageMetadata": usage}


class PayloadTests(unittest.TestCase):
    def test_thinking_level_is_sent_only_when_given(self):
        provider = GeminiProvider(api_key="k")
        self.assertNotIn("thinkingConfig", provider.build_payload("q", "s")["generationConfig"])
        config = provider.build_payload("q", "s", thinking_level="low")["generationConfig"]
        self.assertEqual(config["thinkingConfig"], {"thinkingLevel": "low"})


class ParseTests(unittest.TestCase):
    def test_thinking_tokens_count_as_billed_output(self):
        answer, counts, finish = GeminiProvider.parse_response(ok_body(answer=5, thoughts=122))
        self.assertEqual(counts, {"input_tokens": 10, "thinking_tokens": 122, "output_tokens": 127})
        self.assertEqual(finish, "STOP")

    def test_no_thinking_field_means_zero(self):
        _, counts, _ = GeminiProvider.parse_response(ok_body(answer=5))
        self.assertEqual(counts["thinking_tokens"], 0)
        self.assertEqual(counts["output_tokens"], 5)

    def test_thought_parts_are_excluded_from_the_answer(self):
        body = ok_body()
        body["candidates"][0]["content"]["parts"].insert(0, {"text": "secret reasoning", "thought": True})
        answer, _, _ = GeminiProvider.parse_response(body)
        self.assertEqual(answer, "Hello.")

    def test_blocked_reply_raises(self):
        with self.assertRaises(ValueError):
            GeminiProvider.parse_response({"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []})


class RetryTests(unittest.TestCase):
    def test_retries_on_503_then_succeeds(self):
        provider = GeminiProvider(api_key="k", backoff_seconds=0)
        responses = [FakeResponse(503), FakeResponse(503), FakeResponse(200, ok_body())]
        with mock.patch("gateway.llm.requests.post", side_effect=responses) as post:
            result = provider.generate("q", "s", "m", thinking_level="low")
        self.assertEqual(post.call_count, 3)
        self.assertEqual(result.status, "generated")
        self.assertIn("3 attempts", result.message)

    def test_gives_up_after_max_attempts(self):
        provider = GeminiProvider(api_key="k", backoff_seconds=0, max_attempts=2)
        with mock.patch("gateway.llm.requests.post", side_effect=[FakeResponse(429), FakeResponse(429)]) as post:
            result = provider.generate("q", "s", "m")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(result.status, "error")
        self.assertIn("rate limit", result.message)

    def test_404_is_not_retried(self):
        provider = GeminiProvider(api_key="k", backoff_seconds=0)
        with mock.patch("gateway.llm.requests.post", side_effect=[FakeResponse(404)]) as post:
            result = provider.generate("q", "s", "m")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result.status, "error")

    def test_max_tokens_is_flagged(self):
        provider = GeminiProvider(api_key="k")
        with mock.patch("gateway.llm.requests.post", side_effect=[FakeResponse(200, ok_body(finish="MAX_TOKENS"))]):
            result = provider.generate("q", "s", "m")
        self.assertEqual(result.status, "generated")
        self.assertIn("output cap", result.message)


if __name__ == "__main__":
    unittest.main()
