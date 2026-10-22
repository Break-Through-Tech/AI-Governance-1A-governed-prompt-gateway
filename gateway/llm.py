"""Stage 5: produce the answer.

A *provider* is any object with ``name`` and
``generate(query, system_prompt, model, *, thinking_level=None)``. Two are included:

* ``SimulatedProvider``: no network. Used for the full workload run, where we only
  need token counts and routing decisions, not thousands of real API calls.
* ``GeminiProvider``: a live call for the demo. Adapted from Joohye's
  ``banking_chatbot/gemini.py`` on ``feature/semantic-cache``.

Both return a ``GenerationResult``. Errors come back as a result with
``status="error"`` rather than an exception, so the pipeline can still write a
complete decision record.
"""

from __future__ import annotations

import os
import time
from typing import Protocol

import requests

from .cost import count_tokens
from .env import load_dotenv
from .schema import GenerationResult


class Provider(Protocol):
    name: str

    def generate(self, query: str, system_prompt: str, model: str, *,
                 thinking_level: str | None = None) -> GenerationResult: ...


class SimulatedProvider:
    """Returns a placeholder answer and *estimated* token counts. Never touches the network."""

    name = "simulated"

    def __init__(self, expected_output_tokens: int = 120) -> None:
        self.expected_output_tokens = expected_output_tokens

    def generate(self, query: str, system_prompt: str, model: str, *,
                 thinking_level: str | None = None) -> GenerationResult:
        started = time.perf_counter()
        answer = f"[simulated {model}] Placeholder answer for: {query[:80]}"
        return GenerationResult(
            status="generated",
            provider=self.name,
            model=model,
            answer=answer,
            input_tokens=count_tokens(system_prompt) + count_tokens(query),
            output_tokens=self.expected_output_tokens,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            simulated=True,
            finish_reason="STOP",
        )


class GeminiProvider:
    """Live Gemini REST call. The API key never enters the prompt or the record.

    Transient failures (429 rate limit, 5xx overload) are retried a few times with
    a growing pause, which is what keeps a live demo alive on the free tier.
    """

    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    RETRY_STATUSES = {429, 500, 502, 503, 504}
    STATUS_MESSAGES = {
        400: "Gemini rejected the request. Check the model name, thinking level, and payload.",
        401: "Gemini rejected the API key.",
        403: "Gemini rejected the API key or project access.",
        404: "This Gemini model is not available to this key. Run the models list to pick another.",
        429: "Gemini rate limit or free quota reached, even after retries.",
        503: "Gemini reports high demand for this model, even after retries.",
    }

    def __init__(self, api_key: str | None = None, *, max_output_tokens: int = 1024,
                 temperature: float = 0.2, timeout: tuple[float, float] = (5, 40),
                 max_attempts: int = 3, backoff_seconds: float = 1.0) -> None:
        if api_key is None:
            load_dotenv()  # fills GEMINI_API_KEY from .env unless the shell already set it
            api_key = os.environ.get("GEMINI_API_KEY", "")
        self.api_key = api_key.strip()
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    def _error(self, model: str, message: str, started: float) -> GenerationResult:
        return GenerationResult(status="error", provider=self.name, model=model, message=message,
                                elapsed_ms=(time.perf_counter() - started) * 1000, simulated=False)

    def build_payload(self, query: str, system_prompt: str, thinking_level: str | None = None) -> dict:
        config: dict = {"temperature": self.temperature, "maxOutputTokens": self.max_output_tokens}
        if thinking_level:
            config["thinkingConfig"] = {"thinkingLevel": thinking_level}
        return {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": query[:2000]}]}],
            "generationConfig": config,
        }

    @staticmethod
    def parse_response(data: dict) -> tuple[str, dict, str | None]:
        """Return (answer, token counts, finish reason). Raises ValueError on a blocked or empty reply."""
        candidates = data.get("candidates") or []
        if (data.get("promptFeedback") or {}).get("blockReason") or not candidates:
            raise ValueError("Gemini returned no answer; the request may have been blocked.")
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        answer = "\n".join(p["text"] for p in parts if isinstance(p.get("text"), str) and not p.get("thought")).strip()
        if not answer:
            raise ValueError("Gemini returned an empty answer.")
        usage = data.get("usageMetadata") or {}
        answer_tokens = usage.get("candidatesTokenCount")
        thinking = usage.get("thoughtsTokenCount") or 0
        counts = {
            "input_tokens": usage.get("promptTokenCount"),
            "thinking_tokens": thinking,
            # Billed output is answer + thinking. Keep None if the API gave no answer count.
            "output_tokens": answer_tokens + thinking if isinstance(answer_tokens, int) else None,
        }
        return answer, counts, candidate.get("finishReason")

    def generate(self, query: str, system_prompt: str, model: str, *,
                 thinking_level: str | None = None) -> GenerationResult:
        started = time.perf_counter()
        if not self.api_key:
            return self._error(model, "No Gemini API key. Put it in .env or set GEMINI_API_KEY.", started)
        payload = self.build_payload(query, system_prompt, thinking_level)
        attempts = 0
        response = None
        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            try:
                response = requests.post(
                    self.URL.format(model=model),
                    headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                    json=payload, timeout=self.timeout,
                )
            except requests.Timeout:
                return self._error(model, "Gemini timed out.", started)
            except requests.RequestException:
                return self._error(model, "Could not connect to Gemini.", started)
            if response.status_code not in self.RETRY_STATUSES or attempt == self.max_attempts:
                break
            time.sleep(self.backoff_seconds * attempt)  # 1s, 2s, ...

        if response.status_code != 200:
            message = self.STATUS_MESSAGES.get(response.status_code, "Gemini is temporarily unavailable.")
            return self._error(model, message, started)
        try:
            answer, counts, finish = self.parse_response(response.json())
        except ValueError as exc:
            return self._error(model, str(exc), started)
        except (KeyError, TypeError, AttributeError):
            return self._error(model, "Gemini returned an unexpected response.", started)

        result = GenerationResult(
            status="generated", provider=self.name, model=model, answer=answer,
            elapsed_ms=(time.perf_counter() - started) * 1000, simulated=False,
            finish_reason=finish, **counts,
        )
        if finish == "MAX_TOKENS":
            result.message = f"Answer hit the {self.max_output_tokens}-token output cap and may be incomplete."
        if attempts > 1:
            result.message = (result.message + " " if result.message else "") + f"Succeeded after {attempts} attempts."
        return result
