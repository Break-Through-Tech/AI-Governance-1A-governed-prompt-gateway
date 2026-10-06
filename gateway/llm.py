"""Stage 5: produce the answer.

A *provider* is any object with ``name`` and ``generate(query, system_prompt, model)``.
Two are included:

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
from .schema import GenerationResult


class Provider(Protocol):
    name: str

    def generate(self, query: str, system_prompt: str, model: str) -> GenerationResult: ...


class SimulatedProvider:
    """Returns a placeholder answer and *estimated* token counts. Never touches the network."""

    name = "simulated"

    def __init__(self, expected_output_tokens: int = 120) -> None:
        self.expected_output_tokens = expected_output_tokens

    def generate(self, query: str, system_prompt: str, model: str) -> GenerationResult:
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
        )


class GeminiProvider:
    """Live Gemini REST call. The API key never enters the prompt or the record."""

    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    STATUS_MESSAGES = {
        400: "Gemini rejected the request. Check the model name and payload.",
        401: "Gemini rejected the API key.",
        403: "Gemini rejected the API key or project access.",
        404: "This Gemini model is not available for this API.",
        429: "Gemini rate limit or free quota reached. Wait and retry.",
    }

    def __init__(self, api_key: str | None = None, *, max_output_tokens: int = 384,
                 temperature: float = 0.2, timeout: tuple[float, float] = (5, 20)) -> None:
        self.api_key = (api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")).strip()
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout = timeout

    def _error(self, model: str, message: str, started: float) -> GenerationResult:
        return GenerationResult(status="error", provider=self.name, model=model, message=message,
                                elapsed_ms=(time.perf_counter() - started) * 1000, simulated=False)

    def build_payload(self, query: str, system_prompt: str) -> dict:
        return {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": query[:2000]}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

    def generate(self, query: str, system_prompt: str, model: str) -> GenerationResult:
        started = time.perf_counter()
        if not self.api_key:
            return self._error(model, "No Gemini API key. Set GEMINI_API_KEY or use the simulated provider.", started)
        try:
            response = requests.post(
                self.URL.format(model=model),
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=self.build_payload(query, system_prompt),
                timeout=self.timeout,
            )
        except requests.Timeout:
            return self._error(model, "Gemini timed out.", started)
        except requests.RequestException:
            return self._error(model, "Could not connect to Gemini.", started)

        if response.status_code != 200:
            message = self.STATUS_MESSAGES.get(response.status_code, "Gemini is temporarily unavailable.")
            return self._error(model, message, started)

        try:
            data = response.json()
            candidates = data.get("candidates") or []
            if (data.get("promptFeedback") or {}).get("blockReason") or not candidates:
                return self._error(model, "Gemini returned no answer; the request may have been blocked.", started)
            parts = candidates[0].get("content", {}).get("parts", [])
            answer = "\n".join(p["text"] for p in parts if isinstance(p.get("text"), str)).strip()
            usage = data.get("usageMetadata") or {}
        except (ValueError, KeyError, TypeError, AttributeError):
            return self._error(model, "Gemini returned an unexpected response.", started)
        if not answer:
            return self._error(model, "Gemini returned an empty answer.", started)

        return GenerationResult(
            status="generated",
            provider=self.name,
            model=model,
            answer=answer,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            simulated=False,
        )
