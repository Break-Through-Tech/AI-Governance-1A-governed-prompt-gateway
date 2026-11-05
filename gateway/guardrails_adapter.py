"""Stage 1 adapter: Team A's input policy behind ``check_guardrails``.

Team A's guardrail (``guardrails.py`` at the repo root) asks an LLM whether a
question breaks one of four bank policies, using NVIDIA's NeMo Guardrails
library. That library cannot be loaded next to this pipeline: it needs
``langchain-core`` below 0.4 while LangGraph 1.2 needs 1.4 or newer, and its
LangChain dependency does not import on Python 3.14.

What NeMo's "self check input" rail does is small, though: fill the question
into a prompt, send it to a model, and block when the model says Yes. This
adapter does exactly that with the gateway's own Gemini client, and reads both
the rules from Team A's own file, ``config2/prompts.yml``, so that file stays
the single source of the policy.

Which model answers is the one place this differs from their ``config2/config.yml``.
They name ``gemini-3.5-flash``. Measured on 2026-11-04 with the free tier:

* that model is limited to 20 requests per day per project, so the 21st
  question of the day could not be checked at all;
* it spends 200 to 500 billed thinking tokens to say Yes or No.

The gateway's own tier-1 model gave the same verdict on all ten test questions
with about 3 output tokens each, so the check runs on the default tier unless
``GUARDRAILS_MODEL`` (or the ``model`` argument) says otherwise. Because that
model is in the price table, the cost of every check is recorded too.

Three things are added on top of their prompt. The model is asked to name the
rule number when it answers Yes, so the customer can be told *why*. When the
question contains a placeholder left by ``masking.py``, the model is told the
real value is already gone, so a masked question can still be answered. And a
check that cannot run, or gives no clear verdict, blocks the question: an input
gate that opens when it breaks is not a gate.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from .config import GatewayConfig, load_config
from .cost import estimate_cost
from .llm import GeminiProvider
from .schema import GuardrailDecision

POLICY_DIR = Path(__file__).resolve().parents[1] / "config2"
TASK = "self_check_input"
MAX_QUERY_CHARS = 2000  # same cap the answer path applies to a question

# Appended after Team A's prompt, which ends with "Answer (Yes or No):".
RULE_NUMBER_REQUEST = (
    "\nIf the answer is Yes, also give the number of the first policy that is violated, "
    "for example: Yes 4. If the answer is No, reply with only: No."
)
# Added only when the question contains a placeholder left by ``masking.py``. Without it the model
# reads "[CARD NUMBER]" as a customer sharing a card number and blocks a question whose sensitive
# value is already gone. Pass ``masked_values_are_safe=False`` to keep Team A's stricter reading.
MASKED_VALUE_NOTE = (
    "\nNote: bracketed placeholders such as [CARD NUMBER], [PIN], [PASSWORD], [ACCOUNT NUMBER] or [SSN] "
    "mark values that were already removed from the user input before it reached you. A placeholder "
    "is not sensitive information, so it does not violate policy 4. Any sensitive value that is "
    "still written out does."
)
_MASK_TOKEN = re.compile(r"\[(?:CARD NUMBER|PIN|PASSWORD|ACCOUNT NUMBER|SSN)\]")
_PLACEHOLDER = re.compile(r"\{\{\s*user_input\s*\}\}")
_RULE = re.compile(r"^\s*(\d+)\.\s*([^:\n]+):\s*(.+)$", re.MULTILINE)
_VERDICT = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def load_policy(policy_dir: Path = POLICY_DIR) -> tuple[str, str, dict[int, tuple[str, str]]]:
    """Return (prompt template, model name, {rule number: (title, text)}) from Team A's files."""
    prompts = yaml.safe_load((policy_dir / "prompts.yml").read_text())["prompts"]
    template = next(p["content"] for p in prompts if p["task"] == TASK)
    models = yaml.safe_load((policy_dir / "config.yml").read_text())["models"]
    model = next(m["model"] for m in models if m["type"] == TASK)
    rules = {int(number): (title.strip(), text.strip()) for number, title, text in _RULE.findall(template)}
    if not _PLACEHOLDER.search(template):
        raise ValueError("Team A's prompt has no {{ user_input }} placeholder.")
    return template, model, rules


class InputGuardrails:
    """One Gemini call per question: does it break Team A's input policy?"""

    name = "llm-self-check"

    def __init__(self, provider: GeminiProvider | None = None, *, policy_dir: Path = POLICY_DIR,
                 config: GatewayConfig | None = None, model: str | None = None,
                 thinking_level: str | None = None, masked_values_are_safe: bool = True) -> None:
        self.masked_values_are_safe = masked_values_are_safe
        self.template, self.policy_model, self.rules = load_policy(policy_dir)  # policy_model: Team A's choice
        config = config or load_config()
        default_tier = config.tier(config.rules.default_tier)
        self.model = model or os.environ.get("GUARDRAILS_MODEL", "").strip() or default_tier.model
        # If the check runs on one of the gateway's tiers, its price and thinking setting are known.
        self.tier = next((tier for tier in config.tiers.values() if tier.model == self.model), None)
        self.thinking_level = thinking_level or (self.tier.thinking_level if self.tier else None)
        # temperature 0: the same question should get the same verdict every time.
        self.provider = provider or GeminiProvider(temperature=0.0, max_query_chars=8000)

    def render(self, query: str) -> str:
        """Fill the question into Team A's prompt. The lambda keeps backslashes in it literal."""
        prompt = _PLACEHOLDER.sub(lambda _: query[:MAX_QUERY_CHARS], self.template)
        if self.masked_values_are_safe and _MASK_TOKEN.search(query):
            prompt += MASKED_VALUE_NOTE
        return prompt + RULE_NUMBER_REQUEST

    def _blocked(self, reason: str, result=None, **extra) -> GuardrailDecision:
        return self._decision("block", reason, result, **extra)

    def _decision(self, status: str, reason: str, result=None, **extra) -> GuardrailDecision:
        usage = {}
        if result is not None:
            usage = dict(input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                         elapsed_ms=round(result.elapsed_ms, 1))
            if self.tier and isinstance(result.input_tokens, int) and isinstance(result.output_tokens, int):
                usage["cost_usd"] = estimate_cost(result.input_tokens, result.output_tokens, self.tier)
        return GuardrailDecision(status=status, reason=reason, model=self.model, **usage, **extra)

    def check(self, query: str) -> GuardrailDecision:
        result = self.provider.generate(self.render(query), "", self.model, thinking_level=self.thinking_level)
        if result.status != "generated":
            return self._blocked(f"the input safety check could not run ({result.message.rstrip('.')})", result)

        verdicts = _VERDICT.findall(result.answer)
        if not verdicts:
            return self._blocked("the input safety check gave no clear verdict", result)
        # A model may echo "Answer (Yes or No):" before answering, so the last yes/no is the verdict.
        if verdicts[-1].lower() == "no":
            return self._decision("allow", "passed the input policy check", result)

        tail = result.answer[result.answer.lower().rfind("yes"):]
        number = next((int(n) for n in re.findall(r"\d+", tail) if int(n) in self.rules), None)
        if number is None:
            return self._blocked("it goes against the bank's input policy", result)
        title, text = self.rules[number]
        text = text[0].lower() + text[1:].rstrip(".")
        return self._blocked(f"it goes against the {title} policy ({text})", result, policy=title)
