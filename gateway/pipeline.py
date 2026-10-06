"""Glue: run one query through all six stages and return the decision record.

Stages owned by teammates are stub functions that always let the query through.
To plug in a real implementation, replace the stub body; the record shape and
the order of stages do not change.
"""

from __future__ import annotations

from .config import GatewayConfig, load_config
from .llm import Provider, SimulatedProvider
from .prompts import system_prompt
from .router import route
from .schema import (
    CacheDecision,
    ComplianceDecision,
    DecisionRecord,
    GuardrailDecision,
    IntentDecision,
)


# ---- stubs for stages other teammates own ----------------------------------

def check_guardrails(query: str) -> GuardrailDecision:  # Team A
    return GuardrailDecision()


def lookup_cache(query: str) -> CacheDecision:  # Team B
    return CacheDecision()


def classify_intent(query: str, override: str | None = None) -> IntentDecision:  # Team C
    if override:
        return IntentDecision(label=override, confidence=1.0, reason="intent supplied by caller")
    return IntentDecision(reason="stub: classifier not implemented yet")


def check_compliance(answer: str) -> ComplianceDecision:  # Team D
    return ComplianceDecision()


# ---- the request path --------------------------------------------------------

def process(
    query: str,
    *,
    intent: str | None = None,
    provider: Provider | None = None,
    config: GatewayConfig | None = None,
) -> DecisionRecord:
    """Run ``query`` through the gateway and return its audit record."""
    config = config or load_config()
    provider = provider or SimulatedProvider(config.rules.expected_output_tokens)
    record = DecisionRecord(query=query)

    record.input_guardrails = check_guardrails(query)
    if record.input_guardrails.status == "block":
        record.final_status = "blocked"
        return record  # blocked questions never reach a model

    record.cache = lookup_cache(query)
    if record.cache.status == "hit":
        record.final_status = "cached"
        return record  # cached questions never reach a model either

    record.intent = classify_intent(query, intent)
    prompt = system_prompt(record.intent.prompt_template)
    record.router = route(
        query,
        intent=record.intent.label,
        risk=record.input_guardrails.status,
        system_prompt=prompt,
        config=config,
    )

    record.generation = provider.generate(query, prompt, record.router.model)
    if record.generation.status != "generated":
        record.final_status = "error"
        return record

    record.output_compliance = check_compliance(record.generation.answer)
    record.final_status = "answered" if record.output_compliance.status == "pass" else "blocked"
    return record
