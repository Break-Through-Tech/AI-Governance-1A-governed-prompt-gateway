"""Stage 4: pick the cheapest model tier that can handle the query.

Version 1 is rule-based on purpose. Rules are explainable (every decision
carries a ``reason``), they run in microseconds, and they need no training
data. A learned complexity model can replace ``complexity_signals`` later
without changing the function signature.
"""

from __future__ import annotations

import re

from .config import GatewayConfig, RoutingRules, load_config
from .cost import count_tokens, estimate_cost
from .schema import RouterDecision

# Each entry is (signal name, pattern). One match = one complexity point.
COMPLEXITY_SIGNALS: list[tuple[str, re.Pattern[str]]] = [
    ("multiple_questions", re.compile(r"\?.*\?", re.DOTALL)),
    ("numeric_reasoning", re.compile(
        r"\b(?:calculate|compute|how much|total|percent|interest|apr|rate)\b|%", re.IGNORECASE)),
    ("comparison", re.compile(
        r"\b(?:compare|versus|vs\.?|difference between|better|which (?:one|should))\b", re.IGNORECASE)),
    ("multi_step", re.compile(
        r"\b(?:and then|after that|step by step|first\b.*\bthen)\b", re.IGNORECASE | re.DOTALL)),
]


def complexity_signals(query: str, query_tokens: int, rules: RoutingRules) -> list[str]:
    """Names of every complexity signal present in the query."""
    found = [name for name, pattern in COMPLEXITY_SIGNALS if pattern.search(query)]
    if query_tokens > rules.long_query_tokens:
        found.append("long_query")
    return found


def route(
    query: str,
    *,
    intent: str = "unknown",
    risk: str = "allow",
    system_prompt: str = "",
    config: GatewayConfig | None = None,
) -> RouterDecision:
    """Choose a tier and estimate its cost against the baseline tier.

    ``risk`` is the policy engine's verdict (allow | escalate). A ``block``
    verdict never reaches the router; the pipeline stops before this stage.
    """
    config = config or load_config()
    rules = config.rules

    query_tokens = count_tokens(query)
    signals = complexity_signals(query, query_tokens, rules)
    score = len(signals)

    # First rule that fires wins, so order matters: policy > intent > complexity > default.
    if risk == "escalate":
        tier_name = rules.escalation_tier
        reason = "policy engine escalated this request"
    elif intent in rules.intent_tiers:
        tier_name = rules.intent_tiers[intent]
        reason = f"intent {intent!r} is mapped to {tier_name}"
    elif score >= rules.complexity_threshold:
        tier_name = rules.escalation_tier
        reason = f"{score} complexity signals: {', '.join(signals)}"
    else:
        tier_name = rules.default_tier
        reason = f"simple query ({score} complexity signal{'s' if score != 1 else ''}), default tier"

    tier = config.tier(tier_name)
    baseline = config.tier(rules.baseline_tier)

    # The model sees the system prompt too, so it counts toward input cost.
    input_tokens = query_tokens + count_tokens(system_prompt)
    output_tokens = rules.expected_output_tokens
    cost = estimate_cost(input_tokens, output_tokens, tier)
    baseline_cost = estimate_cost(input_tokens, output_tokens, baseline)

    return RouterDecision(
        tier=tier.name,
        model=tier.model,
        reason=reason,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_cost_usd=cost,
        baseline_tier=baseline.name,
        baseline_cost_usd=baseline_cost,
        savings_usd=round(baseline_cost - cost, 8),
        complexity_score=score,
        signals=signals,
    )
