"""The six stage functions, one per team, in pipeline order.

This file is the plug-in point. A teammate replaces the body of their stub and
nothing else in the package has to change: the graph (``graph.py``) only calls
these names, and the record shape (``schema.py``) is fixed.

Stages we own (router, generation) live in their own modules and are imported
here so that every stage can be found from one place.
"""

from __future__ import annotations

from typing import Protocol

from .router import route  # noqa: F401  (stage 4, re-exported for discoverability)
from .schema import (
    CacheDecision,
    ComplianceDecision,
    GenerationResult,
    GuardrailDecision,
    IntentDecision,
)


# ---- stage 1: Team A --------------------------------------------------------

def check_guardrails(query: str) -> GuardrailDecision:
    """Stub: allows everything. Team A replaces the body with the policy engine."""
    return GuardrailDecision()


# ---- stage 2: Team B (adapter in cache_adapter.py) --------------------------

class Cache(Protocol):
    """What the pipeline needs from a cache. ``gateway.cache_adapter.GatewayCache`` implements it."""

    name: str

    def lookup(self, query: str, *, provider_name: str) -> tuple[CacheDecision, GenerationResult | None]: ...

    def store(self, query: str, generation: GenerationResult) -> bool: ...


def lookup_cache(query: str, cache: Cache | None, provider_name: str) -> tuple[CacheDecision, GenerationResult | None]:
    """Without a cache object the stage records a bypass rather than pretending it ran."""
    if cache is None:
        return CacheDecision(status="bypass", reason="no cache configured"), None
    return cache.lookup(query, provider_name=provider_name)


# ---- stage 3: Team C --------------------------------------------------------

def classify_intent(query: str, override: str | None = None) -> IntentDecision:
    """Stub: trusts the caller's override, otherwise returns ``unknown``."""
    if override:
        return IntentDecision(label=override, confidence=1.0, reason="intent supplied by caller")
    return IntentDecision(reason="stub: classifier not implemented yet")


# ---- stage 6: Team D --------------------------------------------------------

def check_compliance(answer: str) -> ComplianceDecision:
    """Stub: passes everything. Team D replaces the body with the output checker."""
    return ComplianceDecision()
