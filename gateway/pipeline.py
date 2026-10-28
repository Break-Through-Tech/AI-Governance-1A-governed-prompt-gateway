"""Entry point: run one query through the gateway and return its decision record.

The order of stages is defined in ``graph.py``; the stage functions live in
``stages.py``, ``router.py`` and ``llm.py``. This module only turns the caller's
arguments into a graph run, and the run's final state into a ``DecisionRecord``.
"""

from __future__ import annotations

from .audit import AuditLog
from .config import GatewayConfig, load_config
from .graph import GRAPH, STAGE_KEYS, Deps
from .llm import Provider, SimulatedProvider
from .schema import DecisionRecord
from .stages import Cache


def process(
    query: str,
    *,
    intent: str | None = None,
    provider: Provider | None = None,
    config: GatewayConfig | None = None,
    audit: AuditLog | None = None,
    cache: Cache | None = None,
) -> DecisionRecord:
    """Run ``query`` through the gateway, append the record to ``audit`` if given, and return it.

    Every outcome is logged, including blocked, cached and error outcomes. An
    audit log that only recorded successes would not be an audit log.
    """
    config = config or load_config()
    provider = provider or SimulatedProvider(config.rules.expected_output_tokens)

    state = GRAPH.invoke(
        {"query": query, "intent_override": intent},
        context=Deps(provider=provider, config=config, cache=cache),
    )

    record = DecisionRecord(
        query=query,
        final_status=state.get("final_status", "pending"),
        **{key: state.get(key) for key in STAGE_KEYS},  # a key the run never reached stays None
    )
    if audit is not None:
        audit.append(record)
    return record
