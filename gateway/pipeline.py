"""Entry point: run one query through the gateway and return its decision record.

The order of stages is defined in ``graph.py``; the stage functions live in
``stages.py``, ``router.py`` and ``llm.py``. This module only turns the caller's
arguments into a graph run, and the run's final state into a ``DecisionRecord``.
"""

from __future__ import annotations

from collections.abc import Callable

from .audit import AuditLog
from .config import GatewayConfig, load_config
from .graph import GRAPH, STAGE_KEYS, Deps, GatewayState
from .llm import Provider, SimulatedProvider
from .response import customer_response
from .schema import DecisionRecord
from .stages import Cache, InputGuard


def process(
    query: str,
    *,
    intent: str | None = None,
    provider: Provider | None = None,
    config: GatewayConfig | None = None,
    audit: AuditLog | None = None,
    cache: Cache | None = None,
    guardrails: InputGuard | None = None,
    on_step: Callable[[str, GatewayState], None] | None = None,
) -> DecisionRecord:
    """Run ``query`` through the gateway, append the record to ``audit`` if given, and return it.

    Every outcome is logged, including blocked, cached and error outcomes. An
    audit log that only recorded successes would not be an audit log.

    ``on_step(node_name, state)`` is called after each node finishes, with the
    state as it stands at that moment. The demo page uses it to light up the
    graph one node at a time; the CLI and the tests leave it unset.
    """
    config = config or load_config()
    provider = provider or SimulatedProvider(config.rules.expected_output_tokens)

    # Stream the run node by node instead of invoke(): same final state, but each node's
    # update arrives as soon as it is produced, so a listener can follow the run live.
    state: GatewayState = {"query": query, "intent_override": intent}
    deps = Deps(provider=provider, config=config, cache=cache, guardrails=guardrails)
    for chunk in GRAPH.stream(dict(state), context=deps, stream_mode="updates"):
        for node, update in chunk.items():  # one node per chunk in this graph
            state.update(update or {})  # a node that returns {} arrives as None
            if on_step is not None:
                on_step(node, state)

    record = DecisionRecord(
        query=state["query"],  # masked by stage 1
        final_status=state.get("final_status", "pending"),
        **{key: state.get(key) for key in STAGE_KEYS},  # a key the run never reached stays None
    )
    record.response = customer_response(record)
    if audit is not None:
        audit.append(record)
    return record
