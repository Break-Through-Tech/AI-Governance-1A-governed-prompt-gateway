"""The decision record: one JSON object the gateway emits for every request.

Each stage of the gateway fills in exactly one field of ``DecisionRecord``.
The record is the contract between teammates (everyone writes to their own
field and reads the fields before theirs) and it doubles as the audit log.

Stage owners:
    input_guardrails   Team A   (toxicity, jailbreak, policy verdict)
    cache              Team B   (semantic cache of pre-approved answers)
    intent             Team C   (intent label + system prompt template)
    router             us       (model tier + cost estimate)
    generation         us       (what the LLM or simulator returned)
    output_compliance  Team D   (advice / guarantee scanner)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class GuardrailDecision:
    """Stage 1. Verdict of the input screen and policy engine."""

    status: str = "allow"  # allow | block | escalate
    toxicity: float | None = None
    jailbreak: bool | None = None
    reason: str = "stub: guardrails not implemented yet"


@dataclass
class CacheDecision:
    """Stage 2. Did a pre-approved answer already cover this question?"""

    status: str = "miss"  # hit | miss | bypass
    similarity: float | None = None
    entry_id: str | None = None
    reason: str = "stub: cache not implemented yet"


@dataclass
class IntentDecision:
    """Stage 3. Which banking intent, and which system prompt template to use."""

    label: str = "unknown"
    confidence: float = 0.0
    prompt_template: str = "default"
    reason: str = ""


@dataclass
class RouterDecision:
    """Stage 4. Which model tier handles this query, why, and what it costs."""

    tier: str
    model: str
    reason: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    baseline_tier: str
    baseline_cost_usd: float
    savings_usd: float
    complexity_score: int = 0
    signals: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Stage 5. What the LLM (or the simulator) returned."""

    status: str  # generated | error
    provider: str  # simulated | gemini
    model: str
    answer: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_ms: float = 0.0
    simulated: bool = True
    message: str = ""  # user-safe error text when status == "error"


@dataclass
class ComplianceDecision:
    """Stage 6. Does the answer contain unauthorized advice or guarantees?"""

    status: str = "pass"  # pass | fail
    flags: list[str] = field(default_factory=list)
    reason: str = "stub: compliance not implemented yet"


@dataclass
class DecisionRecord:
    """The full audit record for one request. ``None`` means the stage never ran."""

    query: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    input_guardrails: GuardrailDecision | None = None
    cache: CacheDecision | None = None
    intent: IntentDecision | None = None
    router: RouterDecision | None = None
    generation: GenerationResult | None = None
    output_compliance: ComplianceDecision | None = None
    final_status: str = "pending"  # pending | answered | blocked | cached | error

    def to_dict(self) -> dict:
        """Nested plain dict; ``asdict`` recurses into the stage dataclasses."""
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
