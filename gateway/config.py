"""Load ``config/tiers.yaml`` into typed, read-only objects.

Keeping prices and routing rules in a file means the team can change a tier or
a price without touching Python, and the ``version`` string lets us say exactly
which table produced a given decision record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "tiers.yaml"


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    input_per_million: float
    output_per_million: float
    description: str = ""


@dataclass(frozen=True)
class RoutingRules:
    default_tier: str
    baseline_tier: str
    escalation_tier: str
    expected_output_tokens: int
    long_query_tokens: int
    complexity_threshold: int
    intent_tiers: dict[str, str]


@dataclass(frozen=True)
class GatewayConfig:
    version: str
    currency: str
    tiers: dict[str, Tier]
    rules: RoutingRules

    def tier(self, name: str) -> Tier:
        try:
            return self.tiers[name]
        except KeyError:
            raise ValueError(f"Unknown tier {name!r}. Known tiers: {sorted(self.tiers)}") from None


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> GatewayConfig:
    with open(path, "rb") as handle:
        raw = yaml.safe_load(handle)  # safe_load: plain data only, never arbitrary objects

    tiers = {name: Tier(name=name, **spec) for name, spec in raw["tiers"].items()}
    routing = raw["routing"]
    rules = RoutingRules(
        default_tier=routing["default_tier"],
        baseline_tier=routing["baseline_tier"],
        escalation_tier=routing["escalation_tier"],
        expected_output_tokens=int(routing["expected_output_tokens"]),
        long_query_tokens=int(routing["long_query_tokens"]),
        complexity_threshold=int(routing["complexity_threshold"]),
        intent_tiers=dict(routing.get("intent_tiers") or {}),
    )
    config = GatewayConfig(version=str(raw["version"]), currency=raw["currency"], tiers=tiers, rules=rules)

    # Fail at load time, not mid-request, if a rule names a tier that does not exist.
    for name in (rules.default_tier, rules.baseline_tier, rules.escalation_tier, *rules.intent_tiers.values()):
        config.tier(name)
    return config
