"""Evaluate the router against hand-labeled queries.

    python -m gateway.router_eval data/router_eval.csv
    python -m gateway.router_eval data/router_eval.csv --json

The CSV needs ``query``, ``intent`` (may be empty) and ``expected_tier``. The
report gives accuracy, a confusion table, and every misroute with the router's
own reason, so you can see *why* a rule fired or failed to fire.

Two kinds of error, with different costs:
    upgrade    expected a cheap tier, router chose a dearer one  -> wasted money
    downgrade  expected a dear tier, router chose a cheaper one  -> risk of a weak answer
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .config import GatewayConfig, load_config
from .prompts import system_prompt
from .router import route


def load_cases(path: Path | str, config: GatewayConfig) -> list[dict]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"query", "expected_tier"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} needs columns {sorted(required)}; found {reader.fieldnames}")
        cases = []
        for number, row in enumerate(reader, start=2):  # row 1 is the header
            query = (row.get("query") or "").strip()
            expected = (row.get("expected_tier") or "").strip()
            if not query:
                continue
            if expected not in config.tiers:
                raise ValueError(f"{path} row {number}: expected_tier {expected!r} is not one of {sorted(config.tiers)}")
            cases.append({"query": query, "intent": (row.get("intent") or "").strip() or None, "expected_tier": expected})
    if not cases:
        raise ValueError(f"{path} contains no cases")
    return cases


def evaluate(cases: list[dict], config: GatewayConfig | None = None) -> dict:
    """Route every case and compare with its label."""
    config = config or load_config()
    tier_order = list(config.tiers)  # YAML order: cheapest first
    rank = {name: index for index, name in enumerate(tier_order)}
    confusion: dict[str, Counter[str]] = {name: Counter() for name in tier_order}
    misroutes = []
    prompt = system_prompt("default")

    for case in cases:
        decision = route(case["query"], intent=case["intent"] or "unknown", system_prompt=prompt, config=config)
        expected, predicted = case["expected_tier"], decision.tier
        confusion[expected][predicted] += 1
        if predicted != expected:
            misroutes.append({
                "query": case["query"],
                "intent": case["intent"],
                "expected": expected,
                "predicted": predicted,
                "kind": "upgrade" if rank[predicted] > rank[expected] else "downgrade",
                "reason": decision.reason,
                "signals": decision.signals,
            })

    total = len(cases)
    correct = total - len(misroutes)
    kinds = Counter(m["kind"] for m in misroutes)
    return {
        "cases": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "upgrades": kinds.get("upgrade", 0),
        "downgrades": kinds.get("downgrade", 0),
        "confusion": {expected: dict(row) for expected, row in confusion.items()},
        "tiers": tier_order,
        "misroutes": misroutes,
        "config_version": config.version,
    }


def format_report(result: dict, show_all: bool = False) -> str:
    tiers = result["tiers"]
    lines = [
        f"Cases: {result['cases']}   Correct: {result['correct']}   Accuracy: {100 * result['accuracy']:.1f}%"
        f"   (price table {result['config_version']})",
        f"Upgrades (wasted cost): {result['upgrades']}   Downgrades (risk): {result['downgrades']}",
        "",
        "Confusion (rows = expected, columns = router chose):",
        "  " + " " * 10 + "".join(f"{t:>10}" for t in tiers),
    ]
    for expected in tiers:
        row = result["confusion"][expected]
        lines.append(f"  {expected:<10}" + "".join(f"{row.get(t, 0):>10}" for t in tiers))
    if result["misroutes"]:
        lines += ["", "Misroutes:"]
        for m in result["misroutes"]:
            intent = f" [{m['intent']}]" if m["intent"] else ""
            lines.append(f"  {m['kind']:<9} expected {m['expected']}, got {m['predicted']}: {m['query'][:70]}{intent}")
            lines.append(f"            reason: {m['reason']}")
    else:
        lines += ["", "No misroutes."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score the router against a labeled CSV.")
    parser.add_argument("cases", type=Path)
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON instead of a report.")
    args = parser.parse_args(argv)
    config = load_config()
    result = evaluate(load_cases(args.cases, config), config)
    print(json.dumps(result, indent=2) if args.json else format_report(result))


if __name__ == "__main__":
    main()
