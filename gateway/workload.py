"""Run a batch of queries through the gateway and summarize the result.

This turns "one query, one record" into "one workload, one headline number".

    python -m gateway.workload run data/workload_sample.csv
    python -m gateway.workload summarize var/decisions.jsonl

``run`` processes every query with the simulated provider (no API calls), appends
each decision record to a JSON Lines file, and prints a summary. ``summarize``
re-reads any such file, including the live audit log, and prints the summary.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .audit import AuditLog
from .config import ROOT, GatewayConfig, load_config
from .cost import count_tokens, estimate_cost
from .llm import Provider, SimulatedProvider
from .pipeline import process
from .prompts import system_prompt
from .schema import DecisionRecord

DEFAULT_OUT_DIR = ROOT / "var"


# ---- loading queries ---------------------------------------------------------

def load_queries(path: Path | str) -> list[dict]:
    """Read queries from a CSV (needs a ``query`` column, optional ``intent``) or a text file (one per line)."""
    path = Path(path)
    rows: list[dict] = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "query" not in reader.fieldnames:
                raise ValueError(f"{path} needs a 'query' column; found {reader.fieldnames}")
            for row in reader:
                query = (row.get("query") or "").strip()
                if query:
                    rows.append({"query": query, "intent": (row.get("intent") or "").strip() or None})
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append({"query": line.strip(), "intent": None})
    if not rows:
        raise ValueError(f"{path} contains no queries")
    return rows


# ---- running -----------------------------------------------------------------

def run_workload(
    queries: Iterable[dict],
    *,
    provider: Provider | None = None,
    config: GatewayConfig | None = None,
    audit: AuditLog | None = None,
    cache=None,
    progress_every: int = 0,
) -> list[DecisionRecord]:
    """Process each query and return the records. ``audit`` receives every record as it is produced."""
    config = config or load_config()
    provider = provider or SimulatedProvider(config.rules.expected_output_tokens)
    records: list[DecisionRecord] = []
    for index, item in enumerate(queries, start=1):
        record = process(item["query"], intent=item.get("intent"), provider=provider, config=config, audit=audit,
                         cache=cache)
        records.append(record)
        if progress_every and index % progress_every == 0:
            print(f"  {index} processed", file=sys.stderr)
    return records


# ---- summarizing -------------------------------------------------------------

def _as_dict(record: DecisionRecord | dict) -> dict:
    return record.to_dict() if isinstance(record, DecisionRecord) else record


def _baseline_cost(record: dict, config: GatewayConfig) -> float:
    """What this query would have cost with no gateway: straight to the baseline tier.

    Blocked and cached queries never reach the router, so they carry no estimate.
    They still count in the baseline, because avoiding them is where the savings come from.
    """
    router = record.get("router")
    if router:
        return float(router["baseline_cost_usd"])
    tokens = count_tokens(record["query"]) + count_tokens(system_prompt("default"))
    baseline = config.tier(config.rules.baseline_tier)
    return estimate_cost(tokens, baseline.expected_output_tokens or config.rules.expected_output_tokens, baseline)


def summarize(records: Iterable[DecisionRecord | dict], config: GatewayConfig | None = None) -> dict:
    """Aggregate a batch of decision records into the dashboard numbers."""
    config = config or load_config()
    outcomes: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    model_calls = 0
    input_tokens = output_tokens = 0
    gateway_cost = baseline_cost = 0.0
    total = 0

    for raw in records:
        record = _as_dict(raw)
        total += 1
        outcomes[record.get("final_status", "unknown")] += 1
        baseline_cost += _baseline_cost(record, config)

        router = record.get("router")
        generation = record.get("generation")
        if router:
            tiers[router["tier"]] += 1
        # A cache hit also carries a generation block but no router block: it cost no model call.
        if router and generation and generation.get("status") == "generated":
            model_calls += 1
            # Prefer real token counts (live calls) over the router's estimate.
            inp = generation.get("input_tokens")
            out = generation.get("output_tokens")
            if inp is None or out is None:
                inp, out = router["estimated_input_tokens"], router["estimated_output_tokens"]
            input_tokens += inp
            output_tokens += out
            gateway_cost += estimate_cost(inp, out, config.tier(router["tier"]))

    savings = baseline_cost - gateway_cost
    return {
        "requests": total,
        "outcomes": dict(outcomes),
        "tier_split": dict(tiers),
        "model_calls": model_calls,
        "tokens": {"input": input_tokens, "output": output_tokens},
        "cost": {
            "gateway_usd": round(gateway_cost, 6),
            "baseline_usd": round(baseline_cost, 6),
            "savings_usd": round(savings, 6),
            "savings_pct": round(100 * savings / baseline_cost, 1) if baseline_cost else 0.0,
        },
        "config_version": config.version,
    }


def format_summary(summary: dict) -> str:
    """Human-readable version of ``summarize`` output."""
    n = summary["requests"] or 1
    lines = [f"Requests: {summary['requests']}   (price table {summary['config_version']})", "Outcomes:"]
    for name, count in sorted(summary["outcomes"].items()):
        lines.append(f"  {name:<9}{count:>7}  ({100 * count / n:5.1f}%)")
    lines.append("Tier split:")
    for name, count in sorted(summary["tier_split"].items()):
        lines.append(f"  {name:<9}{count:>7}  ({100 * count / n:5.1f}%)")
    cost = summary["cost"]
    lines += [
        f"Model calls: {summary['model_calls']}",
        f"Tokens: {summary['tokens']['input']} in / {summary['tokens']['output']} out",
        f"Cost through gateway: ${cost['gateway_usd']:.6f}",
        f"Cost with no gateway: ${cost['baseline_usd']:.6f}  (everything to the baseline tier)",
        f"Savings: ${cost['savings_usd']:.6f}  ({cost['savings_pct']}%)",
    ]
    return "\n".join(lines)


# ---- command line ------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a workload through the gateway or summarize a results file.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Process every query in a CSV or text file (simulated, no API calls).")
    run.add_argument("queries", type=Path)
    run.add_argument("--out", type=Path, default=None, help="Results file (JSON Lines). Default: var/workload_<timestamp>.jsonl")
    run.add_argument("--limit", type=int, default=None, help="Only process the first N queries.")

    summ = commands.add_parser("summarize", help="Summarize an existing JSON Lines results or audit file.")
    summ.add_argument("records", type=Path)

    args = parser.parse_args(argv)
    config = load_config()

    if args.command == "run":
        queries = load_queries(args.queries)
        if args.limit:
            queries = queries[: args.limit]
        out = args.out or DEFAULT_OUT_DIR / f"workload_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        audit = AuditLog(out)
        print(f"Running {len(queries)} queries from {args.queries} -> {out}", file=sys.stderr)
        records = run_workload(queries, config=config, audit=audit, progress_every=100)
        print(format_summary(summarize(records, config)))
    else:
        print(format_summary(summarize(AuditLog(args.records).read(), config)))


if __name__ == "__main__":
    main()
