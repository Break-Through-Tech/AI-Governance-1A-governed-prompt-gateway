"""Command line entry point.

    python -m gateway "How do I reset my PIN?"
    python -m gateway "Compare APR on both cards" --intent investment_question
    GEMINI_API_KEY=... python -m gateway "How do I reset my PIN?" --live
    python -m gateway "test" --no-log          # do not write to the audit log
    python -m gateway "How do I reset my PIN?" --cache semantic   # second run is a cache hit
    python -m gateway "My card is 4111 1111 1111 1111" --guardrails --live   # blocked before the model

Every run appends its decision record to var/decisions.jsonl unless --no-log is given.
"""

import argparse
from pathlib import Path

from .audit import DEFAULT_LOG_PATH, AuditLog
from .llm import GeminiProvider
from .pipeline import process


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one query through the gateway and print its decision record.")
    parser.add_argument("query")
    parser.add_argument("--intent", default=None, help="Override the intent label (the classifier is a stub for now).")
    parser.add_argument("--live", action="store_true", help="Call Gemini for real instead of simulating. Needs GEMINI_API_KEY.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH, help="Audit log path (JSON Lines).")
    parser.add_argument("--no-log", action="store_true", help="Print the record without appending it to the audit log.")
    parser.add_argument("--guardrails", action="store_true",
                        help="Check the question against Team A's input policy first. One extra Gemini call; needs GEMINI_API_KEY.")
    parser.add_argument("--cache", default="off", choices=["off", "exact", "semantic-shadow", "semantic"],
                        help="Response cache mode (default off). Entries live in var/gateway_cache.sqlite3.")
    args = parser.parse_args()

    provider = GeminiProvider() if args.live else None
    audit = None if args.no_log else AuditLog(args.log)
    cache = None
    if args.cache != "off":
        from .cache_adapter import GatewayCache  # imports her package, so only when asked
        cache = GatewayCache(mode=args.cache)
    guardrails = None
    if args.guardrails:
        from .guardrails_adapter import InputGuardrails
        guardrails = InputGuardrails()
    record = process(args.query, intent=args.intent, provider=provider, audit=audit, cache=cache,
                     guardrails=guardrails)
    print(record.to_json())
    if audit is not None:
        print(f"\nAppended to {audit.path} ({len(audit)} records total)")


if __name__ == "__main__":
    main()
