"""Command line entry point.

    python -m gateway "How do I reset my PIN?"
    python -m gateway "Compare APR on both cards" --intent investment_question
    GEMINI_API_KEY=... python -m gateway "How do I reset my PIN?" --live
    python -m gateway "test" --no-log          # do not write to the audit log

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
    args = parser.parse_args()

    provider = GeminiProvider() if args.live else None
    audit = None if args.no_log else AuditLog(args.log)
    record = process(args.query, intent=args.intent, provider=provider, audit=audit)
    print(record.to_json())
    if audit is not None:
        print(f"\nAppended to {audit.path} ({len(audit)} records total)")


if __name__ == "__main__":
    main()
