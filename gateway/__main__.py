"""Command line entry point.

    python -m gateway "How do I reset my PIN?"
    python -m gateway "Compare APR on both cards" --intent investment_question
    GEMINI_API_KEY=... python -m gateway "How do I reset my PIN?" --live
"""

import argparse

from .llm import GeminiProvider
from .pipeline import process


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one query through the gateway and print its decision record.")
    parser.add_argument("query")
    parser.add_argument("--intent", default=None, help="Override the intent label (the classifier is a stub for now).")
    parser.add_argument("--live", action="store_true", help="Call Gemini for real instead of simulating. Needs GEMINI_API_KEY.")
    args = parser.parse_args()

    provider = GeminiProvider() if args.live else None
    record = process(args.query, intent=args.intent, provider=provider)
    print(record.to_json())


if __name__ == "__main__":
    main()
