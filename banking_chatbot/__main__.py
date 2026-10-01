"""Run with python -m banking_chatbot prepare|search|evaluate."""

import argparse
import csv
import json
from pathlib import Path

from .data import DEFAULT_CLEAN, DEFAULT_RAW, DEFAULT_REPORT, ROOT, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and search the synthetic banking FAQ locally.")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Clean the CSV and produce quality reports")
    prep.add_argument("--input", type=Path, default=DEFAULT_RAW)
    prep.add_argument("--output", type=Path, default=DEFAULT_CLEAN)
    prep.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    prep.add_argument("--encoding", default="auto", help="auto tries UTF-8, then Windows-1252")
    search = commands.add_parser("search", help="Retrieve up to three complete question/answer pairs")
    search.add_argument("query")
    search.add_argument("--knowledge", type=Path, default=DEFAULT_CLEAN)
    search.add_argument("--top-k", type=int, default=3)
    search.add_argument("--threshold", type=float, default=None)
    evaluation = commands.add_parser("evaluate", help="Run the small retrieval development set")
    evaluation.add_argument("--knowledge", type=Path, default=DEFAULT_CLEAN)
    evaluation.add_argument("--cases", type=Path, default=ROOT / "data/banking/retrieval_dev.json")
    evaluation.add_argument("--output", type=Path, default=ROOT / "reports/retrieval_dev.json")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            report = prepare(args.input, args.output, args.report, args.encoding)
            print(json.dumps({"cleaned_records": report["cleaned_records"], "output": str(args.output), "report": str(args.report)}, indent=2))
        elif args.command == "search":
            from .retrieval import DEFAULT_THRESHOLD, Retriever
            result = Retriever(args.knowledge).search(
                args.query, top_k=args.top_k,
                threshold=DEFAULT_THRESHOLD if args.threshold is None else args.threshold,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            from .evaluation import evaluate
            report = evaluate(args.knowledge, args.cases)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"report": str(args.output), "default_metrics": report["default_metrics"], "recommended_threshold": report["recommended_threshold"]}, indent=2))
    except (OSError, ValueError, csv.Error, LookupError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
