"""Prepare complete question/answer records without silently rewriting facts."""

import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data/banking/raw/Dataset_Banking_chatbot.csv"
DEFAULT_CLEAN = ROOT / "data/banking/processed/knowledge.jsonl"
DEFAULT_REPORT = ROOT / "reports/banking_data_quality.json"


def normalize(text: str) -> str:
    """Normalize typography and spacing while retaining case and wording."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    return " ".join(text.split())


def record_id(question: str, answer: str) -> str:
    pair = [normalize(question).casefold(), normalize(answer).casefold()]
    return "faq_" + hashlib.sha256(json.dumps(pair, ensure_ascii=False).encode()).hexdigest()[:16]


def review_flags(answer: str) -> list[str]:
    flags = []
    if re.search(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b|\b\d[\d ()-]{7,}\d\b", answer):
        flags.append("contact_detail")
    if re.search(r"\b(our|we|us|XYZ Bank)\b", answer, re.I):
        flags.append("fictional_bank_claim")
    return flags


def prepare(input_path: Path = DEFAULT_RAW, output_path: Path = DEFAULT_CLEAN,
            report_path: Path = DEFAULT_REPORT, encoding: str = "auto") -> dict:
    input_path, output_path, report_path = map(Path, (input_path, output_path, report_path))
    report_md = report_path.with_suffix(".md")
    paths = [p.resolve() for p in (input_path, output_path, report_path, report_md)]
    if len(set(paths)) != len(paths):
        raise ValueError("Input, cleaned output, JSON report, and Markdown report must be distinct paths.")
    raw = input_path.read_bytes()
    if encoding == "auto":
        try:
            decoded = raw.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            decoded = raw.decode("cp1252")
            encoding = "cp1252"
    else:
        decoded = raw.decode(encoding)
    reader = csv.DictReader(io.StringIO(decoded), strict=True)
    if reader.fieldnames != ["Query", "Response"]:
        raise ValueError(f"Expected CSV columns ['Query', 'Response']; found {reader.fieldnames!r}.")
    records, dropped, duplicates = {}, [], []
    changed, row_count = 0, 0
    for row_count, row in enumerate(reader, start=1):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV record {row_count}: expected exactly two fields.")
        question, answer = normalize(row["Query"]), normalize(row["Response"])
        if not question or not answer:
            dropped.append(row_count)
            continue
        changed += (question != row["Query"] or answer != row["Response"])
        rid = record_id(question, answer)
        if rid in records:
            records[rid]["source_rows"].append(row_count)
            duplicates.append({"source_row": row_count, "kept_id": rid})
            continue
        records[rid] = {
            "id": rid, "question": question, "answer": answer,
            "source_file": input_path.name, "source_rows": [row_count],
            "review_flags": review_flags(answer),
        }
    if not records:
        raise ValueError("The input contains no usable question/answer pairs.")
    groups = defaultdict(list)
    for record in records.values():
        groups[record["question"].casefold()].append(record)
    variants = []
    for group in groups.values():
        if len(group) > 1:
            variants.append({"question": group[0]["question"], "record_ids": [r["id"] for r in group]})
            for record in group:
                record["review_flags"].append("multiple_answers_for_question")
    ordered = sorted(records.values(), key=lambda r: r["id"])
    report = {
        "schema_version": 1,
        "source_file": input_path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_encoding": encoding,
        "input_rows": row_count,
        "cleaned_records": len(records),
        "normalized_rows": changed,
        "empty_rows_removed": dropped,
        "duplicate_pairs_removed": duplicates,
        "questions_with_multiple_answers": variants,
        "flagged_records": [
            {"id": r["id"], "source_rows": r["source_rows"], "flags": r["review_flags"]}
            for r in ordered if r["review_flags"]
        ],
        "limitations": [
            "Synthetic educational content; not verified policies of a real bank.",
            "Different answers to the same question are retained and flagged, not declared contradictions.",
            "Heuristic review flags are not a complete factual or contradiction check.",
            "Source row numbers count CSV data records starting at 1, excluding the header.",
        ],
    }
    for path in (output_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Banking dataset quality report", "",
        f"Source: `{input_path.name}`; decoded as `{encoding}`.", "",
        f"- Input records: {row_count}", f"- Cleaned records: {len(records)}",
        f"- Empty records removed: {len(dropped)}",
        f"- Duplicate pairs removed: {len(duplicates)}",
        f"- Records with normalized typography/spacing: {changed}",
        f"- Questions with multiple answer variants: {len(variants)}", "",
        "## Questions needing review", "",
    ]
    lines += [f"- {v['question']} ({len(v['record_ids'])} answer variants retained)" for v in variants] or ["None."]
    lines += ["", "## Contact details", ""]
    lines += [f"- Source rows {r['source_rows']}: {r['answer']}" for r in ordered if "contact_detail" in r["review_flags"]] or ["None detected."]
    lines += ["", "## Limits", ""] + [f"- {note}" for note in report["limitations"]]
    lines += ["", "See the JSON report for all flagged record IDs and the source checksum.", ""]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return report


def load_records(path: Path = DEFAULT_CLEAN) -> list[dict]:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Knowledge file contains no records. Run the prepare command first.")
    ids = set()
    for record in records:
        if not isinstance(record, dict) or any(
            not isinstance(record.get(key), str) or not record[key].strip()
            for key in ("id", "question", "answer")
        ):
            raise ValueError("Each knowledge record must have nonempty string id, question, and answer fields.")
        if record["id"] in ids:
            raise ValueError(f"Duplicate knowledge record ID: {record['id']}")
        ids.add(record["id"])
    return records
