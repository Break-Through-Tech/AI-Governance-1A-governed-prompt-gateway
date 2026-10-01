"""Small, explicit development-set calibration; not a held-out quality benchmark."""

import hashlib
import json
from pathlib import Path

from .retrieval import DEFAULT_THRESHOLD, Retriever


def evaluate(knowledge_path: Path, cases_path: Path) -> dict:
    retriever = Retriever(knowledge_path)
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    if not cases or not any(c["expected_source_rows"] for c in cases) or not any(not c["expected_source_rows"] for c in cases):
        raise ValueError("Development cases must include supported and unsupported questions.")
    source_rows = {row for r in retriever.records for row in r.get("source_rows", [])}
    for case in cases:
        if not set(case["expected_source_rows"]) <= source_rows:
            raise ValueError("Development case references missing source rows; use the corresponding knowledge file.")

    def run(threshold):
        details = []
        for case in cases:
            result = retriever.search(case["query"], threshold=threshold)
            expected = set(case["expected_source_rows"])
            matches = result["matches"]
            correct = [bool(expected.intersection(m["source_rows"])) for m in matches]
            details.append({
                "query": case["query"], "expected_source_rows": sorted(expected),
                "status": result["status"],
                "top_1_correct": bool(correct and correct[0]),
                "top_3_correct": any(correct),
                "matches": [{"id": m["id"], "source_rows": m["source_rows"], "score": m["score"]} for m in matches],
            })
        supported = [d for d in details if d["expected_source_rows"]]
        unsupported = [d for d in details if not d["expected_source_rows"]]
        recall = sum(d["top_3_correct"] for d in supported) / len(supported)
        rejection = sum(d["status"] == "no_match" for d in unsupported) / len(unsupported)
        metrics = {
            "threshold": threshold, "supported_cases": len(supported), "unsupported_cases": len(unsupported),
            "supported_top_1_accuracy": sum(d["top_1_correct"] for d in supported) / len(supported),
            "supported_recall_at_3": recall,
            "unsupported_rejection_rate": rejection,
            "balanced_score": (recall + rejection) / 2,
        }
        return metrics, details

    sweep = [run(round(i / 100, 2))[0] for i in range(10, 81, 5)]
    best = max(sweep, key=lambda m: (m["balanced_score"], m["unsupported_rejection_rate"], -m["threshold"]))
    metrics, details = run(DEFAULT_THRESHOLD)
    return {
        "purpose": "Development-set threshold calibration only; these cases are not an independent test set.",
        "knowledge_sha256": hashlib.sha256(Path(knowledge_path).read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(Path(cases_path).read_bytes()).hexdigest(),
        "retrieval": "Question-only word unigram/bigram TF-IDF, English stop words, cosine similarity, top 3",
        "selection_rule": "Maximize mean(supported recall@3, unsupported rejection); ties prefer rejection, then lower threshold.",
        "default_metrics": metrics, "recommended_threshold": best["threshold"],
        "threshold_sweep": sweep, "cases_at_default": details,
        "limitations": [
            "Tiny manually authored development set; results do not establish general chatbot quality.",
            "Lexical overlap can accept unsupported queries, and synonyms can miss relevant answers.",
            "Retrieval is not a safety classifier or proof that a question can be answered.",
            "Expected matches refer to 1-based data record numbers in the supplied Kaggle CSV.",
        ],
    }
