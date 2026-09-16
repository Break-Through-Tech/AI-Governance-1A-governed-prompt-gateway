"""Question-only TF-IDF retrieval. Similarity is not factual confidence."""

import math
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import DEFAULT_CLEAN, load_records, normalize

# Selected with the small, explicitly labeled development set; see reports/retrieval_dev.json.
DEFAULT_THRESHOLD = 0.50


class Retriever:
    def __init__(self, knowledge_path: Path = DEFAULT_CLEAN):
        self.records = load_records(knowledge_path)
        self.vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", stop_words="english",
            ngram_range=(1, 2), sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform([r["question"] for r in self.records])

    def search(self, query: str, top_k: int = 3, threshold: float = DEFAULT_THRESHOLD) -> dict:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a finite number between 0 and 1.")
        query = normalize(query)
        if not query:
            raise ValueError("Please enter a nonempty question.")
        vector = self.vectorizer.transform([query])
        scores = cosine_similarity(vector, self.matrix).ravel()
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], self.records[i]["id"]))
        matches = []
        for i in ranked:
            score = max(0.0, min(1.0, float(scores[i])))
            if score <= 0 or score < threshold:
                continue
            matches.append({**self.records[i], "score": round(score, 6)})
            if len(matches) == top_k:
                break
        return {
            "status": "matched" if matches else "no_match",
            "query": query, "threshold": threshold, "matches": matches,
            "message": (
                "Retrieved synthetic reference material; no answer has been generated."
                if matches else
                "I couldn't find a sufficiently close banking FAQ. Please rephrase or give more details."
            ),
        }
