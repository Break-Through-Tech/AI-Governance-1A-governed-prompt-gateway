"""Small, deterministic local embeddings for semantic response caching."""

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class Embedder(ABC):
    """Versioned embedding interface used by the response cache."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return an immutable identifier for the model and preprocessing."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one finite, unit-normalized vector per input string."""


_CANONICAL_TERMS = {
    "alter": "change",
    "changing": "change",
    "modify": "change",
    "reset": "change",
    "updated": "change",
    "updating": "change",
    "update": "change",
}


def _canonicalize(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return " ".join(_CANONICAL_TERMS.get(word, word) for word in words)


class LocalHashingEmbedder(Embedder):
    """Dependency-light local text embedder with no network or model download.

    Word and character n-gram features provide robust matching for close FAQ
    paraphrases. The version changes whenever preprocessing or dimensions do.
    """

    _VERSION = "local-hashing-word-char-v1-4096"

    def __init__(self) -> None:
        self._word = HashingVectorizer(
            n_features=2048,
            alternate_sign=False,
            norm=None,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._character = HashingVectorizer(
            analyzer="char_wb",
            n_features=2048,
            alternate_sign=False,
            norm=None,
            ngram_range=(3, 5),
        )

    @property
    def version(self) -> str:
        return self._VERSION

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValueError("texts must be a sequence of strings.")
        if not texts:
            return np.empty((0, 4096), dtype=np.float32)
        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("each text must be a nonempty string.")

        canonical = [_canonicalize(text) for text in texts]
        word = self._word.transform(canonical).astype(np.float32).toarray()
        character = self._character.transform(canonical).astype(np.float32).toarray()
        # Prefer semantic word overlap while retaining typo/inflection tolerance.
        vectors = np.concatenate((word * 0.8, character * 0.2), axis=1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
            raise ValueError("embedding produced an invalid zero or non-finite vector.")
        return np.asarray(vectors / norms, dtype=np.float32)


def validate_embeddings(vectors: np.ndarray, expected_rows: int) -> np.ndarray:
    """Validate and normalize vectors returned by any Embedder implementation."""

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != expected_rows or array.shape[1] < 1:
        raise ValueError("embedder returned an unexpected matrix shape.")
    if not np.all(np.isfinite(array)):
        raise ValueError("embedder returned non-finite values.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("embedder returned a zero vector.")
    return np.asarray(array / norms, dtype=np.float32)
