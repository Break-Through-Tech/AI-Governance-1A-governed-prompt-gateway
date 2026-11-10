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
    "activating": "activate",
    "reset": "change",
    "resetting": "change",
    "alter": "change",
    "changing": "change",
    "branches": "branch",
    "fees": "fee",
    "charges": "charge",
    "modify": "change",
    "updated": "change",
    "updating": "change",
    "update": "change",
}


# Normalize the framing of a how-to question as a unit. Keeping a "howto"
# token distinguishes a request for instructions from a direct action request.
_HOW_TO = re.compile(
    r"^(?:how (?:do|can|should) i|how to|"
    r"what (?:is the (?:procedure|process|way)|are the steps) (?:to|for))\s+"
)
_DEFINITION = re.compile(r"^(?:what is|what s|(?:can|could) you explain|explain|define)\s+")
_POLITE_PREFIX = re.compile(r"^(?:please )?(?:(?:(?:could|can) you )?(?:tell|show) me )?")
_BANKING_PHRASES = {
    "personal identification number": "pin",
    "automated teller machine": "atm",
    "annual percentage rate": "apr",
    "mobile application": "mobile app",
    "telephone number": "phone number",
    "e mail": "email",
}
# Limit removal to articles and first-person possessives in how-to questions.
# Broad English stop lists also remove meaningful words such as "where" and "why".
_ARTICLES = {"a", "an", "the"}


_SUBJECT_NOUNS = {"branches": "branch", "centers": "center", "offices": "office", "desks": "desk"}

# Each pattern extracts what the question is about. Named suffixes retain
# restrictions after the predicate, e.g. a day, channel or location.
_QUESTION_FRAMES = (
    ("hours", r"what are (?:the )?(?:(?:opening|operating|business) )?hours (?:for|of) (?P<subject>.+)"),
    ("hours", r"what are (?P<subject>.+?) (?:(?:opening|operating|business) )?hours(?P<suffix> .+)?"),
    ("hours", r"when (?:is|are) (?P<subject>.+?) open(?P<suffix> .+)?"),
    ("fee", r"how much (?:does|do) (?P<subject>.+?) cost(?P<suffix> .+)?"),
    ("fee", r"what (?:is|s) (?:the |a )?(?:fee|cost|charge) (?:for|of) (?P<subject>.+)"),
    ("duration", r"how long (?:does|do) (?P<subject>.+?) take(?P<suffix> .+)?"),
    ("duration", r"what (?:is|s) (?:the )?processing time for (?P<subject>.+)"),
)
_FRAME_PATTERNS = tuple((kind, re.compile(pattern)) for kind, pattern in _QUESTION_FRAMES)


def _question_words(text: str) -> str:
    words = " ".join(re.findall(r"[^\W_]+", text.casefold()))
    return _POLITE_PREFIX.sub("", words)


def question_frame(text: str) -> tuple[str, str] | None:
    """Return the question type and ordered subject, including its qualifiers."""
    words = _question_words(text)
    for kind, pattern in _FRAME_PATTERNS:
        match = pattern.fullmatch(words)
        if match is None:
            continue
        subject = match["subject"] + (match.groupdict().get("suffix") or "")
        # In the single-bank FAQ, "your" refers to the bank being asked.
        # Preserve "my", named banks, places, weekdays and product qualifiers.
        if kind == "hours":
            subject = subject.removeprefix("your ")
        subject = re.sub(r"^(?:a|an|the) ", "", subject)
        subject = " ".join(_SUBJECT_NOUNS.get(word, word) for word in subject.split())
        subject = re.sub(r"\bbank branch\b", "branch", subject)
        return kind, subject
    return None


def _canonicalize(text: str) -> str:
    frame = question_frame(text)
    if frame is not None:
        return " ".join(frame)
    words = _HOW_TO.sub("howto ", _question_words(text))
    words = _DEFINITION.sub("definition ", words)
    for phrase, term in _BANKING_PHRASES.items():
        words = re.sub(r"\b" + phrase + r"\b", term, words)
    tokens = words.split()
    ignored = _ARTICLES | {"my"} if tokens[:1] == ["howto"] else _ARTICLES
    return " ".join(_CANONICAL_TERMS.get(word, word) for word in tokens if word not in ignored) or text.casefold()


class LocalHashingEmbedder(Embedder):
    """Dependency-light local text embedder with no network or model download.

    Word and character n-gram features provide robust matching for close FAQ
    paraphrases. The version changes whenever preprocessing or dimensions do.
    """

    _VERSION = "local-hashing-word-char-v3-4096"

    def __init__(self) -> None:
        self._word = HashingVectorizer(
            n_features=2048,
            alternate_sign=False,
            norm=None,
            token_pattern=r"(?u)\b\w+\b",
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
