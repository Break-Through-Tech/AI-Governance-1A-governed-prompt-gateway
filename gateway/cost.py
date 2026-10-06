"""Token counting and cost arithmetic.

The whole cost story rests on two tiny functions: how many tokens is this text,
and what does a tier charge for them. Everything else is multiplication.
"""

from __future__ import annotations

from .config import Tier

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # tiktoken not installed, or its vocabulary file could not be fetched
    _ENCODING = None

CHARS_PER_TOKEN = 4  # rough English average; used only if tiktoken is unavailable


def count_tokens(text: str) -> int:
    """Estimate tokens for ``text``.

    tiktoken's ``cl100k_base`` is an OpenAI tokenizer, not Gemini's, so this is an
    estimate for planning and comparison. The live provider reports real counts.
    """
    if not text:
        return 0
    if _ENCODING is None:
        return max(1, len(text) // CHARS_PER_TOKEN)
    return len(_ENCODING.encode(text))


def estimate_cost(input_tokens: int, output_tokens: int, tier: Tier) -> float:
    """USD cost of a call with these token counts on this tier."""
    cost = (input_tokens * tier.input_per_million + output_tokens * tier.output_per_million) / 1_000_000
    return round(cost, 8)
