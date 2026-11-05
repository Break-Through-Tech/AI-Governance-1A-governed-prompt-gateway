"""Remove sensitive values from a question before any other part of the gateway sees it.

Masking is the first thing stage 1 does. It is plain pattern matching: no model
call, no network, the same result every time. What comes out is the only
version of the question that exists from then on. Team A's policy check, the
cache, the model and the audit log all receive the masked text, so a card
number a customer types is never sent to a model provider or written to disk.

    My card number is 4111 1111 1111 1111 and my PIN is 1234, is it active?
    My card number is [CARD NUMBER] and my PIN is [PIN], is it active?

The patterns are deliberately simple and err toward masking too much. A long
reference number may be masked as a card number; that costs nothing. Things
this does not catch: values written out in words, and names or addresses.
"""

from __future__ import annotations

import re

# Each entry: (what to call it, pattern, replacement). Patterns with a keyword keep the
# keyword (group 1) and replace only the value, so the question still reads naturally.
_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("social security number", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    # 13 to 19 digits, optionally separated by spaces or hyphens: the length of a payment card number.
    ("card number", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), "[CARD NUMBER]"),
    ("account number", re.compile(
        r"\b((?:account|acct|routing|iban|sort code)\s*(?:number|no\.?|#)?\s*(?:is|:|=)?\s*)\d[\d -]{4,}\d",
        re.IGNORECASE), r"\1[ACCOUNT NUMBER]"),
    ("PIN", re.compile(
        r"\b((?:pin|passcode|cvv2?|cvc|security code)\s*(?:number|code|no\.?|#)?\s*(?:is|was|:|=)?\s*)\d[\d -]*\d",
        re.IGNORECASE), r"\1[PIN]"),
    # "my password is hunter2", but not "my password is not working".
    ("password", re.compile(
        r"\b((?:password|passwd|passphrase)\s*(?:is|was|:|=)\s*)"
        r"(?!(?:not|wrong|incorrect|invalid|expired|locked|blocked|too|being|still|no|correct|required|the|a|an|my)\b)"
        r"[^\s,;]+", re.IGNORECASE), r"\1[PASSWORD]"),
]


def mask_sensitive(text: str) -> tuple[str, list[str]]:
    """Return the masked text and the kinds of value that were removed, in the order found."""
    removed: list[str] = []
    for kind, pattern, replacement in _RULES:
        text, count = pattern.subn(replacement, text)
        if count:
            removed.append(kind)
    return text, removed
