"""Verifiable grounding check.

A model's yes/no "is this supported?" verdict is noisy: the same correct answer can be judged
supported on one call and unsupported on the next. Instead the checker must give a quote for
every claim, and *code* confirms the quote really occurs in the passages. A made-up quote fails,
and a real one passes no matter how the checker felt on that call.

Matching is exact when possible. Because models sometimes reword a quote slightly, a quote also
passes if at least FUZZY_MIN of its words appear within one short stretch of a passage AND every
number in the quote appears there too (limits, prices and times are what a wrong answer gets wrong).
"""
import re
from collections import Counter

MIN_QUOTE_CHARS = 6   # shorter quotes ("the", "no") would match almost anything
FUZZY_MIN = 0.85      # share of the quote's words that must appear in the window
WINDOW_SLACK = 4      # extra words allowed in the window beyond the quote's length
MIN_FUZZY_WORDS = 4   # very short quotes must match exactly

_TOKEN = re.compile(r"[a-z0-9]+(?:[./][a-z0-9]+)*|[$%]")


def _norm(text: str) -> str:
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def _fuzzy_match(part: str, ctx_tokens: list[str]) -> bool:
    words = _TOKEN.findall(part)
    if len(words) < MIN_FUZZY_WORDS:
        return False
    need = Counter(words)
    numbers = {w for w in words if any(ch.isdigit() for ch in w)}
    size = len(words) + WINDOW_SLACK
    for i in range(max(1, len(ctx_tokens) - size + 1)):
        window = ctx_tokens[i:i + size]
        have = Counter(window)
        if sum(min(n, have[w]) for w, n in need.items()) / len(words) >= FUZZY_MIN and numbers <= set(window):
            return True
    return False


def quote_in_context(quote: str | None, context_norm: str) -> bool:
    """True if every '...'-separated part of the quote is (near-)verbatim in the context."""
    if not quote:
        return False
    parts = [_norm(p) for p in re.split(r"\.\.\.|…", quote)]
    parts = [p for p in parts if p]
    if not parts:
        return False
    ctx_tokens = _TOKEN.findall(context_norm)
    return all(len(p) >= MIN_QUOTE_CHARS and (p in context_norm or _fuzzy_match(p, ctx_tokens))
               for p in parts)


def unsupported_claims(claims: list[dict], context: str) -> list[str]:
    """Claims without a verifiable quote. An empty claim list counts as unverified."""
    if not claims:
        return ["the answer as a whole could not be verified"]
    ctx = _norm(context)
    return [str(c.get("claim", "?")) for c in claims if not quote_in_context(c.get("quote"), ctx)]
