"""Lightweight diagnostics for free-generation behavior across training stages."""

from __future__ import annotations

import re
from statistics import median


DISMISSIVE_RE = re.compile(r"\b(гугл|документац|википед|не ной|сам разбер|иди читай|не парься|сам виноват)\w*", re.IGNORECASE)
REFUSAL_RE = re.compile(r"\b(не могу|не буду|не знаю|без понятия|отстань)\b", re.IGNORECASE)
WORD_RE = re.compile(r"\w+", re.UNICODE)


def _repeated_bigram_fraction(text: str) -> float:
    words = WORD_RE.findall(text.lower())
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)


def _summarize(records: list[dict]) -> dict[str, float | int]:
    """Summarize surface behavior without pretending it measures factual quality."""

    generations = [record.get("generated", "") for record in records]
    chars = [len(text) for text in generations]
    words = [len(WORD_RE.findall(text)) for text in generations]
    tokens = [int(record.get("completion_tokens", 0)) for record in records]
    target_chars = [len(record.get("target_source_rejected", "")) for record in records if "target_source_rejected" in record]
    return {
        "count": len(generations),
        "chars_mean": sum(chars) / len(chars) if chars else 0.0,
        "chars_median": float(median(chars)) if chars else 0.0,
        "words_mean": sum(words) / len(words) if words else 0.0,
        "completion_tokens_mean": sum(tokens) / len(tokens) if tokens else 0.0,
        "completion_tokens_median": float(median(tokens)) if tokens else 0.0,
        "eos_rate": sum(bool(record.get("finished_by_eos")) for record in records) / len(records) if records else 0.0,
        "max_token_cap_rate": sum(bool(record.get("hit_max_new_tokens")) for record in records) / len(records) if records else 0.0,
        "think_rate": sum("<think>" in text.lower() for text in generations) / len(generations) if generations else 0.0,
        "empty_rate": sum(not text.strip() for text in generations) / len(generations) if generations else 0.0,
        "dismissive_keyword_rate": sum(bool(DISMISSIVE_RE.search(text)) for text in generations) / len(generations) if generations else 0.0,
        "refusal_keyword_rate": sum(bool(REFUSAL_RE.search(text)) for text in generations) / len(generations) if generations else 0.0,
        "repeated_bigram_fraction_mean": sum(_repeated_bigram_fraction(text) for text in generations) / len(generations) if generations else 0.0,
        "target_chars_mean": sum(target_chars) / len(target_chars) if target_chars else 0.0,
        "at_or_below_target_chars_rate": sum(
            len(record.get("generated", "")) <= len(record.get("target_source_rejected", ""))
            for record in records
            if "target_source_rejected" in record
        ) / len(target_chars) if target_chars else 0.0,
    }


def summarize_generations(records: list[dict]) -> dict[str, dict[str, float | int]]:
    """Summarize each generation mode separately for comparable stage reports."""

    modes = sorted({str(record.get("mode", "unknown")) for record in records})
    return {mode: _summarize([record for record in records if record.get("mode", "unknown") == mode]) for mode in modes}
