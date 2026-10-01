"""Fuzzy search helper built on rapidfuzz.

Used everywhere we need to turn a human-typed name ("general", "aliyyan")
into the best-matching Discord object (channel / member / role).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, TypeVar

from rapidfuzz import fuzz, process

T = TypeVar("T")


@dataclass
class FuzzyResult:
    item: object
    score: float  # 0-100
    matched_on: str


def fuzzy_search(
    query: str,
    candidates: Sequence[T],
    *,
    key=lambda x: str(x),
    score_cutoff: float = 50.0,
    limit: int = 5,
) -> list[FuzzyResult]:
    """Return the best-matching candidates for `query`.

    `key` extracts the comparable string from each candidate.
    Uses token_set_ratio (order-independent, robust to partial names),
    with partial_ratio as a fallback for short queries that might match
    a substring of a longer name (e.g. "revive" in "🔔 Chat Revive").
    """
    if not query or not candidates:
        return []
    choices = {i: key(c) for i, c in enumerate(candidates)}
    # Primary: token_set_ratio (order-independent).
    matches = process.extract(
        query,
        choices,
        scorer=fuzz.token_set_ratio,
        score_cutoff=score_cutoff,
        limit=limit,
    )
    # Fallback: partial_ratio (substring match) — catches short queries
    # that token_set_ratio misses. Take the best score for each candidate.
    partial_matches = process.extract(
        query,
        choices,
        scorer=fuzz.partial_ratio,
        score_cutoff=max(score_cutoff, 60.0),
        limit=limit,
    )
    # Merge: keep the best score per candidate index.
    best: dict[int, tuple[str, float]] = {}
    for matched_str, score, idx in list(matches) + list(partial_matches):
        if idx not in best or score > best[idx][1]:
            best[idx] = (matched_str, float(score))
    results: list[FuzzyResult] = []
    for idx, (matched_str, score) in best.items():
        results.append(FuzzyResult(item=candidates[idx], score=score, matched_on=matched_str))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


def best_match(
    query: str,
    candidates: Sequence[T],
    *,
    key=lambda x: str(x),
    score_cutoff: float = 60.0,
) -> FuzzyResult | None:
    res = fuzzy_search(query, candidates, key=key, score_cutoff=score_cutoff, limit=1)
    return res[0] if res else None
