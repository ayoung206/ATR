"""Shared cross-encoder reranking helpers for ATR's retrieval views."""
from __future__ import annotations

import threading
from typing import Any, List, Sequence, TypeVar


T = TypeVar("T")


def candidate_count(top_k: int, corpus_size: int, multiplier: int = 6) -> int:
    """Number of dense candidates to recall before cross-encoder reranking."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if corpus_size < 0:
        raise ValueError("corpus_size must be non-negative")
    if multiplier < 1:
        raise ValueError("rerank candidate multiplier must be at least 1")
    return min(corpus_size, top_k * multiplier)


class CrossEncoderCandidateReranker:
    """Thread-safe adapter around a ``compute_score`` cross-encoder backend."""

    def __init__(self, backend: Any, candidate_multiplier: int = 6) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        self.backend = backend
        self.candidate_multiplier = candidate_multiplier
        self._lock = threading.Lock()

    def recall_size(self, top_k: int, corpus_size: int) -> int:
        return candidate_count(top_k, corpus_size, self.candidate_multiplier)

    def select(
        self,
        query: str,
        candidates: Sequence[T],
        passages: Sequence[str],
        top_k: int,
    ) -> List[T]:
        """Return candidates ordered by descending cross-encoder score."""
        if len(candidates) != len(passages):
            raise ValueError("candidates and passages must have the same length")
        if top_k <= 0 or not candidates:
            return []

        pairs = [[query, passage] for passage in passages]
        with self._lock:
            scores = list(self.backend.compute_score(pairs))
        if len(scores) != len(candidates):
            raise RuntimeError(
                "reranker returned a different number of scores than candidates"
            )

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order[:top_k]]
