"""Unit tests for ATR's model-independent cross-encoder ordering contract."""
from __future__ import annotations

import unittest

from atr.offline.reranking import CrossEncoderCandidateReranker, candidate_count


class _RecordingBackend:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def compute_score(self, pairs):
        self.pairs = pairs
        return self.scores


class CandidateCountTest(unittest.TestCase):
    def test_overfetches_by_multiplier_and_caps_at_corpus(self):
        self.assertEqual(candidate_count(5, 100, multiplier=6), 30)
        self.assertEqual(candidate_count(5, 12, multiplier=6), 12)

    def test_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            candidate_count(-1, 10)
        with self.assertRaises(ValueError):
            candidate_count(1, 10, multiplier=0)


class CrossEncoderCandidateRerankerTest(unittest.TestCase):
    def test_orders_candidates_by_cross_encoder_score(self):
        backend = _RecordingBackend([0.1, 0.9, 0.4])
        reranker = CrossEncoderCandidateReranker(backend, candidate_multiplier=6)

        result = reranker.select(
            "capital of France",
            ["a", "b", "c"],
            ["Berlin", "Paris", "Madrid"],
            top_k=2,
        )

        self.assertEqual(result, ["b", "c"])
        self.assertEqual(
            backend.pairs,
            [
                ["capital of France", "Berlin"],
                ["capital of France", "Paris"],
                ["capital of France", "Madrid"],
            ],
        )

    def test_preserves_dense_order_for_equal_scores(self):
        backend = _RecordingBackend([0.5, 0.5, 0.2])
        reranker = CrossEncoderCandidateReranker(backend)
        self.assertEqual(
            reranker.select(
                "q", ["first", "second", "third"], ["1", "2", "3"], 2
            ),
            ["first", "second"],
        )

    def test_rejects_backend_score_count_mismatch(self):
        reranker = CrossEncoderCandidateReranker(_RecordingBackend([0.1]))
        with self.assertRaises(RuntimeError):
            reranker.select("q", ["a", "b"], ["A", "B"], 2)


if __name__ == "__main__":
    unittest.main()
