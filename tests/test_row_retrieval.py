"""Regression tests for the paper-aligned row retrieval contract."""
from __future__ import annotations

import unittest

import faiss
import numpy as np
import pandas as pd

from atr.offline.multiview_index import RowIndex


class _QueryEmbedder:
    def encode(self, texts):
        return np.repeat(np.array([[1.0, 0.0]], dtype=np.float32), len(texts), axis=0)


class _RecordingReranker:
    def __init__(self):
        self.recall_calls = []
        self.candidate_count = 0

    def recall_size(self, top_k, available):
        self.recall_calls.append((top_k, available))
        return min(top_k * 6, available)

    def select(self, query, candidates, texts, top_k):
        del query, texts
        self.candidate_count = len(candidates)
        return candidates[:top_k]


def _entry(table_id, row_idx):
    return {
        "table_id": table_id,
        "row_idx": row_idx,
        "row_text": f"Table {table_id} | value: {row_idx}",
        "row_dict": {"value": str(row_idx)},
    }


class RowRetrievalTest(unittest.TestCase):
    def test_table_filter_is_applied_inside_faiss_before_top_k(self):
        row_index = RowIndex(_QueryEmbedder())
        distractor_count = 120
        target_count = 12
        row_index.entries = [
            *[_entry("other", i) for i in range(distractor_count)],
            *[_entry("target", i) for i in range(target_count)],
        ]

        vectors = np.asarray(
            [[1.0, 0.0]] * distractor_count
            + [[0.50 - i * 0.01, 0.0] for i in range(target_count)],
            dtype=np.float32,
        )
        row_index._index = faiss.IndexFlatIP(2)
        row_index._index.add(vectors)

        hits = row_index.retrieve("question", top_k=10, table_id="target")

        self.assertEqual(len(hits), 10)
        self.assertTrue(all(hit["table_id"] == "target" for hit in hits))
        self.assertEqual([hit["row_idx"] for hit in hits], list(range(10)))

    def test_reranker_recall_pool_uses_target_table_population(self):
        reranker = _RecordingReranker()
        row_index = RowIndex(_QueryEmbedder(), reranker=reranker)
        row_index.entries = [
            *[_entry("other", i) for i in range(120)],
            *[_entry("target", i) for i in range(12)],
        ]
        vectors = np.asarray(
            [[1.0, 0.0]] * 120
            + [[0.50 - i * 0.01, 0.0] for i in range(12)],
            dtype=np.float32,
        )
        row_index._index = faiss.IndexFlatIP(2)
        row_index._index.add(vectors)

        hits = row_index.retrieve("question", top_k=10, table_id="target")

        self.assertEqual(reranker.recall_calls, [(10, 12)])
        self.assertEqual(reranker.candidate_count, 12)
        self.assertEqual(len(hits), 10)

    def test_default_indexing_keeps_all_rows_and_full_row_text(self):
        long_value = "x" * 700
        frame = pd.DataFrame({
            "id": range(150),
            "description": [long_value] * 150,
        })
        row_index = RowIndex(_QueryEmbedder())

        row_index.add_table("large", frame)

        self.assertEqual(len(row_index.entries), 150)
        self.assertEqual(row_index.entries[-1]["row_idx"], 149)
        self.assertIn(long_value, row_index.entries[-1]["row_text"])

    def test_explicit_safety_limits_are_still_enforced(self):
        frame = pd.DataFrame({"id": range(150)})
        row_index = RowIndex(
            _QueryEmbedder(), budget=120, per_table_quota=110, max_row_chars=20
        )

        row_index.add_table("first", frame)
        row_index.add_table("second", frame)

        self.assertEqual(len(row_index.entries), 120)
        self.assertEqual(sum(e["table_id"] == "first" for e in row_index.entries), 110)
        self.assertEqual(sum(e["table_id"] == "second" for e in row_index.entries), 10)
        self.assertTrue(all(len(e["row_text"]) <= 20 for e in row_index.entries))


if __name__ == "__main__":
    unittest.main()
