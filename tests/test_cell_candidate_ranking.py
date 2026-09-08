"""Final entity top-K must not depend on the first retrieved schema column."""
import unittest
from types import SimpleNamespace

import numpy as np

from atr.offline.multiview_index import CellIndex, MultiviewIndex
from atr.offline.reranking import CrossEncoderCandidateReranker


class CellCandidateRankingTest(unittest.TestCase):
    def make_index(self, rerank=True):
        index = object.__new__(MultiviewIndex)
        pools = {
            col: [{"col_name": col, "value": f"value-{i:02}", "table_id": "t"}
                  for i in range(start, start + 15)]
            for col, start in (("A", 0), ("B", 15))
        }
        calls = []

        def retrieve(entity, column, top_k):
            calls.append((entity, column, top_k))
            return pools[column][:top_k]

        index.retrieve_schema = lambda *a, **k: [{"col_name": "A"}, {"col_name": "B"}]
        index.retrieve_cells = retrieve
        cell = object.__new__(CellIndex)
        index.cell_index = cell
        pairs_seen = []

        def scores(pairs):
            pairs_seen.extend(pairs)
            return [float(text.split("value-")[1][:2]) * (-1 if q == "low" else 1)
                    for q, text in pairs]

        cell.reranker = CrossEncoderCandidateReranker(SimpleNamespace(compute_score=scores)) if rerank else None

        def encode(texts):
            return np.array([
                [-1.0 if text == "low" else 1.0] if "value-" not in text
                else [float(text.split("value-")[1][:2])]
                for text in texts
            ], dtype=np.float32)

        cell.embedder = SimpleNamespace(encode=encode)
        return index, pools, calls, pairs_seen

    def test_global_top15_includes_later_column_in_both_modes(self):
        for rerank in (True, False):
            with self.subTest(rerank=rerank):
                index, _, _, pairs = self.make_index(rerank)
                _, raw = index.schema_cell_retrieval("question", ["high", "low"])
                self.assertEqual([c["value"] for c in raw["high"]],
                                 [f"value-{i:02}" for i in range(29, 14, -1)])
                self.assertEqual([c["value"] for c in raw["low"]],
                                 [f"value-{i:02}" for i in range(15)])
                if rerank:
                    self.assertEqual(len(pairs), 60)
                    self.assertEqual({q for q, _ in pairs}, {"high", "low"})

    def test_schema_order_duplicates_and_outside_C_do_not_change_ranking(self):
        index, pools, calls, pairs = self.make_index()
        pools["A"] += [pools["B"][0], pools["B"][0],
                       {"col_name": "Secret", "value": "value-99", "table_id": "t"}]
        index.retrieve_cells = lambda entity, column, top_k: pools[column]
        _, first = index.schema_cell_retrieval("q", ["high"])
        index.retrieve_schema = lambda *a, **k: [{"col_name": c} for c in ("B", "A", "B")]
        _, second = index.schema_cell_retrieval("q", ["high"])
        self.assertEqual(first, second)
        self.assertEqual(len(pairs), 60)  # 30 unique eligible pairs per call
        self.assertTrue(all("Secret" not in passage for _, passage in pairs))

    def test_empty_and_zero_budget_skip_candidate_scoring(self):
        index, pools, calls, pairs = self.make_index()
        _, raw = index.schema_cell_retrieval("q", ["high"], cell_top_k=0)
        self.assertEqual(raw, {"high": []})
        self.assertEqual(calls, [])
        pools["A"].clear()
        pools["B"].clear()
        _, raw = index.schema_cell_retrieval("q", ["high"])
        self.assertEqual(raw, {"high": []})
        self.assertEqual(pairs, [])
        with self.assertRaises(ValueError):
            index.schema_cell_retrieval("q", ["high"], cell_top_k=-1)
