"""Regression tests for sub-query-scoped ValueLinker fallback history."""
from __future__ import annotations

import unittest

from atr.online.value_linker import HybridValueLinker


class ValueLinkerHistoryTest(unittest.TestCase):
    def setUp(self):
        self.linker = HybridValueLinker(
            lambda _messages: '{"matched_value":"no_match","confidence":0}'
        )
        self.schema = [{"col_name": "City", "dtype": "text"}]
        self.candidates = {
            "Saint-Etienne": [
                {"col_name": "City", "value": "Saint Etienne"}
            ]
        }

    def _link(self, history):
        return self.linker.link(
            entity_mentions=["Saint-Etienne"],
            schema_columns=self.schema,
            V_raw=self.candidates,
            history_H=history,
            current_sub_query="current question",
        )[0]

    def test_fresh_attempt_starts_with_unconstrained_fallback(self):
        linked = self._link([])
        self.assertEqual(linked.fallback_level, 1)
        self.assertFalse(linked.is_matched)

    def test_unrelated_sub_query_and_non_linker_failures_are_ignored(self):
        history = [
            {
                "sub_query": "different question",
                "route": "HYBRID",
                "value_linker_attempted": True,
            },
            {
                "sub_query": "current question",
                "route": "SQL",
                "requested_route": "SQL",
                "value_linker_attempted": False,
            },
        ]
        linked = self._link(history)
        self.assertEqual(linked.fallback_level, 1)

    def test_effective_text_record_still_advances_linker_fallback(self):
        history = [{
            "sub_query": "current question",
            "route": "TEXT",
            "requested_route": "HYBRID",
            "effective_route": "TEXT",
            "value_linker_attempted": True,
        }]
        linked = self._link(history)
        self.assertEqual(linked.fallback_level, 2)
        self.assertEqual(linked.matched_value, "Saint Etienne")

    def test_two_linker_failures_advance_to_text_reroute(self):
        history = [
            {
                "sub_query": "current question",
                "route": "HYBRID",
                "value_linker_attempted": True,
            },
            {
                "sub_query": "current question",
                "route": "RETRIEVE",
                "value_linker_attempted": True,
            },
        ]
        linked = self._link(history)
        self.assertEqual(linked.fallback_level, 3)
        self.assertTrue(linked.needs_reroute)

    def test_fuzzy_stage_without_candidates_falls_through_to_reroute(self):
        linked = self.linker.link(
            entity_mentions=["missing"],
            schema_columns=self.schema,
            V_raw={"missing": []},
            history_H=[{
                "sub_query": "current question",
                "route": "HYBRID",
                "value_linker_attempted": True,
            }],
            current_sub_query="current question",
        )[0]
        self.assertEqual(linked.fallback_level, 3)
        self.assertTrue(linked.needs_reroute)


if __name__ == "__main__":
    unittest.main()
