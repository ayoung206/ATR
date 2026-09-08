"""Regression tests for sub-query-scoped ValueLinker fallback history."""
from __future__ import annotations

import unittest
from enum import Enum

from atr.online.value_linker import HybridValueLinker


class ValueLinkerHistoryTest(unittest.TestCase):
    def test_rejected_text_at_reroute_level_uses_fuzzy_without_extra_llm_call(self):
        calls = []
        self.linker.llm_fn = lambda messages: calls.append(messages) or '{"matched_value":"no_match"}'
        linked = self._link([
            {"sub_query": "current question", "route": "HYBRID", "value_linker_attempted": True},
            {"sub_query": "current question", "route": "TEXT", "requested_route": "RETRIEVE",
             "effective_route": "TEXT", "value_linker_attempted": True},
        ])
        self.assertEqual(linked.fallback_level, 2)
        self.assertEqual(linked.to_where_clause(), "City LIKE '%Saint Etienne%'")
        self.assertFalse(linked.needs_reroute)
        self.assertEqual(len(calls), 1)

    def test_question_wide_text_failure_blocks_reroute_with_no_candidates(self):
        history = [
            {"sub_query": "earlier question step", "route": "TEXT"},
            {"sub_query": "current question", "route": "HYBRID", "value_linker_attempted": True},
        ]
        linked = self.linker.link(
            ["missing"], self.schema, {}, history, "current question",
        )[0]
        self.assertFalse(linked.needs_reroute)
        self.assertFalse(linked.is_matched)
        self.assertEqual(linked.fallback_level, 1)

    def test_effective_route_overrides_requested_and_legacy_route_fields(self):
        self.candidates = {}
        for effective, reroutes in (("TEXT", False), ("HYBRID", True)):
            with self.subTest(effective=effective):
                linked = self._link([
                    {"sub_query": "current question", "route": "HYBRID", "value_linker_attempted": True},
                    {"route": "TEXT" if reroutes else "HYBRID",
                     "requested_route": "TEXT" if reroutes else "HYBRID",
                     "effective_route": effective, "value_linker_attempted": False},
                ])
                self.assertEqual(linked.needs_reroute, reroutes)

    def test_legacy_text_history_and_llm_exception_do_not_repeat_text(self):
        class LegacyRoute(Enum):
            TEXT = "TEXT"

        def failed_llm(_messages):
            raise RuntimeError("offline")

        self.linker.llm_fn = failed_llm
        self.candidates = {"Saint-Etienne": [{"col_name": "City", "value": "London"}]}
        for record in ("TEXT", LegacyRoute.TEXT, {"route": "TEXT"}):
            with self.subTest(record=record):
                linked = self._link([
                    {"sub_query": "current question", "route": "HYBRID", "value_linker_attempted": True},
                    record,
                ])
                self.assertFalse(linked.needs_reroute)
                self.assertFalse(linked.is_matched)

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
