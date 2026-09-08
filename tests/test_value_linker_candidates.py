"""Grounded pairs must survive LLM selection and every fallback path."""
import json
import unittest

from atr.online.value_linker import HybridValueLinker
from atr.online.constrained_sql import validate_sql_constraints


class ValueLinkerCandidateTest(unittest.TestCase):
    schema = [{"col_name": "City"}, {"col_name": "Club"}]
    candidates = [
        {"col_name": "City", "value": "Paris"},
        {"col_name": "City", "value": "London"},
        {"col_name": "Club", "value": "Saint Etienne"},
    ]

    def link(self, reply, candidates=None, entity="Saint-Etienne", retry=False):
        captured = []

        def llm(messages):
            captured.append(messages[0]["content"])
            if isinstance(reply, Exception):
                raise reply
            return reply if isinstance(reply, str) else json.dumps(reply)

        linked = HybridValueLinker(llm).link(
            [entity], self.schema,
            {entity: self.candidates if candidates is None else candidates},
            [{"sub_query": "q", "value_linker_attempted": True}] if retry else [],
            current_sub_query="q",
        )[0]
        return linked, captured

    def test_out_of_list_value_and_wrong_column_are_rejected(self):
        for reply in (
            {"matched_value": "Atlantis", "matched_column": "City", "confidence": 1},
            {"matched_value": "Saint Etienne", "matched_column": "City", "confidence": 1},
            {"matched_value": "Paris", "matched_column": "Secret", "confidence": 1},
        ):
            with self.subTest(reply=reply):
                linked, _ = self.link(reply)
                self.assertFalse(linked.is_matched)
                self.assertEqual(linked.to_where_clause(), "")

    def test_valid_pair_and_unique_legacy_reply_preserve_column(self):
        for reply in (
            {"matched_value": "Saint Etienne", "matched_column": "Club", "confidence": 1},
            {"matched_value": "Saint Etienne", "confidence": 1},
        ):
            with self.subTest(reply=reply):
                linked, _ = self.link(reply)
                self.assertEqual((linked.column, linked.matched_value), ("Club", "Saint Etienne"))
                self.assertEqual(linked.fallback_level, 0)

    def test_duplicate_values_require_column_selection(self):
        candidates = [{"col_name": col, "value": "Paris"} for col in ("City", "Club")]
        linked, _ = self.link({"matched_value": "Paris", "confidence": 1}, candidates)
        self.assertFalse(linked.is_matched)
        linked, _ = self.link(
            {"matched_value": "Paris", "matched_column": "Club", "confidence": 1}, candidates,
        )
        self.assertEqual((linked.column, linked.matched_value), ("Club", "Paris"))

    def test_fuzzy_and_exception_fallback_preserve_candidate_column(self):
        for reply in ({"matched_value": "no_match"}, RuntimeError("offline")):
            with self.subTest(reply=str(reply)):
                linked, _ = self.link(reply, retry=True)
                self.assertEqual(linked.to_where_clause(), "Club LIKE '%Saint Etienne%'")
                self.assertEqual(validate_sql_constraints(
                    "SELECT City FROM clubs WHERE Club LIKE '%Saint Etienne%'",
                    ["City", "Club"], ["clubs"], [linked],
                ), [])

    def test_columns_outside_C_are_never_prompted_or_used_by_fallback(self):
        for retry in (False, True):
            linked, prompts = self.link(
                {"matched_value": "Saint Etienne", "confidence": 1},
                [{"col_name": "Secret", "value": "Saint Etienne"}], retry=retry,
            )
            self.assertFalse(linked.is_matched)
            self.assertEqual(prompts, [])

    def test_fuzzy_does_not_guess_between_columns_or_match_empty_entity(self):
        candidates = [{"col_name": c, "value": "Paris"} for c in ("City", "Club")]
        for entity in ("Paris", " "):
            linked, _ = self.link({"matched_value": "no_match"}, candidates, entity, retry=True)
            self.assertFalse(linked.is_matched)

    def test_malformed_responses_follow_fallback_without_crashing(self):
        for reply in (
            "[]", "null", "not json",
            {"matched_value": "Paris", "confidence": "invalid"},
            {"matched_value": "Paris", "confidence": None},
            {"matched_value": "Paris", "confidence": float("nan")},
            {"matched_value": "Paris", "confidence": 2},
        ):
            with self.subTest(reply=reply):
                linked, _ = self.link(reply)
                self.assertFalse(linked.is_matched)

    def test_empty_and_malformed_candidates_are_ignored(self):
        linked, prompts = self.link({"matched_value": "no_match"}, [
            None, {}, {"col_name": "City", "value": None},
            {"col_name": "City", "value": " "}, {"col_name": "City", "value": []},
        ], retry=True)
        self.assertFalse(linked.is_matched)
        self.assertEqual(prompts, [])

    def test_numeric_zero_and_duplicate_pair_remain_valid(self):
        candidate = {"col_name": "City", "value": 0}
        linked, _ = self.link({"matched_value": "0", "confidence": 1}, [candidate, candidate])
        self.assertEqual((linked.column, linked.matched_value), ("City", "0"))
