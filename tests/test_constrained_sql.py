"""Regression tests for fail-closed SQL constraint enforcement."""
import unittest

from atr.online import constrained_sql as subject
from atr.online.value_linker import LinkedValue, build_value_bindings_text


def _binding() -> LinkedValue:
    return LinkedValue("1971", "Founded", "1971", 1.0)


def _fuzzy_binding() -> LinkedValue:
    return LinkedValue(
        "Saint-Etienne",
        "City",
        "Saint Etienne",
        0.3,
        fallback_level=2,
    )


class SQLConstraintValidationTest(unittest.TestCase):
    def test_aliases_cannot_authorize_forbidden_columns(self):
        for sql in (
            "SELECT Club AS Secret, Secret FROM clubs WHERE Founded = '1971'",
            "SELECT c.Secret FROM clubs c WHERE c.Founded = '1971' "
            "AND EXISTS (SELECT Club AS Secret FROM clubs)",
            "SELECT Club FROM clubs WHERE Secret = 'x' "
            "AND Founded = '1971' ORDER BY Club",
        ):
            with self.subTest(sql=sql):
                self.assertTrue(subject.validate_sql_constraints(
                    sql, ["Club", "Founded"], ["clubs"], [_binding()],
                ))

    def test_computed_derived_columns_do_not_satisfy_exact_binding(self):
        for sql in (
            "SELECT Club FROM (SELECT Club, '1971' AS Founded FROM clubs) x "
            "WHERE Founded = '1971'",
            "WITH x AS (SELECT Club, '1971' AS Founded FROM clubs) "
            "SELECT Club FROM x WHERE Founded = '1971'",
            "SELECT Club FROM (SELECT Club, Club AS Founded FROM clubs) x "
            "WHERE Founded = '1971'",
            "SELECT Club FROM (SELECT Club, COALESCE(Founded, '1971') AS Founded "
            "FROM clubs) x WHERE Founded = '1971'",
            "SELECT Club FROM (SELECT Club, Founded FROM clubs UNION ALL "
            "SELECT Club, '1971' AS Founded FROM clubs) x WHERE Founded = '1971'",
        ):
            with self.subTest(sql=sql):
                self.assertTrue(subject.validate_sql_constraints(
                    sql, ["Club", "Founded"], ["clubs"], [_binding()],
                ))

    def test_computed_derived_columns_do_not_satisfy_fuzzy_binding(self):
        self.assertTrue(subject.validate_sql_constraints(
            "SELECT Club FROM (SELECT Club, 'Saint Etienne' AS City FROM clubs) x "
            "WHERE City LIKE '%Saint Etienne%'",
            ["Club", "City"], ["clubs"], [_fuzzy_binding()],
        ))

    def test_accepts_real_columns_through_aliases_and_projections(self):
        for sql in (
            "SELECT c.Club AS name FROM clubs c WHERE c.Founded = '1971' ORDER BY name",
            "SELECT Club FROM (SELECT Club, Founded FROM clubs) x WHERE Founded = '1971'",
            "WITH x AS (SELECT Club, Founded AS year FROM clubs), "
            "y AS (SELECT Club, year FROM x) SELECT Club FROM y WHERE year = '1971'",
        ):
            with self.subTest(sql=sql):
                self.assertEqual(subject.validate_sql_constraints(
                    sql, ["Club", "Founded"], ["clubs"], [_binding()],
                ), [])
        self.assertEqual(subject.validate_sql_constraints(
            "SELECT COUNT(*) AS n FROM clubs ORDER BY n", ["Club"], ["clubs"], [],
        ), [])

    def test_table_names_are_resolved_in_their_own_scope(self):
        for sql in (
            "WITH other_table AS (SELECT Club FROM clubs) "
            "SELECT Club FROM foreign_db.other_table",
            "SELECT c.Club FROM clubs c WHERE EXISTS "
            "(WITH secret AS (SELECT Club FROM clubs) SELECT Club FROM secret) "
            "AND EXISTS (SELECT Club FROM secret)",
        ):
            with self.subTest(sql=sql):
                self.assertTrue(subject.validate_sql_constraints(
                    sql, ["Club"], ["clubs"], [],
                ))

    def test_formats_fuzzy_binding_as_like_with_canonical_value(self):
        self.assertEqual(
            build_value_bindings_text([_fuzzy_binding()]),
            "City LIKE '%Saint Etienne%'",
        )

    def test_fuzzy_binding_requires_like_and_canonical_value(self):
        self.assertEqual(
            subject.validate_sql_constraints(
                "SELECT Club FROM clubs WHERE City LIKE '%Saint Etienne%'",
                ["Club", "City"],
                ["clubs"],
                [_fuzzy_binding()],
            ),
            [],
        )

        for invalid_sql in (
            "SELECT Club FROM clubs WHERE City = 'Saint Etienne'",
            "SELECT Club FROM clubs WHERE City LIKE '%Saint-Etienne%'",
            "SELECT Club FROM clubs WHERE City LIKE '%Saint Etienne%' OR 1 = 1",
        ):
            violations = subject.validate_sql_constraints(
                invalid_sql,
                ["Club", "City"],
                ["clubs"],
                [_fuzzy_binding()],
            )
            self.assertTrue(
                any("missing fuzzy LIKE binding" in v for v in violations),
                violations,
            )

    def test_accepts_allowed_columns_table_and_exact_binding(self):
        violations = subject.validate_sql_constraints(
            "SELECT `Club` FROM `clubs` WHERE `Founded` = '1971'",
            ["Club", "Founded"],
            ["clubs"],
            [_binding()],
        )
        self.assertEqual(violations, [])

    def test_exact_binding_requires_mandatory_direct_equality(self):
        invalid_queries = (
            "SELECT Club FROM clubs WHERE Founded != '1971'",
            "SELECT Club FROM clubs WHERE Founded > '1971'",
            "SELECT Club FROM clubs WHERE Founded LIKE '1971'",
            "SELECT Club FROM clubs WHERE Founded IN ('1971', '1972')",
            "SELECT Club FROM clubs WHERE Founded = '1971' OR 1 = 1",
            "SELECT Club FROM clubs "
            "WHERE CASE WHEN Founded = '1971' THEN 1 ELSE 1 END = 1",
            "SELECT Club FROM clubs WHERE EXISTS "
            "(SELECT 1 FROM clubs WHERE Founded = '1971')",
        )
        for sql in invalid_queries:
            with self.subTest(sql=sql):
                violations = subject.validate_sql_constraints(
                    sql,
                    ["Club", "Founded"],
                    ["clubs"],
                    [_binding()],
                )
                self.assertTrue(
                    any("missing exact WHERE binding" in v for v in violations),
                    violations,
                )

    def test_accepts_all_exact_bindings_joined_by_and(self):
        city = LinkedValue("Paris", "City", "Paris", 1.0)
        violations = subject.validate_sql_constraints(
            "SELECT Club FROM clubs "
            "WHERE Founded = '1971' AND City = 'Paris'",
            ["Club", "Founded", "City"],
            ["clubs"],
            [_binding(), city],
        )
        self.assertEqual(violations, [])

    def test_requires_an_allowed_table_reference(self):
        violations = subject.validate_sql_constraints(
            "SELECT Founded WHERE Founded = '1971'",
            ["Founded"],
            ["clubs"],
            [_binding()],
        )
        self.assertIn("query does not reference an allowed table", violations)

    def test_sql_literal_is_escaped_in_prompt_binding(self):
        binding = LinkedValue("O'Reilly", "Publisher", "O'Reilly", 1.0)
        self.assertEqual(binding.to_where_clause(), "Publisher = 'O''Reilly'")
        self.assertEqual(
            subject.validate_sql_constraints(
                "SELECT Title FROM books WHERE Publisher = 'O''Reilly'",
                ["Title", "Publisher"],
                ["books"],
                [binding],
            ),
            [],
        )

    def test_rejects_disallowed_sql_and_missing_binding(self):
        violations = subject.validate_sql_constraints(
            "SELECT Secret FROM other_table WHERE Founded = '1972'; DELETE FROM clubs",
            ["Club", "Founded"],
            ["clubs"],
            [_binding()],
        )
        self.assertIn("exactly one SQL statement is required", violations)

        violations = subject.validate_sql_constraints(
            "SELECT Club FROM clubs WHERE Founded = '1972'",
            ["Club", "Founded"],
            ["clubs"],
            [_binding()],
        )
        self.assertTrue(any("missing exact WHERE binding" in v for v in violations))

    def test_allows_count_star_but_rejects_row_star(self):
        self.assertEqual(
            subject.validate_sql_constraints(
                "SELECT COUNT(*) FROM clubs", ["Club"], ["clubs"], []
            ),
            [],
        )
        violations = subject.validate_sql_constraints(
            "SELECT * FROM clubs", ["Club"], ["clubs"], []
        )
        self.assertIn("SELECT * is not allowed", violations)


class ConstrainedSQLExecutorTest(unittest.TestCase):
    def setUp(self):
        self.columns = [
            {"table_id": "clubs", "col_name": "Club"},
            {"table_id": "clubs", "col_name": "Founded"},
        ]
        self.schema = {
            "table_name": "clubs",
            "columns": [["Club", "object"], ["Founded", "object"]],
        }

    def test_rejects_then_repairs_without_relaxing_constraints(self):
        responses = iter([
            {"sql_str": "SELECT Secret FROM clubs", "sql_execution_result": "wrong"},
            {
                "sql_str": "SELECT Club FROM clubs WHERE Founded = '1971'",
                "sql_execution_result": "Alpha FC",
            },
        ])
        prompts = []

        def fake_service(**kwargs):
            prompts.append(kwargs["query"])
            return next(responses)

        original = subject.get_excel_rag_response_plain
        subject.get_excel_rag_response_plain = fake_service
        try:
            result = subject.ConstrainedSQLExecutor(["clubs"]).execute(
                "Which club was founded in 1971?",
                self.schema,
                self.columns,
                [_binding()],
            )
        finally:
            subject.get_excel_rag_response_plain = original

        self.assertEqual(result, ("Alpha FC", 1.0))
        self.assertEqual(len(prompts), 2)
        self.assertIn("STRICT CONSTRAINT REPAIR REQUIRED", prompts[1])
        self.assertNotIn("unconstrained", prompts[1].lower())

    def test_repairs_derived_binding_bypass_before_accepting_result(self):
        responses = iter([
            {
                "sql_str": "SELECT Club FROM (SELECT Club, '1971' AS Founded "
                           "FROM clubs) x WHERE Founded = '1971'",
                "sql_execution_result": "Wrong FC",
            },
            {
                "sql_str": "SELECT Club FROM clubs WHERE Founded = '1971'",
                "sql_execution_result": "Alpha FC",
            },
        ])
        from unittest.mock import patch
        with patch.object(subject, "get_excel_rag_response_plain", side_effect=lambda **_: next(responses)) as service:
            result = subject.ConstrainedSQLExecutor(["clubs"]).execute(
                "Which club was founded in 1971?", self.schema, self.columns, [_binding()],
            )
        self.assertEqual(result, ("Alpha FC", 1.0))
        self.assertIn("missing exact WHERE binding", service.call_args.kwargs["query"])

    def test_fails_closed_when_binding_never_appears(self):
        def fake_service(**_):
            return {
                "sql_str": "SELECT Club FROM clubs WHERE Founded = '1972'",
                "sql_execution_result": "Wrong FC",
            }

        original = subject.get_excel_rag_response_plain
        subject.get_excel_rag_response_plain = fake_service
        try:
            result = subject.ConstrainedSQLExecutor(["clubs"]).execute(
                "Which club was founded in 1971?",
                self.schema,
                self.columns,
                [_binding()],
            )
        finally:
            subject.get_excel_rag_response_plain = original

        self.assertEqual(result, ("not found", 0.0))

    def test_repairs_non_equality_that_only_mentions_the_exact_value(self):
        responses = iter([
            {
                "sql_str": "SELECT Club FROM clubs WHERE Founded != '1971'",
                "sql_execution_result": "Wrong FC",
            },
            {
                "sql_str": "SELECT Club FROM clubs WHERE Founded = '1971'",
                "sql_execution_result": "Alpha FC",
            },
        ])
        prompts = []

        def fake_service(**kwargs):
            prompts.append(kwargs["query"])
            return next(responses)

        original = subject.get_excel_rag_response_plain
        subject.get_excel_rag_response_plain = fake_service
        try:
            result = subject.ConstrainedSQLExecutor(["clubs"]).execute(
                "Which club was founded in 1971?",
                self.schema,
                self.columns,
                [_binding()],
            )
        finally:
            subject.get_excel_rag_response_plain = original

        self.assertEqual(result, ("Alpha FC", 1.0))
        self.assertIn("missing exact WHERE binding", prompts[1])

    def test_refuses_to_call_service_without_column_constraint(self):
        called = []
        original = subject.get_excel_rag_response_plain
        subject.get_excel_rag_response_plain = lambda **_: called.append(True)
        try:
            result = subject.ConstrainedSQLExecutor(["clubs"]).execute(
                "question", None, [], []
            )
        finally:
            subject.get_excel_rag_response_plain = original

        self.assertEqual(result, ("not found", 0.0))
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
