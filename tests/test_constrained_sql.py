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
