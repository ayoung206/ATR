"""Retrieval-guided SQL with fail-closed structural constraint validation."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from atr.online.value_linker import LinkedValue, build_value_bindings_text
from atr.prompt import CONSTRAINED_SQL_QUERY_TEMPLATE

def _infer_answer_type(sub_query: str) -> str:
    """Heuristic: infer what type of value the query expects in SELECT."""
    q = sub_query.lower().strip()
    if q.startswith(("who ", "which person", "who is", "who was", "whose")):
        return "person name"
    if q.startswith(("what country", "which country", "what nation")):
        return "country name"
    if q.startswith(("what city", "which city", "what town", "where is", "where was", "where are")):
        return "city or location name"
    if q.startswith(("when ", "what year", "what date", "in what year", "what month")):
        return "date or year"
    if any(q.startswith(p) for p in ("how many", "how much", "what is the number", "count")):
        return "integer count or quantity"
    if "age" in q or "how old" in q or "born" in q:
        return "age or year (integer)"
    if "difference" in q or "how long" in q or "how far" in q:
        return "numeric difference"
    if q.startswith(("what is the name", "what was the name", "what is the title")):
        return "name or title string"
    return "(infer from query context)"

logger = logging.getLogger(__name__)

from atr.clients.sql_tool import get_excel_rag_response_plain  # noqa: E402

def _execution_confidence(sql_result: str) -> float:
    """
    c^exec = 1[parse] · 1[non-empty] · stability(o_t)
    Simplified: penalise failed / empty results.
    """
    if not sql_result or not sql_result.strip():
        return 0.0
    lower = sql_result.lower()
    if "error" in lower or "failed" in lower or "exception" in lower:
        return 0.0
    if "empty" in lower or "no result" in lower or sql_result.strip() in ("[]", "{}"):
        return 0.3
    return 1.0

def _normalise_identifier(value: Any) -> str:
    text = str(value or "").strip().strip("`\"'").lower()
    text = re.sub(r"\.(xlsx|xls|csv)$", "", text)
    return re.sub(r"[\s\-]+", "_", text)


def _schema_columns(schema: Optional[Dict[str, Any]]) -> List[str]:
    columns: List[str] = []
    for entry in (schema or {}).get("columns", []) or []:
        if isinstance(entry, dict):
            name = entry.get("col_name") or entry.get("name")
        elif isinstance(entry, (list, tuple)) and entry:
            name = entry[0]
        else:
            name = entry
        if name:
            columns.append(str(name))
    return columns


def _clean_sql(sql: str) -> str:
    text = str(sql or "").strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text).strip()


def validate_sql_constraints(
    sql: str,
    allowed_columns: List[str],
    allowed_tables: List[str],
    linked_values: List[LinkedValue],
) -> List[str]:
    """Return violations of ``C``/``V*``; empty means admissible SQL."""
    sql = _clean_sql(sql)
    if not sql:
        return ["SQL service did not return sql_str"]
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        return ["sqlglot is required for strict SQL constraint validation"]

    try:
        statements = [s for s in sqlglot.parse(sql, read="mysql") if s is not None]
    except Exception as exc:
        return [f"SQL parse failed: {exc}"]
    if len(statements) != 1:
        return ["exactly one SQL statement is required"]
    tree = statements[0]
    if not isinstance(tree, exp.Query) or tree.find(exp.Into):
        return ["only one read-only SELECT query is allowed"]

    violations: List[str] = []
    allowed_column_set = {_normalise_identifier(c) for c in allowed_columns}
    allowed_table_set = {_normalise_identifier(t) for t in allowed_tables if t}
    cte_names = {
        _normalise_identifier(cte.alias_or_name) for cte in tree.find_all(exp.CTE)
    }
    used_tables = {
        _normalise_identifier(table.name)
        for table in tree.find_all(exp.Table)
        if _normalise_identifier(table.name) not in cte_names
    }
    disallowed_tables = sorted(used_tables - allowed_table_set)
    if disallowed_tables:
        violations.append(f"disallowed tables: {', '.join(disallowed_tables)}")

    aliases = {
        _normalise_identifier(selection.alias)
        for select in tree.find_all(exp.Select)
        for selection in select.expressions
        if selection.alias
    }
    used_columns = {
        _normalise_identifier(column.name)
        for column in tree.find_all(exp.Column)
        if column.name != "*"
    }
    disallowed_columns = sorted(used_columns - allowed_column_set - aliases)
    if disallowed_columns:
        violations.append(f"disallowed columns: {', '.join(disallowed_columns)}")

    stars = list(tree.find_all(exp.Star))
    if any(star.find_ancestor(exp.Count) is None for star in stars):
        violations.append("SELECT * is not allowed")
    if not (used_columns & allowed_column_set) and not stars:
        violations.append("query does not reference an allowed column")

    where_nodes = list(tree.find_all(exp.Where))
    for linked in linked_values:
        if not linked.is_matched or linked.matched_value is None:
            continue
        wanted_column = _normalise_identifier(linked.column)
        wanted_value = str(linked.matched_value)
        binding_found = False
        for where in where_nodes:
            for predicate in where.find_all(exp.Predicate):
                predicate_columns = {
                    _normalise_identifier(column.name)
                    for column in predicate.find_all(exp.Column)
                }
                literal_values = [
                    str(literal.this) for literal in predicate.find_all(exp.Literal)
                ]
                if wanted_column not in predicate_columns:
                    continue
                if wanted_value in literal_values:
                    binding_found = True
                    break
                if linked.fallback_level == 2 and any(
                    wanted_value in literal.strip("%") for literal in literal_values
                ):
                    binding_found = True
                    break
            if binding_found:
                break
        if not binding_found:
            violations.append(
                f"missing exact WHERE binding: {linked.column}={wanted_value!r}"
            )
    return violations

class ConstrainedSQLExecutor:
    """
    §3.5  Retrieval-Guided Constrained SQL Executor.

    Wraps the Flask SQL service (get_excel_rag_response_plain) with:
      - Column constraint injection and AST enforcement        (Principle 1)
      - Value binding injection and WHERE enforcement           (Principle 2)
      - Table context from restored schema                     (Principle 3)
      - Fail-closed repair on invalid SQL or empty execution    (Principle 4)
    """

    def __init__(
        self,
        table_name_list: List[str],
        max_retries: int = 2,
    ) -> None:
        self.table_name_list = table_name_list
        self.max_retries = max_retries

    def execute(
        self,
        sub_query: str,
        schema: Optional[Dict[str, Any]],
        allowed_columns: List[Dict[str, Any]],
        linked_values: List[LinkedValue],
        retrieval_evidence: str = "",
    ) -> Tuple[str, float]:
        """
        Generate and execute constrained SQL.

        Args:
            sub_query:          natural-language sub-query
            schema:             restored table schema from View 2, used to inject
                                table name context into the query
            allowed_columns:    column entries from Schema Index (View 3)
            linked_values:      grounded value bindings V* from Constrained Value Linking
            retrieval_evidence: schema/cell evidence text supplied to SQL generation

        Returns:
            (sql_execution_result, execution_confidence)
        """
        col_names = [str(c["col_name"]) for c in allowed_columns if c.get("col_name")]
        if not col_names:
            col_names = _schema_columns(schema)
        col_names = list(dict.fromkeys(col_names))
        if not col_names:
            logger.warning("ConstrainedSQL: refusing execution without column constraint C")
            return "not found", 0.0

        allowed_tables = [
            str(c.get("table_id") or c.get("table_name") or c.get("source"))
            for c in allowed_columns
            if c.get("table_id") or c.get("table_name") or c.get("source")
        ]
        if schema and schema.get("table_name"):
            allowed_tables.append(str(schema["table_name"]))
        if not allowed_tables:
            allowed_tables = list(self.table_name_list)
        allowed_tables = list(dict.fromkeys(allowed_tables))
        if not allowed_tables:
            logger.warning("ConstrainedSQL: refusing execution without a table constraint")
            return "not found", 0.0

        value_bindings_text = build_value_bindings_text(linked_values)

        # Principle 3: inject table name from restored schema (View 2)
        table_context = ""
        if schema and schema.get("table_name"):
            table_context = f"\nTarget table: {schema['table_name']}"

        answer_type_hint = _infer_answer_type(sub_query)

        def _build_query(repair: str = "") -> str:
            evidence_snippet = retrieval_evidence[:800] if retrieval_evidence else "(none)"
            base = CONSTRAINED_SQL_QUERY_TEMPLATE.format(
                original_query=sub_query,
                allowed_columns=", ".join(col_names) if col_names else "(all)",
                value_bindings=value_bindings_text,
                answer_type_hint=answer_type_hint,
                text_evidence=evidence_snippet,
            )
            return base + table_context + repair

        enriched_query = _build_query()
        sql_result = ""

        for attempt in range(1, self.max_retries + 1):
            response = get_excel_rag_response_plain(
                table_name_list=self.table_name_list,
                query=enriched_query,
            )
            sql_result = str(response.get("sql_execution_result", ""))
            sql_str = str(response.get("sql_str", ""))
            violations = validate_sql_constraints(
                sql_str,
                allowed_columns=col_names,
                allowed_tables=allowed_tables,
                linked_values=linked_values,
            )
            if violations:
                logger.warning(
                    "ConstrainedSQL rejected generated SQL: %s",
                    "; ".join(violations),
                )
                if attempt < self.max_retries:
                    enriched_query = _build_query(
                        "\n\nSTRICT CONSTRAINT REPAIR REQUIRED\n"
                        f"Rejected SQL: {_clean_sql(sql_str)}\n"
                        f"Violations: {'; '.join(violations)}\n"
                        "Generate a new SQL query without relaxing C or V*."
                    )
                continue
            c_exec = _execution_confidence(sql_result)

            logger.debug(
                f"ConstrainedSQL attempt {attempt}: "
                f"c_exec={c_exec:.2f}, result={str(sql_result)[:120]}"
            )

            if c_exec != 1.0:
                if attempt < self.max_retries:
                    logger.info(
                        f"ConstrainedSQL: empty/failed result, retrying without "
                        f"relaxing constraints (attempt {attempt}/{self.max_retries})"
                    )
                    enriched_query = _build_query(
                        "\n\nExecution was empty or failed. Generate a different "
                        "query that still satisfies every constraint above."
                    )
                continue

            return sql_result, c_exec

        # When all retries exhaust without c_exec=1.0, signal
        # "not found" explicitly rather than returning an empty/last
        # sql_result string. The verifier's evidence-fusion path will
        # then treat this as a definitive negative result instead of
        # asking the LLM to infer from absence.
        return "not found", 0.0
