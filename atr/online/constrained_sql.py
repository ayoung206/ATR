"""Retrieval-guided constrained SQL: column set C and value bindings V* are injected into the NL query before SQL generation; result is regenerated when no linked value appears in the output."""
from __future__ import annotations

import logging
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

def _has_mismatch(
    sql_result: str,
    linked_values: List[LinkedValue],
    retrieval_evidence: str,
) -> bool:
    """
    Principle (4): detect result-evidence conflict.

    A mismatch is flagged when:
      - Linked values are available AND none of their surface forms appear
        in the SQL result (wrong table / wrong column binding).
      - OR the SQL result is empty while retrieval evidence contains data
        (suggests the query hit the wrong place).

    Returns True if the result should be retried.
    """
    # Only check when SQL returned something non-trivial
    if not sql_result or sql_result.strip() in ("[]", "{}", ""):
        return False

    matched_values = [
        lv.matched_value for lv in linked_values
        if lv.is_matched and lv.matched_value
    ]
    if not matched_values:
        return False

    sql_lower = sql_result.lower()
    any_present = any(v.lower() in sql_lower for v in matched_values)
    if not any_present:
        logger.debug(
            f"Mismatch: linked values {matched_values} absent from SQL result"
        )
        return True

    return False

class ConstrainedSQLExecutor:
    """
    §3.5  Retrieval-Guided Constrained SQL Executor.

    Wraps the Flask SQL service (get_excel_rag_response_plain) with:
      - Column constraint injection into the query text        (Principle 1)
      - Value constraint injection (grounded WHERE conditions) (Principle 2)
      - Table context from restored schema                     (Principle 3)
      - Automatic retry on empty result or evidence mismatch   (Principle 4)
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
            retrieval_evidence: schema/cell evidence text, used for mismatch detection

        Returns:
            (sql_execution_result, execution_confidence)
        """
        col_names = [c["col_name"] for c in allowed_columns]
        value_bindings_text = build_value_bindings_text(linked_values)

        # Principle 3: inject table name from restored schema (View 2)
        table_context = ""
        if schema and schema.get("table_name"):
            table_context = f"\nTarget table: {schema['table_name']}"

        answer_type_hint = _infer_answer_type(sub_query)

        def _build_query(value_bindings: str) -> str:
            evidence_snippet = retrieval_evidence[:800] if retrieval_evidence else "(none)"
            base = CONSTRAINED_SQL_QUERY_TEMPLATE.format(
                original_query=sub_query,
                allowed_columns=", ".join(col_names) if col_names else "(all)",
                value_bindings=value_bindings,
                answer_type_hint=answer_type_hint,
                text_evidence=evidence_snippet,
            )
            return base + table_context

        enriched_query = _build_query(value_bindings_text)
        sql_result = ""

        for attempt in range(1, self.max_retries + 1):
            response = get_excel_rag_response_plain(
                table_name_list=self.table_name_list,
                query=enriched_query,
            )
            sql_result = str(response.get("sql_execution_result", ""))
            c_exec = _execution_confidence(sql_result)

            logger.debug(
                f"ConstrainedSQL attempt {attempt}: "
                f"c_exec={c_exec:.2f}, result={str(sql_result)[:120]}"
            )

            if c_exec == 0.0:
                # Principle (4) retry path A: empty / parse error
                if attempt < self.max_retries:
                    logger.info(
                        f"ConstrainedSQL: empty/failed result, relaxing value "
                        f"constraint (attempt {attempt}/{self.max_retries})"
                    )
                    enriched_query = _build_query(
                        "(unconstrained — retry after empty result)"
                    )
                continue

            # Principle (4) retry path B: result-evidence mismatch
            if _has_mismatch(sql_result, linked_values, retrieval_evidence):
                if attempt < self.max_retries:
                    logger.info(
                        f"ConstrainedSQL: result-evidence mismatch detected, "
                        f"retrying with relaxed value constraint "
                        f"(attempt {attempt}/{self.max_retries})"
                    )
                    enriched_query = _build_query(
                        "(unconstrained — retry after mismatch)"
                    )
                    continue

            return sql_result, c_exec

        # When all retries exhaust without c_exec=1.0, signal
        # "not found" explicitly rather than returning an empty/last
        # sql_result string. The verifier's evidence-fusion path will
        # then treat this as a definitive negative result instead of
        # asking the LLM to infer from absence.
        return "not found", 0.0
