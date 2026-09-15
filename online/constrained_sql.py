"""Retrieval-guided SQL with fail-closed structural constraint validation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from clients.sql_tool import get_excel_rag_response_plain
from online.value_linker import LinkedValue, build_value_bindings_text
from prompt import CONSTRAINED_SQL_QUERY_TEMPLATE


def _infer_answer_type(sub_query: str) -> str:
    """Heuristic: infer what type of value the query expects in SELECT."""
    q = sub_query.lower().strip()
    if q.startswith(("who ", "which person", "who is", "who was", "whose")):
        return "person name"
    if q.startswith(("what country", "which country", "what nation")):
        return "country name"
    if q.startswith(
        ("what city", "which city", "what town", "where is", "where was", "where are")
    ):
        return "city or location name"
    if q.startswith(("when ", "what year", "what date", "in what year", "what month")):
        return "date or year"
    if any(
        q.startswith(p) for p in ("how many", "how much", "what is the number", "count")
    ):
        return "integer count or quantity"
    if "age" in q or "how old" in q or "born" in q:
        return "age or year (integer)"
    if "difference" in q or "how long" in q or "how far" in q:
        return "numeric difference"
    if q.startswith(("what is the name", "what was the name", "what is the title")):
        return "name or title string"
    return "(infer from query context)"


logger = logging.getLogger(__name__)


def _execution_confidence(sql_result: Any, *, error: Any = None) -> float:
    """
    c^exec = 1[parse] · 1[non-empty] · stability(o_t)
    Use response-level errors and empty containers, not words inside row data.
    Legacy text responses remain supported; the reference service's explicit
    failure prefix is recognized only outside JSON-encoded data.
    """
    if error:
        return 0.0
    if sql_result is None:
        return 0.0
    if isinstance(sql_result, str):
        text = sql_result.strip()
        if not text:
            return 0.0
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            if text.startswith(("[", "{")):
                return 0.0
            if text.lower().startswith("sql execution failed:"):
                return 0.0
            return 1.0
    else:
        data = sql_result
    if data is None or (isinstance(data, (list, tuple, dict)) and not data):
        return 0.3
    return 1.0


def _normalise_identifier(value: Any) -> str:
    # SQL identifiers are already physical names. Do not turn an unapproved
    # physical table/column into an approved one by sanitizing the SQL AST.
    return str(value or "").strip().strip("`\"'").lower()


def sql_service_identifier(value: Any) -> str:
    """Map logical names using the reference SQL service's transfer_name.

    Apply only to trusted schema/C/V* names, never to returned SQL identifiers.
    The service uses this same rule for table and column names.
    """
    name = str(value or "").split(".")[0]
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    if len(name) > 2:
        name = name.strip("_")
    name = name.lower()
    if not name:
        raise ValueError("Empty SQL service identifier")
    if name[0].isdigit():
        name = "t_" + name
    if len(name) > 64:
        name = (
            name[:20].rstrip("_")
            + "_"
            + hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        )
    return name


def sql_table_identifier(value: Any) -> str:
    """Full-stem table names: interior dots are data, not file extensions.

    SQL-service ingestion must use the same rule. Column naming deliberately
    keeps the reference service's separate ordered duplicate-column contract.
    """
    stem = re.sub(r"\.(xlsx|xls|csv)$", "", str(value or ""), flags=re.IGNORECASE)
    return sql_service_identifier(stem.replace(".", "_"))


def _checked_name_map(names: List[str], *, table_names: bool = False) -> Dict[str, str]:
    result: Dict[str, str] = {}
    owners: Dict[str, str] = {}
    for raw in names:
        physical = (
            sql_table_identifier(raw) if table_names else sql_service_identifier(raw)
        )
        if physical in owners and owners[physical] != raw:
            # The service suffixes colliding columns by their full schema order.
            # A retrieved subset cannot safely reconstruct that mapping.
            raise ValueError(f"Ambiguous SQL service identifier: {physical}")
        owners[physical] = raw
        result[raw] = physical
    return result


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


def _full_column_name_map(schema: Dict[str, Any]) -> Dict[str, str]:
    """Mirror reference transfer_df_columns using the full ordered schema."""
    result: Dict[str, str] = {}
    seen: Dict[str, int] = {}
    for raw in _schema_columns(schema):
        base = sql_service_identifier(raw)
        count = seen.get(base, 0)
        physical = base if count == 0 else f"{base}_{count}"
        seen[base] = count + 1
        if raw in result or physical in result.values():
            raise ValueError("Full schema has ambiguous duplicate column identifiers")
        result[raw] = physical
    return result


def _clean_sql(sql: str) -> str:
    text = str(sql or "").strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text).strip()


def validate_sql_constraints(
    sql: str,
    allowed_columns: List[str],
    allowed_tables: List[str],
    linked_values: List[LinkedValue],
    table_columns: Optional[Dict[str, List[str]]] = None,
    full_table_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    """Return violations of ``C``/``V*``; empty means admissible SQL."""
    sql = _clean_sql(sql)
    if not sql:
        return ["SQL service did not return sql_str"]
    try:
        import sqlglot
        from sqlglot import exp
        from sqlglot.optimizer.qualify import qualify
        from sqlglot.optimizer.scope import Scope, traverse_scope
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
    try:
        # Validate every supplied column identifier, including unused columns.
        _ = {c: sql_service_identifier(c) for c in allowed_columns}
        table_map = _checked_name_map(
            [t for t in allowed_tables if t], table_names=True
        )
        if table_columns is None:
            if len(table_map) != 1:
                raise ValueError("Multi-table C requires table-specific columns")
            table_columns = {next(iter(table_map)): allowed_columns}
        if set(table_columns) != set(table_map):
            raise ValueError("Table-specific C must match the allowed table scope")
        scoped_maps = {}
        for table, columns in table_columns.items():
            if full_table_schemas is not None and columns:
                if table not in full_table_schemas:
                    raise ValueError(f"Missing full column schema: {table}")
                mapping = _full_column_name_map(full_table_schemas[table])
                if not set(columns).issubset(mapping):
                    raise ValueError(f"C contains columns outside full schema: {table}")
                scoped_maps[table] = {c: mapping[c] for c in columns}
            else:
                if any("." in c for c in columns):
                    raise ValueError("Dotted column requires full ordered schema")
                scoped_maps[table] = _checked_name_map(columns)
        physical_schema = {
            table_map[t]: set(mapping.values()) for t, mapping in scoped_maps.items()
        }
        if set().union(*(set(cols) for cols in table_columns.values())) != set(
            allowed_columns
        ):
            raise ValueError("Table-specific C does not match allowed columns")
    except ValueError as exc:
        return [str(exc)]
    allowed_column_set = set().union(*physical_schema.values())
    allowed_table_set = set(table_map.values())
    # Resolve actual sources per scope. A CTE alias in one scope must never
    # hide a physical table with the same name in another scope.
    try:
        physical_tables = [
            source
            for scope in traverse_scope(tree)
            for _, source in scope.selected_sources.values()
            if isinstance(source, exp.Table)
        ]
    except Exception as exc:
        return [f"SQL source resolution failed: {exc}"]
    used_tables = {_normalise_identifier(table.name) for table in physical_tables}
    if any(table.db or table.catalog for table in physical_tables):
        violations.append("database-qualified tables cannot be verified against C")
    disallowed_tables = sorted(used_tables - allowed_table_set)
    if disallowed_tables:
        violations.append(f"disallowed tables: {', '.join(disallowed_tables)}")
    if not (used_tables & allowed_table_set):
        violations.append("query does not reference an allowed table")
    for table in used_tables & allowed_table_set:
        if not physical_schema[table]:
            violations.append(f"table has no authorized columns in C: {table}")
    if violations:
        return violations

    used_columns = {
        _normalise_identifier(column.name)
        for column in tree.find_all(exp.Column)
        if column.name != "*"
    }
    stars = list(tree.find_all(exp.Star))
    if any(star.find_ancestor(exp.Count) is None for star in stars):
        violations.append("SELECT * is not allowed")
    if not (used_columns & allowed_column_set) and not stars:
        violations.append("query does not reference an allowed column")

    # Qualify a validation-only copy with the closed retrieved schema. Do not
    # expand aliases: SELECT Club AS Secret, Secret must not rewrite a real
    # forbidden column into Club. ORDER BY output aliases remain supported.
    try:
        checked = tree.copy()
        for identifier in checked.find_all(exp.Identifier):
            identifier.set("this", identifier.this.lower())
        schema = {
            table.name.lower(): {
                c: "UNKNOWN" for c in physical_schema[table.name.lower()]
            }
            for table in physical_tables
        }
        tree = qualify(
            checked,
            dialect="mysql",
            schema=schema,
            infer_schema=False,
            expand_alias_refs=False,
            expand_stars=False,
        )
        scopes = list(traverse_scope(tree))
        outer_scope = scopes[-1] if scopes else None
    except Exception as exc:
        violations.append(f"column/source constraint resolution failed: {exc}")
        return violations

    def _base_column(node: Any, scope: Any) -> Optional[Tuple[str, str]]:
        """Prove that a binding refers to a real column, not a computed alias.

        Derived sources are supported only for single-source row-preserving
        projections. Aggregates, unions, windows and other transformations
        fail closed; repair can express the binding on the base table instead.
        """
        if not isinstance(node, exp.Column) or scope is None:
            return None
        source = scope.sources.get(node.table)
        if isinstance(source, exp.Table):
            return (
                _normalise_identifier(source.name),
                _normalise_identifier(node.name),
            )
        if not isinstance(source, Scope) or not isinstance(
            source.expression, exp.Select
        ):
            return None
        select = source.expression
        if (
            len(source.selected_sources) != 1
            or any(
                select.args.get(key)
                for key in (
                    "joins",
                    "group",
                    "having",
                    "qualify",
                    "distinct",
                    "limit",
                    "offset",
                )
            )
            or select.find(exp.AggFunc, exp.Window)
        ):
            return None
        outputs = [s for s in select.expressions if s.alias_or_name == node.name]
        if len(outputs) != 1:
            return None
        projection = outputs[0]
        if isinstance(projection, exp.Alias):
            projection = projection.this
        return _base_column(projection, source)

    # A grounded value is a hard row constraint, not merely a token that may
    # appear somewhere in the AST. Only a direct predicate in the outer WHERE
    # can satisfy V*: exact bindings require `column = literal`, fuzzy bindings
    # require `column LIKE literal`, and the predicate must stay on an AND-only
    # path to WHERE so OR/NOT/CASE cannot make it optional.
    outer_where = tree.args.get("where") if isinstance(tree, exp.Select) else None

    def _literal_value(node: Any) -> Optional[str]:
        if isinstance(node, exp.Literal):
            return str(node.this)
        if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal):
            return f"-{node.this.this}"
        return None

    def _mandatory_in_outer_where(predicate: Any) -> bool:
        if outer_where is None:
            return False
        node = predicate.parent
        while node is not None and node is not outer_where:
            if not isinstance(node, (exp.And, exp.Paren)):
                return False
            node = node.parent
        return node is outer_where

    def _matches_exact(predicate: Any, column: Tuple[str, str], value: str) -> bool:
        if not isinstance(predicate, exp.EQ) or not _mandatory_in_outer_where(
            predicate
        ):
            return False
        left_column = _base_column(predicate.this, outer_scope)
        right_column = _base_column(predicate.expression, outer_scope)
        left_value = _literal_value(predicate.this)
        right_value = _literal_value(predicate.expression)
        return (left_column == column and right_value == value) or (
            right_column == column and left_value == value
        )

    def _matches_fuzzy(predicate: Any, column: Tuple[str, str], pattern: str) -> bool:
        return (
            isinstance(predicate, exp.Like)
            and _mandatory_in_outer_where(predicate)
            and isinstance(predicate.this, exp.Column)
            and _base_column(predicate.this, outer_scope) == column
            and _literal_value(predicate.expression) == pattern
        )

    predicates = list(outer_where.find_all(exp.Predicate)) if outer_where else []
    for linked in linked_values:
        if not linked.is_matched or linked.matched_value is None:
            continue
        candidates = [
            (table_map[t], mapping[linked.column])
            for t, mapping in scoped_maps.items()
            if linked.column in mapping
            and (not linked.table_id or t == linked.table_id)
        ]
        if not candidates:
            violations.append(f"linked column is outside C: {linked.column}")
            continue
        if len(candidates) != 1:
            violations.append(f"ambiguous linked column table: {linked.column}")
            continue
        wanted_column = candidates[0]
        wanted_value = str(linked.matched_value)
        expected_fuzzy_pattern = f"%{wanted_value}%"
        if linked.fallback_level == 2:
            binding_found = any(
                _matches_fuzzy(predicate, wanted_column, expected_fuzzy_pattern)
                for predicate in predicates
            )
        else:
            binding_found = any(
                _matches_exact(predicate, wanted_column, wanted_value)
                for predicate in predicates
            )
        if not binding_found:
            if linked.fallback_level == 2:
                violations.append(
                    "missing fuzzy LIKE binding: "
                    f"{linked.column} LIKE {expected_fuzzy_pattern!r} "
                    "as a mandatory outer WHERE predicate"
                )
            else:
                violations.append(
                    f"missing exact WHERE binding: {linked.column}={wanted_value!r} "
                    "as a mandatory equality predicate"
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
        table_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.table_name_list = table_name_list
        self.max_retries = max_retries
        self.table_schemas = table_schemas or {}

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
            logger.warning(
                "ConstrainedSQL: refusing execution without column constraint C"
            )
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
            logger.warning(
                "ConstrainedSQL: refusing execution without a table constraint"
            )
            return "not found", 0.0

        try:
            column_map = {c: sql_service_identifier(c) for c in col_names}
            table_map = _checked_name_map(allowed_tables, table_names=True)
            table_columns = {table: [] for table in allowed_tables}
            entries = [c for c in allowed_columns if c.get("col_name")]
            if not entries:
                entries = [{"col_name": c} for c in col_names]
            for entry in entries:
                table = (
                    entry.get("table_id")
                    or entry.get("table_name")
                    or entry.get("source")
                )
                if not table:
                    if len(allowed_tables) != 1:
                        raise ValueError(
                            "Column without table provenance in multi-table C"
                        )
                    table = allowed_tables[0]
                table_columns[str(table)].append(str(entry["col_name"]))
            full_schemas = {}
            scoped_maps = {}
            for table, columns in table_columns.items():
                if not columns:
                    scoped_maps[table] = {}
                    continue
                full = self.table_schemas.get(table)
                if full is None and schema and str(schema.get("table_name")) == table:
                    full = schema
                if full is None:
                    raise ValueError(f"Missing full column schema: {table}")
                full_schemas[table] = full
                mapping = _full_column_name_map(full)
                if not set(columns).issubset(mapping):
                    raise ValueError(f"C contains columns outside full schema: {table}")
                scoped_maps[table] = {c: mapping[c] for c in columns}
            physical_schema = {
                table_map[t]: set(m.values()) for t, m in scoped_maps.items()
            }
        except ValueError as exc:
            logger.warning("ConstrainedSQL: %s", exc)
            return "not found", 0.0
        # Generation and validation share the same retrieved table scope.
        # Constructor hints are only a fallback when retrieval supplies none;
        # sending them unconditionally made the service query a different DB
        # schema from the one against which its SQL was then validated.
        service_tables = list(table_map.values())
        physical_values = []
        for linked in linked_values:
            if linked.is_matched and linked.column not in column_map:
                logger.warning(
                    "ConstrainedSQL: linked column is outside C: %s", linked.column
                )
                return "not found", 0.0
            physical_column = linked.column
            if linked.is_matched:
                candidates = [
                    (table_map[t], m[linked.column])
                    for t, m in scoped_maps.items()
                    if linked.column in m
                    and (not linked.table_id or t == linked.table_id)
                ]
                if len(candidates) != 1:
                    logger.warning(
                        "ConstrainedSQL: ambiguous linked column table: %s",
                        linked.column,
                    )
                    return "not found", 0.0
                owner, physical_column = candidates[0]
                if len(service_tables) > 1:
                    physical_column = f"{owner}.{physical_column}"
            physical_values.append(
                LinkedValue(
                    linked.entity,
                    physical_column,
                    linked.matched_value,
                    linked.confidence,
                    linked.fallback_level,
                )
            )
        value_bindings_text = build_value_bindings_text(physical_values)

        # Principle 3: inject table name from restored schema (View 2)
        table_context = ""
        if schema and schema.get("table_name"):
            table_context = f"\nTarget table: {table_map[str(schema['table_name'])]}"

        answer_type_hint = _infer_answer_type(sub_query)

        def _build_query(repair: str = "") -> str:
            evidence_snippet = (
                retrieval_evidence[:800] if retrieval_evidence else "(none)"
            )
            base = CONSTRAINED_SQL_QUERY_TEMPLATE.format(
                original_query=sub_query,
                allowed_columns=", ".join(
                    f"{table}.{column}"
                    for table, columns in physical_schema.items()
                    for column in sorted(columns)
                ),
                value_bindings=value_bindings_text,
                answer_type_hint=answer_type_hint,
                text_evidence=evidence_snippet,
            )
            return (
                base
                + "\nAllowed tables: "
                + ", ".join(service_tables)
                + table_context
                + repair
            )

        enriched_query = _build_query()
        sql_result = ""

        for attempt in range(1, self.max_retries + 1):
            response = get_excel_rag_response_plain(
                table_name_list=list(service_tables),
                query=enriched_query,
                constraints={
                    "allowed_columns": col_names,
                    "allowed_tables": allowed_tables,
                    "table_columns": table_columns,
                    "full_table_schemas": full_schemas,
                    "linked_values": [
                        dict(
                            entity=v.entity,
                            column=v.column,
                            matched_value=v.matched_value,
                            confidence=v.confidence,
                            fallback_level=v.fallback_level,
                            table_id=v.table_id,
                        )
                        for v in linked_values
                    ],
                },
            )
            raw_sql_result = response.get("sql_execution_result", "")
            sql_result = str(raw_sql_result)
            sql_str = str(response.get("sql_str", ""))
            if response.get("sql_service_failure") or (
                response.get("error") and not sql_str.strip()
            ):
                logger.warning(
                    "ConstrainedSQL: service failure; skipping SQL repair: %s",
                    response.get("error", "unknown service failure"),
                )
                return "not found", 0.0
            violations = validate_sql_constraints(
                sql_str,
                allowed_columns=col_names,
                allowed_tables=allowed_tables,
                linked_values=linked_values,
                table_columns=table_columns,
                full_table_schemas=full_schemas,
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
            c_exec = _execution_confidence(raw_sql_result, error=response.get("error"))

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
