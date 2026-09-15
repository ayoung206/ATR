"""Pre-execution guard adapter for the external TableRAG Flask service.

The service must accept sql_validator= and call it on the exact SQL immediately
before fetchall. A rejected callback must prevent DB execution. See
README.md (SQL service); this adapter does not alter NL2SQL prompts.
"""

from online.constrained_sql import sql_table_identifier, validate_sql_constraints
from online.value_linker import LinkedValue


class SQLConstraintViolation(ValueError):
    pass


def call_sql_llm(system_prompt, user_prompt):
    """Send the external service's unchanged prompts through ATR's LLM client."""
    import os

    from clients.chat_utils import get_chat_result
    from config import config_mapping

    backbone = os.getenv("ATR_SQL_BACKBONE", "gemini")
    if backbone not in config_mapping:
        raise ValueError(f"Unknown ATR_SQL_BACKBONE: {backbone}")
    response = get_chat_result(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        llm_config=config_mapping[backbone],
    )
    return response.content


def register_guarded_route(app, process_request, base_path="/get_tablerag_response"):
    from flask import jsonify, request

    def guarded_request():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="Expected JSON object"), 400
        try:
            constraints = body["constraints"]
            query, tables = body["query"], body["table_name_list"]
            if not isinstance(query, str) or not isinstance(tables, list):
                raise ValueError("Invalid query or tables")
            required = {
                "allowed_columns",
                "allowed_tables",
                "table_columns",
                "full_table_schemas",
                "linked_values",
            }
            if not isinstance(constraints, dict) or set(constraints) != required:
                raise ValueError("Invalid constraint contract")
            if not constraints["allowed_columns"] or not constraints["allowed_tables"]:
                raise ValueError("Empty constraint scope")
            if set(tables) != {
                sql_table_identifier(t) for t in constraints["allowed_tables"]
            }:
                raise ValueError("Requested tables differ from constraint scope")
            kwargs = {k: v for k, v in constraints.items() if k != "linked_values"}
            kwargs["linked_values"] = [
                LinkedValue(**v) for v in constraints["linked_values"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

        checked_sql = []

        def validate_before_execution(sql):
            violations = validate_sql_constraints(sql, **kwargs)
            if violations:
                raise SQLConstraintViolation("; ".join(violations))
            checked_sql.append(sql)

        # No fallback to the old generate-and-execute API. An unpatched
        # service rejects this keyword before it can generate or execute SQL.
        response = process_request(
            tables, query, sql_validator=validate_before_execution
        )
        if not isinstance(response, dict):
            return jsonify(error="Invalid guarded service response"), 500
        if not response.get("error") and response.get("sql_str") not in checked_sql:
            return jsonify(error="Service did not validate the returned SQL"), 500
        response = dict(response, atr_guard_version=1)
        return jsonify(response)

    app.add_url_rule(
        base_path.rstrip("/") + "/atr-guarded-v1",
        "atr_guarded_v1",
        guarded_request,
        methods=["POST"],
    )
