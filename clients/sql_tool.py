"""Client for the Flask SQL service behind ATR's SQL and HYBRID primitives.

The service takes a natural-language (or SQL) query plus the tables it may
touch, generates SQL, executes it against MySQL, and returns the rows. ATR
reaches it over HTTP so the executor can live on a different host from the
agent loop.

Point ``SQL_SERVICE_URL`` at the endpoint. The reference implementation is the
one released with TableRAG (Yu et al., EMNLP 2025). Constrained ATR calls
require the pre-execution adapter described in README.md (SQL service);
the original unguarded API remains available only to plain callers.

Every call returns a dict. On failure it is a well-formed *failure envelope*
with the same keys as a success, so callers can read
``response["sql_execution_result"]`` unconditionally and let the confidence
heuristic in :mod:`online.constrained_sql` route around the error.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from clients.chat_utils import init_logger
from config import sql_service_url

logger = init_logger(name="sql_tool", level=logging.INFO, log_file=None)

_RESPONSE_KEYS = (
    "query",
    "table_name_list",
    "sql_str",
    "sql_execution_result",
    "nl2sql_prompt",
    "nl2sql_response",
)


def _endpoint_is_usable(url: str) -> bool:
    if not url:
        return False
    parts = urlparse(url)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def _clip(text: str, limit: int = 500) -> str:
    """One-line preview of a server response, for logs."""
    flat = (text or "").replace("\n", "\\n")
    return flat if len(flat) <= limit else flat[:limit] + "...(truncated)"


def _failure(
    reason: str,
    query: str = "",
    table_name_list: Optional[List[str]] = None,
    status_code: Optional[int] = None,
    body: str = "",
) -> Dict[str, Any]:
    """Failure envelope shaped like a successful response."""
    envelope: Dict[str, Any] = dict.fromkeys(_RESPONSE_KEYS, "")
    envelope["query"] = query
    envelope["table_name_list"] = list(table_name_list or [])
    envelope["sql_execution_result"] = f"SQL execution failed: {reason}"
    envelope["error"] = reason
    # This is an HTTP/service failure, not a SQL query to repair. A read
    # timeout may leave the server working; do not multiply its LLM requests.
    envelope["sql_service_failure"] = True
    if status_code is not None:
        envelope["sql_service_http_status"] = status_code
    if body:
        envelope["sql_service_response_preview"] = _clip(body)
    return envelope


def get_excel_rag_response_plain(
    table_name_list: Optional[List[str]] = None,
    query: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ask the SQL service to answer ``query`` over ``table_name_list``.

    Only connection-establishment timeouts are retried. After a request may
    have reached the service, timeouts, 5xx and malformed replies fail without
    resubmission: this API has no server-side idempotency/cancellation contract.
    """
    endpoint = sql_service_url
    if constraints is not None and endpoint:
        # A distinct route is essential: legacy endpoints ignore unknown JSON
        # fields and would execute before the client can validate anything.
        endpoint = endpoint.rstrip("/") + "/atr-guarded-v1"
    if not _endpoint_is_usable(endpoint):
        logger.error(
            "SQL_SERVICE_URL is unset or malformed; expected a full http(s) endpoint."
        )
        return _failure(
            "SQL service endpoint is unset or malformed", query or "", table_name_list
        )

    tables = list(table_name_list or [])
    question = query or ""
    body = {"table_name_list": tables, "query": question}
    if constraints is not None:
        body["constraints"] = constraints

    attempts = max(1, int(os.getenv("ATR_SQL_RETRIES", "3")))
    backoff = max(0.1, float(os.getenv("ATR_SQL_BACKOFF_SEC", "0.5")))
    # Reference service can make four 30-second Vertex attempts plus backoff.
    # These are socket timeouts, not a guaranteed end-to-end deadline.
    timeout = float(os.getenv("ATR_SQL_TIMEOUT_SEC", "180"))
    connect_timeout = float(os.getenv("ATR_SQL_CONNECT_TIMEOUT_SEC", "5"))
    verify_tls = os.getenv("ATR_SQL_VERIFY_TLS", "1") != "0"

    for attempt in range(1, attempts + 1):
        remaining = attempts - attempt
        pause = backoff * (2 ** (attempt - 1))

        try:
            reply = requests.post(
                endpoint,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=(connect_timeout, timeout),
                verify=verify_tls,
            )
        except (
            requests.exceptions.MissingSchema,
            requests.exceptions.InvalidSchema,
            requests.exceptions.InvalidURL,
        ) as bad_url:
            reason = f"Invalid SQL service URL '{endpoint}': {bad_url}"
            logger.error(reason)
            return _failure(reason, question, tables)
        except requests.exceptions.ConnectTimeout as unreachable:
            # requests identifies this as a failure to establish a connection,
            # before the service has accepted a request.
            reason = f"SQL service connection timed out: {unreachable}"
            logger.error("%s (attempt %d/%d)", reason, attempt, attempts)
            if remaining:
                time.sleep(pause)
                continue
            return _failure(reason, question, tables)
        except requests.exceptions.RequestException as unreachable:
            reason = f"SQL service unreachable: {unreachable}"
            logger.error("%s (attempt %d/%d)", reason, attempt, attempts)
            return _failure(reason, question, tables)

        status = reply.status_code

        if 400 <= status < 500:
            reason = f"SQL service rejected the request (HTTP {status})"
            logger.error("%s: %s", reason, _clip(reply.text))
            return _failure(reason, question, tables, status, reply.text)

        if status >= 500:
            reason = f"SQL service error (HTTP {status})"
            logger.error(
                "%s (attempt %d/%d): %s", reason, attempt, attempts, _clip(reply.text)
            )
            return _failure(reason, question, tables, status, reply.text)

        try:
            payload = reply.json()
        except ValueError:
            reason = "SQL service returned a non-JSON body"
            logger.error(
                "%s (attempt %d/%d): %s", reason, attempt, attempts, _clip(reply.text)
            )
            return _failure(reason, question, tables, status, reply.text)

        if isinstance(payload, dict):
            if constraints is not None and payload.get("atr_guard_version") != 1:
                return _failure(
                    "SQL service did not acknowledge pre-execution validation",
                    question,
                    tables,
                    status,
                )
            if not all(key in payload for key in ("sql_str", "sql_execution_result")):
                return _failure(
                    "SQL service response is missing required fields",
                    question,
                    tables,
                    status,
                )
            result = payload["sql_execution_result"]
            # Reference service serializes rows as a JSON string inside the
            # outer response. Some versions slice that string at 1,000 chars.
            # An intact outer envelope does not prove its row data is intact.
            if isinstance(result, str) and result.lstrip().startswith(("[", "{")):
                try:
                    json.loads(result)
                except ValueError:
                    return _failure(
                        "SQL service returned truncated or malformed JSON row data",
                        question,
                        tables,
                        status,
                    )
            return payload

        reason = f"SQL service returned {type(payload).__name__}, expected an object"
        logger.error(reason)
        return _failure(reason, question, tables, status, reply.text)

    return _failure("SQL service did not answer after retries", question, tables)
