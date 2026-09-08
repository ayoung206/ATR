"""HybridValueLinker: embedding candidates from cell index, LLM-verified to a grounded value, with a 3-level fallback (unconstrained / LIKE-fuzzy / reroute to TEXT)."""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional

from atr.prompt import VALUE_LINKER_PROMPT

logger = logging.getLogger(__name__)

NO_MATCH = "no_match"

# Routes that require value linking (used to interpret history_H)
_VL_ROUTES = {"HYBRID", "RETRIEVE"}


def _route_name(route: Any) -> str:
    value = getattr(route, "value", route)
    return str(value or "").upper()


def _text_route_failed(history_H: List[Dict[str, Any]]) -> bool:
    """Paper H persists across the question: do not repeat rejected TEXT.

    Use the effective primitive, not the router's original request. Legacy
    records contain only route (or the route itself).
    """
    return any(
        _route_name(
            item.get("effective_route") or item.get("route")
            if isinstance(item, dict) else item
        ) == "TEXT"
        for item in history_H or []
    )


def _prior_value_linker_failures(
    history_H: List[Dict[str, Any]],
    current_sub_query: str,
) -> int:
    """Count rejected attempts that actually invoked linking for this q_t."""
    count = 0
    for item in history_H or []:
        if not isinstance(item, dict):
            if _route_name(item) in _VL_ROUTES:
                count += 1
            continue

        failed_query = str(item.get("sub_query", "") or "")
        if current_sub_query and failed_query and failed_query != current_sub_query:
            continue

        attempted = item.get("value_linker_attempted")
        if attempted is not None:
            count += int(bool(attempted))
            continue

        # Backward compatibility for traces written before the explicit flag.
        route = item.get("requested_route") or item.get("route")
        if _route_name(route) in _VL_ROUTES:
            count += 1
    return count

def _parse_json(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {}
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}

class LinkedValue:
    """Result of value linking for a single (entity, column) pair."""

    def __init__(
        self,
        entity: str,
        column: str,
        matched_value: Optional[str],
        confidence: float,
        fallback_level: int = 0,  # 0=exact, 1=unconstrained, 2=fuzzy, 3=reroute
    ) -> None:
        self.entity = entity
        self.column = column
        self.matched_value = matched_value
        self.confidence = confidence
        self.fallback_level = fallback_level

    @property
    def is_matched(self) -> bool:
        return self.matched_value is not None and self.matched_value != NO_MATCH

    @property
    def needs_reroute(self) -> bool:
        return self.fallback_level >= 3

    def to_where_clause(self) -> str:
        """Build a SQL WHERE snippet for this linked value."""
        if not self.is_matched:
            return ""
        escaped_value = str(self.matched_value).replace("'", "''")
        if self.fallback_level == 2:
            return f"{self.column} LIKE '%{escaped_value}%'"
        return f"{self.column} = '{escaped_value}'"

class HybridValueLinker:
    """
    §3.5  Constrained Value Linking.

    V* = {(e_i, c_i, v*_i, conf_i)}

    Usage:
        linker = HybridValueLinker(llm_fn)
        V_star = linker.link(entity_mentions, schema_columns, V_raw, history_H)
    """

    def __init__(
        self,
        llm_fn: Callable[[List[Dict]], str],
        cell_top_k: int = 15,
    ) -> None:
        self.llm_fn = llm_fn
        self.cell_top_k = cell_top_k

    def link(
        self,
        entity_mentions: List[str],
        schema_columns: List[Dict[str, Any]],
        V_raw: Dict[str, List[Dict[str, Any]]],
        history_H: List[Dict[str, Any]],
        current_sub_query: str = "",
    ) -> List[LinkedValue]:
        """
        For each entity mention, run Stage 1 (already done → V_raw) then Stage 2 (LLM).

        Args:
            entity_mentions:  list of entity strings from the sub-query
            schema_columns:   retrieved schema column entries (View 3)
            V_raw:            {entity: [cell candidates]} from View 4 Stage 1
            history_H:        rejected execution attempts
            current_sub_query: q_t used to scope history to the active sub-query

        Returns:
            List of LinkedValue objects.
        """
        prior_vl_failures = _prior_value_linker_failures(
            history_H, current_sub_query
        )
        # A fresh no-match uses level 1. Each rejected attempt that actually
        # ran the linker advances exactly one step: unconstrained → fuzzy →
        # reroute. Unrelated sub-queries and SQL/TEXT-only failures do not
        # consume this fallback ladder.
        min_fallback = min(prior_vl_failures + 1, 3)
        allow_text_fallback = not _text_route_failed(history_H)

        if prior_vl_failures > 0:
            logger.debug(
                f"ValueLinker: {prior_vl_failures} prior VL failure(s) → "
                f"min_fallback={min_fallback}"
            )

        # Process every entity mention, not just up to the
        # first reroute signal. Otherwise multi-entity questions lose the SQL
        # constraints for entities after the first failing one. The reroute
        # signal is still propagated (last result may have needs_reroute=True),
        # but downstream HYBRID/SQL gets bindings for all linkable entities.
        results: List[LinkedValue] = []
        for entity in entity_mentions:
            candidates = V_raw.get(entity, [])
            linked = self._link_one(
                entity, candidates, schema_columns, min_fallback, allow_text_fallback
            )
            results.append(linked)
            # Do NOT break: continue linking remaining entities.
        return results

    def _link_one(
        self,
        entity: str,
        candidates: List[Dict[str, Any]],
        schema_columns: List[Dict[str, Any]],
        min_fallback: int = 0,
        allow_text_fallback: bool = True,
    ) -> LinkedValue:
        """Stage 2: LLM verification for a single entity (multi-column aware)."""
        # Only indexed pairs whose column belongs to retrieved C may be used
        # by either the LLM or fallback. Keep canonical values and columns
        # together throughout linking.
        schema_by_col = {
            c["col_name"]: c for c in schema_columns
            if isinstance(c, dict) and isinstance(c.get("col_name"), str)
            and c["col_name"].strip()
        }
        valid_candidates = []
        seen_pairs = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            col, val = candidate.get("col_name"), candidate.get("value")
            if not isinstance(col, str) or col not in schema_by_col:
                continue
            if not isinstance(val, (str, int, float)) or isinstance(val, bool):
                continue
            if isinstance(val, float) and not math.isfinite(val):
                continue
            val = str(val)
            if not val.strip() or val == NO_MATCH or (col, val) in seen_pairs:
                continue
            seen_pairs.add((col, val))
            valid_candidates.append({**candidate, "value": val})
        candidates = valid_candidates
        if not candidates:
            logger.debug(f"No candidates for entity '{entity}'")
            return self._apply_fallback(entity, "", [], min_fallback, allow_text_fallback)

        # Show candidates with their column context (multi-column)
        candidates_lines = [
            f"  {i}. [col: {c['col_name']}] {c['value']}"
            for i, c in enumerate(candidates, 1)
        ]
        candidates_text = "\n".join(candidates_lines)

        # Use the most common column among top candidates for schema context
        col_counts: Dict[str, int] = {}
        for c in candidates:
            col_counts[c.get("col_name", "")] = col_counts.get(c.get("col_name", ""), 0) + 1
        primary_col = max(col_counts, key=col_counts.__getitem__) if col_counts else ""
        schema_col = schema_by_col.get(primary_col, {})

        prompt = VALUE_LINKER_PROMPT.format(
            entity=entity,
            column_name="(see candidate list below)",
            column_type=schema_col.get("dtype", "text"),
            column_examples=schema_col.get("examples", ""),
            candidates=candidates_text,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response_text = self.llm_fn(messages)
        except Exception as exc:
            logger.error(f"ValueLinker LLM call failed: {exc}")
            return self._apply_fallback(
                entity, primary_col, candidates, min_fallback, allow_text_fallback
            )

        parsed = _parse_json(response_text)
        matched_value: str = parsed.get("matched_value", NO_MATCH)
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError, OverflowError):
            confidence = -1.0
        matched_column = parsed.get("matched_column")
        matches = [
            c for c in candidates if c["value"] == matched_value
            and (matched_column is None or c["col_name"] == matched_column)
        ]
        # Legacy value-only replies remain valid only when the column is
        # unambiguous. Never invent a column or trust an out-of-list value.
        if len(matches) != 1 or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return self._apply_fallback(
                entity, primary_col, candidates, min_fallback, allow_text_fallback
            )
        matched_col = matches[0]["col_name"]
        matched_value = matches[0]["value"]

        # INFO-level (lifted from debug) so that ablation studies can quantify
        # how often the linker rewrites the surface entity into a canonical
        # cell value. Compare entity == matched_value (no rewrite) vs entity
        # != matched_value (lexical-mismatch recovery).
        logger.info(
            f"ValueLinker: '{entity}' → '{matched_value}' "
            f"(conf={confidence:.2f}, col={matched_col})"
        )
        return LinkedValue(entity, matched_col, matched_value, confidence, fallback_level=0)

    def _apply_fallback(
        self,
        entity: str,
        col_name: str,
        candidates: List[Dict[str, Any]],
        min_fallback: int = 0,
        allow_text_fallback: bool = True,
    ) -> LinkedValue:
        """
        §3.5 'No match', a 3-level fallback respecting min_fallback:
          Level 1: remove value constraint (unconstrained SQL)
          Level 2: fuzzy LIKE matching
          Level 3: reroute to TEXT
        """
        # Level 1, unconstrained SQL: drop value constraint for this entity
        if min_fallback <= 1:
            logger.debug(
                f"ValueLinker: '{entity}' → unconstrained SQL (fallback level 1)"
            )
            return LinkedValue(entity, col_name, None, 0.0, fallback_level=1)

        # Level 2, fuzzy: substring match with normalization.
        # Normalise hyphens and whitespace so e.g.
        # "Saint-Etienne" matches "Saint Etienne" (and vice versa).
        if min_fallback <= 2 or not allow_text_fallback:
            def _norm_fuzzy(s: str) -> str:
                return re.sub(r"[-_\s]+", " ", s.lower()).strip()
            entity_norm = _norm_fuzzy(entity)
            for candidate in candidates:
                val = candidate["value"]
                val_norm = _norm_fuzzy(str(val))
                if entity_norm and val_norm and (entity_norm in val_norm or val_norm in entity_norm):
                    if len({c["col_name"] for c in candidates if c["value"] == val}) != 1:
                        break
                    logger.debug(f"ValueLinker: fuzzy match '{entity}' → '{val}'")
                    return LinkedValue(entity, candidate["col_name"], val, 0.3, fallback_level=2)

        if not allow_text_fallback:
            logger.debug(
                "ValueLinker: TEXT already rejected; no unambiguous fuzzy "
                "match for '%s', dropping this entity's binding", entity,
            )
            return LinkedValue(entity, col_name, None, 0.0, fallback_level=1)

        # Level 3: signal reroute to TEXT only if it has not failed in H.
        logger.info(
            f"ValueLinker: no match for '{entity}' (min_fallback={min_fallback}) "
            "and is signalling a reroute to TEXT"
        )
        return LinkedValue(entity, col_name, None, 0.0, fallback_level=3)

def build_value_bindings_text(linked_values: List[LinkedValue]) -> str:
    """Format V* for injection into SQL query prompts (§3.5)."""
    bindings = []
    for lv in linked_values:
        clause = lv.to_where_clause()
        if clause:
            bindings.append(clause)
    return ", ".join(bindings) if bindings else "(none)"
