"""HybridValueLinker: embedding candidates from cell index, LLM-verified to a grounded value, with a 3-level fallback (unconstrained / LIKE-fuzzy / reroute to TEXT)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from atr.prompt import VALUE_LINKER_PROMPT

logger = logging.getLogger(__name__)

NO_MATCH = "no_match"

# Routes that require value linking (used to interpret history_H)
_VL_ROUTES = {"HYBRID", "RETRIEVE"}

def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
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
        if self.fallback_level == 2:
            return f"{self.column} LIKE '%{self.matched_value}%'"
        return f"{self.column} = '{self.matched_value}'"

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
    ) -> List[LinkedValue]:
        """
        For each entity mention, run Stage 1 (already done → V_raw) then Stage 2 (LLM).

        Args:
            entity_mentions:  list of entity strings from the sub-query
            schema_columns:   retrieved schema column entries (View 3)
            V_raw:            {entity: [cell candidates]} from View 4 Stage 1
            history_H:        failure history: used to escalate min_fallback_level
                              so already-failed linking strategies are skipped

        Returns:
            List of LinkedValue objects.
        """
        # §3.5 "skip already-failed attempts":
        # count how many routes requiring value linking have already failed
        prior_vl_failures = sum(
            1 for r in history_H
            if str(r.get("route", "")) in _VL_ROUTES
        )
        # Gradual escalation, keyed on how many times linking has already failed.
        # 0 failures → start at level 0 (strict match first)
        # 1 failure  → start at level 1 (skip strict, try unconstrained SQL)
        # 2 failures → start at level 2 (try fuzzy LIKE)
        # 3+ failures → start at level 3 (reroute to TEXT)
        # The previous `* 2` doubling caused the first VL failure to skip past
        # fuzzy-LIKE entirely, missing legitimate fuzzy matches that would
        # have rescued the route on retry.
        min_fallback = min(prior_vl_failures, 3)

        if min_fallback > 0:
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
            linked = self._link_one(entity, candidates, schema_columns, min_fallback)
            results.append(linked)
            # Do NOT break: continue linking remaining entities.
        return results

    def _link_one(
        self,
        entity: str,
        candidates: List[Dict[str, Any]],
        schema_columns: List[Dict[str, Any]],
        min_fallback: int = 0,
    ) -> LinkedValue:
        """Stage 2: LLM verification for a single entity (multi-column aware)."""
        if not candidates:
            fallback = max(min_fallback, 1)
            logger.debug(f"No candidates for entity '{entity}' → fallback level {fallback}")
            return LinkedValue(entity, "", None, 0.0, fallback_level=fallback)

        # Build schema lookup for quick type/examples access
        schema_by_col: Dict[str, Dict] = {c["col_name"]: c for c in schema_columns}

        # Show candidates with their column context (multi-column)
        seen_pairs: set = set()
        candidates_lines = []
        for i, c in enumerate(candidates):
            col = c.get("col_name", "")
            val = c.get("value", "")
            pair = (col, val)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            candidates_lines.append(f"  {len(candidates_lines)+1}. [col: {col}] {val}")
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
            fallback = max(min_fallback, 1)
            return LinkedValue(entity, primary_col, None, 0.0, fallback_level=fallback)

        parsed = _parse_json(response_text)
        matched_value: str = parsed.get("matched_value", NO_MATCH)
        confidence: float = float(parsed.get("confidence", 0.0))

        if matched_value == NO_MATCH or not matched_value:
            candidate_values = [c["value"] for c in candidates]
            return self._apply_fallback(entity, primary_col, candidate_values, min_fallback)

        # Resolve matched column: find which candidate this value came from
        matched_col = primary_col
        for c in candidates:
            if c.get("value", "") == matched_value:
                matched_col = c.get("col_name", primary_col)
                break

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
        candidate_values: List[str],
        min_fallback: int = 0,
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
        if min_fallback <= 2:
            def _norm_fuzzy(s: str) -> str:
                return re.sub(r"[-_\s]+", " ", s.lower()).strip()
            entity_norm = _norm_fuzzy(entity)
            for val in candidate_values:
                val_norm = _norm_fuzzy(str(val))
                if entity_norm in val_norm or val_norm in entity_norm:
                    logger.debug(f"ValueLinker: fuzzy match '{entity}' → '{val}'")
                    return LinkedValue(entity, col_name, val, 0.3, fallback_level=2)

        # Level 3: signal reroute to TEXT
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
