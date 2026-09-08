"""Table backend router: TEXT, SQL, RETRIEVE, or HYBRID per sub-query.

Variants: HeuristicRouter, LLMRouter, LearnedRouter (DistilBERT 4-class).
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

class Route(str, Enum):
    TEXT = "TEXT"
    SQL = "SQL"
    RETRIEVE = "RETRIEVE"
    HYBRID = "HYBRID"

# Compatibility chain used by BaseRouter and explicit FixedRouter ablations.
# Normal online retries call ``reselect`` and therefore re-run the active
# policy on the paper-defined input with the updated failure history H.
_ESCALATION: Dict[Route, Route] = {
    Route.SQL: Route.HYBRID,        # SQL failed → try HYBRID (constrained variant)
    Route.HYBRID: Route.RETRIEVE,   # HYBRID failed → RETRIEVE (cell lookup)
    Route.RETRIEVE: Route.TEXT,     # RETRIEVE failed → TEXT (free-form read)
    Route.TEXT: Route.SQL,          # cycle: TEXT failed → back to SQL
}

_AGGREGATE_OPERATORS = {"count", "aggregate", "sort", "arithmetic", "compare"}
_LOOKUP_OPERATORS = {"lookup", "filter"}


def _first_available_route(
    failed_routes: Any,
    order: tuple[Route, ...] = (
        Route.RETRIEVE, Route.SQL, Route.HYBRID, Route.TEXT,
    ),
) -> Route:
    failed = {
        route.value if isinstance(route, Route) else str(route)
        for route in failed_routes
        if route
    }
    for route in order:
        if route.value not in failed:
            return route
    return Route.TEXT


def _failed_route_names(
    history_H: Optional[List[Dict[str, Any]]],
    sub_query: str,
) -> set[str]:
    """Return routes rejected for this sub-query, while retaining global H.

    Older traces did not record ``sub_query`` on failures; those entries stay
    applicable for backwards compatibility.
    """
    failed = set()
    for item in history_H or []:
        if not isinstance(item, dict):
            route, failed_query = item, ""
        else:
            route = item.get("route")
            failed_query = item.get("sub_query", "")
        if failed_query and failed_query != sub_query:
            continue
        if isinstance(route, Route):
            route = route.value
        if route:
            failed.add(str(route))
    return failed


def _schema_feature(schema: Optional[Dict[str, Any]]) -> str:
    """Serialize the restored View-2 schema for the student-router input."""
    if not schema:
        return "none"
    table_name = schema.get("table_name") or schema.get("table_id") or "unknown"
    columns = []
    for entry in schema.get("columns", []) or []:
        if isinstance(entry, dict):
            name = entry.get("col_name") or entry.get("name") or ""
            dtype = entry.get("dtype") or entry.get("type") or ""
            examples = entry.get("examples") or ""
        elif isinstance(entry, (list, tuple)) and entry:
            name = entry[0]
            dtype = entry[1] if len(entry) > 1 else ""
            examples = entry[2] if len(entry) > 2 else ""
        else:
            name, dtype, examples = str(entry), "", ""
        if name:
            description = f"{name}:{dtype}" if dtype else str(name)
            if examples:
                description += f" examples={examples}"
            columns.append(description)
    column_text = ", ".join(columns) if columns else "unknown"
    return f"table={table_name}; columns={column_text}"


def _history_feature(history_H: Optional[List[Dict[str, Any]]]) -> str:
    """Serialize failed routes and verdicts in attempt order."""
    if not history_H:
        return "none"
    failures = []
    for item in history_H:
        if isinstance(item, dict):
            route = item.get("route")
            requested_route = item.get("requested_route")
            verdict = item.get("verdict")
            failed_query = item.get("sub_query", "")
        else:
            route, requested_route, verdict, failed_query = item, None, None, ""
        if isinstance(route, Route):
            route = route.value
        if isinstance(requested_route, Route):
            requested_route = requested_route.value
        if not route:
            continue
        route_label = str(route)
        if requested_route and str(requested_route) != route_label:
            route_label = f"{requested_route}->{route_label}"
        detail = route_label if verdict is None else f"{route_label}:{verdict}"
        failures.append(f"{failed_query} => {detail}" if failed_query else detail)
    return ", ".join(failures) if failures else "none"


def build_router_input(
    sub_query: str,
    expected_operator: str,
    required_modalities: str,
    entity_mentions: Optional[List[str]],
    need_global_table_view: bool,
    uncertainty: float,
    schema: Optional[Dict[str, Any]],
    history_H: Optional[List[Dict[str, Any]]],
) -> str:
    """Build the paper-defined ``(q_t, meta_t, schema_t, H)`` input."""
    entities = ", ".join(entity_mentions or []) or "none"
    has_schema = "yes" if schema else "no"
    try:
        uncertainty_text = f"{float(uncertainty):.3f}"
    except (TypeError, ValueError):
        uncertainty_text = "0.000"
    return (
        f"[QUERY] {sub_query} "
        f"[OP] {expected_operator or 'lookup'} "
        f"[MOD] {required_modalities or 'both'} "
        f"[ENT] {entities} "
        f"[SCHEMA] {has_schema} "
        f"[GLOBAL] {'yes' if need_global_table_view else 'no'} "
        f"[UNCERTAINTY] {uncertainty_text} "
        f"[FAILED] {_history_feature(history_H)} "
        f"[SCHEMA_DETAIL] {_schema_feature(schema)}"
    )


class BaseRouter:
    """Common interface for all router variants."""

    def route(
        self,
        sub_query: str,
        meta: Any,
        schema: Optional[Dict[str, Any]],
        history_H: List[Dict[str, Any]],
    ) -> Route:
        raise NotImplementedError

    def escalate(
        self,
        current_route: Route,
        history_H: Optional[List[Dict[str, Any]]] = None,
    ) -> Route:
        """Compatibility fallback for callers that explicitly request a chain.

        Paper-aligned online inference uses :meth:`reselect`, which invokes the
        active policy again with updated ``H``.
        """
        next_route = _ESCALATION.get(current_route, Route.TEXT)
        logger.info(f"EscalatePolicy: {current_route} → {next_route}")
        return next_route

    def reselect(
        self,
        sub_query: str,
        meta: Any,
        schema: Optional[Dict[str, Any]],
        history_H: List[Dict[str, Any]],
        current_route: Route,
    ) -> Route:
        """Re-invoke the routing policy with updated failed-route history."""
        return self.route(sub_query, meta, schema, history_H)

class HeuristicRouter(BaseRouter):
    """
    §3.4  Heuristic router based on operator cues and schema signal.

    Input:
      sub_query: the current sub-query text
      meta: SubQuery metadata (expected_operator, required_modalities,
                    entity_mentions, need_global_table_view, uncertainty)
      schema: restored schema dict from View 2 (None if no table chunk found)
      history_H: list of FailureRecord dicts for the current question
    """

    def route(
        self,
        sub_query: str,
        meta: Any,
        schema: Optional[Dict[str, Any]],
        history_H: List[Dict[str, Any]],
    ) -> Route:
        op = (meta.expected_operator or "lookup").lower()
        modalities = (meta.required_modalities or "both").lower()
        entity_mentions = meta.entity_mentions or []
        need_global = meta.need_global_table_view

        has_schema = schema is not None
        has_entities = bool(entity_mentions)
        is_aggregate = op in _AGGREGATE_OPERATORS
        is_lookup = op in _LOOKUP_OPERATORS

        failed_routes = _failed_route_names(history_H, sub_query)

        # ── Rule 1, TEXT: descriptive / no table signal ──────────────────
        # Only when the question is unambiguously textual (modalities="text")
        # OR when no table evidence is available at all.
        if modalities == "text" or (not has_schema and not has_entities):
            if Route.TEXT not in failed_routes:
                logger.debug("Router → TEXT (weak table signal)")
                return Route.TEXT

        # ── Rule 1.5, RETRIEVE: single-cell lookup (cheaper than HYBRID) ──
        # Disabled by default. Sending single-cell lookups straight to
        # RETRIEVE saves one HYBRID call, but RETRIEVE on its own verifies far
        # less often than HYBRID on dev, so the saving costs more first-attempt
        # successes than it is worth. Such lookups fall through to Rule 2
        # instead. Set ATR_KEEP_RULE_1_5=1 to re-enable it for an ablation.
        import os
        if os.environ.get("ATR_KEEP_RULE_1_5", "0") == "1" and (
            has_schema
            and is_lookup
            and len(entity_mentions) == 1
            and not need_global
            and modalities == "table"
            and Route.RETRIEVE not in failed_routes
        ):
            logger.debug("Router → RETRIEVE (Rule 1.5, ablation only)")
            return Route.RETRIEVE

        # ── Rule 2, HYBRID: default for entity-grounded table queries ────
        # When schema and entities are both present, HYBRID verifies more
        # often on dev than either RETRIEVE or SQL alone: it pairs cell linking
        # with constrained SQL execution, which is what a cross-modal lookup
        # over a known table needs. Restricting this rule to mixed-modality
        # sub-queries, so that pure-table ones fall to RETRIEVE or SQL, cost
        # accuracy on WTQ, so the rule is left ungated (§3.4).
        if has_schema and has_entities and Route.HYBRID not in failed_routes:
            logger.debug("Router → HYBRID (schema + entities, default)")
            return Route.HYBRID

        # ── Rule 3, SQL: aggregation without entity grounding ────────────
        # Pure aggregation (count/sum/avg/sort/arithmetic/compare) over a
        # known table without specific entities to link.
        if is_aggregate and has_schema and not need_global:
            if Route.SQL not in failed_routes:
                logger.debug("Router → SQL (aggregate, no entity grounding needed)")
                return Route.SQL

        # ── Rule 4, RETRIEVE: escalation fallback / unstructured lookup ──
        # Now that RETRIEVE has T2024-style row retrieval (per-table BM25),
        # it is a meaningful escalation target after HYBRID/SQL exhausted,
        # OR a primary path when entities are present but no schema is
        # available (rare).
        if has_entities and Route.RETRIEVE not in failed_routes:
            if (Route.HYBRID in failed_routes
                    or Route.SQL in failed_routes
                    or not has_schema):
                logger.debug("Router → RETRIEVE (escalation / unstructured lookup)")
                return Route.RETRIEVE

        # ── Rule 5: SQL fallback ──────────────────────────────────────────
        if has_schema and Route.SQL not in failed_routes:
            logger.debug("Router → SQL (fallback: has schema)")
            return Route.SQL

        # ── Rule 6: TEXT last resort ──────────────────────────────────────
        fallback = _first_available_route(
            failed_routes,
            (Route.TEXT, Route.RETRIEVE, Route.SQL, Route.HYBRID),
        )
        logger.debug(f"Router → {fallback.value} (last available route)")
        return fallback

class LLMRouter(BaseRouter):
    """
    §3.4  LLM-based Router.

    Prompts an LLM to select TEXT / SQL / RETRIEVE / HYBRID given the
    sub-query semantics and metadata. Falls back to RETRIEVE on parse failure.
    """

    def __init__(self, llm_fn: Callable[[List[Dict]], str]) -> None:
        self.llm_fn = llm_fn

    def route(
        self,
        sub_query: str,
        meta: Any,
        schema: Optional[Dict[str, Any]],
        history_H: List[Dict[str, Any]],
    ) -> Route:
        from atr.prompt import ROUTER_PROMPT

        failed_routes = sorted(_failed_route_names(history_H, sub_query))
        has_schema = "yes" if schema is not None else "no"
        failed_set = {r.upper() for r in failed_routes}

        prompt = ROUTER_PROMPT.format(
            sub_query=sub_query,
            expected_operator=meta.expected_operator or "lookup",
            required_modalities=meta.required_modalities or "both",
            entity_mentions=", ".join(meta.entity_mentions or []) or "none",
            has_schema=has_schema,
            need_global_table_view=(
                "yes" if getattr(meta, "need_global_table_view", False) else "no"
            ),
            uncertainty=getattr(meta, "uncertainty", 0.0),
            schema_detail=_schema_feature(schema),
            failed_routes=", ".join(failed_routes) if failed_routes else "none",
            failure_history=_history_feature(history_H),
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_fn(messages)
        except Exception as exc:
            fallback = _first_available_route(failed_set)
            logger.error(
                f"LLMRouter call failed: {exc}; defaulting to {fallback.value}"
            )
            return fallback

        # Parse: pick the first known route name in the response (longest first)
        response_upper = response.strip().upper()
        for name in ("HYBRID", "RETRIEVE", "SQL", "TEXT"):
            if name in response_upper and name not in failed_set:
                logger.debug(f"LLMRouter → {name}")
                return Route(name)

        fallback = _first_available_route(failed_set)
        logger.warning(
            f"LLMRouter: cannot parse route from '{response}'; "
            f"defaulting to {fallback.value}"
        )
        return fallback

class LearnedRouter(BaseRouter):
    """
    §3.4  Learned Router: DistilBERT fine-tuned 4-class classifier.

    Input text implements the paper tuple ``(q_t, meta_t, schema_t, H)``.

    Label mapping: TEXT=0, SQL=1, RETRIEVE=2, HYBRID=3

    Train with: tools/train_router.py
    """

    LABEL2ROUTE = {0: Route.TEXT, 1: Route.SQL, 2: Route.RETRIEVE, 3: Route.HYBRID}
    ROUTE2LABEL = {v: k for k, v in LABEL2ROUTE.items()}

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        try:
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            import torch
        except ImportError as e:
            raise ImportError(
                "LearnedRouter requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            ) from e

        self._torch = torch
        self.device = device
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_path, num_labels=4
        )
        self.model.to(device)
        self.model.eval()
        if getattr(self.model.config, "router_input_version", 1) < 2:
            logger.warning(
                "LearnedRouter checkpoint predates schema/history input v2; "
                "retrain it with tools/train_router.py for paper-aligned routing."
            )
        # Cache of the most recent forward pass so `escalate()` can reuse the
        # logits without re-running the DistilBERT forward. Reset on every
        # call to `route()`.
        self._last_logits = None
        logger.info(f"LearnedRouter loaded from {model_path} on {device}")

    @staticmethod
    def featurise(sub_query: str, meta: Any, schema: Optional[Dict]) -> str:
        """Convert sub-query + metadata into a single classification input string."""
        return build_router_input(
            sub_query=sub_query,
            expected_operator=meta.expected_operator or "lookup",
            required_modalities=meta.required_modalities or "both",
            entity_mentions=meta.entity_mentions or [],
            need_global_table_view=bool(
                getattr(meta, "need_global_table_view", False)
            ),
            uncertainty=getattr(meta, "uncertainty", 0.0),
            schema=schema,
            history_H=[],
        )

    def route(
        self,
        sub_query: str,
        meta: Any,
        schema: Optional[Dict[str, Any]],
        history_H: List[Dict[str, Any]],
    ) -> Route:
        text = build_router_input(
            sub_query=sub_query,
            expected_operator=meta.expected_operator or "lookup",
            required_modalities=meta.required_modalities or "both",
            entity_mentions=meta.entity_mentions or [],
            need_global_table_view=bool(
                getattr(meta, "need_global_table_view", False)
            ),
            uncertainty=getattr(meta, "uncertainty", 0.0),
            schema=schema,
            history_H=history_H,
        )
        failed_routes = _failed_route_names(history_H, sub_query)

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)

        with self._torch.no_grad():
            logits = self.model(**inputs).logits[0]

        # Cache for the legacy ``escalate`` compatibility method. Normal
        # retries call ``reselect`` and perform a fresh H-conditioned forward.
        self._last_logits = logits

        # Pick highest-scoring non-failed route
        order = logits.argsort(descending=True).tolist()
        for idx in order:
            candidate = self.LABEL2ROUTE[idx]
            if candidate.value not in failed_routes:
                logger.debug(f"LearnedRouter → {candidate} (logit={logits[idx]:.3f})")
                return candidate

        return Route.TEXT

    def escalate(
        self,
        current_route: Route,
        history_H: Optional[List[Dict[str, Any]]] = None,
    ) -> Route:
        """Compatibility path: choose the best cached non-failed student route."""
        history_H = history_H or []
        failed = {current_route.value}
        failed.update(h.get("route") for h in history_H if h.get("route"))
        if self._last_logits is not None:
            for idx in self._last_logits.argsort(descending=True).tolist():
                candidate = self.LABEL2ROUTE[idx]
                if candidate.value not in failed:
                    logger.info(
                        f"LearnedRouter.escalate {current_route} → {candidate} "
                        "(student-ranked)"
                    )
                    return candidate
        return Route.TEXT

    def reselect(
        self,
        sub_query: str,
        meta: Any,
        schema: Optional[Dict[str, Any]],
        history_H: List[Dict[str, Any]],
        current_route: Route,
    ) -> Route:
        """Run the student again after encoding the updated history ``H``."""
        return self.route(sub_query, meta, schema, history_H)

class FixedRouter(BaseRouter):
    """Ablation router: returns the fixed route as first choice.

    Escalation behaviour is controlled by `escalate_chain`:
      - None (default): no escalation, stays on the fixed route on retry.
        Used with `--no_escalation` for "pure forced route" ablation.
      - "standard": after first failure, cycles through the other three routes
        in order [HYBRID, SQL, RETRIEVE, TEXT] minus `fixed_route`. Used to
        measure "first choice = X, with full escalation safety net", separates
        the cost of a bad first choice from the cost of no escalation at all.
    """

    def __init__(
        self,
        fixed_route: Route,
        escalate_chain: Optional[str] = None,
    ) -> None:
        self.fixed_route = fixed_route
        self.escalate_chain = escalate_chain
        all_routes = [Route.HYBRID, Route.SQL, Route.RETRIEVE, Route.TEXT]
        self._chain = [r for r in all_routes if r != fixed_route]
        self._next_idx = 0  # per-sub-query escalation counter
        logger.info(
            f"FixedRouter initialized: first={fixed_route.value}  "
            f"escalate_chain={escalate_chain}  "
            f"chain={[r.value for r in self._chain] if escalate_chain else 'none'}"
        )

    def route(
        self,
        sub_query: str,
        meta: Any,
        schema: Optional[Dict[str, Any]],
        history_H: List[Dict[str, Any]],
    ) -> Route:
        # Reset escalation counter for each new sub-query.
        self._next_idx = 0
        return self.fixed_route

    def escalate(
        self,
        current_route: Route,
        history_H: Optional[List[Dict[str, Any]]] = None,
    ) -> Route:
        if self.escalate_chain != "standard":
            # Legacy behaviour: stay on the fixed route.
            return self.fixed_route
        if self._next_idx < len(self._chain):
            r = self._chain[self._next_idx]
            self._next_idx += 1
            logger.info(
                f"FixedRouter EscalatePolicy: {current_route} → {r}  "
                f"({self._next_idx}/{len(self._chain)})"
            )
            return r
        # Chain exhausted: stay on TEXT (the universal fallback).
        return Route.TEXT

    def reselect(
        self,
        sub_query: str,
        meta: Any,
        schema: Optional[Dict[str, Any]],
        history_H: List[Dict[str, Any]],
        current_route: Route,
    ) -> Route:
        """Keep fixed-router ablations on their explicitly requested chain."""
        return self.escalate(current_route, history_H=history_H)

def build_router(
    router_type: str,
    llm_fn: Optional[Callable] = None,
    model_path: Optional[str] = None,
    device: str = "cpu",
    force_route: Optional[str] = None,
    fixed_escalate_chain: Optional[str] = None,
) -> BaseRouter:
    """
    Factory function: select router by name.

    Args:
        router_type:  "heuristic" | "llm" | "learned" | "fixed"
        llm_fn:       required for "llm"
        model_path:   required for "learned"
        device:       torch device for "learned"
        force_route:  required for "fixed": TEXT|SQL|RETRIEVE|HYBRID
        fixed_escalate_chain: optional for "fixed": None|"standard" (cycle
                              through other 3 routes on each escalate() call)
    """
    if router_type == "heuristic":
        return HeuristicRouter()
    if router_type == "llm":
        if llm_fn is None:
            raise ValueError("LLMRouter requires llm_fn")
        return LLMRouter(llm_fn)
    if router_type == "learned":
        if model_path is None:
            raise ValueError("LearnedRouter requires model_path")
        return LearnedRouter(model_path, device=device)
    if router_type == "fixed":
        if not force_route:
            raise ValueError("FixedRouter requires force_route")
        return FixedRouter(Route(force_route.upper()),
                           escalate_chain=fixed_escalate_chain)
    raise ValueError(f"Unknown router_type '{router_type}'. Choose: heuristic | llm | learned | fixed")
