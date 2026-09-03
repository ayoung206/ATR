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

# Escalation order (§3.6), specific-first: a failed route escalates to the
# next most constrained alternative, ending at free-form text reading. Trying
# structured routes before TEXT beats the reverse order on dev, and this is
# the chain used for the reported results.
# Each entry maps the current route to the one tried next.
_ESCALATION: Dict[Route, Route] = {
    Route.SQL: Route.HYBRID,        # SQL failed → try HYBRID (constrained variant)
    Route.HYBRID: Route.RETRIEVE,   # HYBRID failed → RETRIEVE (cell lookup)
    Route.RETRIEVE: Route.TEXT,     # RETRIEVE failed → TEXT (free-form read)
    Route.TEXT: Route.SQL,          # cycle: TEXT failed → back to SQL
}

_AGGREGATE_OPERATORS = {"count", "aggregate", "sort", "arithmetic", "compare"}
_LOOKUP_OPERATORS = {"lookup", "filter"}

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
        """§3.6 EscalatePolicy. Default: fixed global chain (TEXT → HYBRID
        → SQL → RETRIEVE).  Subclasses (e.g., LearnedRouter) may override to
        provide adaptive ordering using their own confidence ranking; the
        history_H argument lets them skip any routes already tried.
        """
        next_route = _ESCALATION.get(current_route, Route.TEXT)
        logger.info(f"EscalatePolicy: {current_route} → {next_route}")
        return next_route

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

        failed_routes = {r.get("route") for r in history_H}

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
        logger.debug("Router → TEXT (last resort)")
        return Route.TEXT

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

        failed_routes = [r.get("route", "") for r in history_H if r.get("route")]
        has_schema = "yes" if schema is not None else "no"

        prompt = ROUTER_PROMPT.format(
            sub_query=sub_query,
            expected_operator=meta.expected_operator or "lookup",
            required_modalities=meta.required_modalities or "both",
            entity_mentions=", ".join(meta.entity_mentions or []) or "none",
            has_schema=has_schema,
            failed_routes=", ".join(failed_routes) if failed_routes else "none",
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_fn(messages)
        except Exception as exc:
            logger.error(f"LLMRouter call failed: {exc}; defaulting to RETRIEVE")
            return Route.RETRIEVE

        # Parse: pick the first known route name in the response (longest first)
        response_upper = response.strip().upper()
        failed_set = {r.upper() for r in failed_routes}
        for name in ("HYBRID", "RETRIEVE", "SQL", "TEXT"):
            if name in response_upper and name not in failed_set:
                logger.debug(f"LLMRouter → {name}")
                return Route(name)

        logger.warning(f"LLMRouter: cannot parse route from '{response}'; defaulting to RETRIEVE")
        return Route.RETRIEVE

class LearnedRouter(BaseRouter):
    """
    §3.4  Learned Router: DistilBERT fine-tuned 4-class classifier.

    Input text format:
      "[QUERY] {sub_query} [OP] {op} [MOD] {mod} [ENT] {entities} [SCHEMA] {yes|no}"

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
        # Cache of the most recent forward pass so `escalate()` can reuse the
        # logits without re-running the DistilBERT forward. Reset on every
        # call to `route()`.
        self._last_logits = None
        logger.info(f"LearnedRouter loaded from {model_path} on {device}")

    @staticmethod
    def featurise(sub_query: str, meta: Any, schema: Optional[Dict]) -> str:
        """Convert sub-query + metadata into a single classification input string."""
        op = meta.expected_operator or "lookup"
        mod = meta.required_modalities or "both"
        entities = ", ".join(meta.entity_mentions or []) or "none"
        has_schema = "yes" if schema is not None else "no"
        return (
            f"[QUERY] {sub_query} "
            f"[OP] {op} "
            f"[MOD] {mod} "
            f"[ENT] {entities} "
            f"[SCHEMA] {has_schema}"
        )

    def route(
        self,
        sub_query: str,
        meta: Any,
        schema: Optional[Dict[str, Any]],
        history_H: List[Dict[str, Any]],
    ) -> Route:
        text = self.featurise(sub_query, meta, schema)
        failed_routes = {r.get("route") for r in history_H if r.get("route")}

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)

        with self._torch.no_grad():
            logits = self.model(**inputs).logits[0]

        # Cache for adaptive escalation: escalate() reuses these logits
        # so the escalation order respects per-sub-query router ranking
        # rather than a fixed global chain.
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
        """Fixed specific-first escalation chain
        (SQL $\\to$ HYBRID $\\to$ RETRIEVE $\\to$ TEXT), skipping any route
        already attempted in this sub-query.

        On dev this fixed chain outperforms a logit-ranked adaptive variant,
        presumably because it orders by information yield given a verifier
        rejection rather than by first-attempt prior. The set of already-attempted routes is read
        from `history_H` so the chain never revisits a failed primitive.
        """
        history_H = history_H or []
        failed = {current_route.value}
        failed.update(h.get("route") for h in history_H if h.get("route"))

        # Walk the global _ESCALATION chain, skipping anything we've already
        # tried this sub-query.
        nxt = _ESCALATION.get(current_route, Route.TEXT)
        for _ in range(4):
            if nxt.value not in failed:
                logger.info(
                    f"LearnedRouter.escalate {current_route} → {nxt} (fixed chain)"
                )
                return nxt
            failed.add(nxt.value)
            nxt = _ESCALATION.get(nxt, Route.TEXT)
        # All 4 routes already attempted, degenerate, return TEXT as final fallback
        return Route.TEXT

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
