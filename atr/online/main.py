from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from atr.clients.chat_utils import get_chat_result, init_logger                   # noqa: E402
from atr.config import (  # noqa: E402
    config_mapping,
    MAX_ITER,
    SCHEMA_TOP_K,
    CELL_TOP_K,
    ROW_TOP_K,
    VERIFIER_THRESHOLD,
    RERANK_CANDIDATE_MULTIPLIER,
)
from atr.offline.multiview_index import MultiviewIndex                      # noqa: E402
from atr.online.decomposer import QueryDecomposer, SubQuery              # noqa: E402
from atr.online.router import build_router, Route                           # noqa: E402
from atr.online.value_linker import HybridValueLinker                       # noqa: E402
from atr.online.constrained_sql import ConstrainedSQLExecutor               # noqa: E402
from atr.online.verifier import EvidenceFusionVerifier                   # noqa: E402

logger = logging.getLogger(__name__)


def _is_valid_answer(answer: Any) -> bool:
    if answer is None:
        return False
    text = str(answer).strip()
    return bool(text) and text.lower() not in {
        "none", "null", "n/a", "na", "not found", "error",
    }


def _is_verified_answer(answer: Any, execution_confidence: float) -> bool:
    """Only verifier-supported, non-empty answers may become final answers."""
    return execution_confidence == 1.0 and _is_valid_answer(answer)


def _should_stop(
    execution_confidence: float,
    uncertainty: float,
    verifier_threshold: float,
) -> bool:
    """Paper stop rule: verifier support and residual uncertainty below tau."""
    return execution_confidence == 1.0 and uncertainty < verifier_threshold


def _oracle_match(pred: Any, gold: Any) -> bool:
    """Lenient gold-match for the oracle verifier (upper-bound) ablation.

    True when the normalized prediction equals the gold, fully contains the
    gold's tokens, or one is a substring of the other. Deliberately generous:
    it credits any produced candidate that a perfect verifier could have
    recognized as correct.
    """
    def _norm(s: Any) -> str:
        s = re.sub(r"\b(a|an|the)\b", " ", str(s or "").lower())
        s = re.sub(r"[^\w\s]", " ", s)
        return " ".join(s.split())
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return False
    if p == g or g in p or p in g:
        return True
    g_tokens = set(g.split())
    return bool(g_tokens) and g_tokens.issubset(set(p.split()))


def _make_llm_fn(llm_config: Dict) -> Callable[[List[Dict]], str]:
    """Build a callable(messages) → text that wraps get_chat_result."""
    def llm_fn(messages: List[Dict]) -> str:
        response = get_chat_result(messages=messages, llm_config=llm_config)
        if hasattr(response, "content"):
            return response.content or ""
        if isinstance(response, dict):
            return response.get("content", "")
        return str(response)
    return llm_fn

def _normalize_table_key(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0].strip()
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", stem)
    normalized = re.sub(r"_+", "_", normalized).strip("_").lower()
    return normalized

def _table_name_variants(table_name: str) -> List[str]:
    """Generate lookup-key variants for a table name (same logic as TableRAG (Yu et al., 2025))."""
    if not table_name:
        return []
    stem = os.path.splitext(os.path.basename(table_name))[0].strip()
    normalized = _normalize_table_key(stem)
    variants: List[str] = [stem, stem.lower(), normalized]
    for candidate in [normalized, stem.lower()]:
        if not candidate:
            continue
        if candidate.startswith("t_"):
            variants.append(candidate[2:])
        elif candidate and candidate[0].isdigit():
            variants.append(f"t_{candidate}")
    seen: List[str] = []
    for v in variants:
        if v and v not in seen:
            seen.append(v)
    return seen

def _select_document_chunks(
    chunks: List[Dict],
    table_id_hint: str,
    top_n: int = 3,
) -> List[Dict]:
    """
    Post-retrieval document selection:
      1. Deduplicate by (source, text) pairs.
      2. Take top_n; if none are "text" type, swap last slot with first text chunk found.
      3. If table_id_hint given, ensure at least one chunk from that table is included.
    """
    # Step 1: deduplicate
    seen: set = set()
    deduped: List[Dict] = []
    for c in chunks:
        key = (c.get("source", ""), c.get("text", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    selected = deduped[:top_n]
    remaining = deduped[top_n:]

    # Step 2: ensure at least one text chunk in selection
    if selected and not any(c.get("type") == "text" for c in selected):
        for candidate in remaining:
            if candidate.get("type") == "text":
                selected[-1] = candidate
                break

    # Step 3: prefer chunk matching table_id_hint
    if table_id_hint:
        hint_variants = set(_table_name_variants(table_id_hint))
        hint_variants.add(table_id_hint.lower())

        def _matches_hint(c: Dict) -> bool:
            src = c.get("source", "").lower()
            tid = c.get("table_id", "").lower()
            return any(v in src or v == tid for v in hint_variants)

        if not any(_matches_hint(c) for c in selected):
            for candidate in deduped:
                if _matches_hint(candidate) and candidate not in selected:
                    if selected:
                        selected[-1] = candidate
                    else:
                        selected.append(candidate)
                    break

    return selected

def _chunks_to_text(chunks: List[Dict]) -> str:
    return "\n\n".join(
        f"[Source: {c.get('source', '?')}]\n{c.get('text', '')}"
        for c in chunks
    )

def _build_schema_preview(table_chunks: List[Dict], max_rows: int = 3) -> str:
    """#1: compact "column headers + sample rows" hint for the decomposer.

    Picks the top-1 table chunk and returns header + separator + first
    `max_rows` data rows in the original markdown form. Returns "" when no
    table chunk is available: caller treats as a no-op slot.
    """
    if not table_chunks:
        return ""
    chunk = table_chunks[0]
    text = chunk.get("text") or chunk.get("content") or ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.lstrip().startswith("|")),
        None,
    )
    if header_idx is None:
        return ""
    snippet = "\n".join(lines[header_idx : header_idx + 2 + max_rows])
    src = chunk.get("source") or chunk.get("table_id") or "?"
    return (
        "\n═══════════════════════════════════════════════════════════════\n"
        f"SCHEMA PREVIEW (from candidate table: {src})\n"
        "═══════════════════════════════════════════════════════════════\n"
        f"{snippet}\n"
        "(↑ Use these column headers and sample rows to decide whether the\n"
        " sub-query is a single-cell lookup or requires COUNT/aggregate.)\n"
    )

def _row_hits_to_markdown(hits: List[Dict], max_rows: int = ROW_TOP_K) -> str:
    """Phase F-A: render View 4 row-component hits as a compact markdown listing.

    Each row becomes a single block of `column: value` lines plus a header
    pointing back to its table_id and row index. The LLM then sees only
    rows that matched the sub-query/entity, rather than entire tables.
    """
    if not hits:
        return ""
    out: List[str] = []
    for h in hits[:max_rows]:
        tid = h.get("table_id", "?")
        ridx = h.get("row_idx", "?")
        rd = h.get("row_dict", {})
        lines = [f"[Table {tid}, row {ridx}]"]
        for col, val in rd.items():
            if val == "" or val is None:
                continue
            lines.append(f"  {col}: {val}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def _retrieve_row_context(
    index: MultiviewIndex,
    query: str,
    chunks: List[Dict],
    top_k: int = ROW_TOP_K,
) -> str:
    """Paper RETRIEVE primitive: rank exactly top-K rows against q_t."""
    target_table = next(
        (
            chunk.get("table_id") or chunk.get("source")
            for chunk in chunks
            if chunk.get("type") == "table"
        ),
        None,
    )
    hits = index.retrieve_rows(query, top_k=top_k, table_id=target_table)
    return _row_hits_to_markdown(hits, max_rows=top_k)

def _table_chunks_to_markdown(chunks: List[Dict], max_chars: int = 4000) -> str:
    """#3+(b): concatenate table-type chunk text for verifier fusion.

    Returns up to `max_chars` characters of markdown table chunks so the LLM
    fusion can cross-validate SQL=0/empty cases against actual cell values.
    """
    out: List[str] = []
    total = 0
    for c in chunks:
        if c.get("type") != "table":
            continue
        t = c.get("text") or c.get("content") or ""
        if not t:
            continue
        if total + len(t) > max_chars:
            t = t[: max(0, max_chars - total)]
        src = c.get("source") or c.get("table_id") or "?"
        out.append(f"[Source: {src}]\n{t}")
        total += len(t)
        if total >= max_chars:
            break
    return "\n\n".join(out)

def _schema_entries_to_text(entries: List[Dict]) -> str:
    lines = []
    for e in entries:
        lines.append(
            f"Table: {e.get('table_id','?')} | Column: {e['col_name']} "
            f"({e.get('dtype','')}) | Examples: {e.get('examples','')}"
        )
    return "\n".join(lines)

def _cell_entries_to_text(linked_values: Any) -> str:
    lines = []
    for lv in linked_values:
        if lv.is_matched:
            lines.append(f"Entity '{lv.entity}' → {lv.column} = '{lv.matched_value}'")
    return "\n".join(lines) if lines else "(no values matched)"

_ARITH_PATTERN = re.compile(
    r"\b(percentage\s+change|percent\s+change|change|difference|"
    r"average|mean|increase|decrease|growth|decline|ratio|proportion|"
    r"sum|total|how\s+much\s+(?:is|did|was)|how\s+many|number\s+of|count\s+of)\b",
    re.I,
)

def _is_arithmetic_q(text: str) -> bool:
    return bool(_ARITH_PATTERN.search(text or ""))

def _arithmetic_hint(text: str) -> str:
    """D1: formula hint based on detected arithmetic intent."""
    t = (text or "").lower()
    if "percentage change" in t or "percent change" in t or "% change" in t:
        return ("Compute (new_value - old_value) / old_value * 100. "
                "Output as a percent value with sign (e.g., '-12.14 percent').")
    if any(k in t for k in ("how many", "number of", "count of", "count the")):
        return ("Use SELECT COUNT(*) FROM ... WHERE <condition> with the threshold "
                "applied strictly. Return a single integer scalar.")
    if any(k in t for k in ("change", "difference", "increase", "decrease", "growth", "decline")):
        return ("Compute new_value - old_value. Preserve unit (e.g., million / thousand / percent). "
                "Keep the negative sign if the result is negative.")
    if "average" in t or "mean" in t:
        return "Compute AVG of the relevant values. Preserve unit."
    if "ratio" in t or "proportion" in t:
        return "Compute as decimal (e.g., 0.11), unless 'percentage' is asked."
    if any(k in t for k in ("sum", "total")):
        return "Compute SUM of the relevant values. Preserve unit."
    return ("Perform the requested computation in SQL. "
            "Return the computed scalar with unit, NOT raw rows.")

def _column_type_hint(schema_entries: List[Dict]) -> str:
    """E8: hint about column types, when VARCHAR holds numeric strings, suggest REPLACE/CAST."""
    suggestions = []
    for e in schema_entries[:5]:
        col = e.get("col_name", "")
        dtype = (e.get("dtype") or "").upper()
        examples = str(e.get("examples", "") or "")
        # Detect VARCHAR columns storing money/numeric strings
        looks_money = any(t in examples for t in ("$", ",")) or bool(
            re.search(r"\d[,.]?\d", examples)
        )
        if "VARCHAR" in dtype or "TEXT" in dtype or "CHAR" in dtype:
            if looks_money:
                suggestions.append(
                    f"Column `{col}` stores numbers as strings (e.g., {examples[:60]!r}). "
                    f"Use REPLACE(REPLACE(`{col}`, '$', ''), ',', '') and CAST(... AS DECIMAL) "
                    f"before arithmetic."
                )
    return "\n".join(suggestions) if suggestions else ""

def _schema_columns_hint(schema_entries: List[Dict]) -> str:
    """E1-light: top schema columns with dtype, no value linker LLM call."""
    if not schema_entries:
        return ""
    parts = []
    for e in schema_entries[:5]:
        parts.append(f"`{e.get('col_name','?')}` ({e.get('dtype','')})")
    return "Top relevant columns: " + ", ".join(parts)

def _augment_sql_query(
    sub_query: str,
    schema_entries: Optional[List[Dict]] = None,
    enable_arith: bool = True,
    enable_schema_hint: bool = True,
    enable_type_hint: bool = True,
) -> str:
    """Wrap sub_query with NL2SQL-friendly hints (D1 + E1-light + E8).

    Pure string transformation; no LLM calls. T2025 NL2SQL receives this
    augmented query and uses the extra context when generating SQL.
    """
    blocks: List[str] = [sub_query.strip()]

    if enable_arith and _is_arithmetic_q(sub_query):
        blocks.append("IMPORTANT: " + _arithmetic_hint(sub_query) +
                      " Do NOT return raw rows; compute the answer in SQL.")

    if schema_entries:
        if enable_schema_hint:
            sh = _schema_columns_hint(schema_entries)
            if sh:
                blocks.append("CONTEXT: " + sh)
        if enable_type_hint:
            th = _column_type_hint(schema_entries)
            if th:
                blocks.append("TYPE NOTE:\n" + th)

    return "\n\n".join(blocks)

class AgenticTableRAGAgent:
    """
    Agentic TableRAG online inference agent implementing Algorithm 1 (§3.1).

    Combines:
      - TableRAG (Chen et al., 2024)'s Schema/Cell retrieval strength  (Views 3, 4)
      - TableRAG (Yu et al., 2025)'s document-loop exploration strength (Views 1, 2, 5)
    Connected by a learnable router (§3.4).
    """

    def __init__(
        self,
        multiview_index: MultiviewIndex,
        backbone: str = "gemini",
        max_iter: int = MAX_ITER,
        router_type: str = "heuristic",
        router_model_path: Optional[str] = None,
        router_device: str = "cuda",
        verifier_threshold: float = VERIFIER_THRESHOLD,
        force_route: Optional[str] = None,
        no_decomposition: bool = False,
        final_synthesis: bool = False,
        no_escalation: bool = False,
        no_schema_preview: bool = False,
        legacy_fast_path: bool = False,
        fixed_escalate_chain: Optional[str] = None,
        no_value_linker: bool = False,
        decomposer_backbone: Optional[str] = None,
        verifier_backbone: Optional[str] = None,
        oracle_verifier: bool = False,
    ) -> None:
        self.index = multiview_index
        self.max_iter = max_iter
        self.no_decomposition = no_decomposition
        self.final_synthesis = final_synthesis
        self.no_escalation = no_escalation
        # Ablation flags, disabling post-fix behaviours to recover pre-fix variants:
        # • no_schema_preview: skip pre-retrieval; pass empty schema_preview
        #                        to decomposer (reverts #1)
        # • legacy_fast_path: verifier.fuse() uses unconditional SQL scalar
        #                        fast-path; skips markdown cross-validation (reverts #3+(b))
        # • no_value_linker: bypass HybridValueLinker for RETRIEVE/HYBRID
        #                        routes; pass empty linked_values to ConstrainedSQL,
        #                        isolating the value-linking contribution. Column
        #                        constraint C from schema retrieval is still applied.
        self.no_schema_preview = no_schema_preview
        self.legacy_fast_path = legacy_fast_path
        self.no_value_linker = no_value_linker
        llm_config = config_mapping[backbone]
        llm_fn = _make_llm_fn(llm_config)
        self._llm_fn = llm_fn

        # Decomposer-sensitivity ablation: optionally drive ONLY the query
        # decomposer with a different backbone, holding router/executor/
        # verifier/synthesis at `backbone`. Isolates how much ATR depends on
        # the decomposer model's capability (vs. the decomposition mechanism,
        # which the --no_decomposition ablation already measures).
        if decomposer_backbone and decomposer_backbone != backbone:
            decomposer_llm_fn = _make_llm_fn(config_mapping[decomposer_backbone])
            logger.info("Decomposer backbone overridden: %s (pipeline backbone: %s)",
                        decomposer_backbone, backbone)
        else:
            decomposer_llm_fn = llm_fn
        self._decomposer_backbone = decomposer_backbone or backbone

        self.verifier_threshold = verifier_threshold
        self.decomposer = QueryDecomposer(decomposer_llm_fn)
        self.router = build_router(
            router_type=router_type,
            llm_fn=llm_fn,
            model_path=router_model_path,
            device=router_device,
            force_route=force_route,
            fixed_escalate_chain=fixed_escalate_chain,
        )
        self.value_linker = HybridValueLinker(llm_fn, cell_top_k=CELL_TOP_K)

        # Verifier-sensitivity ablation: optionally drive ONLY the verdict
        # (EvidenceFusionVerifier.verify) with a different backbone, holding
        # fusion, route answering, decomposition, and routing at `backbone`.
        # Isolates how much ATR depends on the verifier model's accuracy.
        if verifier_backbone and verifier_backbone != backbone:
            verify_llm_fn = _make_llm_fn(config_mapping[verifier_backbone])
            logger.info("Verifier (verdict) backbone overridden: %s (pipeline backbone: %s)",
                        verifier_backbone, backbone)
        else:
            verify_llm_fn = llm_fn
        self._verifier_backbone = verifier_backbone or backbone
        # Oracle verifier (upper bound): after a question runs normally, if any
        # answer ATR produced matches the gold, return it. Measures the accuracy
        # ceiling a perfect verifier could reach over ATR's produced candidates.
        self.oracle_verifier = oracle_verifier
        self.verifier = EvidenceFusionVerifier(llm_fn, verify_llm_fn=verify_llm_fn)

    def run_single(
        self,
        question: str,
        table_name_list: Optional[List[str]] = None,
        question_id: str = "Q",
        gold_answer: Optional[str] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Algorithm 1: Agentic TableRAG Online Inference.

        Args:
            question:         original user question q
            table_name_list:  candidate table names for SQL executor (View 5)
            question_id:      for logging

        Returns:
            final answer string
        """
        # Line 1: z_0 ← ∅; H ← ∅
        z: List[Dict[str, str]] = []   # intermediate {sub_query, answer}
        H: List[Dict[str, Any]] = []   # failure history
        best_answer: str = ""
        # Oracle-verifier ablation: every answer produced this question, so a
        # perfect verifier's reachable accuracy can be measured post-hoc.
        # Thread-local list (not an instance attribute) to stay thread-safe
        # under BatchRunner's multi-worker execution.
        oracle_candidates: List[str] = []
        verifier_decisions: List[Dict[str, Any]] = []

        # Feature 3: expand table_name_list with normalized variants
        raw_table_names = table_name_list or []
        expanded_table_names: List[str] = []
        for tn in raw_table_names:
            for v in _table_name_variants(tn):
                if v not in expanded_table_names:
                    expanded_table_names.append(v)
        if not expanded_table_names:
            expanded_table_names = raw_table_names

        sql_executor = ConstrainedSQLExecutor(
            table_name_list=expanded_table_names
        )
        table_id_hint = raw_table_names[0] if raw_table_names else ""

        consecutive_failures = 0   # tracks consecutive "not found" steps

        # ── #1: pre-retrieval to give the decomposer a schema preview ────────
        # Uses the original question once, before any decomposition, so the
        # decomposer can see column headers + sample rows of the candidate
        # table and disambiguate "how many" Qs that are cell-lookups vs
        # COUNT(*) aggregates. Empty string when no table chunks are available
        # OR when --no_schema_preview is set (ablation).
        if self.no_schema_preview:
            schema_preview = ""
        else:
            try:
                pre_chunks_raw = self.index.retrieve_documents(question, top_k=10)
                pre_chunks = _select_document_chunks(pre_chunks_raw, table_id_hint, top_n=5)
                pre_table_chunks = [c for c in pre_chunks if c.get("type") == "table"]
                schema_preview = _build_schema_preview(pre_table_chunks)
            except Exception as exc:
                logger.warning(f"[{question_id}] schema preview pre-retrieval failed: {exc}")
                schema_preview = ""

        # Ablation: w/o Decomposition → cap iterations to 1, use raw question
        effective_max_iter = 1 if self.no_decomposition else self.max_iter

        for t in range(1, effective_max_iter + 1):
            logger.info(f"[{question_id}] ── Iteration {t}/{effective_max_iter} ──")

            # ── Line 3: Decompose (with failure recovery hint) ───────────────
            if self.no_decomposition:
                sub_q = SubQuery(
                    sub_query=question,
                    required_modalities="both",
                    expected_operator="lookup",
                    need_global_table_view=False,
                    entity_mentions=[],
                    uncertainty=0.0,
                )
            else:
                sub_q = self.decomposer.decompose(
                    question, z,
                    table_id=table_id_hint,
                    consecutive_failures=consecutive_failures,
                    schema_preview=schema_preview,
                )
            logger.info(f"[{question_id}] sub_query='{sub_q.sub_query}'  "
                        f"op={sub_q.expected_operator}  "
                        f"entities={sub_q.entity_mentions}  "
                        f"consec_failures={consecutive_failures}")

            # ── Line 4: TERMINATE check ──────────────────────────────────────
            if sub_q.is_terminate:
                logger.info(f"[{question_id}] Decomposer signalled TERMINATE.")
                break

            # ── Line 5–8: Document Retrieval + Schema Restoration ────────────
            raw_chunks = self.index.retrieve_documents(sub_q.sub_query, top_k=10)
            chunks = _select_document_chunks(raw_chunks, table_id_hint, top_n=5)
            table_chunks = [c for c in chunks if c.get("type") == "table"]
            schema: Optional[Dict] = None
            if table_chunks:
                schema = self.index.restore_schema_via_mapping(table_chunks)

            # ── Line 9: Route selection ──────────────────────────────────────
            route = self.router.route(sub_q.sub_query, sub_q, schema, H)
            logger.info(f"[{question_id}] Route = {route.value}")

            # ── Execute + Verify (with escalation loop) ──────────────────────
            y_t, c_exec = self._execute_and_verify(
                sub_q=sub_q,
                original_question=question,
                schema=schema,
                chunks=chunks,
                route=route,
                sql_executor=sql_executor,
                H=H,
                question_id=question_id,
                oracle_candidates=oracle_candidates,
                decisions=verifier_decisions if trace is not None else None,
            )

            step_failed = not _is_verified_answer(y_t, c_exec)
            if step_failed:
                consecutive_failures += 1
                logger.info(
                    f"[{question_id}] Step {t} failed verification or returned no answer. "
                    f"consecutive_failures={consecutive_failures}"
                )
            else:
                consecutive_failures = 0
                best_answer = y_t

            z.append({
                "sub_query": sub_q.sub_query,
                "answer": y_t or "not found",
                "failed": step_failed,
            })

            # ── §3.6 Stop Controller ───────────────────────────────────────
            # Require BOTH (a) verified execution AND (b) decomposer
            # uncertainty below the configured paper threshold tau. For
            # multi-hop questions the decomposer often *under*-estimates
            # uncertainty after the first successful step, causing early
            # exit before all bridging sub-queries are resolved. Trust
            # TERMINATE signal (handled above as q_t == TERMINATE break)
            # as the primary stop trigger; this controller is only a
            # secondary safety net for confident terminal answers.
            if _should_stop(c_exec, sub_q.uncertainty, self.verifier_threshold):
                logger.info(
                    f"[{question_id}] Stop Controller: verified + "
                    f"uncertainty={sub_q.uncertainty:.2f} < "
                    f"{self.verifier_threshold:.2f}, stopping early"
                )
                break

        # ── Final answer selection (Line 29) ────────────────────────────────
        # Final Answer Synthesis (multi-hop aggregation only): only
        # invoke FS when there are multiple verifier-accepted sub-answers to
        # fuse. Rejected answers remain in z for decomposition diagnostics but
        # cannot influence the final answer.
        accepted_z = [entry for entry in z if not entry.get("failed", False)]
        final_answer: str = ""
        if self.final_synthesis and len(accepted_z) >= 2:
            synthesized = self._synthesize_final(question, accepted_z, question_id)
            if synthesized:
                final_answer = synthesized
        # Prefer best_answer (last verified non-failed); fall back through z.
        if not final_answer:
            if _is_valid_answer(best_answer):
                final_answer = best_answer
            else:
                for entry in reversed(accepted_z):
                    a = entry.get("answer", "")
                    if _is_valid_answer(a):
                        final_answer = a
                        break
        if not final_answer:
            final_answer = "not found"

        if trace is not None:
            trace.update({
                "question_id": question_id,
                "verifier_decisions": verifier_decisions,
                "n_escalations": sum(
                    1 for decision in verifier_decisions
                    if not decision["accepted"]
                ),
                "final_answer": final_answer,
            })

        # Oracle verifier (upper bound): if any answer ATR produced this
        # question matches the gold, a perfect verifier could have selected or
        # stopped on it; return that candidate so the run measures the ceiling.
        if self.oracle_verifier and gold_answer:
            for cand in [final_answer, best_answer, *(e.get("answer", "") for e in z), *oracle_candidates]:
                if _oracle_match(cand, gold_answer):
                    if cand != final_answer:
                        logger.info(
                            f"[{question_id}] Oracle verifier recovered "
                            f"'{str(cand)[:60]}' (pipeline returned "
                            f"'{str(final_answer)[:60]}')"
                        )
                    return cand

        return final_answer

    def _synthesize_final(
        self,
        question: str,
        z: List[Dict[str, str]],
        question_id: str,
    ) -> str:
        """Post-loop multi-hop aggregation: re-read original question + sub-answers."""
        from atr.prompt import FINAL_SYNTHESIS_PROMPT

        history_lines = []
        for i, entry in enumerate(z, 1):
            ans = entry.get("answer", "")
            failed = entry.get("failed", False)
            tag = " [FAILED]" if failed else ""
            history_lines.append(
                f"Step {i}{tag}: Q: {entry.get('sub_query', '')}\n          A: {ans}"
            )
        history_text = "\n".join(history_lines)

        prompt_text = FINAL_SYNTHESIS_PROMPT.format(
            question=question,
            history=history_text,
            output_marker="<final_answer>:",
        )
        try:
            response = self._llm_fn([{"role": "user", "content": prompt_text}])
        except Exception as exc:
            logger.error(f"[{question_id}] Final synthesis failed: {exc}")
            return ""

        # Strip output marker if present
        text = response.strip()
        for marker in ["<final_answer>:", "<final_answer>", "Final answer:", "Answer:"]:
            if text.lower().startswith(marker.lower()):
                text = text[len(marker):].strip()
        # Take first line only
        text = text.split("\n")[0].strip()
        # Remove trailing punctuation
        text = text.rstrip(".")
        logger.info(f"[{question_id}] Final synthesis → '{text[:80]}'")
        return text

    def _execute_and_verify(
        self,
        sub_q: Any,
        original_question: str,
        schema: Optional[Dict],
        chunks: List[Dict],
        route: Route,
        sql_executor: ConstrainedSQLExecutor,
        H: List[Dict],
        question_id: str,
        oracle_candidates: Optional[List[str]] = None,
        decisions: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, float]:
        """
        Execute by route, verify, and escalate if needed (Lines 10–26 of Algo 1).
        Returns (answer, execution_confidence).
        """
        text_evidence = _chunks_to_text(chunks)

        # When --no_escalation is set (ablation): execute the router's first
        # choice once, verify, and return regardless of verdict. This isolates
        # the router's intrinsic contribution without the escalation safety net.
        # Otherwise allow up to 4 attempts so the escalation chain can reach
        # all 4 primitives if the router's first 3 choices all fail.
        max_attempts = 1 if self.no_escalation else 4

        for attempt in range(max_attempts):
            logger.info(f"[{question_id}] Executing route={route.value} (attempt {attempt+1})")

            y_t, sql_result, schema_cell_ev = self._execute_route(
                sub_q=sub_q,
                original_question=original_question,
                schema=schema,
                chunks=chunks,
                text_evidence=text_evidence,
                route=route,
                sql_executor=sql_executor,
                H=H,
            )
            if oracle_candidates is not None and y_t:
                oracle_candidates.append(y_t)

            # ── Lines 23–25: Verify ──────────────────────────────────────────
            verdict = self.verifier.verify(
                question=sub_q.sub_query,
                answer=y_t,
                text_evidence=text_evidence,
                sql_result=sql_result,
            )
            logger.info(f"[{question_id}] Verifier verdict={verdict}")

            if decisions is not None:
                decisions.append({
                    "question_id": question_id,
                    "sub_query": getattr(sub_q, "sub_query", ""),
                    "expected_operator": getattr(sub_q, "expected_operator", "lookup"),
                    "required_modalities": getattr(sub_q, "required_modalities", "both"),
                    "entity_mentions": getattr(sub_q, "entity_mentions", []) or [],
                    "need_global_table_view": bool(
                        getattr(sub_q, "need_global_table_view", False)
                    ),
                    "uncertainty": getattr(sub_q, "uncertainty", 0.0),
                    "schema": schema,
                    "has_schema": bool(schema),
                    "history_H": [dict(item) for item in H],
                    "route": route.value,
                    "attempt": attempt + 1,
                    "verdict": verdict,
                    "accepted": not self.verifier.is_rejected(verdict),
                    "produced_answer": y_t,
                })

            if not self.verifier.is_rejected(verdict):
                return y_t, 1.0

            if self.no_escalation:
                # Ablation: do not escalate to a different route.
                logger.info(f"[{question_id}] no_escalation=True → returning first-route answer")
                break

            # Record failure and escalate. The history uses the route's *string
            # value* so downstream router code can compare against `Route.value`
            # consistently; the previous mix of enum and string caused
            # `r.get("route")` to mismatch and effectively skip the
            # failed-routes filter on retries.
            H.append({
                "sub_query": sub_q.sub_query,
                "route": route.value,
                "verdict": verdict,
                "step": len(H) + 1,
            })
            new_route = self.router.reselect(
                sub_query=sub_q.sub_query,
                meta=sub_q,
                schema=schema,
                history_H=H,
                current_route=route,
            )
            if new_route == route:
                # Terminal: TEXT and already tried
                break
            route = new_route

        return y_t, 0.0 if not y_t else 0.5

    def _execute_route(
        self,
        sub_q: Any,
        original_question: str,
        schema: Optional[Dict],
        chunks: List[Dict],
        text_evidence: str,
        route: Route,
        sql_executor: ConstrainedSQLExecutor,
        H: List[Dict],
    ) -> Tuple[str, str, str]:
        """
        Execute a single route.
        Returns (answer, sql_result, schema_cell_evidence_text).
        """
        sql_result = ""
        schema_cell_ev = ""

        # ── TEXT (Lines 10–11) ───────────────────────────────────────────────
        if route == Route.TEXT:
            y_t = self.verifier.answer_from_text(
                sub_q.sub_query, text_evidence, original_question=original_question
            )
            return y_t, sql_result, schema_cell_ev

        # ── SQL (Lines 16–17) ────────────────────────────────────────────────
        if route == Route.SQL:
            # E1-light: schema-only retrieval (BGE+FAISS, no cell index, no value
            # linker LLM). Calling `retrieve_schema` directly avoids the per-entity
            # cell loop so SQL-route latency stays low.
            try:
                C_sql = self.index.retrieve_schema(
                    sub_q.sub_query, top_k=SCHEMA_TOP_K
                )
            except Exception as exc:
                logger.warning(f"E1-light schema retrieval failed: {exc}")
                C_sql = []

            augmented_query = _augment_sql_query(
                sub_q.sub_query,
                schema_entries=C_sql,
                enable_arith=True,
                enable_schema_hint=True,
                enable_type_hint=True,
            )
            schema_evidence = _schema_entries_to_text(C_sql)
            sql_result, _ = sql_executor.execute(
                sub_query=augmented_query,
                schema=schema,
                allowed_columns=C_sql,
                linked_values=[],
                retrieval_evidence=schema_evidence,
            )
            y_t = self.verifier.fuse(
                sub_query=sub_q.sub_query,
                original_question=original_question,
                route=route.value,
                text_evidence=text_evidence,
                schema_cell_evidence=schema_evidence,
                sql_result=sql_result,
                expected_operator=sub_q.expected_operator,
                table_chunks_md=_table_chunks_to_markdown(chunks),
                legacy_fast_path=self.legacy_fast_path,
            )
            return y_t, sql_result, schema_evidence

        # ── RETRIEVE / HYBRID: Schema + Cell retrieval first ─────────────────
        C, V_raw = self.index.schema_cell_retrieval(
            query=sub_q.sub_query,
            entity_mentions=sub_q.entity_mentions,
            schema_top_k=SCHEMA_TOP_K,
            cell_top_k=CELL_TOP_K,
        )
        if self.no_value_linker:
            # Ablation: bypass HybridValueLinker. Pass empty linked_values to
            # ConstrainedSQL so the SQL is generated only with column constraint
            # C (from schema retrieval) and no value constraint V^*. Isolates
            # the contribution of value linking against the lexical-mismatch
            # failure mode.
            linked_values = []
        else:
            linked_values = self.value_linker.link(
                entity_mentions=sub_q.entity_mentions,
                schema_columns=C,
                V_raw=V_raw,
                history_H=H,
            )

        schema_cell_ev = "\n".join([
            _schema_entries_to_text(C),
            _cell_entries_to_text(linked_values),
        ])

        # Check for reroute signal from value linker
        if any(lv.needs_reroute for lv in linked_values):
            logger.info("ValueLinker signalled reroute → TEXT fallback")
            y_t = self.verifier.answer_from_text(
                sub_q.sub_query, text_evidence, original_question=original_question
            )
            return y_t, sql_result, schema_cell_ev

        # ── RETRIEVE (Lines 12–15) ───────────────────────────────────────────
        # Phase F-A (Option A): pass View 4 row-component (RowIndex) rows to the synthesis
        # prompt: each row contains every column for one entity, so the LLM
        # can extract the target column directly. Falls back to View 2 (table
        # chunks) when the RowIndex is unavailable (legacy index).
        if route == Route.RETRIEVE:
            schema_info = _schema_entries_to_text(C)
            cell_info = _cell_entries_to_text(linked_values)

            # Prefer View 4 row-component (row-level) over View 2 (chunk-level)
            table_context = ""
            if getattr(self.index, "row_index", None) and self.index.row_index.entries:
                table_context = _retrieve_row_context(
                    self.index,
                    sub_q.sub_query,
                    chunks,
                    top_k=ROW_TOP_K,
                )

            if not table_context:
                table_context = _table_chunks_to_markdown(chunks)

            y_t = self.verifier.answer_from_retrieval(
                sub_query=sub_q.sub_query,
                original_question=original_question,
                schema_info=schema_info,
                cell_info=cell_info,
                table_context=table_context,
            )
            return y_t, sql_result, schema_cell_ev

        # ── HYBRID (Lines 18–21) ─────────────────────────────────────────────
        # D1 + E8 augmentation: arithmetic hint + type hint on the sub_query
        # passed to ConstrainedSQLExecutor. Column constraint (E1) is already
        # provided via `allowed_columns=C` in the constrained template.
        augmented_sub_q = _augment_sql_query(
            sub_q.sub_query,
            schema_entries=C,
            enable_arith=True,
            enable_schema_hint=False,   # E1 already covered by ConstrainedSQL template
            enable_type_hint=True,
        )
        sql_result, c_exec = sql_executor.execute(
            sub_query=augmented_sub_q,
            schema=schema,
            allowed_columns=C,
            linked_values=linked_values,
            retrieval_evidence=schema_cell_ev,
        )
        y_t = self.verifier.fuse(
            sub_query=sub_q.sub_query,
            original_question=original_question,
            route=route.value,
            text_evidence=text_evidence,
            schema_cell_evidence=schema_cell_ev,
            sql_result=sql_result,
            expected_operator=sub_q.expected_operator,
            table_chunks_md=_table_chunks_to_markdown(chunks),
            legacy_fast_path=self.legacy_fast_path,
        )
        return y_t, sql_result, schema_cell_ev

class BatchRunner:
    """
    Batch inference runner with multi-threading support.
    Mirrors the `TableRAG.run()` interface from TableRAG (Yu et al., 2025).
    """

    def __init__(self, agent: AgenticTableRAGAgent) -> None:
        self.agent = agent

    def run(
        self,
        data_file: str,
        save_file: str,
        max_workers: int = 2,
        rerun: bool = False,
        emit_trace: bool = False,
    ) -> None:
        with open(data_file, "r", encoding="utf-8") as f:
            if data_file.endswith(".jsonl"):
                cases = [json.loads(line) for line in f if line.strip()]
            else:
                cases = json.load(f)

        if rerun and os.path.exists(save_file):
            with open(save_file, "r", encoding="utf-8") as f:
                done_questions = {
                    json.loads(line)["question"]
                    for line in f if line.strip()
                }
            logger.info(f"Resume: {len(done_questions)} questions already done in {save_file}")
        else:
            done_questions = set()

        os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
        file_lock = threading.Lock()

        def process(case: Dict):
            q = case.get("question")
            if q in done_questions:
                return None
            qid = case.get("question_id", case.get("question", "?"))
            table_names = [case.get("table_id", "")] if case.get("table_id") else []
            gold = case.get("answer-text") or case.get("answer") or case.get("answers") or ""
            if isinstance(gold, list):
                gold = " ".join(str(g) for g in gold)
            trace: Optional[Dict[str, Any]] = {} if emit_trace else None
            try:
                answer = self.agent.run_single(
                    question=case["question"],
                    table_name_list=table_names,
                    question_id=str(qid),
                    gold_answer=str(gold) if gold else None,
                    trace=trace,
                )
            except Exception as exc:
                logger.error(f"[{qid}] Error: {exc}")
                traceback.print_exc()
                answer = "error"
            result = {**case, "agentic_tablerag_answer": answer}
            if trace is not None:
                result["atr_trace"] = trace
            return result

        with open(save_file, "a" if rerun else "w", encoding="utf-8") as fout:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(process, case): case for case in cases}
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(cases),
                    desc="Agentic TableRAG",
                ):
                    try:
                        result = future.result()
                        if result is None:
                            continue
                        with file_lock:
                            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                            fout.flush()
                    except Exception as exc:
                        logger.error(f"Future failed: {exc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic TableRAG online inference")
    parser.add_argument("--backbone", type=str, default="gemini")
    parser.add_argument(
        "--decomposer_backbone", type=str, default=None,
        help="Ablation: drive ONLY the query decomposer with this backbone "
             "(router/executor/verifier/synthesis stay on --backbone). "
             "Isolates decomposer-model sensitivity. Default None = same as --backbone.",
    )
    parser.add_argument(
        "--verifier_backbone", type=str, default=None,
        help="Ablation: drive ONLY the verifier verdict with this backbone "
             "(fusion, route answering, decomposition, routing stay on --backbone). "
             "Isolates verifier-model sensitivity. Default None = same as --backbone.",
    )
    parser.add_argument(
        "--oracle_verifier", action="store_true",
        help="Ablation: upper bound. After a question runs normally, return any "
             "produced candidate that matches the gold answer, measuring the "
             "accuracy ceiling a perfect verifier could reach.",
    )
    parser.add_argument("--data_file_path", type=str, required=True)
    parser.add_argument("--save_file_path", type=str, default="")
    parser.add_argument("--index_path", type=str, required=True,
                        help="Path prefix for the saved MultiviewIndex (from build_index.py)")
    parser.add_argument("--bge_dir", type=str, default="BAAI",
                        help="Directory containing bge-m3 and bge-reranker-v2-m3")
    parser.add_argument(
        "--reranker_path", type=str, default=None,
        help="Cross-encoder model path (default: <bge_dir>/bge-reranker-v2-m3)",
    )
    parser.add_argument(
        "--rerank_candidate_multiplier", type=int,
        default=RERANK_CANDIDATE_MULTIPLIER,
        help="Dense candidates recalled per result before cross-encoder reranking",
    )
    parser.add_argument(
        "--no_reranker", action="store_true",
        help="Ablation/debug only: disable the BGE cross-encoder reranker",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--max_iter", type=int, default=MAX_ITER)
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Parallel dataset-question workers; sub-queries stay sequential",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--emit_trace", action="store_true",
        help="Write per-question verifier decisions, router inputs, and "
             "failure histories under `atr_trace` for router distillation.",
    )
    parser.add_argument(
        "--router_type", type=str, default="heuristic",
        choices=["heuristic", "llm", "learned", "fixed"],
        help="Router variant: heuristic (rule-based) | llm (prompt-based) | learned (DistilBERT) | fixed (ablation)",
    )
    parser.add_argument(
        "--force_route", type=str, default=None,
        choices=["TEXT", "SQL", "RETRIEVE", "HYBRID"],
        help="Ablation: force one route (only used with --router_type=fixed)",
    )
    parser.add_argument(
        "--no_decomposition", action="store_true",
        help="Ablation: bypass decomposer; treat full question as single sub-query and cap max_iter=1",
    )
    parser.add_argument(
        "--final_synthesis", action="store_true",
        help="Improvement: after the iterative loop, run a final synthesis step that takes the original question + all sub-answers to produce the final answer",
    )
    parser.add_argument(
        "--no_escalation", action="store_true",
        help="Ablation: disable verifier-driven route escalation. Use the router's "
             "first choice only (no fallback to other routes on verifier reject). "
             "Isolates router contribution from the escalation safety net.",
    )
    parser.add_argument(
        "--fixed_escalate_chain", type=str, default=None,
        choices=[None, "standard"],
        help="Only meaningful with --router_type=fixed. None (default) keeps "
             "the legacy 'stay on fixed route' behaviour. 'standard' cycles "
             "through the other 3 routes on each escalate() call, measures "
             "'first choice = X, with full escalation safety net'.",
    )
    parser.add_argument(
        "--no_value_linker", action="store_true",
        help="Ablation: bypass HybridValueLinker for RETRIEVE/HYBRID routes. "
             "Pass empty linked_values to ConstrainedSQL so only the column "
             "constraint C (from schema retrieval) is applied, no value "
             "constraint V*. Isolates the value-linking contribution against "
             "the lexical-mismatch failure mode (e.g., 'tv' vs 'television').",
    )
    parser.add_argument(
        "--router_model_path", type=str, default=None,
        help="Path to fine-tuned DistilBERT router model (required when --router_type=learned)",
    )
    parser.add_argument("--router_device", type=str, default="cuda",
                        help="Device for learned router: cpu | cuda | cuda:0 "
                             "(default cuda: DistilBERT 110M params, ~1ms/forward "
                             "vs ~10ms on CPU. Use cpu only when GPU memory tight.)")
    # Round-5 prompt fixes: default OFF (smoke 30Q ablation showed +10pp from
    # cell-index fix alone but -3.3pp regression when prompt fixes layered on top).
    # Kept as opt-in flags for paper-grade ablation. To reproduce post-fix
    # behavior (decomposer schema preview + markdown fusion) pass both flags.
    parser.add_argument("--use_schema_preview", action="store_true",
                        help="#1: pre-retrieve once; inject column headers + "
                             "sample rows into decomposer prompt. Default OFF.")
    parser.add_argument("--use_markdown_fusion", action="store_true",
                        help="#3+(b): conditional fast-path + EVIDENCE_FUSION_PROMPT "
                             "with markdown chunks (T2025 COMBINE pattern). When "
                             "OFF (default), verifier uses unconditional SQL scalar "
                             "fast-path (legacy).")
    parser.add_argument("--verifier_threshold", type=float, default=VERIFIER_THRESHOLD,
                        help="§3.6 Stop Controller: exit early when verified + uncertainty < threshold")
    args, _ = parser.parse_known_args()

    from datetime import datetime
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save_file_path:
        if not args.rerun:
            p = Path(args.save_file_path)
            args.save_file_path = str(p.parent / f"{p.stem}_{_ts}{p.suffix}")
    else:
        args.save_file_path = str(
            Path(__file__).resolve().parent.parent / "output" / f"{args.router_type}_results_{_ts}.jsonl"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    _logger = init_logger(name="agentic_tablerag", level=logging.INFO, log_file=None)

    bge_model_path = os.path.join(args.bge_dir, "bge-m3")
    index = MultiviewIndex.load(
        save_path=args.index_path,
        bge_model_path=bge_model_path,
        device=args.device,
        require_cuda=args.require_cuda,
        reranker_model_path=args.reranker_path,
        enable_reranker=not args.no_reranker,
        rerank_candidate_multiplier=args.rerank_candidate_multiplier,
    )

    agent = AgenticTableRAGAgent(
        multiview_index=index,
        backbone=args.backbone,
        max_iter=args.max_iter,
        router_type=args.router_type,
        router_model_path=args.router_model_path,
        router_device=args.router_device,
        verifier_threshold=args.verifier_threshold,
        force_route=args.force_route,
        fixed_escalate_chain=args.fixed_escalate_chain,
        no_value_linker=args.no_value_linker,
        no_decomposition=args.no_decomposition,
        decomposer_backbone=args.decomposer_backbone,
        verifier_backbone=args.verifier_backbone,
        oracle_verifier=args.oracle_verifier,
        final_synthesis=args.final_synthesis,
        # Default OFF: opt-in via --use_schema_preview / --use_markdown_fusion.
        # Internal agent attribute names kept inverted for code-path clarity.
        no_schema_preview=not args.use_schema_preview,
        legacy_fast_path=not args.use_markdown_fusion,
        no_escalation=args.no_escalation,
    )

    start = time.time()
    BatchRunner(agent).run(
        data_file=args.data_file_path,
        save_file=args.save_file_path,
        max_workers=args.max_workers,
        rerun=args.rerun,
        emit_trace=args.emit_trace,
    )
    print(f"Done in {time.time() - start:.1f}s")
