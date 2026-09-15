"""Adaptive query decomposition: emits the next sub-query and routing metadata."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from online.parsing import parse_json_object
from prompt import DECOMPOSER_PROMPT

logger = logging.getLogger(__name__)

TERMINATE = "TERMINATE"


@dataclass
class SubQuery:
    """Structured output of a single decomposition step."""

    sub_query: str
    required_modalities: str = "both"  # text | table | both
    expected_operator: str = (
        "lookup"  # lookup|filter|count|compare|aggregate|sort|arithmetic
    )
    need_global_table_view: bool = False
    entity_mentions: List[str] = field(default_factory=list)
    uncertainty: float = 0.0

    @property
    def is_terminate(self) -> bool:
        return self.sub_query.strip().upper() == TERMINATE


def _parse_llm_output(text: str) -> Dict[str, Any]:
    """Extract the decomposer's JSON object, allowing nested metadata."""
    return parse_json_object(text, greedy=True)


class QueryDecomposer:
    """
    §3.3  Adaptive iterative query decomposer.

    qt ~ p(q | q, z_<t)

    Each call to `decompose()` produces the next SubQuery given the current
    intermediate-answer history.
    """

    def __init__(self, llm_fn: Callable[[List[Dict]], str]) -> None:
        """
        Args:
            llm_fn: callable(messages) → response_text string.
        """
        self.llm_fn = llm_fn

    def decompose(
        self,
        question: str,
        history: List[Dict[str, str]],
        table_id: str = "",
        consecutive_failures: int = 0,
        schema_preview: str = "",
    ) -> SubQuery:
        """
        Generate the next sub-query (or TERMINATE).

        Args:
            question:             original user question q
            history:              list of {sub_query, answer, failed} dicts
            table_id:             known table identifier hint
            consecutive_failures: number of consecutive "not found" steps so far
            schema_preview:       optional pre-formatted "column headers + sample
                                  rows" string from the candidate table, lets the
                                  decomposer disambiguate cell-lookup vs COUNT(*)
                                  patterns. Empty string disables the hint.
        """
        history_text = self._format_history(history)
        table_hint = (
            f"\nHint: The answer is expected to be found in the table: {table_id}\n"
            if table_id
            else ""
        )

        # Failure recovery hint injected when consecutive failures detected
        failure_hint = ""
        if consecutive_failures == 1:
            failure_hint = (
                "\n⚠ RECOVERY HINT: The previous retrieval step returned 'not found'. "
                "Your next sub-query MUST try a fundamentally different approach:\n"
                "  • Rephrase using synonyms or broader terms.\n"
                "  • Ask for a related intermediate fact that may unlock the answer.\n"
                "  • Switch between table-based and text-based phrasing.\n"
                "  • Do NOT repeat or trivially rephrase the failed question.\n"
            )
        elif consecutive_failures >= 2:
            failure_hint = (
                "\n⚠ RECOVERY HINT: Multiple consecutive steps returned 'not found'. "
                "Consider one of these recovery strategies:\n"
                "  • Decompose the original question differently from the start.\n"
                "  • Ask a broader question that captures the whole answer in one step.\n"
                "  • If partial information is available in history, synthesise directly "
                "and use TERMINATE — a partial answer is better than 'not found'.\n"
                "  • Try a text-based search (required_modalities='text') as a last resort.\n"
            )

        prompt = DECOMPOSER_PROMPT.format(
            question=question,
            table_hint=table_hint + failure_hint,
            schema_preview=schema_preview,
            history=history_text or "(none yet)",
        )
        messages = [{"role": "user", "content": prompt}]
        retry_suffix = "\n\nIMPORTANT: respond with ONLY a single valid JSON object — no prose, no markdown, no commentary."
        parsed = None
        response_text = ""
        for attempt in range(2):
            try:
                response_text = self.llm_fn(messages)
            except Exception as exc:
                logger.error(f"Decomposer LLM call failed: {exc}")
                return SubQuery(sub_query=TERMINATE)
            parsed = _parse_llm_output(response_text)
            if isinstance(parsed.get("sub_query"), str) and parsed["sub_query"].strip():
                break
            logger.warning(
                f"Decomposer failed to parse JSON (attempt {attempt + 1}). "
                f"Raw: {response_text[:200]}"
            )
            messages = [{"role": "user", "content": prompt + retry_suffix}]
        if (
            not parsed
            or not isinstance(parsed.get("sub_query"), str)
            or not parsed["sub_query"].strip()
        ):
            return SubQuery(sub_query=TERMINATE)

        # Missing/malformed uncertainty is NOT evidence of completion.
        # Preserve valid zero but fail conservatively to maximum uncertainty.
        raw_unc = parsed.get("uncertainty")
        try:
            unc = float(raw_unc) if not isinstance(raw_unc, bool) else 1.0
        except (TypeError, ValueError, OverflowError):
            unc = 1.0
        if not math.isfinite(unc) or not 0.0 <= unc <= 1.0:
            unc = 1.0
        entities = parsed.get("entity_mentions")
        if isinstance(entities, str):
            entities = [entities]
        if not isinstance(entities, list):
            entities = []
        entities = list(
            dict.fromkeys(
                e.strip() for e in entities if isinstance(e, str) and e.strip()
            )
        )
        modalities = parsed.get("required_modalities")
        if modalities not in ("text", "table", "both"):
            modalities = "both"
        operator = parsed.get("expected_operator")
        if operator not in (
            "lookup",
            "filter",
            "count",
            "compare",
            "aggregate",
            "sort",
            "arithmetic",
        ):
            operator = "lookup"
        return SubQuery(
            sub_query=str(parsed.get("sub_query", TERMINATE)),
            required_modalities=modalities,
            expected_operator=operator,
            need_global_table_view=str(
                parsed.get("need_global_table_view", "no")
            ).lower()
            in ("yes", "true"),
            entity_mentions=entities,
            uncertainty=unc,
        )

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        """Emit failed sub-queries explicitly so the LLM can see
        which queries have already been tried and avoid paraphrasing them.

        Each step shows: Q (verbatim), A (answer or ⚠), and a banner listing
        all previously failed sub-query texts. The decomposer can then
        check semantic novelty before emitting a new sub-query.
        """
        if not history:
            return ""
        lines = []
        failed_queries = []
        for i, entry in enumerate(history, 1):
            ans = entry.get("answer", "")
            sq = entry.get("sub_query", "")
            failed = entry.get("failed", False) or ans.strip().lower() in (
                "not found",
                "",
                "error",
            )
            lines.append(f"Step {i}: Q: {sq}")
            if failed:
                lines.append(
                    "        A: NOT FOUND, this retrieval approach yielded no answer"
                )
                if sq:
                    failed_queries.append(sq)
            else:
                lines.append(f"        A: {ans}")
        if failed_queries:
            lines.append("")
            lines.append("⚠ FAILED SUB-QUERIES (do NOT repeat or paraphrase):")
            for fq in failed_queries:
                lines.append(f"  - {fq}")
        return "\n".join(lines)
