"""Evidence fusion and verifier: synthesises a candidate answer and validates it (0 / 0.5 / 1); rejection triggers route escalation SQL → HYBRID → RETRIEVE."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from atr.prompt import EVIDENCE_FUSION_PROMPT, VERIFIER_PROMPT, TEXT_ANSWER_PROMPT, RETRIEVE_ANSWER_PROMPT

logger = logging.getLogger(__name__)

def _normalize_for_grounding(s: str) -> str:
    """Normalize text for substring grounding check.

    Lowercase, strip thousands separators + currency, collapse whitespace,
    keep alphanumerics + decimal + hyphen. Designed to make numeric
    formatting differences ("9,461,105" vs "9461105") match.
    """
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[,\$£€¥]", "", s)
    s = re.sub(r"[^a-z0-9\.\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_numerics(s: str) -> List[float]:
    """Extract all numeric values from a string for numeric grounding."""
    if not s:
        return []
    nums = []
    for m in re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", "")):
        try:
            nums.append(float(m))
        except ValueError:
            pass
    return nums

def _is_grounded(answer: str, evidence: str) -> tuple:
    """Programmatic grounding check (Option D in verifier strengthening).

    Returns (grounded: bool, reason: str). The verdict pipeline demotes
    LLM V=1 to V=0.5 when answer is NOT grounded, i.e., the answer
    string (or its numeric value) cannot be found in the evidence
    pool. Catches pure LLM hallucinations where the LLM judge wrongly
    accepts a fabricated answer.
    """
    if not answer:
        return True, "empty_answer"  # let LLM judge handle empty
    if answer.strip().lower() in ("not found", ""):
        return True, "not_found"  # not-found is its own valid case

    a = _normalize_for_grounding(answer)
    e = _normalize_for_grounding(evidence)

    # Single-char answers ("0", "Y", "N") must literally
    # appear in evidence: don't auto-pass. Previous "too_short" auto-pass
    # let "0" through for questions like "how many?" even when evidence
    # had no "0" anywhere, falsely keeping a wrong V=1.
    if len(a) < 2:
        if a in e:
            return True, "single_char_match"
        # Numeric single-char: also check numeric grounding below before failing
        if a.isdigit():
            e_nums = _extract_numerics(evidence)
            if float(a) in e_nums:
                return True, "single_char_numeric_match"
        return False, "single_char_not_in_evidence"

    # 1. Direct substring (case-insensitive, normalized)
    if a in e:
        return True, "substring"

    # 2. Numeric grounding (with 1% tolerance for floating-point)
    a_nums = _extract_numerics(answer)
    if a_nums:
        e_nums = _extract_numerics(evidence)
        for an in a_nums:
            for en in e_nums:
                if an == en:
                    return True, f"numeric_exact:{an}"
                denom = max(abs(an), abs(en), 1e-9)
                if abs(an - en) / denom < 0.01:
                    return True, f"numeric_close:{an}~{en}"

    # 3. Token overlap (paraphrase tolerance), answer's content tokens
    #    must mostly appear in evidence. The threshold is 0.65 rather than 0.75
    #    to admit legitimate paraphrases like "President of US" vs "US President".
    #    Also accept the single-token case: if the answer is one content token
    #    and it appears in the evidence, treat as grounded.
    a_toks = {t for t in a.split() if len(t) >= 3}
    if a_toks:
        e_toks = set(e.split())
        overlap = a_toks & e_toks
        if len(a_toks) == 1 and len(overlap) == 1:
            return True, "single_token_match"
        if len(overlap) / len(a_toks) >= 0.65:
            return True, f"token_overlap:{len(overlap)}/{len(a_toks)}"

    return False, "not_grounded"

def _extract_sql_scalar(sql_result: str) -> Optional[str]:
    """
    If sql_result contains exactly one scalar value, return it directly.
    Bypasses LLM fusion to prevent paraphrasing (e.g., "9.5 million" instead of "9,461,105").

    Handles common SQL result formats:
      - [[value]]  or  [value]          (nested list)
      - [{"col": value}]                (list of dicts, single row single col)
      - plain string scalar
    Returns None if result is multi-row, multi-column, or ambiguous.
    """
    if not sql_result or not sql_result.strip():
        return None
    s = sql_result.strip()

    # Try JSON parse
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        # Plain string: return as-is only if it's a single token
        if "\n" not in s and len(s) < 200:
            return s.strip()
        return None

    # Unwrap [[value]] → [value]
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], list):
        parsed = parsed[0]

    # [value]: single element list of scalar
    if isinstance(parsed, list) and len(parsed) == 1 and not isinstance(parsed[0], (list, dict)):
        return str(parsed[0]).strip()

    # [{"col": value}]: single row, single column
    if (
        isinstance(parsed, list)
        and len(parsed) == 1
        and isinstance(parsed[0], dict)
        and len(parsed[0]) == 1
    ):
        val = next(iter(parsed[0].values()))
        if not isinstance(val, (list, dict)):
            return str(val).strip()

    # {"col": value}: single dict with one key
    if isinstance(parsed, dict) and len(parsed) == 1:
        val = next(iter(parsed.values()))
        if not isinstance(val, (list, dict)):
            return str(val).strip()

    return None

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

def _extract_answer_tag(text: str) -> str:
    """Pull out the value after '<Answer>:' if present, else return full text."""
    match = re.search(r"<\s*[Aa]nswer\s*>\s*:?\s*(.+)$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

VERDICT_SUPPORT = 1.0
VERDICT_INSUFFICIENT = 0.5
VERDICT_CONFLICT = 0.0

class EvidenceFusionVerifier:
    """
    §3.6  Combined Evidence Fusion + Verifier.

    Fusion:  synthesises a candidate answer y_t from multi-route evidence.
    Verify:  LLM-as-a-judge returns V(r) ∈ {0, 0.5, 1}.
    """

    def __init__(
        self,
        llm_fn: Callable[[List[Dict]], str],
        verify_llm_fn: Optional[Callable[[List[Dict]], str]] = None,
    ) -> None:
        self.llm_fn = llm_fn
        # The verdict (verify()) may run on a separate backbone for the
        # verifier-sensitivity ablation; fusion and route answering always
        # use llm_fn so only the verdict model varies.
        self.verify_llm_fn = verify_llm_fn or llm_fn

    # ── Evidence Fusion ──────────────────────────────────────────────────────

    def fuse(
        self,
        sub_query: str,
        original_question: str,
        route: str,
        text_evidence: str,
        schema_cell_evidence: str,
        sql_result: str,
        expected_operator: str = "lookup",
        table_chunks_md: str = "",
        legacy_fast_path: bool = False,
    ) -> str:
        """
        Synthesise evidence from all available sources into a candidate answer.

        st ~ p_θ(s | q_t, T, C, V*, E_text)

        Conditional fast-path (default): scalar SQL result is returned
        directly only when the operator is a simple lookup/filter AND the
        scalar is non-zero. For count/aggregate/arithmetic OR scalar in
        {"0", "[]", null}, the LLM fusion runs with markdown table chunks
        (Source D) for T2025-style COMBINE cross-validation.

        Legacy fast-path (`legacy_fast_path=True`, ablation): unconditional
        scalar return: reproduces pre-#3+(b) behavior.
        """
        if sql_result and sql_result.strip() not in ("", "(none)"):
            scalar = _extract_sql_scalar(sql_result)
            if scalar:
                if legacy_fast_path:
                    logger.debug(f"fuse: legacy fast-path → '{scalar}'")
                    return scalar
                # Conditional fast-path
                suspicious_scalar = {"0", "0.0", "[]", "null", "none", "None"}
                simple_lookup = expected_operator in ("lookup", "filter")
                if simple_lookup and scalar.strip() not in suspicious_scalar:
                    logger.debug(
                        f"fuse: SQL scalar fast-path → '{scalar}' "
                        f"(op={expected_operator})"
                    )
                    return scalar
                logger.debug(
                    f"fuse: scalar='{scalar}' op={expected_operator}, "
                    f"running LLM fusion with markdown cross-check"
                )

        prompt = EVIDENCE_FUSION_PROMPT.format(
            sub_query=sub_query,
            original_question=original_question or sub_query,
            text_evidence=text_evidence or "(none)",
            schema_cell_evidence=schema_cell_evidence or "(none)",
            sql_result=sql_result or "(none)",
            route=route,
            table_chunks_md=table_chunks_md or "(none)",
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_fn(messages)
        except Exception as exc:
            logger.error(f"Evidence fusion LLM call failed: {exc}")
            return ""
        return _extract_answer_tag(response)

    def answer_from_text(
        self,
        sub_query: str,
        text_evidence: str,
        original_question: str = "",
    ) -> str:
        """TEXT route: answer directly from document chunks."""
        prompt = TEXT_ANSWER_PROMPT.format(
            evidence=text_evidence or "(none)",
            question=sub_query,
            original_question=original_question or sub_query,
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_fn(messages)
        except Exception as exc:
            logger.error(f"Text answer LLM call failed: {exc}")
            return "not found"
        return _extract_answer_tag(response)

    def answer_from_retrieval(
        self,
        sub_query: str,
        schema_info: str,
        cell_info: str,
        original_question: str = "",
        table_context: str = "",
    ) -> str:
        """RETRIEVE route: synthesise from Schema/Cell retrieval + View 2 table chunks.

        Phase F (Option C): augments View 3 (schema) + View 4 (cells) with View 2
        (table chunks) so the LLM sees row-level context, not just isolated cell
        values. Closes the standalone-V=1 gap with TableRAG (Chen et al., 2024)'s retriever role.
        """
        prompt = RETRIEVE_ANSWER_PROMPT.format(
            schema_info=schema_info or "(none)",
            cell_info=cell_info or "(none)",
            table_context=table_context or "(none)",
            question=sub_query,
            original_question=original_question or sub_query,
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_fn(messages)
        except Exception as exc:
            logger.error(f"Retrieval answer LLM call failed: {exc}")
            return "not found"
        return _extract_answer_tag(response)

    # ── Verifier ─────────────────────────────────────────────────────────────

    def verify(
        self,
        question: str,
        answer: str,
        text_evidence: str,
        sql_result: str,
        route_evidence: str = "",
    ) -> float:
        """
        LLM-as-a-judge verification over all evidence used by the route.

        ``text_evidence`` contains the shared document chunks, while
        ``route_evidence`` carries route-specific context such as retrieved
        rows or schema/cell bindings.

        Returns:
            V(r) ∈ {0, 0.5, 1}, conflict / insufficient / support
        """
        if not answer or answer.strip().lower() in ("not found", ""):
            return VERDICT_INSUFFICIENT

        evidence_combined = "\n\n".join(
            filter(None, [text_evidence, route_evidence, sql_result])
        )
        prompt = VERIFIER_PROMPT.format(
            question=question,
            answer=answer,
            evidence=evidence_combined or "(none)",
            sql_result=sql_result or "(none)",
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.verify_llm_fn(messages)
        except Exception as exc:
            logger.error(f"Verifier LLM call failed: {exc}")
            return VERDICT_INSUFFICIENT

        parsed = _parse_json(response)
        raw_verdict = parsed.get("verdict", 0.5)
        try:
            verdict = float(raw_verdict)
        except (TypeError, ValueError):
            verdict = VERDICT_INSUFFICIENT

        reason = parsed.get("reason", "")
        logger.debug(f"Verifier verdict={verdict}, reason={reason}")

        # Phase G+ (Priority 4): programmatic grounding check (Option D).
        # If the LLM judge said V=1, verify that the answer is actually
        # findable in the evidence pool (substring, numeric, or token
        # overlap). Demotes false-positive V=1 to V=0.5 so escalation
        # can fire on hallucinated/wrong-row answers that the LLM judge
        # accepted erroneously.
        if verdict >= VERDICT_SUPPORT:
            grounded, ground_reason = _is_grounded(answer, evidence_combined)
            if not grounded:
                logger.info(
                    f"Verifier V=1 → V=0.5 demoted (ungrounded): "
                    f"answer='{answer[:80]}' reason={ground_reason}"
                )
                verdict = VERDICT_INSUFFICIENT
            else:
                logger.debug(f"V=1 grounded via {ground_reason}")

        return verdict

    def is_rejected(self, verdict: float) -> bool:
        """Reject if conflict (0) or insufficient (0.5)."""
        return verdict < VERDICT_SUPPORT
