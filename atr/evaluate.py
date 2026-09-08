"""
ATR Evaluation Script

Metrics:
  - Exact Match (EM)
  - Normalized EM  (NFKC lower, strip punct, strip articles)
  - Relaxed EM     (substring match)
  - LLM Judge      (Gemini binary 0/1, same EVALUATION_PROMPT as TableRAG (Yu et al., 2025))

Output: Excel (.xlsx) per-row detail + printed summary.

Usage:
  # Rule metrics only (fast, no API call)
  python evaluate.py --result_file_path output/llm_router_results.jsonl --skip_llm_eval

  # Rule + LLM judge
  python evaluate.py --result_file_path output/llm_router_results.jsonl

  # Save to specific file
  python evaluate.py --result_file_path output/llm_router_results.jsonl \\
      --output_file output/eval_llm_router.xlsx
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── EVALUATION_PROMPT (identical to the released TableRAG prompt file) ─────────────────
EVALUATION_PROMPT = """\
We would like to request your feedback on the performance of the AI assistant in response \
to the user question displayed above according to the gold answer. \
Please use the following listed aspects and their descriptions as evaluation criteria:
    - Accuracy and Hallucinations: The assistant's answer is semantically consistent with the \
gold answer; The numerical value and order need to be accurate, and there should be no hallucinations.
    - Completeness: Referring to the reference answers, the assistant's answer should contain \
all the key points needed to answer the user's question; further elaboration on these key points can be omitted.
Please rate whether this answer is suitable for the question. \
Please note that the gold answer can be considered as a correct answer to the question.

The assistant receives an overall score on a scale of 0 OR 1, where 0 means wrong and 1 means correct.
Directly output a line indicating the score of the Assistant.

PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS 0 OR 1 BY STRICTLY FOLLOWING THIS FORMAT: \
"[[score]]", FOR EXAMPLE "Rating: [[1]]":
<start output>
Rating: [[score]]
<end output>

[Question]
{question}

[Gold Answer]
{golden}

[The Start of Assistant's Predicted Answer]
{gen}
"""

# ── Strict EVALUATION_PROMPT: for rigorous reporting ────────────────────────
# Distinguishing properties vs the lenient prompt above:
#  1. Hallucination penalty is binding (any unsupported claim → 0).
#  2. Numeric answers must match within ±1 % AND preserve sign and unit equivalence.
#  3. Multi-span gold requires all spans to be present (incomplete → 0).
#  4. Gold value must appear as a standalone token, not as a substring of unrelated
#     text (e.g., gold "1" inside pred "...2019..." is NOT a valid match).
EVALUATION_PROMPT_STRICT = """\
You are a strict QA evaluator. Score the assistant's prediction against the gold answer
with the criteria below. ALL criteria must pass for score 1; any failure → score 0.

CRITERIA:

1. SEMANTIC EQUIVALENCE
   The prediction must convey the SAME information as the gold answer. Different
   surface forms of the same fact are acceptable (paraphrase, synonym, equivalent
   notation), but the underlying claim must match.

2. NO HALLUCINATION / NO EXTRANEOUS CLAIMS
   The prediction must NOT include factual claims absent from the gold answer or
   the source evidence. Score 0 if the prediction adds:
     - numbers, entities, dates, or amounts not in the gold,
     - explanatory commentary that introduces new facts,
     - a different answer alongside the correct one (mixed answers).
   Wrapping a correct answer in narrative ("There were X items in 2019 ...") is
   acceptable ONLY if the narrative does not assert anything beyond the gold.

3. NUMERIC PRECISION  (when answer is numeric)
   - Value must match within ±1 % of the gold (allow rounding).
   - Sign must match (negative ≠ positive change).
   - Unit must be equivalent. Examples of OK pairs:
       "172 million" ≡ "172M" ≡ "$172,000,000"
       "6.67 percent" ≡ "6.67%"
     Examples of NOT OK:
       "172"          ≠ "172 million"     (unit dropped)
       "172 thousand" ≠ "172 million"     (different scale)
       "+12 %"        ≠ "-12 %"           (sign flipped)

4. COMPLETENESS
   - For multi-span gold (e.g. "A | B | C" or list answers), the prediction must
     contain ALL spans. Missing any span → 0.
   - For single-span gold, the prediction must contain the gold value as a standalone
     unit, not as a substring of an unrelated number/word.
       Example: gold "1" inside pred "There were 3 in 2019" → 0
                (the "1" in "2019" is NOT a standalone match)

5. RELEVANCE
   The prediction must answer the SPECIFIC question asked. A correct fact about a
   different entity / year / column → 0.

OUTPUT FORMAT (binary 0 or 1):
"Rating: [[0]]" if any criterion fails.
"Rating: [[1]]" if all criteria pass.

Output exactly one line:
<start output>
Rating: [[score]]
<end output>

[Question]
{question}

[Gold Answer]
{golden}

[The Start of Assistant's Predicted Answer]
{gen}
"""

# ── Answer field name (our inference saves as agentic_tablerag_answer) ────────
_PRED_FIELD = "agentic_tablerag_answer"

# ── Text cleaning (from hybrid_eval.py) ──────────────────────────────────────

def clean_answer_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    # Drop literal None / null markers (degraded answers)
    if s.lower() in {"none", "null", "n/a", "na"}:
        return ""

    # Strip output markers used by ATR final synthesis / verifier
    s = re.sub(r"^\s*<\s*[Ff]inal[_ ]?[Aa]nswer\s*>\s*:?\s*", "", s)
    s = re.sub(r"^\s*<\s*[Aa]nswer\s*>\s*:?\s*", "", s)
    # Strip preamble phrases: only when present at the very start
    s = re.sub(
        r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|to[a-z\s]*?(?:question|query)\s*is)?\s*[:\-]?\s*",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(
        r"^\s*based\s+on\s+(?:the\s+)?(?:provided\s+)?(?:information|table|data|context|evidence)[^,]*,\s*",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(r"^\s*according\s+to[^,]*,\s*", "", s, flags=re.I)
    # Drop "Sub-query N:" / "Step N:" headers (multi-iter leak)
    s = re.sub(r"^\s*(?:sub[\-\s]query|step)\s*\d+\s*[:\.]\s*", "", s, flags=re.I)
    # Generic leading colon/dash
    s = re.sub(r"^\s*[:\-]\s*", "", s)
    # Trailing sentence terminator
    s = re.sub(r"[\.\!]\s*$", "", s)
    # Trailing "is the answer" tail
    s = re.sub(r"\s+is\s+(?:the\s+)?(?:final\s+)?answer\.?\s*$", "", s, flags=re.I)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Strip wrapping quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_SCALE_WORDS = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}

def _normalize_numbers(text: str) -> str:
    """Convert word numbers and scale suffixes to plain integers."""
    # Remove commas from numbers: "9,461,105" → "9461105"
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    # "9.5 million" / "1.2 billion" → integer
    def _scale_sub(m: re.Match) -> str:
        num = float(m.group(1))
        scale = _SCALE_WORDS[m.group(2).lower()]
        return str(int(num * scale))
    text = re.sub(
        r"([\d.]+)\s*(thousand|million|billion)",
        _scale_sub,
        text,
        flags=re.IGNORECASE,
    )
    # word numbers: "seven" → "7"
    def _word_sub(m: re.Match) -> str:
        return str(_NUMBER_WORDS[m.group(0).lower()])
    pattern = r"\b(" + "|".join(_NUMBER_WORDS) + r")\b"
    text = re.sub(pattern, _word_sub, text, flags=re.IGNORECASE)
    return text

def normalize_for_em(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = _normalize_numbers(normalized)
    normalized = re.sub(r"[^0-9a-z\s]", " ", normalized)
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def normalize_for_em_keep_scale(text: str) -> str:
    """Variant that does NOT multiply scale suffixes, keeps 'X thousand'/'X million'
    as literal tokens. Used as alternative form for relaxed matching when prediction
    omits the scale word but gold includes it (e.g., gold='592 thousand', pred='592').
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    # Remove commas from numbers but skip scale multiplication
    normalized = re.sub(r"(\d),(\d)", r"\1\2", normalized)
    # word numbers ("seven" → "7")
    pattern = r"\b(" + "|".join(_NUMBER_WORDS) + r")\b"
    normalized = re.sub(
        pattern,
        lambda m: str(_NUMBER_WORDS[m.group(0).lower()]),
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"[^0-9a-z\s]", " ", normalized)
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

# ── Multi-span splitter (from scripts/eval_tatqa.py) ─────────────────────────

def _split_answers(text: str) -> List[str]:
    """Smart split for multi-span answers, conservative to avoid splitting prose.

    1. Strong markers: ' | ' or '; ' → always split
    2. Numeric with parenthetical year ('73,260 (2019), 57,768 (2018)') → split
    3. ' and ' → split only if both sides numeric (≤2 parts)
    4. ', ' → split only if all parts ≤4 words AND not date pattern
    """
    t = (text or "").strip()
    if not t:
        return [t]
    for sep in [" | ", "; "]:
        if sep in t:
            parts = [a.strip().strip(".,") for a in t.split(sep) if a.strip()]
            if len(parts) > 1:
                return parts
    if re.search(r"\(\d{4}\),\s+\$?\d", t):
        parts = re.split(r",\s+(?=[\d\$])", t)
        if len(parts) > 1:
            return [a.strip() for a in parts if a.strip()]
    if " and " in t:
        parts = [a.strip().strip(".,") for a in t.split(" and ") if a.strip()]
        if len(parts) == 2 and all(re.search(r"\d", p) for p in parts):
            return parts
    if ", " in t and not re.search(r"\d{1,3},\d{3}", t):
        if re.search(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b",
            t,
        ):
            return [t]
        parts = [a.strip().strip(".") for a in t.split(", ") if a.strip()]
        if len(parts) >= 2 and all(len(p.split()) <= 4 for p in parts):
            return parts
    return [t]

# ── Rule metrics (from hybrid_eval.py) ───────────────────────────────────────

def _token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token-level F1 over normalized strings.

    Uses Counter (multiset) so repeated tokens are matched once each, matching
    the SQuAD/HotpotQA convention.
    """
    from collections import Counter
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_rule_metrics(cases: List[Dict]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    total = len(cases)
    evaluated = missing_gold = 0
    exact_correct = normalized_correct = relaxed_correct = 0
    f1_sum = 0.0

    for case in cases:
        gold_raw = case.get("answer-text")
        pred_raw = case.get(_PRED_FIELD, "")
        gold_clean = clean_answer_text(gold_raw)
        pred_clean = clean_answer_text(pred_raw)

        exact_match = normalized_em = relaxed_match = 0
        f1_score = 0.0

        if gold_raw is None:
            missing_gold += 1
        else:
            evaluated += 1
            norm_gold = normalize_for_em(gold_clean)
            norm_pred = normalize_for_em(pred_clean)

            exact_match   = int(gold_clean != "" and pred_clean == gold_clean)
            normalized_em = int(norm_gold != "" and norm_pred == norm_gold)
            # Token-overlap relaxed EM: match if ≥80% of gold tokens appear in pred
            def _token_overlap(p: str, g: str) -> float:
                g_tokens = set(g.split())
                if not g_tokens:
                    return 0.0
                p_tokens = set(p.split())
                return len(p_tokens & g_tokens) / len(g_tokens)

            # Word-boundary substring match: prevents false positives like
            # gold="1" matching pred="...2019..." where "1" is a substring of
            # "2019". Numeric / short golds need standalone-token matching.
            def _word_in(needle: str, haystack: str) -> bool:
                if not needle or not haystack:
                    return False
                return bool(re.search(r"\b" + re.escape(needle) + r"\b", haystack))

            def _check(p: str, g: str) -> bool:
                if not g:
                    return False
                return (
                    p == g
                    or _word_in(g, p)
                    or _word_in(p, g)
                    or _token_overlap(p, g) >= 0.8
                )

            # Try TWO normalizations and OR, this handles the asymmetric case
            # where gold says "592 thousand" but pred says just "592" (or vice
            # versa). The default normalize_for_em multiplies scale words, which
            # makes "592 thousand"→"592000" but leaves "592"→"592" → mismatch.
            # The keep_scale variant treats "thousand"/"million" as a literal token
            # so "592" matches "592 thousand" as a word-token subset.
            norm_gold_ks = normalize_for_em_keep_scale(gold_clean)
            norm_pred_ks = normalize_for_em_keep_scale(pred_clean)
            relaxed_match = int(
                _check(norm_pred, norm_gold)
                or _check(norm_pred_ks, norm_gold_ks)
            )

            f1_score = _token_f1(norm_pred, norm_gold) if norm_gold else 0.0

            exact_correct     += exact_match
            normalized_correct += normalized_em
            relaxed_correct   += relaxed_match
            f1_sum            += f1_score

        rows.append({
            "query":        case.get("question", ""),
            "golden":       gold_raw if gold_raw is not None else "",
            "gen":          pred_raw,
            "table":        case.get("table_id", ""),
            "exact_match":  exact_match,
            "normalized_em": normalized_em,
            "relaxed_match": relaxed_match,
            "f1":           f1_score,
        })

    exact_acc    = exact_correct     / evaluated if evaluated else 0.0
    norm_acc     = normalized_correct / evaluated if evaluated else 0.0
    relaxed_acc  = relaxed_correct    / evaluated if evaluated else 0.0
    f1_acc       = f1_sum             / evaluated if evaluated else 0.0

    summary = {
        "total_rows":             float(total),
        "evaluated_rows":         float(evaluated),
        "missing_gold_rows":      float(missing_gold),
        "exact_accuracy":         exact_acc,
        "normalized_em_accuracy": norm_acc,
        "relaxed_accuracy":       relaxed_acc,
        "token_f1":               f1_acc,
    }
    return pd.DataFrame(rows), summary

def print_rule_summary(summary: Dict[str, float]) -> None:
    print(f"  total_rows            : {int(summary['total_rows'])}")
    print(f"  evaluated_rows        : {int(summary['evaluated_rows'])}")
    print(f"  missing_gold_rows     : {int(summary['missing_gold_rows'])}")
    print(f"  exact_accuracy        : {summary['exact_accuracy']:.4f}")
    print(f"  normalized_em_accuracy: {summary['normalized_em_accuracy']:.4f}")
    print(f"  relaxed_accuracy      : {summary['relaxed_accuracy']:.4f}")
    print(f"  token_f1              : {summary['token_f1']:.4f}")

# ── LLM judge (Gemini via chat_utils, same EVALUATION_PROMPT) ────────────────

def _make_llm_fn():
    from atr.clients.chat_utils import get_chat_result
    from atr.config import config_mapping
    llm_config = config_mapping["gemini"]

    def llm_fn(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        r = get_chat_result(messages=messages, llm_config=llm_config)
        if hasattr(r, "content"):
            return r.content or ""
        if isinstance(r, dict):
            return r.get("content", "")
        return str(r)

    return llm_fn

def parse_llm_score(content: str) -> float:
    m = re.search(r"\[\[\s*([01](?:\.0+)?)\s*\]\]", content)
    if m:
        return float(m.group(1))
    m = re.search(r"rating\s*[:：]?\s*([01])\b", content, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    stripped = content.strip()
    if stripped in {"0", "1"}:
        return float(stripped)
    raise ValueError(f"Cannot parse LLM score: {content[:200]!r}")

def _single_llm_eval(
    case: Dict,
    llm_fn,
    strict: bool = False,
) -> Tuple[float, str]:
    golden = clean_answer_text(case.get("answer-text", ""))
    gen    = clean_answer_text(case.get(_PRED_FIELD, ""))
    ques   = case.get("question", "")
    template = EVALUATION_PROMPT_STRICT if strict else EVALUATION_PROMPT
    prompt = template.format(question=ques, golden=golden, gen=gen)
    content = llm_fn(prompt)
    score = parse_llm_score(content)
    return score, content

def llm_eval(
    cases: List[Dict],
    max_workers: int = 10,
    strict: bool = False,
) -> Tuple[List[Optional[float]], List[str], Dict[str, float]]:
    llm_fn = _make_llm_fn()
    scores: Dict[int, Optional[float]] = {}
    raws:   Dict[int, str]             = {}
    errors: Dict[int, str]             = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_single_llm_eval, case, llm_fn, strict): idx
                   for idx, case in enumerate(cases)}
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                score, raw = fut.result()
                scores[idx] = score
                raws[idx]   = raw
            except Exception as exc:
                scores[idx] = None
                raws[idx]   = ""
                errors[idx] = f"{type(exc).__name__}: {exc}"
                logger.warning(f"LLM eval failed for case {idx}: {exc}")

    score_list = [scores.get(i) for i in range(len(cases))]
    raw_list   = [raws.get(i, "")  for i in range(len(cases))]
    success    = sum(1 for s in score_list if s is not None)
    failed     = len(cases) - success
    final      = sum(s for s in score_list if s is not None) / success if success else float("nan")
    stats = {
        "total":   float(len(cases)),
        "success": float(success),
        "failed":  float(failed),
        "final_score": final,
    }
    return score_list, raw_list, stats

def print_llm_summary(stats: Dict[str, float]) -> None:
    total   = int(stats.get("total", 0))
    success = int(stats.get("success", 0))
    failed  = int(stats.get("failed", 0))
    final   = stats.get("final_score", float("nan"))
    print(f"  llm_total_rows  : {total}")
    print(f"  llm_success_rows: {success}")
    print(f"  llm_failed_rows : {failed}")
    if success > 0:
        print(f"  llm_score       : {final:.4f}  ({success}/{total} evaluated)")
    else:
        print("  llm_score       : N/A")

# ── Output helpers ────────────────────────────────────────────────────────────

def build_output_path(result_file: str, output_file: str = "") -> Path:
    from datetime import datetime
    eval_dir = Path(__file__).resolve().parent / "output"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if output_file:
        p = Path(output_file)
        return p if p.is_absolute() else eval_dir / p.name
    stem = Path(result_file).stem if result_file else "evaluation"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return eval_dir / f"{stem}_eval_{ts}.xlsx"

def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_excel(path, index=False)
        return path
    except Exception as exc:
        if "openpyxl" not in str(exc).lower():
            raise
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"openpyxl not installed, saved CSV: {csv_path}")
        return csv_path

def read_in_lines(path: str) -> List[Dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="ATR evaluation")
    parser.add_argument("--result_file_path", required=True, help="Inference JSONL output")
    parser.add_argument("--output_file", default="", help="Output xlsx path (default: output/<stem>_eval.xlsx)")
    parser.add_argument("--skip_llm_eval", action="store_true", help="Rule metrics only, no LLM judge")
    parser.add_argument("--max_workers", type=int, default=8, help="LLM eval parallelism")
    parser.add_argument("--strict", action="store_true",
                        help="Use strict EVALUATION_PROMPT (hallucination penalty + numeric "
                             "precision + completeness). Default: lenient prompt.")
    args = parser.parse_args()

    cases = read_in_lines(args.result_file_path)
    if not cases:
        print("No records found: exiting.")
        return

    # ── Rule metrics ──────────────────────────────────────────────────────────
    rule_df, rule_summary = compute_rule_metrics(cases)
    output_path = build_output_path(args.result_file_path, args.output_file)

    print(f"\n{'='*52}")
    print(f"  Results : {args.result_file_path}")
    print(f"{'='*52}")
    print("[Rule Metrics]")
    print_rule_summary(rule_summary)

    final_df = rule_df.copy()

    # ── LLM judge ─────────────────────────────────────────────────────────────
    if not args.skip_llm_eval:
        mode = "STRICT" if args.strict else "lenient"
        print(f"\n[LLM Judge]  running ... (mode: {mode})")
        score_list, raw_list, llm_stats = llm_eval(
            cases, max_workers=args.max_workers, strict=args.strict
        )
        print_llm_summary(llm_stats)
        final_df["llm_score"]     = score_list
        final_df["llm_eval_raw"]  = raw_list

    saved = save_dataframe(final_df, output_path)
    print(f"\n  Saved → {saved}")
    print(f"{'='*52}\n")

if __name__ == "__main__":
    main()
