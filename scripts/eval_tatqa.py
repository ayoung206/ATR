"""
Evaluate ATR outputs on TAT-QA using both metrics:
  1. Official TAT-QA EM + F1 (tatqa_eval.py)
  2. ATR relaxed accuracy (substring/token-overlap)

Usage:
  python scripts/eval_tatqa.py --result_file output/tatqa_results.jsonl
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TATQA = Path(os.getenv("TATQA_DIR", str(REPO / "data" / "tatqa")))
if TATQA.exists():
    sys.path.insert(0, str(TATQA))

GOLD_PATH = TATQA / "dataset_raw/tatqa_dataset_dev.json"

SCALE_TOKENS = ["thousand", "million", "billion", "percent", "%", "k", "m"]

def _extract_scale(text: str) -> str:
    """Heuristic: look for scale word in answer text."""
    t = text.lower()
    if "billion" in t:
        return "billion"
    if "million" in t:
        return "million"
    if "thousand" in t:
        return "thousand"
    if "percent" in t or "%" in t:
        return "percent"
    return ""

def _strip_scale_words(text: str) -> str:
    """Remove trailing scale words from answer."""
    t = text.strip()
    for w in ["million", "billion", "thousand", "percent", "%"]:
        if t.lower().endswith(w):
            t = t[: -len(w)].strip()
    return t

def _split_answers(text: str) -> list:
    """Smart split for multi-span answers, conservative to avoid over-splitting prose.

    Strategy:
      1. Strong markers: ' | ' or '; ' → always split
      2. ' and ': only split if BOTH parts are <=3 words (e.g., '19,911 and 15,916')
      3. Numeric with parenthetical year ('73,260 (2019), 57,768 (2018)') → split
      4. ', ': only split if ALL parts are <=4 words AND no embedded ',XXX' number
    """
    t = text.strip()
    if not t:
        return [t]
    # Strong list markers
    for sep in [" | ", "; "]:
        if sep in t:
            parts = [a.strip().strip(".,") for a in t.split(sep) if a.strip()]
            if len(parts) > 1:
                return parts
    # Numeric with parenthetical: '73,260 (2019), 57,768 (2018)'
    if re.search(r"\(\d{4}\),\s+\$?\d", t):
        parts = re.split(r",\s+(?=[\d\$])", t)
        if len(parts) > 1:
            return [a.strip() for a in parts if a.strip()]
    # ' and ': only when both sides look numeric (avoid 'Greece and Turkey' false positive)
    if " and " in t:
        parts = [a.strip().strip(".,") for a in t.split(" and ") if a.strip()]
        if len(parts) == 2 and all(re.search(r"\d", p) for p in parts):
            return parts
    # ', ': only if ALL parts are <=4 words AND not date pattern (e.g., 'January 31, 2019')
    if ", " in t and not re.search(r"\d{1,3},\d{3}", t):
        # Skip if pattern looks like a date (Month Day, Year)
        if re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b", t):
            return [t]
        parts = [a.strip() for a in t.split(", ") if a.strip()]
        if len(parts) > 1 and all(len(p.split()) <= 4 for p in parts):
            return parts
    return [t]

def to_official_predictions(results: list) -> dict:
    """Convert ATR results → tatqa_eval predictions JSON."""
    preds = {}
    for r in results:
        qid = r.get("question_id")
        ans = str(r.get("agentic_tablerag_answer", "")).strip()
        if not qid:
            continue
        scale = _extract_scale(ans)
        clean = _strip_scale_words(ans)
        ans_list = _split_answers(clean)
        preds[qid] = [ans_list, scale]
    return preds

# ── ATR relaxed metric (mirrors evaluate.py) ─────────────────────

def _normalize_for_em(text: str) -> str:
    n = unicodedata.normalize("NFKC", str(text)).lower()
    n = re.sub(r"[^0-9a-z\s\-\.]", " ", n)
    n = re.sub(r"\b(a|an|the)\b", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n

def _is_relaxed(pred: str, gold: str) -> bool:
    g = _normalize_for_em(gold)
    p = _normalize_for_em(pred)
    if not g:
        return False
    if p == g:
        return True
    if g in p:
        return True
    if p and p in g:
        return True
    g_tok = set(g.split())
    p_tok = set(p.split())
    if g_tok and len(p_tok & g_tok) / len(g_tok) >= 0.8:
        return True
    return False

def relaxed_metric(results: list) -> dict:
    total = correct_em = correct_norm = correct_relaxed = 0
    for r in results:
        gold = str(r.get("answer-text", ""))
        pred = str(r.get("agentic_tablerag_answer", ""))
        if not gold:
            continue
        total += 1
        gn, pn = _normalize_for_em(gold), _normalize_for_em(pred)
        if gn == pn:
            correct_norm += 1
        if str(gold).strip() == str(pred).strip():
            correct_em += 1
        if _is_relaxed(pred, gold):
            correct_relaxed += 1
    return {
        "total": total,
        "exact_em": correct_em / total if total else 0,
        "normalized_em": correct_norm / total if total else 0,
        "relaxed_em": correct_relaxed / total if total else 0,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", required=True)
    parser.add_argument("--out_pred_json", default=None,
                        help="Where to write the official-format prediction JSON")
    args = parser.parse_args()

    results = [json.loads(l) for l in open(args.result_file) if l.strip()]
    print(f"Loaded {len(results)} records from {args.result_file}")

    # 1. ATR metric
    print("\n=== ATR Relaxed Metric ===")
    rm = relaxed_metric(results)
    print(f"  total       : {rm['total']}")
    print(f"  exact EM    : {rm['exact_em']:.4f}")
    print(f"  normalized  : {rm['normalized_em']:.4f}")
    print(f"  relaxed EM  : {rm['relaxed_em']:.4f}")

    # 2. Official TAT-QA metric
    print("\n=== Official TAT-QA EM + F1 ===")
    preds = to_official_predictions(results)
    pred_path = args.out_pred_json or args.result_file.replace(".jsonl", "_tatqa_pred.json")
    pred_path = os.path.abspath(pred_path)
    with open(pred_path, "w") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f"  wrote predictions: {pred_path}  ({len(preds)} entries)")

    # Run official evaluator (external dependency, NExTplusplus/TAT-QA)
    try:
        from tatqa_eval import evaluate_prediction_file
    except ImportError:
        print(
            "\n[skipped] Official TAT-QA evaluator not on PYTHONPATH.\n"
            "  Install it once:\n"
            "    git clone https://github.com/NExTplusplus/TAT-QA <somewhere>/TAT-QA\n"
            "    export TATQA_DIR=<somewhere>/TAT-QA\n"
            "  Then re-run this script. The ATR relaxed metric above is "
            "still valid; only the official EM/F1 numbers were skipped.",
            file=sys.stderr,
        )
        return
    if not GOLD_PATH.exists():
        print(
            f"\n[skipped] TAT-QA gold file not found at {GOLD_PATH}.\n"
            f"  Set TATQA_DIR to a TAT-QA checkout (see message above).",
            file=sys.stderr,
        )
        return
    cwd_save = os.getcwd()
    os.chdir(TATQA)  # eval imports tatqa_metric relative
    try:
        evaluate_prediction_file(str(GOLD_PATH), pred_path)
    finally:
        os.chdir(cwd_save)

if __name__ == "__main__":
    main()
