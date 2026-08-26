"""
Naive LLM baseline: no retrieval, no agent scaffolding.

Two modes:

  --mode closed_book
      Feed the question only. The model answers from parametric knowledge.
      Pure floor: "what does the LLM get with no context at all?"

  --mode full_context
      Feed the question + the *gold* table (rendered as CSV/markdown) and, if
      a passage directory is given, the *gold* linked passages, all dumped
      directly into the prompt. This is the standard "no-retrieval" /
      "full-table" setting (cf. ODYSSEY's no-retrieval baseline 58.7/68.2 on
      HybridQA). It uses gold table/passage selection, so it is NOT a
      retrieval-matched competitor to ATR / NaiveRAG / ReAct; it is a
      reference point for the *reading/reasoning* layer: how much does the
      retrieval + router + verifier + decomposition machinery add over just
      handing a long-context LLM the relevant table and text?

Output JSONL uses the `agentic_tablerag_answer` field so `evaluate.py` /
`scripts/eval_tatqa.py` work unchanged.

Usage:
  # Closed-book (any dataset; no extra dirs needed)
  python baselines/naive_llm.py --mode closed_book \\
      --backbone gemini --data_file_path <data.json> \\
      --save_file_path ./output/naivellm_closedbook_<dataset>.jsonl

  # Full gold context: HybridQA (table dir + linked-passage dir)
  python baselines/naive_llm.py --mode full_context \\
      --backbone gemini --data_file_path dataset/HybridQA/my_dev.json \\
      --table_dir dataset/HybridQA/dev_excel \\
      --doc_dir   dataset/HybridQA/dev_doc \\
      --save_file_path ./output/naivellm_fullctx_hybridqa.jsonl

  # Full gold context: WTQ (csv path stored in record's _wtq.context, under a root)
  python baselines/naive_llm.py --mode full_context \\
      --backbone gemini --data_file_path $DATA_DIR/wtq_unseen_flat.json \\
      --table_root <wtq-dataset-root> --table_path_key _wtq.context \\
      --save_file_path ./output/naivellm_fullctx_wtq.jsonl

  # Full gold context: TAT-QA (table + paragraphs embedded in the record's _tatqa dict)
  python baselines/naive_llm.py --mode full_context \\
      --backbone gemini --data_file_path $DATA_DIR/tatqa_dev_flat.json \\
      --embedded_tatqa \\
      --save_file_path ./output/naivellm_fullctx_tatqa.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from atr.clients.chat_utils import get_chat_result
from atr.config import config_mapping

logger = logging.getLogger("naive_llm")

CLOSED_BOOK_PROMPT = """\
Answer the following question with the minimal answer only — a name, number,
date, or short phrase. Do not explain. If you do not know, answer exactly: not found.

QUESTION: {question}

OUTPUT FORMAT
<Answer>: [minimal answer only]
"""

FULL_CONTEXT_PROMPT = """\
You are answering a question about a heterogeneous document containing a table
and (optionally) text passages. Everything you need is provided below — do NOT
use external knowledge.

═══════════════════════════════════════════════════════════════
TABLE
═══════════════════════════════════════════════════════════════
{table}

{passages_block}═══════════════════════════════════════════════════════════════
QUESTION
═══════════════════════════════════════════════════════════════
{question}

═══════════════════════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════════════════════
1. Use ONLY the table (and passages, if any) above.
2. Extract the minimal answer — a name, number, date, or short phrase. For a
   numeric answer, give the number (with its unit/scale if the question implies
   one); for multi-span answers, separate spans with ", ".
3. If the provided context is insufficient, answer exactly: not found.

OUTPUT FORMAT
<Answer>: [minimal answer only]
"""

_PASSAGES_HEADER = (
    "═══════════════════════════════════════════════════════════════\n"
    "LINKED PASSAGES\n"
    "═══════════════════════════════════════════════════════════════\n"
)

def _make_llm_fn(llm_config: Dict):
    def fn(messages: List[Dict]) -> str:
        resp = get_chat_result(messages=messages, llm_config=llm_config)
        if hasattr(resp, "content"):
            return resp.content or ""
        if isinstance(resp, str):
            return resp
        return str(resp) if resp else ""
    return fn

def _extract_answer(response: str) -> str:
    if not response:
        return ""
    m = re.search(r"<\s*Answer\s*>\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    return response.strip().split("\n")[-1].strip()

def _load_dataset(path: str) -> List[Dict]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("data", list(data.values())[0])
    return data

def _existing_qids(save_path: Path) -> set:
    if not save_path.exists():
        return set()
    qids = set()
    with open(save_path) as f:
        for line in f:
            try:
                qids.add(json.loads(line).get("question_id"))
            except Exception:
                continue
    return qids

def _maybe_parse_dict(v) -> Dict:
    """Some flat files store sub-dicts as Python-repr strings (e.g. _wtq, _tatqa)."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                d = parser(v)
                if isinstance(d, dict):
                    return d
            except Exception:
                continue
    return {}

def _dotted_get(record: Dict, dotted_key: str):
    """Resolve 'a.b.c' through nested dicts, parsing repr-string sub-dicts."""
    cur = record
    for part in dotted_key.split("."):
        if isinstance(cur, str):
            cur = _maybe_parse_dict(cur)
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

def _render_table_from_df(df, max_rows: int) -> str:
    n = len(df)
    if max_rows and n > max_rows:
        head = df.iloc[:max_rows]
        body = head.to_csv(index=False)
        return body + f"\n... ({n - max_rows} more rows truncated; table has {n} rows total)"
    return df.to_csv(index=False)

def _load_table_text(
    table_id: str,
    record: Dict,
    *,
    table_dir: Optional[str],
    table_root: Optional[str],
    table_path_key: Optional[str],
    max_rows: int,
) -> str:
    """Return the gold table rendered as CSV text, or '' if not found."""
    import pandas as pd

    candidates: List[Path] = []
    if table_dir:
        for ext in (".csv", ".xlsx", ".xls"):
            candidates.append(Path(table_dir) / f"{table_id}{ext}")
    if table_path_key:
        rel = _dotted_get(record, table_path_key)
        if rel:
            base = Path(table_root) if table_root else Path(".")
            candidates.append(base / str(rel))

    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(path, dtype=str)
            else:
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
            return _render_table_from_df(df, max_rows)
        except Exception as exc:
            logger.warning(f"  failed to read table {path}: {exc}")
    return ""

def _load_passages_text(
    table_id: str,
    *,
    doc_dir: Optional[str],
    max_chars: int,
) -> str:
    """Concatenate HybridQA-style linked passages: <doc_dir>/<table_id>.json = {url: text}."""
    if not doc_dir:
        return ""
    path = Path(doc_dir) / f"{table_id}.json"
    if not path.exists():
        # also try .txt
        txt = Path(doc_dir) / f"{table_id}.txt"
        if txt.exists():
            t = txt.read_text(encoding="utf-8")
            return t[:max_chars]
        return ""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"  failed to read passages {path}: {exc}")
        return ""
    parts: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            title = str(k).split("/")[-1].replace("_", " ")
            parts.append(f"[{title}] {v}")
    elif isinstance(obj, list):
        parts = [str(x) for x in obj]
    else:
        parts = [str(obj)]
    text = "\n\n".join(parts)
    return text[:max_chars]

def _load_tatqa_embedded(record: Dict, max_rows: int, max_chars: int) -> (str, str):
    """TAT-QA flat record: table + paragraphs live inside the _tatqa dict (repr string)."""
    d = _maybe_parse_dict(record.get("_tatqa", {}))
    table_text = ""
    tbl = d.get("table") or d.get("table_rows") or d.get("rows")
    # TAT-QA `table` is usually {"table": [[...row...], ...]} or directly a list of rows
    if isinstance(tbl, dict):
        tbl = tbl.get("table") or tbl.get("rows") or tbl.get("data")
    if isinstance(tbl, list) and tbl:
        try:
            import pandas as pd
            df = pd.DataFrame(tbl[1:], columns=tbl[0]) if len(tbl) > 1 else pd.DataFrame(tbl)
            table_text = _render_table_from_df(df.astype(str), max_rows)
        except Exception:
            table_text = "\n".join(", ".join(str(c) for c in row) for row in tbl[:max_rows + 1])
    paras = d.get("paragraphs") or d.get("paragraph") or []
    if isinstance(paras, list):
        ptexts = []
        for p in paras:
            if isinstance(p, dict):
                ptexts.append(str(p.get("text", "")))
            else:
                ptexts.append(str(p))
        passages_text = "\n\n".join(t for t in ptexts if t.strip())[:max_chars]
    elif isinstance(paras, str):
        passages_text = paras[:max_chars]
    else:
        passages_text = ""
    return table_text, passages_text

def _answer_one(
    record: Dict,
    llm_fn,
    *,
    mode: str,
    table_dir: Optional[str],
    doc_dir: Optional[str],
    table_root: Optional[str],
    table_path_key: Optional[str],
    embedded_tatqa: bool,
    max_rows: int,
    max_context_chars: int,
) -> str:
    question = record.get("question", "")
    if mode == "closed_book":
        prompt = CLOSED_BOOK_PROMPT.format(question=question)
    else:
        tid = record.get("table_id") or record.get("table_name") or ""
        if isinstance(tid, list):
            tid = tid[0] if tid else ""
        tid = str(tid)
        if embedded_tatqa:
            table_text, passages_text = _load_tatqa_embedded(record, max_rows, max_context_chars)
        else:
            table_text = _load_table_text(
                tid, record,
                table_dir=table_dir, table_root=table_root,
                table_path_key=table_path_key, max_rows=max_rows,
            )
            passages_text = _load_passages_text(tid, doc_dir=doc_dir, max_chars=max_context_chars)
        if not table_text and not passages_text:
            logger.warning(f"  [{record.get('question_id')}] no gold context found (table_id={tid!r})")
            table_text = "(table unavailable)"
        # Char budget: table gets priority, passages fill the rest.
        if len(table_text) > max_context_chars:
            table_text = table_text[:max_context_chars] + "\n... (table truncated)"
        remaining = max(0, max_context_chars - len(table_text))
        if len(passages_text) > remaining:
            passages_text = passages_text[:remaining] + "\n... (passages truncated)"
        passages_block = (_PASSAGES_HEADER + passages_text + "\n\n") if passages_text.strip() else ""
        prompt = FULL_CONTEXT_PROMPT.format(
            table=table_text or "(no table)",
            passages_block=passages_block,
            question=question,
        )
    try:
        response = llm_fn([{"role": "user", "content": prompt}])
    except Exception as exc:
        logger.warning(f"  LLM call failed: {exc}")
        return ""
    return _extract_answer(response)

def main() -> None:
    parser = argparse.ArgumentParser(description="Naive LLM baseline (closed-book / full gold context)")
    parser.add_argument("--mode", choices=["closed_book", "full_context"], required=True)
    parser.add_argument("--backbone", default="gemini")
    parser.add_argument("--data_file_path", required=True)
    parser.add_argument("--save_file_path", required=True)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--rerun", action="store_true", help="Resume: skip question_ids already in save_file_path")
    # full_context options
    parser.add_argument("--table_dir", default=None, help="Dir with <table_id>.csv / .xlsx (gold tables)")
    parser.add_argument("--doc_dir", default=None, help="Dir with <table_id>.json linked-passage dicts (HybridQA-style)")
    parser.add_argument("--table_root", default=None, help="Root to prepend to a relative table path from --table_path_key")
    parser.add_argument("--table_path_key", default=None, help="Dotted record key holding a relative csv path (e.g. _wtq.context)")
    parser.add_argument("--embedded_tatqa", action="store_true", help="Read table+paragraphs from the record's _tatqa dict")
    parser.add_argument("--max_rows", type=int, default=500, help="Max table rows to include (rest truncated)")
    parser.add_argument("--max_context_chars", type=int, default=600_000, help="Char budget for table+passages (~150k tokens)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")

    if args.mode == "full_context" and not (args.table_dir or args.table_path_key or args.embedded_tatqa):
        parser.error("--mode full_context requires one of: --table_dir, --table_path_key, --embedded_tatqa")

    cfg = config_mapping[args.backbone]
    llm_fn = _make_llm_fn(cfg)

    data = _load_dataset(args.data_file_path)
    logger.info(f"Loaded {len(data)} questions from {args.data_file_path}  (mode={args.mode})")

    save_path = Path(args.save_file_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    done_qids = _existing_qids(save_path) if args.rerun else set()
    if done_qids:
        logger.info(f"Resuming: {len(done_qids)} already done")
    pending = [d for d in data if d.get("question_id") not in done_qids]

    def process(record: Dict) -> Dict:
        qid = record.get("question_id", "")
        question = record.get("question", "")
        gold = record.get("answer-text", "")
        tid = record.get("table_id") or record.get("table_name") or ""
        if isinstance(tid, list):
            tid = tid[0] if tid else ""
        try:
            answer = _answer_one(
                record, llm_fn,
                mode=args.mode, table_dir=args.table_dir, doc_dir=args.doc_dir,
                table_root=args.table_root, table_path_key=args.table_path_key,
                embedded_tatqa=args.embedded_tatqa,
                max_rows=args.max_rows, max_context_chars=args.max_context_chars,
            )
        except Exception as exc:
            logger.warning(f"[{qid}] failed: {exc}")
            answer = ""
        return {
            "question_id": qid,
            "question": question,
            "table_id": str(tid),
            "answer-text": gold,
            "agentic_tablerag_answer": answer,  # reuse ATR field for evaluate.py
        }

    start = time.time()
    out_f = open(save_path, "a")
    completed = 0
    if args.max_workers > 1:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(process, rec): rec for rec in pending}
            for fut in as_completed(futures):
                out_f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                out_f.flush()
                completed += 1
                if completed % 25 == 0:
                    elapsed = time.time() - start
                    rate = completed / max(elapsed, 1)
                    eta = (len(pending) - completed) / max(rate, 1e-3)
                    logger.info(f"  [{completed}/{len(pending)}]  {rate*60:.1f} Q/min  ETA {eta/60:.1f} min")
    else:
        for rec in pending:
            out_f.write(json.dumps(process(rec), ensure_ascii=False) + "\n")
            out_f.flush()
            completed += 1
    out_f.close()
    logger.info(f"Done: {completed} new records → {save_path}")

if __name__ == "__main__":
    main()
