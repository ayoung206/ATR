"""
NaiveRAG-SC@N: NaiveRAG with self-consistency majority vote.

Runs the same NaiveRAG retrieve+generate prompt N times at temperature T,
then majority-votes the N candidate answers (after SQuAD-style normalization;
ties broken by first occurrence). Designed as a compute-equalized baseline
for ATR: NaiveRAG = 1 LLM call/Q, ATR ≈ 5-9 LLM calls/Q, NaiveRAG-SC@5
≈ 5 LLM calls/Q: answers the "ATR just uses more compute" critique.

Re-uses naiverag.naiverag_answer for retrieval + prompt construction, but
bypasses chat_utils.get_chat_result (which hard-codes temperature=0.1) and
calls the OpenAI-compatible API directly with custom temperature.

Usage:
  python baselines/naiverag_sc.py \\
      --backbone gemini \\
      --data_file_path data/my_dev.json \\
      --index_path ./index/hybridqa_multiview_celln50 \\
      --bge_dir ../models \\
      --device cuda --require_cuda \\
      --top_k 5 --sc 5 --temperature 0.7 \\
      --save_file_path ./output/naiverag_sc5_hybridqa.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
from atr.offline.multiview_index import MultiviewIndex
from atr.config import config_mapping
from atr.clients.chat_utils import _resolve_api_key
from atr.baselines.naiverag import (
    NAIVERAG_PROMPT,
    _extract_answer,
    _load_dataset,
    _existing_qids,
)

logger = logging.getLogger("naiverag_sc")

_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())

def _make_sc_llm_fn(llm_config: Dict, temperature: float):
    """Direct OpenAI client call so we can override temperature (chat_utils hard-codes 0.1).

    Uses chat_utils._resolve_api_key so Vertex Gemini's OAuth2 flow works
    (service account → fresh access token). For pure OpenAI (gpt-4o-mini)
    backbones it just returns the static api_key.
    """
    api_key = _resolve_api_key(llm_config)
    client = OpenAI(api_key=api_key, base_url=llm_config.get("url", ""))
    model = llm_config.get("model", "google/gemini-2.5-flash")

    # Vertex OAuth tokens expire ~1h. Re-resolve if a call returns 401.
    state = {"client": client, "key": api_key}

    def fn(messages: List[Dict]) -> str:
        for attempt in range(2):
            try:
                resp = state["client"].chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                msg = str(exc)
                if "401" in msg and attempt == 0:
                    logger.info("Token expired or invalid: refreshing")
                    state["key"] = _resolve_api_key(llm_config)
                    state["client"] = OpenAI(api_key=state["key"],
                                             base_url=llm_config.get("url", ""))
                    continue
                logger.warning(f"LLM call failed: {exc}")
                return ""
        return ""
    return fn

def _retrieve_context(question: str, table_id: str, index: MultiviewIndex,
                      top_k: int) -> str:
    """Mirror naiverag.naiverag_answer's retrieval + table_id promotion."""
    chunks = index.retrieve_documents(question, top_k=top_k)
    if table_id:
        from atr.online.main import _table_name_variants  # type: ignore
        variants = set(_table_name_variants(table_id))
        variants.add(table_id.lower())
        if not any(c.get("table_id", "").lower() in variants for c in chunks):
            try:
                doc_chunks = index.doc_retriever.chunks
                doc_smap = index.doc_retriever.chunk_schema_map
                for i in range(len(doc_chunks)):
                    tid = doc_smap.get(i, {}).get("table_id", "").lower()
                    if tid and tid in variants:
                        chunks = [{
                            "text": doc_chunks[i],
                            "table_id": tid,
                            "type": index.doc_retriever.chunk_type[i],
                        }] + chunks[: top_k - 1]
                        break
            except Exception:
                pass
    parts = [f"[Chunk {i}] {c.get('text','')}" for i, c in enumerate(chunks[:top_k], 1)]
    return "\n\n".join(parts) if parts else "(no context)"

def naiverag_sc_answer(question: str, table_id: str, index: MultiviewIndex,
                       llm_fn, top_k: int, sc: int) -> Dict:
    context = _retrieve_context(question, table_id, index, top_k)
    prompt = NAIVERAG_PROMPT.format(top_k=top_k, context=context, question=question)
    candidates: List[str] = []
    for _ in range(sc):
        resp = llm_fn([{"role": "user", "content": prompt}])
        candidates.append(_extract_answer(resp))

    # Majority vote on normalized form, ties → first occurrence wins
    norms = [_norm(c) for c in candidates]
    counts = Counter(norms)
    if not counts:
        return {"answer": "", "candidates": candidates, "vote_counts": {}}
    # most_common is stable on insertion order for ties since Python 3.7
    winner_norm, _ = counts.most_common(1)[0]
    # Pick the first original candidate whose normalized form matches
    winner = next(c for c, n in zip(candidates, norms) if n == winner_norm)
    return {
        "answer": winner,
        "candidates": candidates,
        "vote_counts": dict(counts),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="gemini")
    parser.add_argument("--data_file_path", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--bge_dir", default="BAAI")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sc", type=int, default=5,
                        help="Self-consistency samples (N runs at >0 temperature)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for the N SC runs")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--save_file_path", required=True)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")

    bge_path = os.path.join(args.bge_dir, "bge-m3")
    index = MultiviewIndex.load(save_path=args.index_path,
                                bge_model_path=bge_path,
                                device=args.device,
                                require_cuda=args.require_cuda)
    logger.info(f"Loaded index from {args.index_path} on {args.device}")

    cfg = config_mapping[args.backbone]
    llm_fn = _make_sc_llm_fn(cfg, args.temperature)
    logger.info(f"NaiveRAG-SC@{args.sc} (T={args.temperature}) on {cfg.get('model','?')}")

    data = _load_dataset(args.data_file_path)
    save_path = Path(args.save_file_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_qids(save_path) if args.rerun else set()
    pending = [d for d in data if d.get("question_id") not in done]
    logger.info(f"Pending {len(pending)}/{len(data)} (resume={args.rerun})")

    def process(record: Dict) -> Dict:
        qid = record.get("question_id", "")
        question = record.get("question", "")
        gold = record.get("answer-text", "")
        tid = record.get("table_id") or record.get("table_name") or ""
        if isinstance(tid, list):
            tid = tid[0] if tid else ""
        try:
            out = naiverag_sc_answer(question, str(tid), index, llm_fn,
                                     args.top_k, args.sc)
        except Exception as exc:
            logger.warning(f"[{qid}] failed: {exc}")
            out = {"answer": "", "candidates": [], "vote_counts": {}}
        return {
            "question_id": qid,
            "question": question,
            "table_id": str(tid),
            "answer-text": gold,
            "agentic_tablerag_answer": out["answer"],   # field reused by evaluate.py
            "sc_candidates": out["candidates"],
            "sc_vote_counts": out["vote_counts"],
            "sc_n": args.sc,
            "sc_temperature": args.temperature,
        }

    start = time.time()
    out_f = open(save_path, "a")
    completed = 0
    if args.max_workers > 1:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(process, r): r for r in pending}
            for fut in as_completed(futures):
                out_f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                out_f.flush()
                completed += 1
                if completed % 25 == 0:
                    el = time.time() - start
                    rate = completed / max(el, 1)
                    eta = (len(pending) - completed) / max(rate, 0.001)
                    logger.info(f"  [{completed}/{len(pending)}]  "
                                f"{rate*60:.1f} Q/min  ETA {eta/60:.1f} min")
    else:
        for r in pending:
            out_f.write(json.dumps(process(r), ensure_ascii=False) + "\n")
            out_f.flush()
            completed += 1
    out_f.close()
    logger.info(f"Done: {completed} new records → {save_path}")

if __name__ == "__main__":
    main()
