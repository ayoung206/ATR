"""
NaiveRAG baseline: minimal retrieve-and-generate.

For each question:
  1. Retrieve top-K chunks from the existing MultiviewIndex (Views 1+2).
  2. Concatenate as context.
  3. Single LLM call to extract the answer.

No iteration, no decomposition, no SQL execution, no routing: the simplest
retrieval-augmented baseline. This is the *floor* for ATR comparison: any
table-aware system should beat NaiveRAG.

Usage:
  python baselines/naiverag.py \\
      --backbone gemini \\
      --data_file_path <data.json> \\
      --index_path ./index/<dataset>_multiview \\
      --bge_dir ../models \\
      --device cpu \\
      --top_k 5 \\
      --save_file_path ./output/naiverag_<dataset>.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from atr.clients.chat_utils import get_chat_result
from atr.offline.multiview_index import MultiviewIndex
from atr.config import config_mapping

logger = logging.getLogger("naiverag")

NAIVERAG_PROMPT = """\
You are answering a question about a heterogeneous document containing text and a table.

═══════════════════════════════════════════════════════════════
RETRIEVED CONTEXT (top-{top_k} chunks, may include table markdown and document text)
═══════════════════════════════════════════════════════════════
{context}

═══════════════════════════════════════════════════════════════
QUESTION
═══════════════════════════════════════════════════════════════
{question}

═══════════════════════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════════════════════
1. Read the retrieved context carefully — it may contain a markdown table and / or
   passages of text. The answer is grounded in this context only; do NOT use
   external knowledge.
2. Extract the minimal answer (a name, number, date, or short phrase).
3. If the context does not contain enough information, answer exactly: not found.

OUTPUT FORMAT
<Answer>: [minimal answer only]
"""

def _make_llm_fn(llm_config: Dict):
    def fn(messages: List[Dict]) -> str:
        resp = get_chat_result(messages=messages, llm_config=llm_config)
        # OpenAI SDK returns ChatCompletionMessage object → extract content
        if hasattr(resp, "content"):
            return resp.content or ""
        if isinstance(resp, str):
            return resp
        return str(resp) if resp else ""
    return fn

def _extract_answer(response: str) -> str:
    if not response:
        return ""
    # Look for "<Answer>: ..." marker; fall back to whole response.
    import re
    m = re.search(r"<\s*Answer\s*>\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    return response.strip().split("\n")[-1].strip()

def naiverag_answer(
    question: str,
    table_id: str,
    index: MultiviewIndex,
    llm_fn,
    top_k: int = 5,
) -> str:
    """Retrieve + generate, single LLM call."""
    # Use the same retrieval call as ATR's online phase.
    chunks = index.retrieve_documents(question, top_k=top_k)

    # Prefer chunks from the gold table when table_id is supplied (HybridQA convention).
    if table_id:
        from atr.online.main import _table_name_variants  # type: ignore
        variants = set(_table_name_variants(table_id))
        variants.add(table_id.lower())
        # Promote a chunk from the right table to the top if not already there.
        if not any(c.get("table_id", "").lower() in variants for c in chunks):
            try:
                doc_chunks = index.doc_retriever.chunks
                doc_smap = index.doc_retriever.chunk_schema_map
                for idx_ in range(len(doc_chunks)):
                    tid = doc_smap.get(idx_, {}).get("table_id", "").lower()
                    if tid and tid in variants:
                        chunks = [{
                            "text": doc_chunks[idx_],
                            "table_id": tid,
                            "type": index.doc_retriever.chunk_type[idx_],
                        }] + chunks[: top_k - 1]
                        break
            except Exception:
                pass

    context_parts = []
    for i, c in enumerate(chunks[:top_k], 1):
        text = c.get("text", "")
        context_parts.append(f"[Chunk {i}] {text}")
    context = "\n\n".join(context_parts) if context_parts else "(no context)"

    prompt = NAIVERAG_PROMPT.format(
        top_k=top_k, context=context, question=question
    )
    try:
        response = llm_fn([{"role": "user", "content": prompt}])
    except Exception as exc:
        logger.warning(f"LLM call failed: {exc}")
        return ""
    return _extract_answer(response)

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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="gemini")
    parser.add_argument("--data_file_path", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--bge_dir", default="BAAI")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--save_file_path", required=True)
    parser.add_argument("--rerun", action="store_true",
                        help="Resume from existing save_file_path (skip done qids)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    bge_path = os.path.join(args.bge_dir, "bge-m3")
    index = MultiviewIndex.load(
        save_path=args.index_path,
        bge_model_path=bge_path,
        device=args.device,
        require_cuda=args.require_cuda,
    )
    logger.info(f"Loaded MultiviewIndex from {args.index_path} on {args.device}")

    cfg = config_mapping[args.backbone]
    llm_fn = _make_llm_fn(cfg)

    data = _load_dataset(args.data_file_path)
    logger.info(f"Loaded {len(data)} questions from {args.data_file_path}")

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
        # HybridQA-style table_id; some datasets use other keys.
        tid = record.get("table_id") or record.get("table_name") or ""
        if isinstance(tid, list):
            tid = tid[0] if tid else ""
        try:
            answer = naiverag_answer(
                question=question,
                table_id=str(tid),
                index=index,
                llm_fn=llm_fn,
                top_k=args.top_k,
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
                rec_out = fut.result()
                out_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                out_f.flush()
                completed += 1
                if completed % 25 == 0:
                    elapsed = time.time() - start
                    rate = completed / max(elapsed, 1)
                    eta = (len(pending) - completed) / max(rate, 0.001)
                    logger.info(
                        f"  [{completed}/{len(pending)}]  "
                        f"{rate*60:.1f} Q/min  ETA {eta/60:.1f} min"
                    )
    else:
        for rec in pending:
            rec_out = process(rec)
            out_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            out_f.flush()
            completed += 1
    out_f.close()
    logger.info(f"Done: {completed} new records → {save_path}")

if __name__ == "__main__":
    main()
