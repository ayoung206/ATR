"""
ReAct baseline: standard Thought → Action → Observation loop with tools.

The agent is given a question and a set of tools. At each step the LLM
reasons in a "Thought:" line, then issues an "Action:" by calling one of
the tools. The runtime parses the action, executes the tool, and feeds
the "Observation:" back to the LLM. The loop ends when the LLM emits
"Final Answer:".

Tools (mirror ATR's retrieval primitives so the comparison is fair):
  retrieve_docs(query)         → top-k document/table chunks
  retrieve_schema(query)       → top columns from Schema Index
  retrieve_cells(query, col?)  → top (col, value) pairs from Cell Index
  execute_sql(sql)             → SQL execution result via Flask service

This is a *strong* retrieval baseline: same retrieval primitives as ATR,
but no explicit decomposition, no router, no escalation, no verifier.
LLM controls the loop directly.

Usage:
  python baselines/react.py \\
      --backbone gemini \\
      --data_file_path <data.json> \\
      --index_path ./index/<dataset>_multiview \\
      --bge_dir ../models \\
      --device cuda:1 \\
      --max_steps 10 \\
      --save_file_path ./output/react_<dataset>.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from atr.clients.chat_utils import get_chat_result
from atr.offline.multiview_index import MultiviewIndex
from atr.config import config_mapping
from atr.clients.sql_tool import get_excel_rag_response_plain

logger = logging.getLogger("react")

REACT_SYSTEM_PROMPT = """\
You are a ReAct agent answering a question about a heterogeneous document with text and a table.

You have access to four tools. At each step, output exactly ONE of:
  Thought: <brief reasoning>
  Action: <tool_name>(<arguments>)
OR (when ready)
  Final Answer: <minimal answer — name, number, date, or short phrase>

═══════════════════════════════════════════════════════════════
TOOLS
═══════════════════════════════════════════════════════════════
retrieve_docs(query)
  → Returns top-5 document/table chunks matching the natural-language query.
  Best for: finding passages or table fragments mentioning specific entities.

retrieve_schema(query)
  → Returns top column entries (name + type + sample values) from the table schema.
  Best for: discovering relevant columns before issuing SQL.

retrieve_cells(query)
  → Returns top (column, value) cell-index entries matching the query.
  Best for: grounding entity mentions to actual cell values
  (e.g., "tv" → "television" if that is what is stored).

execute_sql(sql)
  → Executes the SQL on the table. Returns rows or an error.
  Use AFTER you've identified the table columns and necessary value bindings.
  Use plain MySQL; the table name appears in chunks (e.g., "List_of_X_0").

═══════════════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════════════
1. Output EXACTLY ONE line beginning with "Thought:", "Action:", or "Final Answer:".
2. After "Action:", wait for the next "Observation:" before continuing.
3. Use tools sparingly — 3-5 actions are usually enough.
4. The final answer must be MINIMAL: a name, number, date, or short phrase (NOT a full sentence).
5. If the available evidence is insufficient, output: Final Answer: not found
"""

REACT_TASK_PROMPT = """\
Question: {question}
{table_hint}
{trace}
Continue. Output exactly one line: Thought / Action / Final Answer.
"""

class ReActTools:
    """ReAct tools, table_id-aware (mirror ATR's known gold-table hint)."""

    def __init__(self, index: MultiviewIndex, table_name_list: List[str], table_id: str = ""):
        self.index = index
        self.table_name_list = table_name_list
        self.table_id_hint = table_id or ""
        # Pre-compute table_id variants for chunk promotion (mirror ATR helper).
        try:
            from atr.online.main import _table_name_variants  # type: ignore
            self._tid_variants = set(_table_name_variants(table_id)) if table_id else set()
            if table_id:
                self._tid_variants.add(table_id.lower())
        except Exception:
            self._tid_variants = {table_id.lower()} if table_id else set()

    def _promote_gold_table_chunk(self, chunks: List[Dict]) -> List[Dict]:
        """If gold table_id is known and not in chunks, prepend a chunk from it."""
        if not self._tid_variants or not chunks:
            return chunks
        # Check chunk source / text for gold table presence
        for c in chunks:
            src = c.get("source", "").lower()
            text = c.get("text", "")[:200].lower()
            for v in self._tid_variants:
                if v and (v in src or v in text):
                    return chunks  # already present
        # Hunt internal index for a chunk from the gold table
        try:
            doc_chunks = self.index.doc_retriever.chunks
            doc_smap = self.index.doc_retriever.chunk_schema_map
            doc_src = self.index.doc_retriever.chunk_source
            doc_type = self.index.doc_retriever.chunk_type
            for i in range(len(doc_chunks)):
                tid = doc_smap.get(i, {}).get("table_id", "").lower()
                if tid and tid in self._tid_variants:
                    promoted = {
                        "text": doc_chunks[i],
                        "source": doc_src[i],
                        "type": doc_type[i],
                    }
                    return [promoted] + chunks[: max(len(chunks) - 1, 0)]
        except Exception:
            pass
        return chunks

    def retrieve_docs(self, query: str, top_k: int = 5) -> str:
        chunks = self.index.retrieve_documents(query, top_k=top_k)
        chunks = self._promote_gold_table_chunk(chunks)
        if not chunks:
            return "(no chunks)"
        out = []
        for i, c in enumerate(chunks[:top_k], 1):
            text = c.get("text", "")
            out.append(f"[{i}] {text[:600]}")
        return "\n".join(out)

    def retrieve_schema(self, query: str, top_k: int = 5) -> str:
        try:
            entries = self.index.retrieve_schema(query, top_k=top_k * 4)
        except Exception as exc:
            return f"(schema retrieval failed: {exc})"
        if not entries:
            return "(no columns)"
        # Filter by gold table_id when known; else return top-k unfiltered.
        if self._tid_variants:
            filtered = [
                e for e in entries
                if (e.get("table_id", "") or "").lower() in self._tid_variants
            ]
            if filtered:
                entries = filtered[:top_k]
            else:
                entries = entries[:top_k]
        else:
            entries = entries[:top_k]
        out = []
        for e in entries:
            tid = e.get("table_id", "")
            col = e.get("col_name", "")
            dtype = e.get("dtype", "")
            ex = e.get("examples", "")
            out.append(f"- table={tid} col={col} ({dtype}): {ex}")
        return "\n".join(out)

    def retrieve_cells(self, query: str, top_k: int = 8) -> str:
        try:
            C, V_raw = self.index.schema_cell_retrieval(
                query=query, entity_mentions=[query],
                schema_top_k=3, cell_top_k=top_k,
            )
        except Exception as exc:
            return f"(cell retrieval failed: {exc})"
        if not V_raw:
            return "(no cells)"
        # V_raw is dict[entity → list[candidate]], flatten and prefer gold-table cells.
        all_cells: List[Dict] = []
        if isinstance(V_raw, dict):
            for cands in V_raw.values():
                if isinstance(cands, list):
                    all_cells.extend(cands)
        elif isinstance(V_raw, list):
            all_cells = V_raw
        if not all_cells:
            return "(no cells)"
        if self._tid_variants:
            preferred = [
                c for c in all_cells
                if (c.get("table_id", "") or "").lower() in self._tid_variants
            ]
            if preferred:
                all_cells = preferred + [c for c in all_cells if c not in preferred]
        out = []
        for c in all_cells[:top_k]:
            tid = c.get("table_id", "")
            col = c.get("col_name", "?")
            val = c.get("value", "?")
            out.append(f"- table={tid} col={col} value={val}")
        return "\n".join(out)

    def execute_sql(self, sql: str) -> str:
        if not self.table_name_list:
            return "(no table_name_list available)"
        try:
            resp = get_excel_rag_response_plain(
                table_name_list=self.table_name_list,
                query=sql,  # NL2SQL service can accept natural text or SQL
            )
            res = resp.get("sql_execution_result", "")
            return str(res)[:1200] if res else "(empty result)"
        except Exception as exc:
            return f"(SQL execution failed: {exc})"

_ACTION_RE = re.compile(
    r"Action:\s*(retrieve_docs|retrieve_schema|retrieve_cells|execute_sql)\s*\((.*)\)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)

def _parse_step(text: str) -> Tuple[str, str]:
    """Return (kind, payload). kind ∈ {action, final, thought, error}."""
    text = text.strip()
    m = _FINAL_RE.search(text)
    if m:
        return "final", m.group(1).strip().split("\n")[0].strip()
    m = _ACTION_RE.search(text)
    if m:
        tool = m.group(1).lower()
        arg = m.group(2).strip()
        # Strip outer quotes
        for q in ('"', "'"):
            if arg.startswith(q) and arg.endswith(q):
                arg = arg[1:-1]
        return "action", f"{tool}|{arg}"
    if text.lower().startswith("thought:"):
        return "thought", text[len("thought:"):].strip()
    return "thought", text  # treat unparseable as thought, force next step

def _make_llm_fn(llm_config: Dict):
    def fn(messages: List[Dict]) -> str:
        resp = get_chat_result(messages=messages, llm_config=llm_config)
        if hasattr(resp, "content"):
            return resp.content or ""
        return str(resp) if resp else ""
    return fn

def react_answer(
    question: str,
    table_id: str,
    index: MultiviewIndex,
    llm_fn,
    max_steps: int = 10,
) -> Dict:
    """Run ReAct loop. Returns dict with answer + trace."""
    table_name_list = [table_id] if table_id else []
    tools = ReActTools(index, table_name_list, table_id=table_id)

    table_hint = f"Table id: {table_id}\n" if table_id else ""
    trace: List[str] = []

    for step in range(max_steps):
        prompt_text = REACT_TASK_PROMPT.format(
            question=question,
            table_hint=table_hint,
            trace=("\n".join(trace) + "\n") if trace else "(no actions yet)\n",
        )
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        try:
            out = llm_fn(messages)
        except Exception as exc:
            logger.warning(f"LLM call failed: {exc}")
            return {"answer": "", "trace": trace, "error": str(exc)}
        kind, payload = _parse_step(out)

        if kind == "final":
            trace.append(f"Final Answer: {payload}")
            return {"answer": payload, "trace": trace, "steps": step + 1}

        if kind == "thought":
            trace.append(f"Thought: {payload[:300]}")
            continue

        # kind == "action"
        tool, arg = payload.split("|", 1)
        trace.append(f"Action: {tool}({arg[:200]})")
        if tool == "retrieve_docs":
            obs = tools.retrieve_docs(arg)
        elif tool == "retrieve_schema":
            obs = tools.retrieve_schema(arg)
        elif tool == "retrieve_cells":
            obs = tools.retrieve_cells(arg)
        elif tool == "execute_sql":
            obs = tools.execute_sql(arg)
        else:
            obs = f"(unknown tool {tool})"
        trace.append(f"Observation: {obs[:600]}")

    return {"answer": "", "trace": trace, "steps": max_steps, "stopped": "max_steps"}

def _load_dataset(path: str) -> List[Dict]:
    text = Path(path).read_text(encoding="utf-8")
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
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--save_file_path", required=True)
    parser.add_argument("--rerun", action="store_true")
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
    cfg = config_mapping[args.backbone]
    llm_fn = _make_llm_fn(cfg)

    data = _load_dataset(args.data_file_path)
    save_path = Path(args.save_file_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    done_qids = _existing_qids(save_path) if args.rerun else set()
    pending = [d for d in data if d.get("question_id") not in done_qids]
    logger.info(f"Loaded {len(data)} questions; {len(pending)} pending after resume filter")

    def process(record: Dict) -> Dict:
        qid = record.get("question_id", "")
        question = record.get("question", "")
        gold = record.get("answer-text", "")
        tid = record.get("table_id") or record.get("table_name") or ""
        if isinstance(tid, list):
            tid = tid[0] if tid else ""
        try:
            res = react_answer(
                question=question,
                table_id=str(tid),
                index=index,
                llm_fn=llm_fn,
                max_steps=args.max_steps,
            )
        except Exception as exc:
            logger.warning(f"[{qid}] failed: {exc}")
            res = {"answer": "", "trace": [], "error": str(exc)}
        return {
            "question_id": qid,
            "question": question,
            "table_id": str(tid),
            "answer-text": gold,
            "agentic_tablerag_answer": res.get("answer", ""),
            "react_trace": res.get("trace", [])[:30],
            "react_steps": res.get("steps"),
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
