"""
Train the Learned Router (§3.4), DistilBERT 4-class classifier.

Two-step workflow
─────────────────
Step 1: Oracle label generation

  (a) generate   : pseudo-oracle from HeuristicRouter (fast bootstrap, no LLM required)
  (b) oracle     : TRUE oracle per §3.4, run all 4 routes on each question,
                   pick argmin Cost(r) s.t. IsCorrect(r, q)
                     Cost(r) = latency_s + ROUTE_BASE_COST[r]
                   Requires a running Flask SQL service and LLM credentials.
  (c) from_inference : extract verified route labels, including restored schema
                       and cumulative H, from an ``--emit_trace`` inference run.

Step 2: Fine-tuning  (train)
  Fine-tune distilbert-base-uncased on oracle-labelled examples.

Step 3: Evaluation  (eval)
  Accuracy + confusion matrix on a held-out split.

Oracle JSONL format (one record per sub-query step):
  {
    "sub_query":            "<str>",
    "expected_operator":    "<str>",
    "required_modalities":  "<str>",
    "entity_mentions":      ["<str>", ...],
    "has_schema":           true | false,
    "oracle_route":         "TEXT" | "SQL" | "RETRIEVE" | "HYBRID"
  }

Usage
─────
# (a) Fast pseudo-oracle
python tools/train_router.py generate \\
    --data_file  $DATA_DIR/hybridqa_shard50_B.json \\
    --out_file   data/oracle_labels.jsonl

# (b) TRUE oracle (§3.4): requires index + LLM
python tools/train_router.py oracle \\
    --data_file   $DATA_DIR/hybridqa_shard50_B.json \\
    --index_path  ./index/multiview \\
    --bge_dir     ./models \\
    --backbone    gemini \\
    --out_file    data/oracle_labels_true.jsonl \\
    --max_questions 100

# Train
python tools/train_router.py train \\
    --oracle_file  data/oracle_labels_true.jsonl \\
    --output_dir   router_model \\
    --epochs       5

# Evaluate
python tools/train_router.py eval \\
    --oracle_file  data/oracle_labels_true.jsonl \\
    --model_dir    router_model
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LABEL2ID = {"TEXT": 0, "SQL": 1, "RETRIEVE": 2, "HYBRID": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# §3.4 base cost per route (proxy for LLM call count)
_ROUTE_BASE_COST: Dict[str, float] = {
    "TEXT": 0.0,     # 1 LLM call
    "SQL": 0.5,      # 1 LLM call + SQL execution
    "RETRIEVE": 1.0, # 2 LLM calls (value linking + synthesis)
    "HYBRID": 1.5,   # 2+ LLM calls + SQL execution
}

_AGGREGATE_OPERATORS = {"count", "aggregate", "sort", "arithmetic", "compare"}

def featurise(record: Dict[str, Any]) -> str:
    from atr.online.router import build_router_input

    schema = record.get("schema")
    if not schema and record.get("has_schema", False):
        schema = {
            "table_name": record.get("table_id", "unknown"),
            "columns": record.get("schema_columns", []),
        }
    history = record.get("history_H", [])
    if not history and record.get("failed_routes"):
        history = [{"route": route} for route in record["failed_routes"]]
    return build_router_input(
        sub_query=record.get("sub_query", ""),
        expected_operator=record.get("expected_operator", "lookup") or "lookup",
        required_modalities=record.get("required_modalities", "both") or "both",
        entity_mentions=record.get("entity_mentions", []),
        need_global_table_view=bool(record.get("need_global_table_view", False)),
        uncertainty=record.get("uncertainty", 0.0),
        schema=schema,
        history_H=history,
    )

def _load_oracle_records(oracle_file: str) -> List[Dict]:
    """Load oracle-label JSONL (for training/eval)."""
    records = []
    with open(oracle_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def _load_qa_dataset(data_file: str) -> List[Dict]:
    """Load QA dataset JSON or JSONL."""
    with open(data_file) as f:
        if data_file.endswith(".jsonl"):
            return [json.loads(l) for l in f if l.strip()]
        return json.load(f)

def _infer_operator(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ("how many", "count", "number of", "total")):
        return "count"
    if any(w in q for w in ("most", "least", "highest", "lowest", "maximum", "minimum", "best", "worst")):
        return "sort"
    if any(w in q for w in ("sum", "average", "mean", "percent")):
        return "aggregate"
    if any(w in q for w in ("greater than", "less than", "more than", "fewer than", "between")):
        return "compare"
    return "lookup"

def _heuristic_label(record: Dict[str, Any]) -> str:
    op = (record.get("expected_operator") or "lookup").lower()
    mod = (record.get("required_modalities") or "both").lower()
    entities = record.get("entity_mentions") or []
    has_schema = record.get("has_schema", False)
    has_entities = bool(entities)
    is_aggregate = op in _AGGREGATE_OPERATORS

    if mod == "text" or (not has_schema and not has_entities):
        return "TEXT"
    if is_aggregate and has_schema and not has_entities:
        return "SQL"
    if is_aggregate and has_entities:
        return "HYBRID"
    if has_entities:
        return "RETRIEVE"
    if has_schema:
        return "SQL"
    return "TEXT"

def label_with_heuristic(data_file: str, out_file: str) -> None:
    """
    Bootstrap oracle labels using HeuristicRouter decisions.
    Fast: no LLM or index required. Use as initial training data.
    """
    data = _load_qa_dataset(data_file)
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    written = 0
    with open(out_file, "w") as fout:
        for item in data:
            question = item.get("question", "")
            if not question:
                continue
            record: Dict[str, Any] = {
                "sub_query": question,
                "expected_operator": _infer_operator(question),
                "required_modalities": "both",
                "entity_mentions": [],
                "has_schema": bool(item.get("table_id")),
            }
            record["oracle_route"] = _heuristic_label(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    logger.info(f"Wrote {written} pseudo-oracle labels → {out_file}")

def _normalize_answer(text: str) -> str:
    """Normalize for exact-match comparison."""
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def _is_correct(pred: str, gold: str) -> bool:
    """Relaxed EM: normalized exact match or substring containment."""
    if not pred or not gold:
        return False
    p, g = _normalize_answer(pred), _normalize_answer(gold)
    return p == g or g in p or p in g

def _make_llm_fn(backbone: str):
    """Build llm_fn (same convention as online/main.py)."""
    from atr.clients.chat_utils import get_chat_result
    from atr.config import config_mapping
    llm_config = config_mapping[backbone]

    def llm_fn(messages: List[Dict]) -> str:
        response = get_chat_result(messages=messages, llm_config=llm_config)
        if hasattr(response, "content"):
            return response.content or ""
        if isinstance(response, dict):
            return response.get("content", "")
        return str(response)

    return llm_fn

def _execute_route_for_oracle(
    route_name: str,
    sub_q: Any,
    schema: Optional[Dict],
    chunks: List[Dict],
    text_evidence: str,
    index: Any,
    value_linker: Any,
    verifier: Any,
    sql_executor: Any,
    schema_top_k: int,
    cell_top_k: int,
) -> str:
    """Execute one route independently and return the answer string."""
    from atr.online.router import Route
    route = Route(route_name)

    if route == Route.TEXT:
        return verifier.answer_from_text(sub_q.sub_query, text_evidence)

    if route == Route.SQL:
        from atr.clients.sql_tool import get_excel_rag_response_plain
        resp = get_excel_rag_response_plain(
            table_name_list=sql_executor.table_name_list,
            query=sub_q.sub_query,
        )
        sql_result = str(resp.get("sql_execution_result", ""))
        return verifier.fuse(
            sub_query=sub_q.sub_query,
            route=route_name,
            text_evidence=text_evidence,
            schema_cell_evidence="",
            sql_result=sql_result,
        )

    # RETRIEVE / HYBRID: schema+cell retrieval first
    C, V_raw = index.schema_cell_retrieval(
        query=sub_q.sub_query,
        entity_mentions=sub_q.entity_mentions or [],
        schema_top_k=schema_top_k,
        cell_top_k=cell_top_k,
    )
    linked_values = value_linker.link(
        entity_mentions=sub_q.entity_mentions or [],
        schema_columns=C,
        V_raw=V_raw,
        history_H=[],
    )

    schema_info = "\n".join(
        f"Table: {e.get('table_id','?')} | Col: {e['col_name']} ({e.get('dtype','')})"
        for e in C
    )
    cell_info = "\n".join(
        f"Entity '{lv.entity}' → {lv.column} = '{lv.matched_value}'"
        for lv in linked_values if lv.is_matched
    ) or "(no values matched)"
    schema_cell_ev = f"{schema_info}\n{cell_info}"

    if any(lv.needs_reroute for lv in linked_values):
        return verifier.answer_from_text(sub_q.sub_query, text_evidence)

    if route == Route.RETRIEVE:
        return verifier.answer_from_retrieval(
            sub_query=sub_q.sub_query,
            schema_info=schema_info,
            cell_info=cell_info,
        )

    # HYBRID
    sql_result, _ = sql_executor.execute(
        sub_query=sub_q.sub_query,
        schema=schema,
        allowed_columns=C,
        linked_values=linked_values,
        retrieval_evidence=schema_cell_ev,
    )
    return verifier.fuse(
        sub_query=sub_q.sub_query,
        route=route_name,
        text_evidence=text_evidence,
        schema_cell_evidence=schema_cell_ev,
        sql_result=sql_result,
    )

def generate_true_oracle(
    data_file: str,
    index_path: str,
    bge_dir: str,
    backbone: str,
    out_file: str,
    device: str = "cpu",
    max_questions: Optional[int] = None,
) -> None:
    """
    §3.4 True oracle label generation.

    For each question in data_file:
      1. Decompose → first sub-query q_t
      2. Run all 4 routes independently, measure latency + correctness
      3. oracle = argmin Cost(r)  s.t.  IsCorrect(r, q)
                  where Cost(r) = latency_s + ROUTE_BASE_COST[r]

    Records that have no correct route are saved with oracle_correct=False
    and can be filtered out before training.
    """
    from atr.offline.multiview_index import MultiviewIndex
    from atr.online.decomposer import QueryDecomposer
    from atr.online.value_linker import HybridValueLinker
    from atr.online.constrained_sql import ConstrainedSQLExecutor
    from atr.online.verifier import EvidenceFusionVerifier
    from atr.config import SCHEMA_TOP_K, CELL_TOP_K

    bge_model_path = os.path.join(bge_dir, "bge-m3")
    logger.info(f"Loading MultiviewIndex from {index_path} ...")
    index = MultiviewIndex.load(index_path, bge_model_path, device=device)

    llm_fn = _make_llm_fn(backbone)
    decomposer = QueryDecomposer(llm_fn)
    value_linker = HybridValueLinker(llm_fn, cell_top_k=CELL_TOP_K)
    verifier = EvidenceFusionVerifier(llm_fn)

    data = _load_qa_dataset(data_file)
    if max_questions:
        data = data[:max_questions]

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    written = skipped = 0

    with open(out_file, "w") as fout:
        for i, item in enumerate(data):
            question = item.get("question", "")
            gold = item.get("answer-text", "")
            table_id = item.get("table_id", "")

            if not question or not gold:
                skipped += 1
                continue

            logger.info(f"[{i+1}/{len(data)}] {question[:80]}")

            # Step 1: Decompose (with table_id hint)
            try:
                sub_q = decomposer.decompose(question, [], table_id=table_id)
            except Exception as exc:
                logger.warning(f"  Decompose failed: {exc}")
                skipped += 1
                continue

            if sub_q.is_terminate:
                skipped += 1
                continue

            # Step 2: Document retrieval + schema restoration
            chunks = index.retrieve_documents(sub_q.sub_query, top_k=5)
            table_chunks = [c for c in chunks if c.get("type") == "table"]
            schema = index.restore_schema_via_mapping(table_chunks) if table_chunks else None
            text_evidence = "\n\n".join(
                f"[{c.get('source','?')}]\n{c.get('text','')}" for c in chunks
            )
            sql_executor = ConstrainedSQLExecutor(
                table_name_list=[table_id] if table_id else []
            )

            # Step 3: Run all 4 routes, measure correctness + cost
            route_results: Dict[str, Dict] = {}
            for route_name in LABEL2ID:  # TEXT, SQL, RETRIEVE, HYBRID
                t0 = time.time()
                try:
                    answer = _execute_route_for_oracle(
                        route_name=route_name,
                        sub_q=sub_q,
                        schema=schema,
                        chunks=chunks,
                        text_evidence=text_evidence,
                        index=index,
                        value_linker=value_linker,
                        verifier=verifier,
                        sql_executor=sql_executor,
                        schema_top_k=SCHEMA_TOP_K,
                        cell_top_k=CELL_TOP_K,
                    )
                except Exception as exc:
                    logger.warning(f"  Route {route_name} error: {exc}")
                    answer = ""
                latency = time.time() - t0
                correct = _is_correct(answer, gold)
                cost = latency + _ROUTE_BASE_COST[route_name]
                route_results[route_name] = {
                    "answer": answer,
                    "is_correct": correct,
                    "latency_s": round(latency, 3),
                    "cost": round(cost, 3),
                }
                logger.info(
                    f"  {route_name:8s}: correct={correct} "
                    f"latency={latency:.2f}s  answer='{answer[:60]}'"
                )

            # Step 4: argmin Cost(r) s.t. IsCorrect(r, q)
            correct_routes = [r for r, v in route_results.items() if v["is_correct"]]
            if correct_routes:
                oracle_route = min(correct_routes, key=lambda r: route_results[r]["cost"])
                oracle_correct = True
            else:
                # Soft label: cheapest route (will be flagged oracle_correct=False)
                oracle_route = min(route_results, key=lambda r: route_results[r]["cost"])
                oracle_correct = False
                logger.info(f"  No correct route, soft-labelling as {oracle_route}")

            rec = {
                "sub_query": sub_q.sub_query,
                "expected_operator": sub_q.expected_operator,
                "required_modalities": sub_q.required_modalities,
                "entity_mentions": sub_q.entity_mentions or [],
                "has_schema": schema is not None,
                "oracle_route": oracle_route,
                "oracle_correct": oracle_correct,
                "question": question,
                "gold_answer": gold,
                "route_results": route_results,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

    n_correct = sum(1 for r in _load_oracle_records(out_file) if r.get("oracle_correct"))
    logger.info(
        f"Oracle labels: {written} written ({n_correct} with correct route, "
        f"{written - n_correct} soft), {skipped} skipped → {out_file}"
    )

def relabel_soft_records(oracle_file: str, out_file: str) -> None:
    """
    Generate improved silver labels for oracle_correct=False records using
    heuristic routing on their actual decomposer metadata (entity_mentions,
    expected_operator, required_modalities, has_schema). True oracle records
    keep their original labels.

    This produces oracle_labels_v3 with a much better class distribution:
      v2: TEXT:273, SQL:14, RETRIEVE:15, HYBRID:3  (RETRIEVE never trained)
      v3: TEXT:~96, RETRIEVE:~162, SQL:~29, HYBRID:~18
    """
    from collections import Counter
    records = _load_oracle_records(oracle_file)
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    label_counts: Counter = Counter()
    source_counts: Counter = Counter()

    with open(out_file, "w") as fout:
        for rec in records:
            if rec.get("oracle_correct"):
                label = rec["oracle_route"]
                source = "oracle"
            else:
                label = _heuristic_label(rec)
                rec = dict(rec)
                rec["oracle_route"] = label
                source = "heuristic_silver"
            label_counts[label] += 1
            source_counts[source] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = sum(label_counts.values())
    logger.info(f"Relabeled {total} records → {out_file}")
    logger.info(f"  Labels : {dict(label_counts)}")
    logger.info(f"  Sources: {dict(source_counts)}")

def load_inference_oracle(inference_log: str, out_file: str) -> None:
    """
    Extract accepted route labels from an inference JSONL.

    Current logs are produced with ``atr.online.main --emit_trace`` and store
    each decision under ``atr_trace.verifier_decisions``.  The legacy flat
    ``verified_route`` format remains accepted for backwards compatibility.
    """
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    written = 0
    with open(inference_log) as fin, open(out_file, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            trace = rec.get("atr_trace") or {}
            decisions = trace.get("verifier_decisions") or []
            if decisions:
                candidates = [d for d in decisions if d.get("accepted") is True]
            elif rec.get("verified_route"):
                candidates = [{**rec, "route": rec["verified_route"]}]
            else:
                candidates = []

            for decision in candidates:
                route = decision.get("route")
                if route not in LABEL2ID:
                    continue
                schema = decision.get("schema", rec.get("schema"))
                oracle_rec = {
                    "question_id": decision.get(
                        "question_id", rec.get("question_id", rec.get("id", ""))
                    ),
                    "question": rec.get("question", ""),
                    "table_id": rec.get("table_id", ""),
                    "sub_query": decision.get(
                        "sub_query", rec.get("sub_query", rec.get("question", ""))
                    ),
                    "expected_operator": decision.get(
                        "expected_operator", rec.get("expected_operator", "lookup")
                    ),
                    "required_modalities": decision.get(
                        "required_modalities", rec.get("required_modalities", "both")
                    ),
                    "entity_mentions": decision.get(
                        "entity_mentions", rec.get("entity_mentions", [])
                    ),
                    "need_global_table_view": decision.get(
                        "need_global_table_view",
                        rec.get("need_global_table_view", False),
                    ),
                    "uncertainty": decision.get(
                        "uncertainty", rec.get("uncertainty", 0.0)
                    ),
                    "schema": schema,
                    "has_schema": bool(schema) or decision.get(
                        "has_schema", rec.get("has_schema", False)
                    ),
                    "history_H": decision.get(
                        "history_H", rec.get("history_H", [])
                    ),
                    "oracle_route": route,
                    "oracle_correct": True,
                    "label_source": "verified_inference",
                }
                fout.write(json.dumps(oracle_rec, ensure_ascii=False) + "\n")
                written += 1
    logger.info(f"Loaded {written} verified-route labels → {out_file}")

def train(
    oracle_file: str,
    output_dir: str,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_len: int = 256,
    val_split: float = 0.1,
    seed: int = 42,
    correct_only: bool = True,
    focal_loss: bool = False,
    focal_gamma: float = 2.0,
) -> None:
    """
    Fine-tune DistilBERT as a 4-class router classifier.

    Args:
        correct_only:  if True, skip records with oracle_correct=False (soft labels).
        focal_loss:    use Focal Loss instead of weighted cross-entropy.
        focal_gamma:   focusing parameter γ for Focal Loss (default 2.0).
    """
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader, random_split
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
        from torch.optim import AdamW
    except ImportError as e:
        raise ImportError("pip install transformers torch") from e

    torch.manual_seed(seed)
    all_records = _load_oracle_records(oracle_file)

    if correct_only:
        records = [r for r in all_records if r.get("oracle_correct", True)]
        logger.info(
            f"Loaded {len(records)} records (filtered from {len(all_records)}, "
            f"kept oracle_correct=True only)"
        )
    else:
        records = all_records
        logger.info(f"Loaded {len(records)} records (including soft labels)")

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    class RouterDataset(Dataset):
        def __init__(self, recs: List[Dict]) -> None:
            self.texts  = [featurise(r) for r in recs]
            self.labels = [LABEL2ID[r["oracle_route"]] for r in recs]

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> Dict:
            enc = tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
            }

    full_ds = RouterDataset(records)
    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)
    model.config.router_input_version = 2

    optimizer    = AdamW(model.parameters(), lr=lr)
    total_steps  = len(train_loader) * epochs
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    import torch.nn as nn

    label_counts = [0] * 4
    for r in records:
        label_counts[LABEL2ID[r["oracle_route"]]] += 1
    max_count = max(label_counts)
    class_weights = torch.tensor(
        [max_count / max(c, 1) for c in label_counts], dtype=torch.float
    ).to(device)
    logger.info(
        f"Class weights: {dict(zip(LABEL2ID.keys(), [round(w.item(), 2) for w in class_weights]))}"
    )

    if focal_loss:
        _gamma = focal_gamma

        class FocalLoss(nn.Module):
            def forward(self, logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
                ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
                pt = torch.exp(-ce)
                return (((1 - pt) ** _gamma) * ce).mean()

        loss_fn: nn.Module = FocalLoss()
        logger.info(f"Using Focal Loss (γ={_gamma})")
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using weighted CrossEntropyLoss")

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch  = {k: v.to(device) for k, v in batch.items()}
                preds  = model(**batch).logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total   += len(batch["labels"])

        val_acc  = correct / total if total else 0.0
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"  → saved best model (val_acc={val_acc:.4f}) to {output_dir}")

    logger.info(f"Training complete. Best val_acc={best_val_acc:.4f}")

def evaluate(oracle_file: str, model_dir: str, max_len: int = 256) -> float:
    try:
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    except ImportError as e:
        raise ImportError("pip install transformers torch") from e

    records   = _load_oracle_records(oracle_file)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    correct = total = 0
    confusion: Dict[str, Dict[str, int]] = {r: {r2: 0 for r2 in LABEL2ID} for r in LABEL2ID}

    with torch.no_grad():
        for rec in records:
            text       = featurise(rec)
            true_label = rec["oracle_route"]
            inputs     = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
            pred_id    = model(**inputs).logits.argmax(dim=-1).item()
            pred_label = ID2LABEL[pred_id]
            confusion[true_label][pred_label] += 1
            if pred_label == true_label:
                correct += 1
            total += 1

    acc = correct / total if total else 0.0
    logger.info(f"Accuracy: {correct}/{total} = {acc:.4f}")
    logger.info("Confusion matrix (rows=true, cols=pred):")
    logger.info("\t" + "\t".join(LABEL2ID.keys()))
    for true_r in LABEL2ID:
        row = "\t".join([str(confusion[true_r][p]) for p in LABEL2ID])
        logger.info(f"{true_r}\t{row}")
    return acc

#
# v4 trainer is an additive prep for learned_router.md Axes 2/3/4/5/6.  It does
# NOT replace the v3 `train()` function above (kept for backwards compat).
#
# Differences vs `train()`:
#   --multi_label         : sigmoid + BCE over all `correct_routes` per Q
#                            (so "TEXT and HYBRID both correct" is multi-hot,
#                            instead of two separate single-label samples).
#   --use_meta_features   : concat operator/modalities/entity_count/has_schema
#                            /need_global/decomposer_uncertainty after [CLS]
#                            and pass through an MLP head.
#   --cost_weight λ       : add λ × Σᵣ P(r) · cost(r) to loss (Axis 3).
#   --curriculum          : 3-stage training (single-correct → multi-correct
#                            → cost penalty).  Disabled if --multi_label off.
#   --exclude_dataset DS  : drop one dataset from training (LOO transfer eval).

OPERATOR_VOCAB = ["lookup", "filter", "count", "aggregate", "sort",
                  "arithmetic", "compare"]
MODALITY_VOCAB = ["table", "text", "both"]
ROUTE_COST_TENSOR = {"TEXT": 1.0, "RETRIEVE": 2.0, "SQL": 3.0, "HYBRID": 5.0}
META_FEATURE_DIM = (
    len(OPERATOR_VOCAB)         # one-hot operator
    + len(MODALITY_VOCAB)       # one-hot modality
    + 4                         # entity_count binned (0/1/2/3+)
    + 1                         # has_schema
    + 1                         # need_global_table_view
    + 1                         # decomposer_uncertainty
)

def _meta_features_vec(rec: Dict[str, Any]):
    """Build the meta-features tensor (length = META_FEATURE_DIM) for one record."""
    import torch
    op  = (rec.get("expected_operator") or "lookup").lower()
    mod = (rec.get("required_modalities") or "both").lower()
    op_oh  = [1.0 if op == o else 0.0 for o in OPERATOR_VOCAB]
    mod_oh = [1.0 if mod == m else 0.0 for m in MODALITY_VOCAB]
    entities = rec.get("entity_mentions") or []
    n_ent = len(entities)
    ent_bin = [
        1.0 if n_ent == 0 else 0.0,
        1.0 if n_ent == 1 else 0.0,
        1.0 if n_ent == 2 else 0.0,
        1.0 if n_ent >= 3 else 0.0,
    ]
    has_schema = 1.0 if rec.get("has_schema") else 0.0
    need_global = 1.0 if rec.get("need_global_table_view") else 0.0
    uncertainty = float(rec.get("decomposer_uncertainty", rec.get("uncertainty", 0.0)) or 0.0)
    feats = op_oh + mod_oh + ent_bin + [has_schema, need_global, uncertainty]
    return torch.tensor(feats, dtype=torch.float)

def _group_by_qid(records: List[Dict]) -> List[Dict]:
    """Multi-label: collapse v3-style multi-record-per-Q into one record per Q.

    Each output record has `correct_routes_multi_hot` (list[float] of length 4).
    Records lacking `correct_routes` fall back to `[oracle_route]`.
    """
    by_qid: Dict[str, Dict] = {}
    for r in records:
        qid = r.get("question_id") or r.get("id") or r.get("query_id") or r.get("question")
        if qid not in by_qid:
            by_qid[qid] = dict(r)
            correct = r.get("correct_routes") or [r["oracle_route"]]
            by_qid[qid]["_correct_set"] = set(correct)
        else:
            correct = r.get("correct_routes") or [r["oracle_route"]]
            by_qid[qid]["_correct_set"].update(correct)
    out = []
    for q, info in by_qid.items():
        info["correct_routes_multi_hot"] = [
            1.0 if route in info["_correct_set"] else 0.0
            for route in ["TEXT", "SQL", "RETRIEVE", "HYBRID"]
        ]
        del info["_correct_set"]
        out.append(info)
    return out

def train_v4(
    oracle_file: str,
    output_dir: str,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_len: int = 256,
    val_split: float = 0.1,
    seed: int = 42,
    multi_label: bool = False,
    use_meta_features: bool = False,
    cost_weight: float = 0.0,
    curriculum: bool = False,
    exclude_dataset: Optional[str] = None,
    correct_only: bool = True,
    focal_loss: bool = False,
    focal_gamma: float = 2.0,
) -> None:
    """v4 trainer: multi-label, meta-features, cost-aware, curriculum, LOO.

    See learned_router.md Axes 2-6.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader, random_split
        from torch.optim import AdamW
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertModel,
            DistilBertForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
    except ImportError as e:
        raise ImportError("pip install transformers torch") from e

    torch.manual_seed(seed)

    # ── 1) Load + filter records ─────────────────────────────────────────────
    all_records = _load_oracle_records(oracle_file)
    if correct_only:
        all_records = [r for r in all_records if r.get("oracle_correct", True)]
    if exclude_dataset:
        before = len(all_records)
        all_records = [r for r in all_records
                       if (r.get("dataset") or "").lower() != exclude_dataset.lower()]
        logger.info(
            f"Excluded dataset='{exclude_dataset}': {before} → {len(all_records)} records"
        )

    if multi_label:
        records = _group_by_qid(all_records)
        logger.info(f"[multi_label] grouped {len(all_records)} records → {len(records)} unique Q")
    else:
        records = all_records
        logger.info(f"Loaded {len(records)} records (single-label)")

    if not records:
        raise SystemExit("No training records after filtering, aborting")

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    # ── 2) Dataset ──────────────────────────────────────────────────────────
    class RouterDatasetV4(Dataset):
        def __init__(self, recs: List[Dict]) -> None:
            self.recs = recs

        def __len__(self) -> int:
            return len(self.recs)

        def __getitem__(self, idx: int) -> Dict:
            r = self.recs[idx]
            text = featurise(r)
            enc = tokenizer(
                text, truncation=True, max_length=max_len,
                padding="max_length", return_tensors="pt",
            )
            item = {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }
            if multi_label:
                item["multi_labels"] = torch.tensor(
                    r["correct_routes_multi_hot"], dtype=torch.float
                )
            else:
                item["labels"] = torch.tensor(
                    LABEL2ID[r["oracle_route"]], dtype=torch.long
                )
            if use_meta_features:
                item["meta"] = _meta_features_vec(r)
            return item

    full_ds = RouterDatasetV4(records)
    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 3) Model architecture ───────────────────────────────────────────────
    if use_meta_features:
        # Custom: DistilBERT encoder → [CLS] embed concat meta → MLP → 4 logits
        class RouterWithMeta(nn.Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.encoder = DistilBertModel.from_pretrained(name)
                hidden = self.encoder.config.hidden_size
                self.head = nn.Sequential(
                    nn.Linear(hidden + META_FEATURE_DIM, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 4),
                )

            def forward(self, input_ids, attention_mask, meta):
                h = self.encoder(input_ids=input_ids,
                                 attention_mask=attention_mask).last_hidden_state[:, 0, :]
                x = torch.cat([h, meta], dim=-1)
                return self.head(x)

        model = RouterWithMeta(model_name).to(device)
    else:
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID,
        ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # ── 4) Loss functions ───────────────────────────────────────────────────
    cost_vec = torch.tensor(
        [ROUTE_COST_TENSOR[ID2LABEL[i]] for i in range(4)],
        dtype=torch.float, device=device,
    )

    if multi_label:
        # Per-class pos_weight = (#neg / #pos) for BCE
        pos = [0.0] * 4
        for r in records:
            for i, v in enumerate(r["correct_routes_multi_hot"]):
                pos[i] += v
        neg = [len(records) - p for p in pos]
        pos_weight = torch.tensor(
            [n / max(p, 1.0) for p, n in zip(pos, neg)],
            dtype=torch.float, device=device,
        )
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        def compute_loss(logits, batch):
            base = bce(logits, batch["multi_labels"])
            if cost_weight > 0:
                p = torch.sigmoid(logits)
                ec = (p * cost_vec).sum(dim=-1).mean()
                return base + cost_weight * ec
            return base
    else:
        # Single-label CE (with class weights or focal)
        label_counts = [0] * 4
        for r in records:
            label_counts[LABEL2ID[r["oracle_route"]]] += 1
        max_count = max(label_counts) or 1
        class_weights = torch.tensor(
            [max_count / max(c, 1) for c in label_counts],
            dtype=torch.float, device=device,
        )
        if focal_loss:
            _gamma = focal_gamma
            def compute_loss(logits, batch):
                ce = F.cross_entropy(logits, batch["labels"],
                                     weight=class_weights, reduction="none")
                pt = torch.exp(-ce)
                base = (((1 - pt) ** _gamma) * ce).mean()
                if cost_weight > 0:
                    p = F.softmax(logits, dim=-1)
                    ec = (p * cost_vec).sum(dim=-1).mean()
                    return base + cost_weight * ec
                return base
        else:
            ce_fn = nn.CrossEntropyLoss(weight=class_weights)
            def compute_loss(logits, batch):
                base = ce_fn(logits, batch["labels"])
                if cost_weight > 0:
                    p = F.softmax(logits, dim=-1)
                    ec = (p * cost_vec).sum(dim=-1).mean()
                    return base + cost_weight * ec
                return base

    logger.info(
        f"v4 config: multi_label={multi_label} meta={use_meta_features} "
        f"cost_w={cost_weight} curriculum={curriculum} "
        f"exclude={exclude_dataset} epochs={epochs}"
    )

    # ── 5) Curriculum stages (only meaningful for multi_label) ──────────────
    # Stage 1: Q with exactly 1 correct route (clear signal)
    # Stage 2: Q with ≥2 correct routes (distribution learning)
    # Stage 3: cost penalty active
    if curriculum and multi_label:
        stages = [
            ("stage1_single_correct", max(1, epochs // 3),
             lambda r: sum(r["correct_routes_multi_hot"]) == 1, 0.0),
            ("stage2_multi_correct", max(1, epochs // 3),
             lambda r: True, 0.0),
            ("stage3_cost_penalty", epochs - 2 * max(1, epochs // 3),
             lambda r: True, cost_weight),
        ]
    else:
        stages = [("all", epochs, lambda r: True, cost_weight)]

    # ── 6) Training loop ────────────────────────────────────────────────────
    best_val_metric = -1.0
    epoch_global = 0
    for stage_name, stage_epochs, stage_filter, stage_cost in stages:
        if curriculum and multi_label:
            stage_recs = [r for r in records if stage_filter(r)]
            if not stage_recs:
                logger.info(f"[{stage_name}] no records match filter, skipping")
                continue
            stage_ds = RouterDatasetV4(stage_recs)
            stage_n_val = max(1, int(len(stage_ds) * val_split))
            stage_n_tr = len(stage_ds) - stage_n_val
            stage_tr, stage_v = random_split(
                stage_ds, [stage_n_tr, stage_n_val],
                generator=torch.Generator().manual_seed(seed),
            )
            tr_loader = DataLoader(stage_tr, batch_size=batch_size, shuffle=True)
            va_loader = DataLoader(stage_v,  batch_size=batch_size)
            logger.info(f"[{stage_name}] {len(stage_recs)} records, {stage_epochs} epochs")
        else:
            tr_loader, va_loader = train_loader, val_loader

        for ep in range(1, stage_epochs + 1):
            epoch_global += 1
            model.train()
            total_loss = 0.0
            for batch in tr_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                if use_meta_features:
                    logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        meta=batch["meta"],
                    )
                else:
                    logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                loss = compute_loss(logits, batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            # Validation
            model.eval()
            correct = total = 0
            multi_top1 = multi_topany = 0
            with torch.no_grad():
                for batch in va_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if use_meta_features:
                        logits = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            meta=batch["meta"],
                        )
                    else:
                        logits = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        ).logits
                    if multi_label:
                        # Top-1 hits any correct route?
                        preds = logits.argmax(dim=-1)
                        for p, mh in zip(preds, batch["multi_labels"]):
                            multi_topany += int(mh[p].item() > 0.5)
                            multi_top1 += int(mh.argmax().item() == p.item())
                            total += 1
                    else:
                        preds = logits.argmax(dim=-1)
                        correct += (preds == batch["labels"]).sum().item()
                        total += len(batch["labels"])

            avg_loss = total_loss / max(1, len(tr_loader))
            if multi_label:
                top_any = multi_topany / total if total else 0.0
                top1 = multi_top1 / total if total else 0.0
                logger.info(
                    f"[{stage_name}] Epoch {epoch_global}/{epochs} loss={avg_loss:.4f} "
                    f"top1={top1:.4f} top_any_correct={top_any:.4f}"
                )
                metric = top_any
            else:
                acc = correct / total if total else 0.0
                logger.info(
                    f"[{stage_name}] Epoch {epoch_global}/{epochs} loss={avg_loss:.4f} val_acc={acc:.4f}"
                )
                metric = acc

            if metric >= best_val_metric:
                best_val_metric = metric
                os.makedirs(output_dir, exist_ok=True)
                if use_meta_features:
                    torch.save(model.state_dict(), os.path.join(output_dir, "router_v4.pt"))
                    # Also save tokenizer + meta-feature config for inference
                    tokenizer.save_pretrained(output_dir)
                    with open(os.path.join(output_dir, "v4_config.json"), "w") as f:
                        json.dump({
                            "model_name":        model_name,
                            "use_meta_features": True,
                            "multi_label":       multi_label,
                            "meta_feature_dim":  META_FEATURE_DIM,
                            "operator_vocab":    OPERATOR_VOCAB,
                            "modality_vocab":    MODALITY_VOCAB,
                        }, f, indent=2)
                else:
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                logger.info(f"  → saved best model (metric={metric:.4f}) to {output_dir}")

    logger.info(f"v4 training complete. Best metric={best_val_metric:.4f}")

def generate_llm_distill(
    data_file: str,
    backbone: str,
    out_file: str,
    excel_dir: Optional[str] = None,
    max_questions: Optional[int] = None,
    sample_seed: int = 42,
    max_workers: int = 8,
) -> None:
    """
    LLM-router distillation (§3.4 training data).

    For each (optionally sampled) question:
      1. Decompose → first sub-query q_t  (1 LLM call)
      2. LLMRouter.route(q_t, H)           (up to 4 LLM calls)
    and record the initial choice plus teacher re-selections under cumulative
    failed-route histories.  When ``excel_dir`` is supplied, the same concrete
    column/type/example schema used online is serialized into every input.

    Output JSONL is directly consumable by `train` / `eval`.
    """
    import random
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from functools import lru_cache

    from atr.online.decomposer import QueryDecomposer
    from atr.online.router import LLMRouter

    llm_fn = _make_llm_fn(backbone)
    decomposer = QueryDecomposer(llm_fn)
    router = LLMRouter(llm_fn)

    data = _load_qa_dataset(data_file)
    data = [d for d in data if d.get("question") and d.get("table_id")]
    if max_questions and max_questions < len(data):
        rng = random.Random(sample_seed)
        data = rng.sample(data, max_questions)
    logger.info(f"LLM-distill over {len(data)} questions (backbone={backbone}, workers={max_workers})")

    @lru_cache(maxsize=None)
    def _schema_from_file(table_id: str) -> Optional[Dict[str, Any]]:
        if not excel_dir:
            return None
        import pandas as pd

        for suffix in (".xlsx", ".xls", ".csv"):
            path = os.path.join(excel_dir, f"{table_id}{suffix}")
            if not os.path.isfile(path):
                continue
            try:
                frame = (
                    pd.read_csv(path, dtype=str)
                    if suffix == ".csv"
                    else pd.read_excel(path, dtype=str)
                )
            except Exception as exc:
                logger.warning(f"  schema load failed for {path}: {exc}")
                return None
            columns = []
            for column in frame.columns:
                examples = frame[column].dropna().unique()[:3].tolist()
                columns.append([
                    str(column),
                    str(frame[column].dtype),
                    ", ".join(str(value) for value in examples),
                ])
            return {"table_name": table_id, "columns": columns}
        return None

    if not excel_dir and not any(
        item.get("schema") or item.get("schema_columns") for item in data
    ):
        logger.warning(
            "Distillation data has no concrete schema columns; pass --excel_dir "
            "for paper-aligned router inputs."
        )

    def _one(item: Dict) -> List[Dict]:
        question = item["question"]
        table_id = item["table_id"]
        try:
            sub_q = decomposer.decompose(question, [], table_id=table_id)
        except Exception as exc:
            logger.warning(f"  decompose failed: {exc}")
            return []
        if getattr(sub_q, "is_terminate", False) or not getattr(sub_q, "sub_query", ""):
            return []
        schema = item.get("schema") or _schema_from_file(table_id) or {
            "table_name": table_id,
            "columns": item.get("schema_columns", []),
        }
        history: List[Dict[str, Any]] = []
        records = []
        for _attempt in range(4):
            try:
                route = router.route(sub_q.sub_query, sub_q, schema, history)
            except Exception as exc:
                logger.warning(f"  route failed: {exc}")
                break
            records.append({
                "sub_query": sub_q.sub_query,
                "expected_operator": sub_q.expected_operator,
                "required_modalities": sub_q.required_modalities,
                "entity_mentions": sub_q.entity_mentions or [],
                "need_global_table_view": sub_q.need_global_table_view,
                "uncertainty": sub_q.uncertainty,
                "schema": schema,
                "has_schema": True,
                "history_H": [dict(failure) for failure in history],
                "oracle_route": route.value,
                "oracle_correct": True,
                "question": question,
                "table_id": table_id,
                "source": "llm_distill",
            })
            history.append({
                "sub_query": sub_q.sub_query,
                "route": route.value,
                "verdict": 0.5,
            })
        return records

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    written = skipped = 0
    label_counts: Dict[str, int] = {}
    with open(out_file, "w") as fout, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, item): i for i, item in enumerate(data)}
        for done, fut in enumerate(as_completed(futures), 1):
            records = fut.result()
            if not records:
                skipped += 1
            else:
                for rec in records:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    label_counts[rec["oracle_route"]] = (
                        label_counts.get(rec["oracle_route"], 0) + 1
                    )
                fout.flush()
            if done % 200 == 0:
                logger.info(f"  [{done}/{len(data)}] written={written} skipped={skipped} {label_counts}")
    logger.info(f"LLM-distill done: {written} written, {skipped} skipped → {out_file}")
    logger.info(f"  label distribution: {label_counts}")

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="Train / evaluate the Learned Router (§3.4)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── generate (pseudo-oracle) ─────────────────────────────────────────────
    gen = sub.add_parser("generate", help="Pseudo-oracle labels via HeuristicRouter (no LLM)")
    gen.add_argument("--data_file", required=True, help="QA dataset JSON/JSONL")
    gen.add_argument("--out_file",  required=True, help="Output oracle JSONL")

    # ── oracle (true oracle, §3.4) ─────────────────────────────────────────
    ora = sub.add_parser(
        "oracle",
        help="TRUE oracle labels: run all 4 routes, pick argmin Cost s.t. IsCorrect (§3.4)",
    )
    ora.add_argument("--data_file",     required=True, help="QA dataset JSON/JSONL")
    ora.add_argument("--index_path",    required=True, help="MultiviewIndex path prefix")
    ora.add_argument("--bge_dir",       required=True, help="Directory containing bge-m3 model")
    ora.add_argument("--backbone",      default="gemini", help="LLM backbone key from config_mapping")
    ora.add_argument("--out_file",      required=True, help="Output oracle JSONL")
    ora.add_argument("--device",        default="cpu")
    ora.add_argument("--max_questions", type=int, default=None,
                     help="Limit number of questions (for testing)")

    # ── distill (LLM-router distillation) ────────────────────────────────────
    dst = sub.add_parser(
        "distill",
        help="LLM-router distillation labels: Decompose + LLMRoute per question (no index)",
    )
    dst.add_argument("--data_file",     required=True, help="QA dataset JSON/JSONL (e.g. HybridQA train.json)")
    dst.add_argument("--backbone",      default="gemini", help="LLM backbone key from config_mapping")
    dst.add_argument("--out_file",      required=True, help="Output oracle JSONL")
    dst.add_argument("--excel_dir",     default=None,
                     help="Table .xlsx/.xls/.csv directory used to serialize actual schemas")
    dst.add_argument("--max_questions", type=int, default=None, help="Sample this many questions (seeded)")
    dst.add_argument("--sample_seed",   type=int, default=42)
    dst.add_argument("--max_workers",   type=int, default=8)

    # ── from_inference ───────────────────────────────────────────────────────
    inf = sub.add_parser("from_inference",
                         help="Extract accepted labels from an --emit_trace inference log")
    inf.add_argument("--inference_log", required=True)
    inf.add_argument("--out_file",      required=True)

    # ── relabel ──────────────────────────────────────────────────────────────
    rl = sub.add_parser(
        "relabel",
        help="Re-label oracle_correct=False records with heuristic (produces oracle_labels_v3)",
    )
    rl.add_argument("--oracle_file", required=True, help="Input oracle JSONL (with oracle_correct field)")
    rl.add_argument("--out_file",    required=True, help="Output oracle JSONL")

    # ── train ────────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Fine-tune DistilBERT router classifier")
    tr.add_argument("--oracle_file",  required=True)
    tr.add_argument("--output_dir",   required=True)
    tr.add_argument("--model_name",   default="distilbert-base-uncased")
    tr.add_argument("--epochs",       type=int,   default=5)
    tr.add_argument("--batch_size",   type=int,   default=32)
    tr.add_argument("--lr",           type=float, default=2e-5)
    tr.add_argument("--max_len",      type=int,   default=256)
    tr.add_argument("--val_split",    type=float, default=0.1)
    tr.add_argument("--seed",         type=int,   default=42)
    tr.add_argument("--all_labels",   action="store_true",
                    help="Include soft labels (oracle_correct=False) in training")
    tr.add_argument("--focal_loss",   action="store_true",
                    help="Use Focal Loss instead of weighted cross-entropy")
    tr.add_argument("--focal_gamma",  type=float, default=2.0,
                    help="Focal Loss γ parameter (default 2.0)")

    # ── train_v4 (multi-label / meta / cost-aware / curriculum / LOO) ────────
    tr4 = sub.add_parser(
        "train_v4",
        help="v4 trainer: multi-label, meta features, cost-aware, curriculum, LOO transfer",
    )
    tr4.add_argument("--oracle_file",  required=True)
    tr4.add_argument("--output_dir",   required=True)
    tr4.add_argument("--model_name",   default="distilbert-base-uncased")
    tr4.add_argument("--epochs",       type=int,   default=10)
    tr4.add_argument("--batch_size",   type=int,   default=32)
    tr4.add_argument("--lr",           type=float, default=2e-5)
    tr4.add_argument("--max_len",      type=int,   default=256)
    tr4.add_argument("--val_split",    type=float, default=0.1)
    tr4.add_argument("--seed",         type=int,   default=42)
    tr4.add_argument("--all_labels",   action="store_true",
                     help="Include soft labels (oracle_correct=False) in training")
    tr4.add_argument("--focal_loss",   action="store_true",
                     help="Single-label only: use Focal Loss instead of weighted CE")
    tr4.add_argument("--focal_gamma",  type=float, default=2.0)
    tr4.add_argument("--multi_label",  action="store_true",
                     help="Axis 4: sigmoid+BCE multi-label classification "
                          "(group oracle records by question_id; positive = any "
                          "correct route)")
    tr4.add_argument("--use_meta_features", action="store_true",
                     help="Axis 2: concat operator/modality/entity_count/has_schema/"
                          "need_global/uncertainty after [CLS]")
    tr4.add_argument("--cost_weight",  type=float, default=0.0,
                     help="Axis 3: λ for expected-cost penalty term in loss")
    tr4.add_argument("--curriculum",   action="store_true",
                     help="Axis 5: 3-stage curriculum (single→multi→cost). "
                          "Requires --multi_label")
    tr4.add_argument("--exclude_dataset", type=str, default=None,
                     choices=["hybridqa", "tatqa", "wtq"],
                     help="Axis 6: drop this dataset from training (LOO transfer eval)")

    # ── eval ─────────────────────────────────────────────────────────────────
    ev = sub.add_parser("eval", help="Evaluate router accuracy on oracle labels")
    ev.add_argument("--oracle_file", required=True)
    ev.add_argument("--model_dir",   required=True)
    ev.add_argument("--max_len",     type=int, default=256)

    args = parser.parse_args()

    if args.cmd == "generate":
        label_with_heuristic(args.data_file, args.out_file)

    elif args.cmd == "relabel":
        relabel_soft_records(args.oracle_file, args.out_file)

    elif args.cmd == "oracle":
        generate_true_oracle(
            data_file=args.data_file,
            index_path=args.index_path,
            bge_dir=args.bge_dir,
            backbone=args.backbone,
            out_file=args.out_file,
            excel_dir=args.excel_dir,
            device=args.device,
            max_questions=args.max_questions,
        )

    elif args.cmd == "distill":
        generate_llm_distill(
            data_file=args.data_file,
            backbone=args.backbone,
            out_file=args.out_file,
            max_questions=args.max_questions,
            sample_seed=args.sample_seed,
            max_workers=args.max_workers,
        )

    elif args.cmd == "from_inference":
        load_inference_oracle(args.inference_log, args.out_file)

    elif args.cmd == "train":
        train(
            oracle_file=args.oracle_file,
            output_dir=args.output_dir,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_len=args.max_len,
            val_split=args.val_split,
            seed=args.seed,
            correct_only=not args.all_labels,
            focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
        )

    elif args.cmd == "train_v4":
        train_v4(
            oracle_file=args.oracle_file,
            output_dir=args.output_dir,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_len=args.max_len,
            val_split=args.val_split,
            seed=args.seed,
            multi_label=args.multi_label,
            use_meta_features=args.use_meta_features,
            cost_weight=args.cost_weight,
            curriculum=args.curriculum,
            exclude_dataset=args.exclude_dataset,
            correct_only=not args.all_labels,
            focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
        )

    elif args.cmd == "eval":
        evaluate(args.oracle_file, args.model_dir, args.max_len)

if __name__ == "__main__":
    main()
