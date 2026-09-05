"""Multiview index construction: text chunks, table chunks + mapping, schema, cell/row index. The relational DB (View 5) is the external Flask/MySQL service."""
from __future__ import annotations

import os
import json
import pickle
import logging
import threading
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

from atr.clients.tool_utils import Embedder, Reranker, excel_to_markdown  # noqa: E402
from atr.offline.reranking import CrossEncoderCandidateReranker  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

logger = logging.getLogger(__name__)

DOCUMENT_CHUNK_SIZE = 512
DOCUMENT_CHUNK_OVERLAP = 64
DOCUMENT_CHUNK_UNIT = "tokens"
INDEX_FORMAT_VERSION = 3


class IncompatibleIndexError(RuntimeError):
    """Raised when saved index artifacts do not satisfy the current contract."""


def _rebuild_error(save_path: str, violations: List[str]) -> IncompatibleIndexError:
    details = "\n".join(f"  - {violation}" for violation in violations)
    return IncompatibleIndexError(
        f"MultiviewIndex at {save_path!r} is incompatible with the current "
        f"ATR index format:\n{details}\n"
        "Rebuild every component together with build_index.py before running "
        "inference. Mixing legacy metadata and FAISS files is not supported."
    )


def _validate_index_payload(save_path: str, payload: Dict[str, Any]) -> None:
    """Validate metadata before loading embedding or reranker models.

    Legacy payloads are rejected rather than relabelled with current defaults.
    That keeps old character chunks, missing row indices, and over-budget cell
    indices from silently participating in paper-aligned runs.
    """
    if not isinstance(payload, dict):
        raise _rebuild_error(save_path, ["metadata payload must be a dict"])

    violations: List[str] = []
    if payload.get("index_format_version") != INDEX_FORMAT_VERSION:
        violations.append(
            "missing or unsupported index_format_version "
            f"(expected {INDEX_FORMAT_VERSION}, got "
            f"{payload.get('index_format_version')!r})"
        )

    required_lists = (
        "doc_chunks",
        "doc_source",
        "doc_type",
        "schema_entries",
        "cell_entries",
        "row_entries",
    )
    for key in required_lists:
        if not isinstance(payload.get(key), list):
            violations.append(f"metadata field {key!r} must be a list")
    if not isinstance(payload.get("doc_schema_map"), dict):
        violations.append("metadata field 'doc_schema_map' must be a dict")
    if not isinstance(payload.get("table_schemas"), dict):
        violations.append("metadata field 'table_schemas' must be a dict")

    chunk_config = payload.get("document_chunk_config")
    expected_chunk_config = {
        "size": DOCUMENT_CHUNK_SIZE,
        "overlap": DOCUMENT_CHUNK_OVERLAP,
        "unit": DOCUMENT_CHUNK_UNIT,
    }
    if chunk_config != expected_chunk_config:
        violations.append(
            "document_chunk_config must be the paper configuration "
            f"{expected_chunk_config!r}, got {chunk_config!r}"
        )

    build_config = payload.get("index_build_config")
    if not isinstance(build_config, dict):
        violations.append("missing index_build_config")
        build_config = {}
    elif build_config.get("document_chunks") != chunk_config:
        violations.append(
            "index_build_config.document_chunks does not match "
            "document_chunk_config"
        )

    cell_config = build_config.get("cell_index", {})
    row_config = build_config.get("row_index", {})
    for name, config in (("cell_index", cell_config), ("row_index", row_config)):
        if not isinstance(config, dict):
            violations.append(f"index_build_config.{name} must be a dict")

    doc_chunks = payload.get("doc_chunks", [])
    doc_sources = payload.get("doc_source", [])
    doc_types = payload.get("doc_type", [])
    if all(isinstance(items, list) for items in (doc_chunks, doc_sources, doc_types)):
        if not (len(doc_chunks) == len(doc_sources) == len(doc_types)):
            violations.append(
                "doc_chunks, doc_source, and doc_type lengths must match"
            )
        schema_map = payload.get("doc_schema_map", {})
        if isinstance(schema_map, dict):
            invalid_chunk_ids = [
                chunk_id
                for chunk_id in schema_map
                if not isinstance(chunk_id, int)
                or chunk_id < 0
                or chunk_id >= len(doc_chunks)
            ]
            if invalid_chunk_ids:
                violations.append("doc_schema_map contains out-of-range chunk ids")
            missing_table_maps = [
                index
                for index, chunk_type in enumerate(doc_types)
                if chunk_type == "table" and index not in schema_map
            ]
            if missing_table_maps:
                violations.append("one or more table chunks lack schema mappings")

    cell_entries = payload.get("cell_entries", [])
    row_entries = payload.get("row_entries", [])
    table_schemas = payload.get("table_schemas", {})
    cell_budget = cell_config.get("budget") if isinstance(cell_config, dict) else None
    cell_quota = (
        cell_config.get("per_table_quota") if isinstance(cell_config, dict) else None
    )
    row_budget = row_config.get("budget") if isinstance(row_config, dict) else None
    row_quota = (
        row_config.get("per_table_quota") if isinstance(row_config, dict) else None
    )
    row_max_chars = (
        row_config.get("max_row_chars") if isinstance(row_config, dict) else None
    )
    if isinstance(row_config, dict):
        for key in ("budget", "per_table_quota", "max_row_chars"):
            if key not in row_config:
                violations.append(f"index_build_config.row_index.{key} is required")
    for name, value in (
        ("cell_index.budget", cell_budget),
        ("cell_index.per_table_quota", cell_quota),
    ):
        if not isinstance(value, int) or value <= 0:
            violations.append(f"index_build_config.{name} must be a positive integer")
    for name, value in (
        ("row_index.budget", row_budget),
        ("row_index.per_table_quota", row_quota),
        ("row_index.max_row_chars", row_max_chars),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            violations.append(
                f"index_build_config.{name} must be null or a positive integer"
            )
    if not isinstance(build_config.get("embedding_model"), str) or not build_config.get(
        "embedding_model"
    ):
        violations.append("index_build_config.embedding_model must be a non-empty string")

    if isinstance(cell_entries, list) and isinstance(cell_budget, int):
        if len(cell_entries) > cell_budget:
            violations.append(
                f"cell entry count {len(cell_entries)} exceeds recorded budget {cell_budget}"
            )
    if isinstance(row_entries, list) and isinstance(row_budget, int):
        if len(row_entries) > row_budget:
            violations.append(
                f"row entry count {len(row_entries)} exceeds recorded budget {row_budget}"
            )
    if isinstance(table_schemas, dict) and table_schemas and not row_entries:
        violations.append("table schemas exist but the required row index is empty")

    for label, entries, quota in (
        ("cell", cell_entries, cell_quota),
        ("row", row_entries, row_quota),
    ):
        if not isinstance(entries, list) or not isinstance(quota, int) or quota <= 0:
            continue
        counts = Counter(entry.get("table_id") for entry in entries if isinstance(entry, dict))
        if any(count > quota for count in counts.values()):
            violations.append(f"{label} entries exceed the recorded per-table quota")

    if violations:
        raise _rebuild_error(save_path, violations)


def _load_validated_faiss_indices(
    save_path: str,
    payload: Dict[str, Any],
) -> Dict[str, Optional[faiss.Index]]:
    """Load FAISS components once and verify their cardinality and dimensions."""
    specifications = {
        "doc": len(payload["doc_chunks"]),
        "schema": len(payload["schema_entries"]),
        "cell": len(payload["cell_entries"]),
        "row": len(payload["row_entries"]),
    }
    loaded: Dict[str, Optional[faiss.Index]] = {}
    violations: List[str] = []
    dimensions = set()
    for component, expected_count in specifications.items():
        path = f"{save_path}.{component}.faiss"
        if not os.path.exists(path):
            loaded[component] = None
            if expected_count:
                violations.append(
                    f"missing {component} FAISS file for {expected_count} metadata entries"
                )
            continue
        try:
            index = faiss.read_index(path)
        except Exception as exc:
            loaded[component] = None
            violations.append(f"cannot read {component} FAISS file: {exc}")
            continue
        loaded[component] = index
        if index.ntotal != expected_count:
            violations.append(
                f"{component} FAISS count {index.ntotal} does not match "
                f"metadata count {expected_count}"
            )
        if index.ntotal:
            dimensions.add(index.d)
    if len(dimensions) > 1:
        violations.append(f"FAISS component dimensions disagree: {sorted(dimensions)}")
    if violations:
        raise _rebuild_error(save_path, violations)
    return loaded


def _schema_entry_to_text(entry: Dict[str, Any]) -> str:
    return (
        f"column: {entry['col_name']} | type: {entry['dtype']} | "
        f"examples: {entry['examples']}"
    )


def _cell_entry_to_text(entry: Dict[str, Any]) -> str:
    return f"column: {entry['col_name']} | value: {entry['value']}"


def _default_reranker_path(bge_model_path: str) -> str:
    return os.path.join(os.path.dirname(bge_model_path), "bge-reranker-v2-m3")


def _load_reranker(
    model_path: str,
    device: str,
    require_cuda: bool,
    candidate_multiplier: int,
) -> CrossEncoderCandidateReranker:
    if candidate_multiplier < 1:
        raise ValueError("rerank candidate multiplier must be at least 1")
    logger.info(
        "Loading cross-encoder reranker from %s (candidate multiplier=%d)",
        model_path,
        candidate_multiplier,
    )
    try:
        backend = Reranker(
            model_name_or_path=model_path,
            device=device,
            require_cuda=require_cuda,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Unable to load BGE reranker from {model_path!r}. Download "
            "BAAI/bge-reranker-v2-m3 there, pass --reranker_path, or use "
            "--no_reranker only for an explicit ablation."
        ) from exc
    return CrossEncoderCandidateReranker(
        backend, candidate_multiplier=candidate_multiplier
    )

def _load_dataframe(file_path: str) -> Optional[pd.DataFrame]:
    """Load a CSV or Excel file into a pandas DataFrame."""
    try:
        if file_path.lower().endswith(".csv"):
            return pd.read_csv(file_path, dtype=str)
        return pd.read_excel(file_path, dtype=str)
    except Exception as exc:
        logger.warning(f"Failed to load {file_path}: {exc}")
        return None

def _extract_schema(df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
    """
    Extract schema in the normalized JSON format from §3.2:
      {"table_name": "...", "columns": [["ColName", "Type", "Examples"], ...]}
    """
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        examples = df[col].dropna().unique()[:3].tolist()
        example_str = ", ".join(str(e) for e in examples)
        columns.append([str(col), dtype, example_str])
    return {"table_name": table_name, "columns": columns}

class SchemaIndex:
    """
    View 3: Schema Index (§3.2)
    Per-column dense recall followed by cross-encoder reranking.
    """

    def __init__(
        self,
        embedder: Embedder,
        reranker: Optional[CrossEncoderCandidateReranker] = None,
    ) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.entries: List[Dict[str, Any]] = []  # [{table_id, col_name, dtype, examples}]
        self._index: Optional[faiss.IndexFlatIP] = None
        self._embeddings: Optional[np.ndarray] = None

    def add_table(self, table_id: str, schema: Dict[str, Any]) -> None:
        for col, dtype, examples in schema["columns"]:
            self.entries.append({
                "table_id": table_id,
                "col_name": col,
                "dtype": dtype,
                "examples": examples,
            })

    def build(self) -> None:
        if not self.entries:
            return
        texts = [_schema_entry_to_text(e) for e in self.entries]
        self._embeddings = self._embed_in_batches(texts)
        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._embeddings.astype(np.float32))
        logger.info(f"SchemaIndex built: {len(self.entries)} column entries.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self._index is None or len(self.entries) == 0:
            return []
        q_emb = self.embedder.encode([query]).astype(np.float32)
        top_k = min(top_k, len(self.entries))
        search_k = (
            self.reranker.recall_size(top_k, len(self.entries))
            if self.reranker else top_k
        )
        _, I = self._index.search(q_emb, search_k)
        candidates = [self.entries[i] for i in I[0] if 0 <= i < len(self.entries)]
        if not self.reranker:
            return candidates[:top_k]
        return self.reranker.select(
            query, candidates, [_schema_entry_to_text(e) for e in candidates], top_k
        )

    def _embed_in_batches(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            vecs.append(self.embedder.encode(texts[i: i + batch_size]))
        return np.vstack(vecs)

class CellIndex:
    """
    View 4: Cell / Row Index (§3.2), a new contribution.
    Cell component: (column, value) pair dense recall + cross-encoder reranking with budget-B
    frequency-aware truncation. Directly supports Constrained Value Linking
    Stage 1 (§3.5). The row component is implemented separately as
    `RowIndex` and registered under the same View 4 conceptual umbrella
    (cells for entity-to-column linking; rows for filtered table context
    fed to the RETRIEVE primitive).
    """

    def __init__(
        self,
        embedder: Embedder,
        budget: int = 10_000,
        per_table_quota: int = 50,
        reranker: Optional[CrossEncoderCandidateReranker] = None,
    ) -> None:
        """
        Args:
            budget:           global safety cap on total cell entries (across all
                              tables). Prevents disk/RAM blow-up on huge corpora.
            per_table_quota:  max (col, value) pairs encoded per table. Mirrors
                              T2024 `max_encode_cell` per-table semantics so every
                              table contributes equally: fixes the previous
                              global-only cap that left 94% of HybridQA tables
                              with zero coverage.
        """
        self.embedder = embedder
        self.reranker = reranker
        self.budget = budget
        self.per_table_quota = per_table_quota
        self.entries: List[Dict[str, Any]] = []  # [{table_id, col_name, value, freq}]
        self._index: Optional[faiss.IndexFlatIP] = None

    def add_table(self, table_id: str, df: pd.DataFrame) -> None:
        """Extract (col, value) pairs with frequency counting.

        Per-table quota (T2024 pattern) + global safety cap. Each table
        contributes up to `per_table_quota` (col, value) pairs by frequency,
        until the global `budget` is exhausted.
        """
        cat_cols = [c for c in df.columns if df[c].dtype == object]
        counter: Counter = Counter()
        for col in cat_cols:
            for val in df[col].dropna().astype(str):
                counter[(col, val)] += 1

        remaining_global = max(0, self.budget - len(self.entries))
        take = min(self.per_table_quota, remaining_global, len(counter))
        for (col, val), freq in counter.most_common(take):
            self.entries.append({
                "table_id": table_id,
                "col_name": col,
                "value": val,
                "freq": freq,
            })

    def build(self) -> None:
        if not self.entries:
            return
        texts = [_cell_entry_to_text(e) for e in self.entries]
        embeddings = self._embed_in_batches(texts)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        logger.info(f"CellIndex built: {len(self.entries)} (col, value) entries.")

    def retrieve(
        self,
        entity: str,
        column: Optional[str] = None,
        top_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Stage 1 of Constrained Value Linking.
        V_e^(c) = TopK_cell-index(e, c; K)
        """
        if self._index is None or len(self.entries) == 0:
            return []
        query_text = f"column: {column} | {entity}" if column else entity
        q_emb = self.embedder.encode([query_text]).astype(np.float32)
        top_k = min(top_k, len(self.entries))
        search_k = (
            self.reranker.recall_size(top_k, len(self.entries))
            if self.reranker else top_k
        )
        _, I = self._index.search(q_emb, search_k)
        candidates = [self.entries[i] for i in I[0] if 0 <= i < len(self.entries)]
        if not self.reranker:
            return candidates[:top_k]
        return self.reranker.select(
            query_text, candidates, [_cell_entry_to_text(e) for e in candidates], top_k
        )

    def _embed_in_batches(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            vecs.append(self.embedder.encode(texts[i: i + batch_size]))
        return np.vstack(vecs)

class RowIndex:
    """
    View 4 row component: RowIndex (§3.2, Phase F-A).

    Per-row text embeddings, indexed for entity-aware dense recall and reranking.
    Closes the standalone-V=1 gap of the RETRIEVE primitive by giving the
    LLM specific table rows (not whole table chunks) so it can locate the
    target column for a filtered entity.

    Each row is serialised as
        "Table {table_id} | col1: val1 | col2: val2 | ..."
    and embedded by BGE-M3. The index returns the top-K rows whose
    concatenated cell values are most similar to the query (or entity).
    """

    def __init__(
        self,
        embedder: Embedder,
        budget: Optional[int] = None,
        per_table_quota: Optional[int] = None,
        max_row_chars: Optional[int] = None,
        reranker: Optional[CrossEncoderCandidateReranker] = None,
    ) -> None:
        for name, value in (
            ("budget", budget),
            ("per_table_quota", per_table_quota),
            ("max_row_chars", max_row_chars),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be None or a positive integer")
        self.embedder = embedder
        self.reranker = reranker
        self.budget = budget
        self.per_table_quota = per_table_quota
        self.max_row_chars = max_row_chars
        # [{table_id, row_idx, row_text, row_dict}]
        self.entries: List[Dict[str, Any]] = []
        self._index: Optional[faiss.IndexFlatIP] = None
        self._table_entry_ids: Dict[str, List[int]] = defaultdict(list)
        self._table_entry_ids_count = 0
        self._cached_table_id: Optional[str] = None
        self._cached_table_index: Optional[faiss.IndexFlatIP] = None
        self._cached_table_global_ids: Optional[np.ndarray] = None
        self._table_search_lock = threading.Lock()

    def add_table(self, table_id: str, df: pd.DataFrame) -> None:
        """Queue table rows for encoding; defaults include every source row."""
        remaining_global = (
            max(0, self.budget - len(self.entries))
            if self.budget is not None else len(df)
        )
        table_limit = self.per_table_quota or len(df)
        take = min(table_limit, remaining_global, len(df))
        if take == 0:
            return

        for row_idx in range(take):
            row = df.iloc[row_idx]
            parts = [f"{col}: {row[col]}" for col in df.columns
                     if pd.notna(row[col]) and str(row[col]).strip()]
            row_text = f"Table {table_id} | " + " | ".join(parts)
            if self.max_row_chars is not None and len(row_text) > self.max_row_chars:
                row_text = row_text[: self.max_row_chars - 1] + "…"

            entry_id = len(self.entries)
            self.entries.append({
                "table_id": table_id,
                "row_idx": int(row_idx),
                "row_text": row_text,
                "row_dict": {c: (str(row[c]) if pd.notna(row[c]) else "")
                             for c in df.columns},
            })
            self._table_entry_ids[table_id].append(entry_id)
        self._table_entry_ids_count = len(self.entries)
        self._clear_table_search_cache()

    def build(self) -> None:
        if not self.entries:
            return
        texts = [e["row_text"] for e in self.entries]
        embeddings = self._embed_in_batches(texts)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        self._clear_table_search_cache()
        logger.info(f"RowIndex built: {len(self.entries)} rows.")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        table_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top-K rows ranked by similarity to the query.
        If `table_id` is given, FAISS searches only that table's row IDs.
        """
        if self._index is None or len(self.entries) == 0 or top_k <= 0:
            return []
        if self._table_entry_ids_count != len(self.entries):
            self._rebuild_table_entry_ids()

        allowed_ids: Optional[np.ndarray] = None
        population = len(self.entries)
        if table_id is not None:
            allowed_ids = np.asarray(
                self._table_entry_ids.get(table_id, []), dtype=np.int64
            )
            population = len(allowed_ids)
            if population == 0:
                return []

        q_emb = self.embedder.encode([query]).astype(np.float32)
        candidate_k = (
            self.reranker.recall_size(top_k, population)
            if self.reranker else min(top_k, population)
        )
        if allowed_ids is None:
            _, indices = self._index.search(q_emb, candidate_k)
        else:
            indices = self._search_table(q_emb, candidate_k, table_id, allowed_ids)

        candidates = []
        for i in indices[0]:
            if i < 0 or i >= len(self.entries):
                continue
            candidates.append(self.entries[i])
        if not self.reranker:
            return candidates[:top_k]
        return self.reranker.select(
            query, candidates, [e["row_text"] for e in candidates], top_k
        )

    def _rebuild_table_entry_ids(self) -> None:
        self._table_entry_ids = defaultdict(list)
        for entry_id, entry in enumerate(self.entries):
            self._table_entry_ids[entry["table_id"]].append(entry_id)
        self._table_entry_ids_count = len(self.entries)
        self._clear_table_search_cache()

    def _search_table(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        table_id: str,
        global_ids: np.ndarray,
    ) -> np.ndarray:
        """Run exact FAISS search over one table and map local IDs to globals."""
        with self._table_search_lock:
            if self._cached_table_id != table_id:
                start = int(global_ids[0])
                contiguous = np.array_equal(
                    global_ids,
                    np.arange(start, start + len(global_ids), dtype=np.int64),
                )
                if contiguous:
                    vectors = self._index.reconstruct_n(start, len(global_ids))
                else:
                    vectors = np.vstack(
                        [
                            self._index.reconstruct(int(entry_id))
                            for entry_id in global_ids
                        ]
                    )
                table_index = faiss.IndexFlatIP(self._index.d)
                table_index.add(np.ascontiguousarray(vectors, dtype=np.float32))
                self._cached_table_id = table_id
                self._cached_table_index = table_index
                self._cached_table_global_ids = global_ids.copy()

            _, local_ids = self._cached_table_index.search(query_embedding, top_k)
            valid = local_ids[0][local_ids[0] >= 0]
            mapped = self._cached_table_global_ids[valid]
            return mapped.reshape(1, -1)

    def _clear_table_search_cache(self) -> None:
        self._cached_table_id = None
        self._cached_table_index = None
        self._cached_table_global_ids = None

    def _embed_in_batches(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            vecs.append(self.embedder.encode(texts[i: i + batch_size]))
        return np.vstack(vecs)

class DocumentRetriever:
    """
    Views 1 & 2 combined.
    View 1 (Text Chunks): T and D̂ (markdown table) dense recall + reranking.
    View 2 (Table Chunks): maintains mapping f: chunk_idx → (table_id, schema).
    """

    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = DOCUMENT_CHUNK_SIZE,
        chunk_overlap: int = DOCUMENT_CHUNK_OVERLAP,
        reranker: Optional[CrossEncoderCandidateReranker] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("document chunk size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "document chunk overlap must be non-negative and smaller than chunk size"
            )
        self.embedder = embedder
        self.reranker = reranker
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_unit = DOCUMENT_CHUNK_UNIT
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_length,
        )
        self.chunks: List[str] = []
        self.chunk_source: List[str] = []   # filename per chunk
        self.chunk_type: List[str] = []     # "text" | "table"
        # mapping f: chunk_idx → {table_id, schema} (View 2)
        self.chunk_schema_map: Dict[int, Dict[str, Any]] = {}
        self._index: Optional[faiss.IndexFlatIP] = None

    def _token_length(self, text: str) -> int:
        """Measure chunks with the same tokenizer used by the BGE-M3 embedder."""
        return len(self.embedder.tokenizer.encode(text, add_special_tokens=False))

    def _split_document(self, filename: str, text: str) -> List[str]:
        """Split content so the filename-prefixed embedding stays within the limit."""
        prefix = f"File name: {filename}\n"
        content_size = self.chunk_size - self._token_length(prefix)
        if content_size <= self.chunk_overlap:
            raise ValueError(
                "filename metadata leaves no room for the configured document "
                "chunk overlap"
            )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=content_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
        )
        return [prefix + split for split in splitter.split_text(text)]

    # ── ingestion ───────────────────────────────────────────────────────────

    def add_text_document(self, filename: str, text: str) -> None:
        """View 1: Add a plain text document."""
        for chunk in self._split_document(filename, text):
            self.chunk_source.append(filename)
            self.chunk_type.append("text")
            self.chunks.append(chunk)

    def add_table_document(
        self,
        filename: str,
        markdown: str,
        table_id: str,
        schema: Dict[str, Any],
    ) -> None:
        """View 2: Add a flattened table chunk with schema mapping."""
        for chunk in self._split_document(filename, markdown):
            idx = len(self.chunks)
            self.chunk_source.append(filename)
            self.chunk_type.append("table")
            # mapping f: chunk_idx → {table_id, schema}
            self.chunk_schema_map[idx] = {"table_id": table_id, "schema": schema}
            self.chunks.append(chunk)

    # ── index build ─────────────────────────────────────────────────────────

    def build(self) -> None:
        if not self.chunks:
            return
        embeddings = self._embed_in_batches(self.chunks)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        logger.info(f"DocumentRetriever built: {len(self.chunks)} chunks.")

    # ── retrieval ───────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Returns list of chunk dicts:
          {text, source, type, schema (if table chunk)}
        """
        if self._index is None or not self.chunks:
            return []
        q_emb = self.embedder.encode([query]).astype(np.float32)
        top_k = min(top_k, len(self.chunks))
        search_k = (
            self.reranker.recall_size(top_k, len(self.chunks))
            if self.reranker else top_k
        )
        _, I = self._index.search(q_emb, search_k)
        results = []
        for i in I[0]:
            if i >= len(self.chunks):
                continue
            entry: Dict[str, Any] = {
                "text": self.chunks[i],
                "source": self.chunk_source[i],
                "type": self.chunk_type[i],
            }
            if i in self.chunk_schema_map:
                entry["table_id"] = self.chunk_schema_map[i]["table_id"]
                entry["schema"] = self.chunk_schema_map[i]["schema"]
            results.append(entry)
        if not self.reranker:
            return results[:top_k]
        return self.reranker.select(
            query, results, [entry["text"] for entry in results], top_k
        )

    def restore_schema_via_mapping(
        self, chunks: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        §3.2 View 2, mapping f: D̂_{i,j} → S(D_i).
        Given retrieved chunks, return the schema of the first table chunk found.
        """
        for chunk in chunks:
            if chunk.get("type") == "table" and "schema" in chunk:
                return chunk["schema"]
        return None

    def _embed_in_batches(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            vecs.append(self.embedder.encode(texts[i: i + batch_size]))
        return np.vstack(vecs)

class MultiviewIndex:
    """
    §3.2  Five-View Multiview Index.

    After `build()`:
      self.doc_retriever    → Views 1 & 2
      self.schema_index     → View 3
      self.cell_index       → View 4 cell component
      self.row_index        → View 4 row component (Phase F-A)
      View 5 (Relational DB) is the external Flask/MySQL service.

    After `save()` / `load()` the same object is fully restored from disk.
    """

    def __init__(
        self,
        excel_dir: str,
        doc_dir: str,
        bge_model_path: str,
        save_path: str,
        budget: int = 10_000,
        per_table_quota: int = 50,
        device: str = "auto",
        require_cuda: bool = False,
        reranker_model_path: Optional[str] = None,
        rerank_candidate_multiplier: int = 6,
        document_chunk_size: int = DOCUMENT_CHUNK_SIZE,
        document_chunk_overlap: int = DOCUMENT_CHUNK_OVERLAP,
        row_budget: Optional[int] = None,
        row_per_table_quota: Optional[int] = None,
        max_row_chars: Optional[int] = None,
    ) -> None:
        self.excel_dir = excel_dir
        self.doc_dir = doc_dir
        self.bge_model_path = bge_model_path
        self.save_path = save_path
        self.budget = budget
        self.per_table_quota = per_table_quota

        self.embedder = Embedder(bge_model_path, device=device, require_cuda=require_cuda)
        self.reranker = (
            _load_reranker(
                reranker_model_path,
                device,
                require_cuda,
                rerank_candidate_multiplier,
            )
            if reranker_model_path else None
        )
        self.doc_retriever = DocumentRetriever(
            self.embedder,
            chunk_size=document_chunk_size,
            chunk_overlap=document_chunk_overlap,
            reranker=self.reranker,
        )
        self.schema_index = SchemaIndex(self.embedder, reranker=self.reranker)
        self.cell_index = CellIndex(
            self.embedder,
            budget=budget,
            per_table_quota=per_table_quota,
            reranker=self.reranker,
        )
        # Phase F-A: View 4 (row component), Row Index (per-row embeddings for RETRIEVE)
        self.row_index = RowIndex(
            self.embedder,
            budget=row_budget,
            per_table_quota=row_per_table_quota,
            max_row_chars=max_row_chars,
            reranker=self.reranker,
        )

        # table_id → schema (for SQL-route schema lookup)
        self.table_schemas: Dict[str, Dict[str, Any]] = {}

    # ── offline build ───────────────────────────────────────────────────────

    def build(self) -> None:
        """
        §3.2: Ingestion and component separation.
        Process all Excel/CSV (Table Component) and JSON (Text Component).
        """
        self._ingest_tables()
        self._ingest_text_docs()
        self.doc_retriever.build()
        self.schema_index.build()
        self.cell_index.build()
        self.row_index.build()  # Phase F-A: View 4 (row component)
        logger.info("MultiviewIndex build complete.")

    def _ingest_tables(self) -> None:
        if not os.path.isdir(self.excel_dir):
            return
        for fname in os.listdir(self.excel_dir):
            fpath = os.path.join(self.excel_dir, fname)
            lower = fname.lower()
            if not (lower.endswith(".xlsx") or lower.endswith(".csv")):
                continue

            df = _load_dataframe(fpath)
            if df is None or df.empty:
                continue

            table_name = os.path.splitext(fname)[0]
            table_id = table_name
            schema = _extract_schema(df, table_name)
            self.table_schemas[table_id] = schema

            # View 2: table markdown chunk + mapping
            if lower.endswith(".xlsx"):
                markdown = excel_to_markdown(fpath)
            else:
                markdown = df.to_markdown(index=False)
            self.doc_retriever.add_table_document(fname, markdown, table_id, schema)

            # View 3: schema column entries
            self.schema_index.add_table(table_id, schema)

            # View 4: cell entries (budget-aware)
            self.cell_index.add_table(table_id, df)

            # View 4 (row component): row entries (Phase F-A, per-row text for RETRIEVE)
            self.row_index.add_table(table_id, df)

    def _ingest_text_docs(self) -> None:
        if not os.path.isdir(self.doc_dir):
            return
        for fname in os.listdir(self.doc_dir):
            fpath = os.path.join(self.doc_dir, fname)
            try:
                if fname.lower().endswith(".json"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        text = "\n".join(f"{k}: {v}" for k, v in data.items())
                    else:
                        text = str(data)
                else:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read()
                self.doc_retriever.add_text_document(fname, text)
            except Exception as exc:
                logger.warning(f"Failed to load text doc {fname}: {exc}")

    # ── online query API (delegated to sub-indexes) ─────────────────────────

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """§3.5 DocumentRetrieval(q_t): Views 1 & 2."""
        return self.doc_retriever.retrieve(query, top_k=top_k)

    def restore_schema_via_mapping(
        self, chunks: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """§3.2 View 2: mapping f."""
        return self.doc_retriever.restore_schema_via_mapping(chunks)

    def retrieve_schema(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """§3.5 SchemaCellRetrieval: View 3."""
        return self.schema_index.retrieve(query, top_k=top_k)

    def retrieve_cells(
        self,
        entity: str,
        column: Optional[str] = None,
        top_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """§3.5 SchemaCellRetrieval: View 4, Stage 1."""
        return self.cell_index.retrieve(entity, column=column, top_k=top_k)

    def retrieve_rows(
        self,
        query: str,
        top_k: int = 10,
        table_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Phase F-A: Row-level retrieval (View 4 (row component)).

        Returns top-K rows ranked by similarity to `query`. If `table_id` is
        given (e.g., recovered from schema retrieval), restrict to that table.
        """
        return self.row_index.retrieve(query, top_k=top_k, table_id=table_id)

    def schema_cell_retrieval(
        self,
        query: str,
        entity_mentions: List[str],
        schema_top_k: int = 5,
        cell_top_k: int = 15,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Combined schema + cell retrieval used by RETRIEVE and HYBRID routes.
        Returns (C, V_raw) where:
          C       = list of relevant column entries (View 3)
          V_raw   = {entity: [candidate cell entries]} (View 4, per entity)
        """
        C = self.retrieve_schema(query, top_k=schema_top_k)
        col_names = [entry["col_name"] for entry in C]

        V_raw: Dict[str, List[Dict[str, Any]]] = {}
        for entity in entity_mentions:
            candidates = []
            for col in col_names:
                candidates.extend(self.retrieve_cells(entity, column=col, top_k=cell_top_k))
            # De-duplicate by (col_name, value) and keep top-K overall
            seen = set()
            deduped = []
            for c in candidates:
                key = (c["col_name"], c["value"])
                if key not in seen:
                    seen.add(key)
                    deduped.append(c)
            V_raw[entity] = deduped[:cell_top_k]

        return C, V_raw

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        payload = {
            "index_format_version": INDEX_FORMAT_VERSION,
            "doc_chunks": self.doc_retriever.chunks,
            "doc_source": self.doc_retriever.chunk_source,
            "doc_type": self.doc_retriever.chunk_type,
            "doc_schema_map": self.doc_retriever.chunk_schema_map,
            "document_chunk_config": {
                "size": self.doc_retriever.chunk_size,
                "overlap": self.doc_retriever.chunk_overlap,
                "unit": self.doc_retriever.chunk_unit,
            },
            "index_build_config": {
                "document_chunks": {
                    "size": self.doc_retriever.chunk_size,
                    "overlap": self.doc_retriever.chunk_overlap,
                    "unit": self.doc_retriever.chunk_unit,
                },
                "cell_index": {
                    "budget": self.cell_index.budget,
                    "per_table_quota": self.cell_index.per_table_quota,
                },
                "row_index": {
                    "budget": self.row_index.budget,
                    "per_table_quota": self.row_index.per_table_quota,
                    "max_row_chars": self.row_index.max_row_chars,
                },
                "embedding_model": os.path.basename(
                    os.path.normpath(getattr(self, "bge_model_path", ""))
                ),
            },
            "schema_entries": self.schema_index.entries,
            "cell_entries": self.cell_index.entries,
            "row_entries": self.row_index.entries,   # Phase F-A: View 4 (row component)
            "table_schemas": self.table_schemas,
        }
        # Save FAISS indexes separately (binary)
        if self.doc_retriever._index is not None:
            faiss.write_index(
                self.doc_retriever._index, self.save_path + ".doc.faiss"
            )
        if self.schema_index._index is not None:
            faiss.write_index(
                self.schema_index._index, self.save_path + ".schema.faiss"
            )
        if self.cell_index._index is not None:
            faiss.write_index(
                self.cell_index._index, self.save_path + ".cell.faiss"
            )
        if self.row_index._index is not None:        # Phase F-A
            faiss.write_index(
                self.row_index._index, self.save_path + ".row.faiss"
            )
        with open(self.save_path + ".meta.pkl", "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"MultiviewIndex saved to {self.save_path}.*")

    @classmethod
    def load(
        cls,
        save_path: str,
        bge_model_path: str,
        device: str = "auto",
        require_cuda: bool = False,
        reranker_model_path: Optional[str] = None,
        enable_reranker: bool = True,
        rerank_candidate_multiplier: int = 6,
    ) -> "MultiviewIndex":
        with open(save_path + ".meta.pkl", "rb") as f:
            payload = pickle.load(f)

        _validate_index_payload(save_path, payload)
        faiss_indices = _load_validated_faiss_indices(save_path, payload)

        embedder = Embedder(bge_model_path, device=device, require_cuda=require_cuda)
        reranker = None
        if enable_reranker:
            resolved_reranker_path = (
                reranker_model_path or _default_reranker_path(bge_model_path)
            )
            reranker = _load_reranker(
                resolved_reranker_path,
                device,
                require_cuda,
                rerank_candidate_multiplier,
            )

        instance = cls.__new__(cls)
        instance.embedder = embedder
        instance.reranker = reranker
        instance.save_path = save_path
        instance.bge_model_path = bge_model_path
        instance.table_schemas = payload["table_schemas"]

        # Restore DocumentRetriever (Views 1+2). Compatibility was validated
        # before loading the heavyweight embedding/reranker models.
        chunk_config = payload.get("document_chunk_config")
        dr = DocumentRetriever(
            embedder,
            chunk_size=int(chunk_config["size"]),
            chunk_overlap=int(chunk_config["overlap"]),
            reranker=reranker,
        )
        dr.chunks = payload["doc_chunks"]
        dr.chunk_source = payload["doc_source"]
        dr.chunk_type = payload["doc_type"]
        dr.chunk_schema_map = payload["doc_schema_map"]
        dr._index = faiss_indices["doc"]
        instance.doc_retriever = dr

        # Restore SchemaIndex (View 3)
        si = SchemaIndex(embedder, reranker=reranker)
        si.entries = payload["schema_entries"]
        si._index = faiss_indices["schema"]
        instance.schema_index = si

        # Restore CellIndex (View 4)
        cell_config = payload["index_build_config"]["cell_index"]
        ci = CellIndex(
            embedder,
            budget=cell_config["budget"],
            per_table_quota=cell_config["per_table_quota"],
            reranker=reranker,
        )
        ci.entries = payload["cell_entries"]
        ci._index = faiss_indices["cell"]
        instance.cell_index = ci

        # Restore RowIndex (View 4 row component).
        row_config = payload["index_build_config"]["row_index"]
        ri = RowIndex(
            embedder,
            budget=row_config["budget"],
            per_table_quota=row_config["per_table_quota"],
            max_row_chars=row_config["max_row_chars"],
            reranker=reranker,
        )
        ri.entries = payload["row_entries"]
        ri._index = faiss_indices["row"]
        ri._rebuild_table_entry_ids()
        instance.row_index = ri

        logger.info(
            f"MultiviewIndex loaded from {save_path}.* "
            f"(row entries: {len(ri.entries)}, reranker: "
            f"{'enabled' if reranker else 'disabled'})"
        )
        return instance
