"""Multiview index construction: text chunks, table chunks + mapping, schema, cell/row index. The relational DB (View 5) is the external Flask/MySQL service."""
from __future__ import annotations

import os
import json
import pickle
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

from atr.clients.tool_utils import Embedder, excel_to_markdown  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

logger = logging.getLogger(__name__)

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
    Per-column embeddings for "which column is relevant?" retrieval.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
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
        texts = [
            f"column: {e['col_name']} | type: {e['dtype']} | examples: {e['examples']}"
            for e in self.entries
        ]
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
        _, I = self._index.search(q_emb, top_k)
        return [self.entries[i] for i in I[0] if i < len(self.entries)]

    def _embed_in_batches(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            vecs.append(self.embedder.encode(texts[i: i + batch_size]))
        return np.vstack(vecs)

class CellIndex:
    """
    View 4: Cell / Row Index (§3.2), a new contribution.
    Cell component: (column, value) pair embeddings with budget-B
    frequency-aware truncation. Directly supports Constrained Value Linking
    Stage 1 (§3.5). The row component is implemented separately as
    `RowIndex` and registered under the same View 4 conceptual umbrella
    (cells for entity-to-column linking; rows for filtered table context
    fed to the RETRIEVE primitive).
    """

    def __init__(
        self,
        embedder: Embedder,
        budget: int = 200_000,
        per_table_quota: int = 50,
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
        texts = [
            f"column: {e['col_name']} | value: {e['value']}"
            for e in self.entries
        ]
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
        _, I = self._index.search(q_emb, top_k)
        return [self.entries[i] for i in I[0] if i < len(self.entries)]

    def _embed_in_batches(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            vecs.append(self.embedder.encode(texts[i: i + batch_size]))
        return np.vstack(vecs)

class RowIndex:
    """
    View 4 row component: RowIndex (§3.2, Phase F-A).

    Per-row text embeddings, indexed for entity-aware retrieval.
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
        budget: int = 500_000,
        per_table_quota: int = 100,
        max_row_chars: int = 600,
    ) -> None:
        self.embedder = embedder
        self.budget = budget
        self.per_table_quota = per_table_quota
        self.max_row_chars = max_row_chars
        # [{table_id, row_idx, row_text, row_dict}]
        self.entries: List[Dict[str, Any]] = []
        self._index: Optional[faiss.IndexFlatIP] = None

    def add_table(self, table_id: str, df: pd.DataFrame) -> None:
        """Encode each row up to `per_table_quota`, respecting the global budget."""
        remaining_global = max(0, self.budget - len(self.entries))
        take = min(self.per_table_quota, remaining_global, len(df))
        if take == 0:
            return

        for row_idx in range(take):
            row = df.iloc[row_idx]
            parts = [f"{col}: {row[col]}" for col in df.columns
                     if pd.notna(row[col]) and str(row[col]).strip()]
            row_text = f"Table {table_id} | " + " | ".join(parts)
            if len(row_text) > self.max_row_chars:
                row_text = row_text[: self.max_row_chars - 1] + "…"

            self.entries.append({
                "table_id": table_id,
                "row_idx": int(row_idx),
                "row_text": row_text,
                "row_dict": {c: (str(row[c]) if pd.notna(row[c]) else "")
                             for c in df.columns},
            })

    def build(self) -> None:
        if not self.entries:
            return
        texts = [e["row_text"] for e in self.entries]
        embeddings = self._embed_in_batches(texts)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        logger.info(f"RowIndex built: {len(self.entries)} rows.")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        table_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top-K rows ranked by similarity to the query.
        If `table_id` is given, restrict to that table only (post-filter).
        """
        if self._index is None or len(self.entries) == 0:
            return []
        q_emb = self.embedder.encode([query]).astype(np.float32)
        # Over-fetch when filtering by table_id so we still get top_k hits
        search_k = min(len(self.entries), top_k * (10 if table_id else 1))
        _, I = self._index.search(q_emb, search_k)

        out = []
        for i in I[0]:
            if i < 0 or i >= len(self.entries):
                continue
            e = self.entries[i]
            if table_id is not None and e["table_id"] != table_id:
                continue
            out.append(e)
            if len(out) >= top_k:
                break
        return out

    def _embed_in_batches(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            vecs.append(self.embedder.encode(texts[i: i + batch_size]))
        return np.vstack(vecs)

class DocumentRetriever:
    """
    Views 1 & 2 combined.
    View 1 (Text Chunks): T and D̂ (markdown table) chunking + dense embedding.
    View 2 (Table Chunks): maintains mapping f: chunk_idx → (table_id, schema).
    """

    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.embedder = embedder
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.chunks: List[str] = []
        self.chunk_source: List[str] = []   # filename per chunk
        self.chunk_type: List[str] = []     # "text" | "table"
        # mapping f: chunk_idx → {table_id, schema} (View 2)
        self.chunk_schema_map: Dict[int, Dict[str, Any]] = {}
        self._index: Optional[faiss.IndexFlatIP] = None

    # ── ingestion ───────────────────────────────────────────────────────────

    def add_text_document(self, filename: str, text: str) -> None:
        """View 1: Add a plain text document."""
        splits = self.splitter.split_text(text)
        for split in splits:
            chunk = f"File name: {filename}\n{split}"
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
        splits = self.splitter.split_text(markdown)
        for split in splits:
            chunk = f"File name: {filename}\n{split}"
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
        _, I = self._index.search(q_emb, top_k)
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
        return results

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
        budget: int = 200_000,
        per_table_quota: int = 50,
        device: str = "auto",
        require_cuda: bool = False,
    ) -> None:
        self.excel_dir = excel_dir
        self.doc_dir = doc_dir
        self.bge_model_path = bge_model_path
        self.save_path = save_path
        self.budget = budget
        self.per_table_quota = per_table_quota

        self.embedder = Embedder(bge_model_path, device=device, require_cuda=require_cuda)
        self.doc_retriever = DocumentRetriever(self.embedder)
        self.schema_index = SchemaIndex(self.embedder)
        self.cell_index = CellIndex(
            self.embedder,
            budget=budget,
            per_table_quota=per_table_quota,
        )
        # Phase F-A: View 4 (row component), Row Index (per-row embeddings for RETRIEVE)
        self.row_index = RowIndex(
            self.embedder,
            budget=max(budget, 500_000),
            per_table_quota=max(per_table_quota, 100),
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
        top_k: int = 5,
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
            "doc_chunks": self.doc_retriever.chunks,
            "doc_source": self.doc_retriever.chunk_source,
            "doc_type": self.doc_retriever.chunk_type,
            "doc_schema_map": self.doc_retriever.chunk_schema_map,
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
    ) -> "MultiviewIndex":
        with open(save_path + ".meta.pkl", "rb") as f:
            payload = pickle.load(f)

        embedder = Embedder(bge_model_path, device=device, require_cuda=require_cuda)

        instance = cls.__new__(cls)
        instance.embedder = embedder
        instance.save_path = save_path
        instance.table_schemas = payload["table_schemas"]

        # Restore DocumentRetriever (Views 1+2)
        dr = DocumentRetriever(embedder)
        dr.chunks = payload["doc_chunks"]
        dr.chunk_source = payload["doc_source"]
        dr.chunk_type = payload["doc_type"]
        dr.chunk_schema_map = payload["doc_schema_map"]
        doc_faiss_path = save_path + ".doc.faiss"
        if os.path.exists(doc_faiss_path):
            dr._index = faiss.read_index(doc_faiss_path)
        instance.doc_retriever = dr

        # Restore SchemaIndex (View 3)
        si = SchemaIndex(embedder)
        si.entries = payload["schema_entries"]
        schema_faiss_path = save_path + ".schema.faiss"
        if os.path.exists(schema_faiss_path):
            si._index = faiss.read_index(schema_faiss_path)
        instance.schema_index = si

        # Restore CellIndex (View 4)
        ci = CellIndex(embedder)
        ci.entries = payload["cell_entries"]
        cell_faiss_path = save_path + ".cell.faiss"
        if os.path.exists(cell_faiss_path):
            ci._index = faiss.read_index(cell_faiss_path)
        instance.cell_index = ci

        # Restore RowIndex (View 4 (row component), Phase F-A; backward-compatible if absent)
        ri = RowIndex(embedder)
        ri.entries = payload.get("row_entries", [])
        row_faiss_path = save_path + ".row.faiss"
        if os.path.exists(row_faiss_path):
            ri._index = faiss.read_index(row_faiss_path)
        instance.row_index = ri

        logger.info(f"MultiviewIndex loaded from {save_path}.* "
                    f"(row entries: {len(ri.entries)})")
        return instance
