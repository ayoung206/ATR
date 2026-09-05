"""Regression tests for the paper-aligned document chunk configuration."""
from __future__ import annotations

import inspect
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np

from atr.offline.multiview_index import (
    DOCUMENT_CHUNK_OVERLAP,
    DOCUMENT_CHUNK_SIZE,
    INDEX_FORMAT_VERSION,
    CellIndex,
    DocumentRetriever,
    IncompatibleIndexError,
    MultiviewIndex,
    _load_validated_faiss_indices,
    _validate_index_payload,
)
from atr.offline import multiview_index as subject


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


class _FakeEmbedder:
    tokenizer = _WhitespaceTokenizer()


def _current_payload(**overrides):
    chunk_config = {"size": 512, "overlap": 64, "unit": "tokens"}
    payload = {
        "index_format_version": INDEX_FORMAT_VERSION,
        "doc_chunks": [],
        "doc_source": [],
        "doc_type": [],
        "doc_schema_map": {},
        "document_chunk_config": chunk_config,
        "index_build_config": {
            "document_chunks": chunk_config,
            "embedding_model": "bge-m3",
            "cell_index": {"budget": 10_000, "per_table_quota": 50},
            "row_index": {
                "budget": None,
                "per_table_quota": None,
                "max_row_chars": None,
            },
        },
        "schema_entries": [],
        "cell_entries": [],
        "row_entries": [],
        "table_schemas": {},
    }
    payload.update(overrides)
    return payload


class DocumentChunkingTest(unittest.TestCase):
    def test_defaults_match_paper_and_measure_tokens(self):
        retriever = DocumentRetriever(_FakeEmbedder())
        retriever.add_text_document(
            "doc.txt", " ".join(f"long_token_{i:03d}" for i in range(300))
        )

        self.assertEqual(retriever.chunk_size, DOCUMENT_CHUNK_SIZE)
        self.assertEqual(retriever.chunk_overlap, DOCUMENT_CHUNK_OVERLAP)
        self.assertEqual(retriever.chunk_unit, "tokens")
        self.assertEqual(len(retriever.chunks), 1)

    def test_prefix_and_content_fit_in_512_tokens_with_64_token_overlap(self):
        retriever = DocumentRetriever(_FakeEmbedder())
        retriever.add_text_document(
            "doc.txt", " ".join(f"token_{i:03d}" for i in range(700))
        )

        tokenized = [chunk.split() for chunk in retriever.chunks]
        self.assertGreater(len(tokenized), 1)
        self.assertTrue(all(len(tokens) <= DOCUMENT_CHUNK_SIZE for tokens in tokenized))
        # The three-token filename prefix is repeated; compare content only.
        first_content = tokenized[0][3:]
        second_content = tokenized[1][3:]
        self.assertEqual(
            first_content[-DOCUMENT_CHUNK_OVERLAP:],
            second_content[:DOCUMENT_CHUNK_OVERLAP],
        )

    def test_rejects_invalid_chunk_configuration(self):
        with self.assertRaises(ValueError):
            DocumentRetriever(_FakeEmbedder(), chunk_size=0)
        with self.assertRaises(ValueError):
            DocumentRetriever(_FakeEmbedder(), chunk_size=64, chunk_overlap=64)

    def test_save_persists_chunk_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            index = MultiviewIndex.__new__(MultiviewIndex)
            index.save_path = str(Path(temp_dir) / "index")
            index.doc_retriever = DocumentRetriever(_FakeEmbedder())
            index.schema_index = SimpleNamespace(entries=[], _index=None)
            index.cell_index = SimpleNamespace(
                entries=[], _index=None, budget=10_000, per_table_quota=50
            )
            index.row_index = SimpleNamespace(
                entries=[],
                _index=None,
                budget=None,
                per_table_quota=None,
                max_row_chars=None,
            )
            index.table_schemas = {}
            index.bge_model_path = "BAAI/bge-m3"

            index.save()

            with open(index.save_path + ".meta.pkl", "rb") as metadata_file:
                payload = pickle.load(metadata_file)
            self.assertEqual(
                payload["document_chunk_config"],
                {"size": 512, "overlap": 64, "unit": "tokens"},
            )
            self.assertEqual(payload["index_format_version"], INDEX_FORMAT_VERSION)
            self.assertEqual(
                payload["index_build_config"]["cell_index"]["budget"], 10_000
            )
            self.assertEqual(
                payload["index_build_config"]["row_index"]["per_table_quota"],
                None,
            )

    def test_rejects_legacy_metadata_instead_of_relabelling_it(self):
        legacy_payload = {
            "doc_chunks": [],
            "doc_source": [],
            "doc_type": [],
            "doc_schema_map": {},
            "schema_entries": [],
            "cell_entries": [],
            "table_schemas": {},
        }
        with self.assertRaisesRegex(IncompatibleIndexError, "Rebuild every component"):
            _validate_index_payload("legacy", legacy_payload)

    def test_rejects_previous_row_index_contract(self):
        payload = _current_payload(index_format_version=2)
        with self.assertRaisesRegex(IncompatibleIndexError, "expected 3"):
            _validate_index_payload("capped-row-index", payload)

    def test_rejects_table_index_without_rows(self):
        payload = _current_payload(
            table_schemas={"clubs": {"table_name": "clubs", "columns": []}}
        )
        with self.assertRaisesRegex(IncompatibleIndexError, "row index is empty"):
            _validate_index_payload("missing-row-index", payload)

    def test_rejects_missing_faiss_file_for_nonempty_metadata(self):
        payload = _current_payload(
            schema_entries=[{"table_id": "clubs", "col_name": "Club"}]
        )
        _validate_index_payload("missing-faiss", payload)
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = str(Path(temp_dir) / "index")
            with self.assertRaisesRegex(IncompatibleIndexError, "missing schema FAISS"):
                _load_validated_faiss_indices(save_path, payload)

    def test_rejects_faiss_metadata_count_mismatch(self):
        payload = _current_payload(
            schema_entries=[
                {"table_id": "clubs", "col_name": "Club"},
                {"table_id": "clubs", "col_name": "Founded"},
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = str(Path(temp_dir) / "index")
            index = faiss.IndexFlatIP(2)
            index.add(np.zeros((1, 2), dtype=np.float32))
            faiss.write_index(index, save_path + ".schema.faiss")
            with self.assertRaisesRegex(IncompatibleIndexError, "does not match"):
                _load_validated_faiss_indices(save_path, payload)

    def test_current_empty_index_round_trips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = str(Path(temp_dir) / "index")
            index = MultiviewIndex.__new__(MultiviewIndex)
            index.save_path = save_path
            index.bge_model_path = "BAAI/bge-m3"
            index.doc_retriever = DocumentRetriever(_FakeEmbedder())
            index.schema_index = SimpleNamespace(entries=[], _index=None)
            index.cell_index = SimpleNamespace(
                entries=[], _index=None, budget=10_000, per_table_quota=50
            )
            index.row_index = SimpleNamespace(
                entries=[],
                _index=None,
                budget=None,
                per_table_quota=None,
                max_row_chars=None,
            )
            index.table_schemas = {}
            index.save()

            original_embedder = subject.Embedder
            subject.Embedder = lambda *_args, **_kwargs: _FakeEmbedder()
            try:
                loaded = MultiviewIndex.load(
                    save_path,
                    bge_model_path="BAAI/bge-m3",
                    enable_reranker=False,
                )
            finally:
                subject.Embedder = original_embedder

            self.assertEqual(loaded.bge_model_path, "BAAI/bge-m3")
            self.assertEqual(loaded.cell_index.budget, 10_000)
            self.assertIsNone(loaded.row_index.per_table_quota)

    def test_cell_index_default_budget_matches_paper(self):
        self.assertEqual(
            inspect.signature(CellIndex).parameters["budget"].default,
            10_000,
        )
        self.assertEqual(
            inspect.signature(MultiviewIndex).parameters["budget"].default,
            10_000,
        )


if __name__ == "__main__":
    unittest.main()
