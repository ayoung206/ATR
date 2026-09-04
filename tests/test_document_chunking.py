"""Regression tests for the paper-aligned document chunk configuration."""
from __future__ import annotations

import inspect
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from atr.offline.multiview_index import (
    DOCUMENT_CHUNK_OVERLAP,
    DOCUMENT_CHUNK_SIZE,
    CellIndex,
    DocumentRetriever,
    MultiviewIndex,
)


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


class _FakeEmbedder:
    tokenizer = _WhitespaceTokenizer()


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
            index.cell_index = SimpleNamespace(entries=[], _index=None)
            index.row_index = SimpleNamespace(entries=[], _index=None)
            index.table_schemas = {}

            index.save()

            with open(index.save_path + ".meta.pkl", "rb") as metadata_file:
                payload = pickle.load(metadata_file)
            self.assertEqual(
                payload["document_chunk_config"],
                {"size": 512, "overlap": 64, "unit": "tokens"},
            )

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
