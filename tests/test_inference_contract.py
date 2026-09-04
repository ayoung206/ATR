"""Regression tests for verifier acceptance and the paper stop rule."""
import inspect
import unittest

from atr.clients.chat_utils import _decoding_kwargs
from atr.config import ROW_TOP_K
from atr.online.main import (
    BatchRunner,
    _is_verified_answer,
    _retrieve_row_context,
    _should_stop,
)


class StopControllerTest(unittest.TestCase):
    def test_uses_configured_threshold(self):
        self.assertFalse(_should_stop(1.0, 0.15, 0.1))
        self.assertTrue(_should_stop(1.0, 0.15, 0.2))

    def test_requires_support_and_strictly_lower_uncertainty(self):
        self.assertFalse(_should_stop(0.5, 0.01, 0.1))
        self.assertFalse(_should_stop(1.0, 0.1, 0.1))
        self.assertTrue(_should_stop(1.0, 0.09, 0.1))


class VerifiedAnswerTest(unittest.TestCase):
    def test_rejected_nonempty_answer_is_not_accepted(self):
        self.assertFalse(_is_verified_answer("plausible but rejected", 0.5))

    def test_supported_valid_answer_is_accepted(self):
        self.assertTrue(_is_verified_answer("supported", 1.0))

    def test_invalid_sentinel_is_not_accepted(self):
        self.assertFalse(_is_verified_answer("not found", 1.0))


class RetrievalProtocolTest(unittest.TestCase):
    def test_row_retrieval_uses_question_once_and_returns_top_10(self):
        class _Index:
            def __init__(self):
                self.calls = []

            def retrieve_rows(self, query, top_k, table_id):
                self.calls.append((query, top_k, table_id))
                return [
                    {
                        "table_id": "clubs",
                        "row_idx": i,
                        "row_dict": {"Club": f"Club {i}"},
                    }
                    for i in range(12)
                ]

        index = _Index()
        context = _retrieve_row_context(
            index,
            "Which clubs were founded in 1971?",
            [{"type": "table", "table_id": "clubs"}],
        )

        self.assertEqual(ROW_TOP_K, 10)
        self.assertEqual(
            index.calls,
            [("Which clubs were founded in 1971?", 10, "clubs")],
        )
        self.assertIn("row 9", context)
        self.assertNotIn("row 10]", context)


class DecodingProtocolTest(unittest.TestCase):
    def test_non_reasoning_backbones_use_temperature_zero(self):
        self.assertEqual(_decoding_kwargs("gemini-2.5-flash")["temperature"], 0.0)
        self.assertEqual(_decoding_kwargs("Qwen3.5")["temperature"], 0.0)

    def test_reasoning_backbone_does_not_receive_temperature(self):
        self.assertNotIn("temperature", _decoding_kwargs("gpt-5.2"))


class WorkerProtocolTest(unittest.TestCase):
    def test_batch_runner_defaults_to_two_question_workers(self):
        default = inspect.signature(BatchRunner.run).parameters["max_workers"].default
        self.assertEqual(default, 2)


if __name__ == "__main__":
    unittest.main()
