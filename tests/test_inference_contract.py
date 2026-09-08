"""Regression tests for verifier acceptance and the paper stop rule."""
import inspect
from types import SimpleNamespace
import unittest

from atr.clients.chat_utils import _decoding_kwargs
from atr.config import ROW_TOP_K
from atr.online.main import (
    AgenticTableRAGAgent,
    BatchRunner,
    _is_verified_answer,
    _retrieve_row_context,
    _should_stop,
)
from atr.online.router import Route
from atr.online.verifier import EvidenceFusionVerifier


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

    def test_retrieve_route_returns_the_rows_used_for_answering(self):
        captured = {}

        class _Index:
            row_index = SimpleNamespace(entries=[object()])

            def schema_cell_retrieval(self, **_kwargs):
                return ([{
                    "table_id": "clubs",
                    "col_name": "Club",
                    "dtype": "text",
                    "examples": "Alpha FC",
                }], {})

            def retrieve_rows(self, query, top_k, table_id):
                return [{
                    "table_id": table_id,
                    "row_idx": 7,
                    "row_dict": {"Club": "Alpha FC", "Founded": "1971"},
                }]

        class _Verifier:
            def answer_from_retrieval(self, **kwargs):
                captured.update(kwargs)
                return "Alpha FC"

        agent = object.__new__(AgenticTableRAGAgent)
        agent.index = _Index()
        agent.verifier = _Verifier()
        agent.no_value_linker = True
        sub_q = SimpleNamespace(
            sub_query="Which club was founded in 1971?",
            entity_mentions=[],
            expected_operator="lookup",
        )

        answer, sql_result, route_evidence, effective_route = agent._execute_route(
            sub_q=sub_q,
            original_question=sub_q.sub_query,
            schema=None,
            chunks=[{"type": "table", "table_id": "clubs"}],
            text_evidence="document evidence",
            route=Route.RETRIEVE,
            sql_executor=object(),
            H=[],
        )

        self.assertEqual(answer, "Alpha FC")
        self.assertEqual(sql_result, "")
        self.assertEqual(effective_route, Route.RETRIEVE)
        self.assertEqual(captured["table_context"], "[Table clubs, row 7]\n  Club: Alpha FC\n  Founded: 1971")
        self.assertIn(captured["table_context"], route_evidence)
        self.assertIn("Column: Club", route_evidence)

    def test_value_linker_text_fallback_reports_effective_route(self):
        class _Index:
            def schema_cell_retrieval(self, **_kwargs):
                return [], {}

        class _Linker:
            def link(self, **_kwargs):
                return [SimpleNamespace(
                    needs_reroute=True,
                    is_matched=False,
                    entity="entity",
                    column="",
                    matched_value=None,
                )]

        class _Verifier:
            def answer_from_text(self, *_args, **_kwargs):
                return "text answer"

        agent = object.__new__(AgenticTableRAGAgent)
        agent.index = _Index()
        agent.value_linker = _Linker()
        agent.verifier = _Verifier()
        agent.no_value_linker = False
        sub_q = SimpleNamespace(
            sub_query="question",
            entity_mentions=["entity"],
        )

        answer, sql_result, evidence, effective_route = agent._execute_route(
            sub_q=sub_q,
            original_question="question",
            schema=None,
            chunks=[],
            text_evidence="text evidence",
            route=Route.HYBRID,
            sql_executor=object(),
            H=[],
        )

        self.assertEqual(answer, "text answer")
        self.assertEqual(sql_result, "")
        self.assertEqual(evidence, "")
        self.assertEqual(effective_route, Route.TEXT)

    def test_failed_history_and_reselection_use_effective_route(self):
        reselect_calls = []

        class _Verifier:
            def verify(self, **_kwargs):
                return 0.0

            def is_rejected(self, verdict):
                return verdict < 1.0

        class _Router:
            def reselect(self, **kwargs):
                reselect_calls.append(kwargs)
                return Route.TEXT

        agent = object.__new__(AgenticTableRAGAgent)
        agent.verifier = _Verifier()
        agent.router = _Router()
        agent.no_escalation = False
        agent.no_value_linker = False
        agent._execute_route = lambda **_kwargs: (
            "rejected text", "", "", Route.TEXT
        )
        sub_q = SimpleNamespace(
            sub_query="current question",
            expected_operator="lookup",
            required_modalities="both",
            entity_mentions=["entity"],
            need_global_table_view=False,
            uncertainty=0.2,
        )
        history = []
        decisions = []

        result = agent._execute_and_verify(
            sub_q=sub_q,
            original_question="question",
            schema=None,
            chunks=[],
            route=Route.HYBRID,
            sql_executor=object(),
            H=history,
            question_id="q1",
            decisions=decisions,
        )

        self.assertEqual(result, ("rejected text", 0.5))
        self.assertEqual(history[0]["route"], "TEXT")
        self.assertEqual(history[0]["requested_route"], "HYBRID")
        self.assertEqual(history[0]["effective_route"], "TEXT")
        self.assertTrue(history[0]["value_linker_attempted"])
        self.assertEqual(decisions[0]["route"], "TEXT")
        self.assertEqual(decisions[0]["requested_route"], "HYBRID")
        self.assertEqual(reselect_calls[0]["current_route"], Route.TEXT)

    def test_execute_and_verify_forwards_route_evidence(self):
        captured = {}

        class _Verifier:
            def verify(self, **kwargs):
                captured.update(kwargs)
                return 1.0

            @staticmethod
            def is_rejected(verdict):
                return verdict < 1.0

        agent = object.__new__(AgenticTableRAGAgent)
        agent.no_escalation = True
        agent.no_value_linker = False
        agent.verifier = _Verifier()
        agent._execute_route = lambda **_kwargs: (
            "Alpha FC",
            "",
            "[Table clubs, row 7]\n  Founded: 1971",
            Route.RETRIEVE,
        )

        answer, confidence = agent._execute_and_verify(
            sub_q=SimpleNamespace(sub_query="Which club?"),
            original_question="Which club?",
            schema=None,
            chunks=[{"source": "clubs", "text": "document evidence"}],
            route=Route.RETRIEVE,
            sql_executor=object(),
            H=[],
            question_id="Q1",
        )

        self.assertEqual((answer, confidence), ("Alpha FC", 1.0))
        self.assertEqual(
            captured["route_evidence"],
            "[Table clubs, row 7]\n  Founded: 1971",
        )


class VerifierEvidenceTest(unittest.TestCase):
    def test_verifier_prompt_combines_document_and_route_evidence(self):
        calls = []

        def judge(messages):
            calls.append(messages)
            return '{"verdict": 1}'

        verifier = EvidenceFusionVerifier(lambda _messages: "", verify_llm_fn=judge)
        verdict = verifier.verify(
            question="Which club?",
            answer="Alpha FC",
            text_evidence="document evidence",
            sql_result="",
            route_evidence="[Table clubs, row 7]\n  Club: Alpha FC\n  Founded: 1971",
        )

        self.assertEqual(verdict, 1.0)
        prompt = calls[0][0]["content"]
        self.assertIn("document evidence", prompt)
        self.assertIn("[Table clubs, row 7]", prompt)


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
