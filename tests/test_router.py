"""Regression tests for the four-class learned router."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from atr.online.router import HeuristicRouter, LLMRouter, LearnedRouter, Route
from atr.tools.train_router import load_inference_oracle


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeTorch:
    @staticmethod
    def no_grad():
        return _NoGrad()


class _FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids="fake", attention_mask="fake")

    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self):
        self.last_text = ""

    def __call__(self, *args, **kwargs):
        self.last_text = args[0]
        return _FakeInputs()


class _Order:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class _Logits:
    def __init__(self, values):
        self.values = values

    def argsort(self, descending=False):
        order = sorted(
            range(len(self.values)),
            key=self.values.__getitem__,
            reverse=descending,
        )
        return _Order(order)

    def __getitem__(self, index):
        return self.values[index]


class _FakeModel:
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(logits=[_Logits(self.values)])


def _router_with_logits(values):
    router = LearnedRouter.__new__(LearnedRouter)
    router._torch = _FakeTorch()
    router.device = "cpu"
    router.tokenizer = _FakeTokenizer()
    router.model = _FakeModel(values)
    router._last_logits = None
    return router


class LearnedRouterTest(unittest.TestCase):
    def setUp(self):
        self.meta = SimpleNamespace(
            expected_operator="lookup",
            required_modalities="table",
            entity_mentions=["1971"],
            need_global_table_view=True,
            uncertainty=0.4,
        )

    def test_retrieve_prediction_is_not_remapped(self):
        router = _router_with_logits([0.1, 0.2, 0.9, 0.3])

        route = router.route("Which club was founded in 1971?", self.meta, {}, [])

        self.assertEqual(route, Route.RETRIEVE)

    def test_failed_retrieve_is_excluded(self):
        router = _router_with_logits([0.1, 0.8, 0.9, 0.3])

        route = router.route(
            "Which club was founded in 1971?",
            self.meta,
            {},
            [{"route": "RETRIEVE", "verdict": 0.5}],
        )

        self.assertEqual(route, Route.SQL)

    def test_failure_from_an_earlier_sub_query_does_not_mask_route(self):
        router = _router_with_logits([0.1, 0.2, 0.9, 0.3])

        route = router.route(
            "Which club was founded in 1971?",
            self.meta,
            {},
            [{
                "sub_query": "What league did the club play in?",
                "route": "RETRIEVE",
                "verdict": 0.5,
            }],
        )

        self.assertEqual(route, Route.RETRIEVE)

    def test_reselect_reinvokes_student_with_schema_and_failure_history(self):
        router = _router_with_logits([0.1, 0.8, 0.9, 0.3])
        schema = {
            "table_name": "clubs",
            "columns": [["Club", "object", "A"], ["Founded", "int", "1971"]],
        }

        first = router.route("Which club was founded in 1971?", self.meta, schema, [])
        second = router.reselect(
            "Which club was founded in 1971?",
            self.meta,
            schema,
            [{"route": "RETRIEVE", "verdict": 0.5}],
            first,
        )

        self.assertEqual(first, Route.RETRIEVE)
        self.assertEqual(second, Route.SQL)
        self.assertEqual(router.model.calls, 2)
        self.assertIn("[GLOBAL] yes", router.tokenizer.last_text)
        self.assertIn("[FAILED] RETRIEVE:0.5", router.tokenizer.last_text)
        self.assertIn("Club:object", router.tokenizer.last_text)


class LLMRouterTest(unittest.TestCase):
    def test_prompt_contains_full_contract_and_never_repeats_failed_route(self):
        captured = []

        def llm_fn(messages):
            captured.append(messages[0]["content"])
            return "RETRIEVE"

        meta = SimpleNamespace(
            expected_operator="lookup",
            required_modalities="table",
            entity_mentions=["1971"],
            need_global_table_view=True,
            uncertainty=0.4,
        )
        schema = {"table_name": "clubs", "columns": [["Club", "object", "A"]]}

        route = LLMRouter(llm_fn).route(
            "Which club was founded in 1971?",
            meta,
            schema,
            [{"route": "RETRIEVE", "verdict": 0.5}],
        )

        self.assertEqual(route, Route.SQL)
        self.assertIn("Need global table view : yes", captured[0])
        self.assertIn("Restored schema        : table=clubs; columns=Club:object", captured[0])
        self.assertIn("Already-failed routes  : [RETRIEVE]", captured[0])
        self.assertIn("Failure history H      : RETRIEVE:0.5", captured[0])


class HeuristicRouterTest(unittest.TestCase):
    def test_reselection_does_not_repeat_text_when_another_route_is_available(self):
        meta = SimpleNamespace(
            expected_operator="lookup",
            required_modalities="text",
            entity_mentions=[],
            need_global_table_view=False,
            uncertainty=0.4,
        )

        route = HeuristicRouter().route(
            "Describe the club",
            meta,
            None,
            [{"route": "TEXT", "verdict": 0.5}],
        )

        self.assertEqual(route, Route.RETRIEVE)


class InferenceLabelTest(unittest.TestCase):
    def test_trace_export_keeps_only_accepted_route_and_full_router_input(self):
        record = {
            "question_id": "q1",
            "question": "Which club was founded in 1971?",
            "table_id": "clubs",
            "atr_trace": {
                "verifier_decisions": [
                    {"route": "RETRIEVE", "accepted": False},
                    {
                        "question_id": "q1",
                        "sub_query": "Which club was founded in 1971?",
                        "expected_operator": "lookup",
                        "required_modalities": "table",
                        "entity_mentions": ["1971"],
                        "need_global_table_view": True,
                        "uncertainty": 0.4,
                        "schema": {
                            "table_name": "clubs",
                            "columns": [["Club", "object"]],
                        },
                        "history_H": [{"route": "RETRIEVE", "verdict": 0.5}],
                        "route": "SQL",
                        "accepted": True,
                    },
                ]
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "trace.jsonl"
            target = Path(tmpdir) / "labels.jsonl"
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")

            load_inference_oracle(str(source), str(target))
            labels = [json.loads(line) for line in target.read_text().splitlines()]

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]["oracle_route"], "SQL")
        self.assertEqual(labels[0]["history_H"][0]["route"], "RETRIEVE")
        self.assertEqual(labels[0]["schema"]["columns"][0][0], "Club")
        self.assertTrue(labels[0]["need_global_table_view"])


if __name__ == "__main__":
    unittest.main()
