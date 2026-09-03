"""Regression tests for the four-class learned router."""
from types import SimpleNamespace
import unittest

from atr.online.router import LearnedRouter, Route


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
    def __call__(self, *args, **kwargs):
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

    def __call__(self, **kwargs):
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


if __name__ == "__main__":
    unittest.main()
