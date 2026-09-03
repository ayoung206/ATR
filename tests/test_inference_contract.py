"""Regression tests for verifier acceptance and the paper stop rule."""
import unittest

from atr.online.main import _is_verified_answer, _should_stop


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


if __name__ == "__main__":
    unittest.main()
