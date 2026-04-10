"""Tests for the core scorers.

These run offline (no network, no API keys) by using the dry-run clients
for LLMJudge / PairwiseComparison. A passing suite here is our smoke test
that the harness is internally consistent.
"""

from __future__ import annotations

import pytest

from evalkit.clients import LocalClient
from evalkit.core import Example
from evalkit.scorers import (
    Contains,
    ExactMatch,
    LLMJudge,
    NumericTolerance,
    PairwiseComparison,
    Regex,
    build_scorer,
)


def ex(expected, input_="q", id_="t1", meta=None):
    return Example(id=id_, input=input_, expected=expected, metadata=meta or {})


# ---------------------------------------------------------------------------
# ExactMatch
# ---------------------------------------------------------------------------


def test_exact_match_case_insensitive_passes():
    r = ExactMatch().score(ex("Positive"), "positive")
    assert r.passed and r.score == 1.0


def test_exact_match_case_sensitive_fails_on_case():
    r = ExactMatch(case_sensitive=True).score(ex("Positive"), "positive")
    assert not r.passed and r.score == 0.0


def test_exact_match_strips_whitespace():
    assert ExactMatch().score(ex("yes"), "  yes  ").passed


# ---------------------------------------------------------------------------
# Contains
# ---------------------------------------------------------------------------


def test_contains_finds_substring():
    assert Contains().score(ex("SELECT"), "select * from users").passed


def test_contains_fails_when_absent():
    assert not Contains().score(ex("DELETE"), "select * from users").passed


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------


def test_regex_with_hardcoded_pattern():
    s = Regex(pattern=r"^\s*SELECT\b")
    assert s.score(ex(None), "SELECT * FROM t").passed
    assert not s.score(ex(None), "UPDATE t SET x=1").passed


def test_regex_pulls_from_example_when_no_pattern():
    assert Regex().score(ex(r"\d{3}-\d{4}"), "call 555-1234 today").passed


# ---------------------------------------------------------------------------
# NumericTolerance
# ---------------------------------------------------------------------------


def test_numeric_tolerance_exact():
    assert NumericTolerance().score(ex("42"), "The answer is 42").passed


def test_numeric_tolerance_within_abs():
    r = NumericTolerance(abs_tol=1.0).score(ex("100"), "about 100.5")
    assert r.passed
    assert 0.4 < r.score < 0.6  # 1 - 0.5/1.0 = 0.5


def test_numeric_tolerance_out_of_range():
    r = NumericTolerance(abs_tol=1.0).score(ex("100"), "about 110")
    assert not r.passed
    assert r.score == 0.0


def test_numeric_tolerance_handles_missing_number():
    r = NumericTolerance().score(ex("10"), "no digits here")
    assert not r.passed and r.score == 0.0


def test_numeric_tolerance_relative():
    r = NumericTolerance(rel_tol=0.1).score(ex("100"), "105")
    assert r.passed  # within 10%


# ---------------------------------------------------------------------------
# LLMJudge (with stub judge)
# ---------------------------------------------------------------------------


def test_llm_judge_with_stub_judge_returns_normalized_score():
    judge = LocalClient()
    scorer = LLMJudge(judge=judge)
    r = scorer.score(ex("Paris", input_="What is the capital of France?"), "Paris")
    assert 0.0 <= r.score <= 1.0
    assert "raw_rating" in r.metadata


def test_llm_judge_requires_judge_client():
    with pytest.raises(ValueError):
        LLMJudge(judge=None).score(ex("x"), "y")


# ---------------------------------------------------------------------------
# PairwiseComparison
# ---------------------------------------------------------------------------


def test_pairwise_comparison_runs_with_stub():
    judge = LocalClient()
    scorer = PairwiseComparison(judge=judge)
    example = ex("ref", input_="question?", meta={"baseline": "old answer"})
    r = scorer.score(example, "new answer")
    assert r.score in (0.0, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------


def test_build_scorer_from_spec():
    s = build_scorer({"type": "exact_match", "case_sensitive": True})
    assert isinstance(s, ExactMatch)
    assert s.case_sensitive is True


def test_build_scorer_unknown_type_raises():
    with pytest.raises(ValueError):
        build_scorer({"type": "nope"})


def test_build_scorer_injects_judge_client():
    judge = LocalClient()
    s = build_scorer({"type": "llm_judge"}, judge_client=judge)
    assert isinstance(s, LLMJudge)
    assert s.judge is judge
