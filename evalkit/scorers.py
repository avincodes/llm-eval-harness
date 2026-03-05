"""Scorer implementations.

Each scorer returns a ScoreResult with `score` normalized to [0, 1] and
a boolean `passed`. The split matters: aggregate metrics (mean score)
and threshold metrics (pass rate) answer different questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from evalkit.core import Example, ScoreResult, Scorer


# ---------------------------------------------------------------------------
# String-matching scorers
# ---------------------------------------------------------------------------


@dataclass
class ExactMatch(Scorer):
    name: str = "exact_match"
    case_sensitive: bool = False
    strip: bool = True

    def score(self, example: Example, prediction: str) -> ScoreResult:
        exp = str(example.expected)
        pred = prediction
        if self.strip:
            exp, pred = exp.strip(), pred.strip()
        if not self.case_sensitive:
            exp, pred = exp.lower(), pred.lower()
        ok = exp == pred
        return ScoreResult(
            name=self.name,
            score=1.0 if ok else 0.0,
            passed=ok,
            rationale=f"expected={exp!r} got={pred!r}",
        )


@dataclass
class Contains(Scorer):
    name: str = "contains"
    case_sensitive: bool = False

    def score(self, example: Example, prediction: str) -> ScoreResult:
        needle = str(example.expected)
        hay = prediction
        if not self.case_sensitive:
            needle, hay = needle.lower(), hay.lower()
        ok = needle in hay
        return ScoreResult(
            name=self.name,
            score=1.0 if ok else 0.0,
            passed=ok,
            rationale=f"needle={needle!r} in prediction={ok}",
        )


@dataclass
class Regex(Scorer):
    """Match prediction against a regex. The regex can either be hardcoded
    on the scorer (a 'pattern' at config time) or pulled from
    example.expected (if the dataset itself specifies per-row patterns)."""

    name: str = "regex"
    pattern: str | None = None
    flags: int = re.IGNORECASE

    def score(self, example: Example, prediction: str) -> ScoreResult:
        pat = self.pattern or str(example.expected)
        m = re.search(pat, prediction, self.flags)
        ok = m is not None
        return ScoreResult(
            name=self.name,
            score=1.0 if ok else 0.0,
            passed=ok,
            rationale=f"pattern={pat!r} match={ok}",
        )


# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class NumericTolerance(Scorer):
    """Extract the first number from the prediction and compare to expected
    within an absolute or relative tolerance. Partial credit is given as
    1 - (error / tolerance), clamped to [0, 1]."""

    name: str = "numeric_tolerance"
    abs_tol: float = 0.0
    rel_tol: float = 0.0

    def _extract(self, s: str) -> float | None:
        m = _NUM_RE.search(str(s))
        return float(m.group(0)) if m else None

    def score(self, example: Example, prediction: str) -> ScoreResult:
        exp = self._extract(example.expected)
        got = self._extract(prediction)
        if exp is None or got is None:
            return ScoreResult(
                name=self.name, score=0.0, passed=False,
                rationale=f"could not parse numbers: exp={example.expected!r} got={prediction!r}",
            )
        err = abs(exp - got)
        tol = max(self.abs_tol, self.rel_tol * abs(exp))
        if tol <= 0:
            ok = err == 0
            partial = 1.0 if ok else 0.0
        else:
            partial = max(0.0, 1.0 - err / tol)
            ok = err <= tol
        return ScoreResult(
            name=self.name,
            score=partial,
            passed=ok,
            rationale=f"exp={exp} got={got} err={err:.4g} tol={tol:.4g}",
        )
