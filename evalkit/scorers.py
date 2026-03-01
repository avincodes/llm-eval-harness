"""Scorer implementations.

Each scorer returns a ScoreResult with `score` normalized to [0, 1] and
a boolean `passed`. Starting with the dumb-but-useful string matchers.
"""

from __future__ import annotations

from dataclasses import dataclass

from evalkit.core import Example, ScoreResult, Scorer


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
