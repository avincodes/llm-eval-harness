"""Scorer implementations.

Each scorer returns a ScoreResult with `score` normalized to [0, 1] and
a boolean `passed`. The split matters: aggregate metrics (mean score)
and threshold metrics (pass rate) answer different questions, and the
regression detector uses `score` for drift and `passed` for counts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from evalkit.core import Example, ScoreResult, Scorer

if TYPE_CHECKING:
    from evalkit.clients import LLMClient  # defined later


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


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------


DEFAULT_JUDGE_PROMPT = """You are an impartial grader. Rate the MODEL ANSWER against the REFERENCE on a scale of 1-5, where:
1 = completely wrong, 5 = semantically equivalent to the reference.

QUESTION:
{question}

REFERENCE:
{reference}

MODEL ANSWER:
{answer}

Respond with a single JSON object: {{"rating": <int 1-5>, "rationale": "<one sentence>"}}
"""


@dataclass
class LLMJudge(Scorer):
    """LLM-as-judge scorer. The judge client is any LLMClient - so you can
    use a stronger/cheaper model for grading than the one under test."""

    name: str = "llm_judge"
    judge: Any = None  # LLMClient — typed loosely until clients lands
    prompt_template: str = DEFAULT_JUDGE_PROMPT
    pass_threshold: float = 0.6

    def _parse(self, text: str) -> tuple[float, str]:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            obj = json.loads(text[start:end])
            rating = float(obj.get("rating", 0))
            rationale = str(obj.get("rationale", ""))
        except (ValueError, json.JSONDecodeError):
            m = re.search(r"\b([1-5])\b", text)
            rating = float(m.group(1)) if m else 0.0
            rationale = text.strip()[:200]
        rating = max(1.0, min(5.0, rating)) if rating else 0.0
        return rating, rationale

    def score(self, example: Example, prediction: str) -> ScoreResult:
        if self.judge is None:
            raise ValueError("LLMJudge requires a judge client")
        question = example.input if isinstance(example.input, str) else json.dumps(example.input)
        prompt = self.prompt_template.format(
            question=question,
            reference=example.expected,
            answer=prediction,
        )
        raw = self.judge.complete(prompt=prompt, temperature=0.0, max_tokens=200)
        rating, rationale = self._parse(raw)
        normalized = (rating - 1) / 4 if rating >= 1 else 0.0
        return ScoreResult(
            name=self.name,
            score=normalized,
            passed=normalized >= self.pass_threshold,
            rationale=rationale,
            metadata={"raw_rating": rating, "raw": raw},
        )


# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------


@dataclass
class PairwiseComparison(Scorer):
    """Compare prediction against a baseline under metadata['baseline']."""

    name: str = "pairwise"
    judge: Any = None
    prompt_template: str = (
        "You are choosing the better of two answers to a question.\n\n"
        "QUESTION: {question}\n\n"
        "ANSWER A (baseline): {a}\n\n"
        "ANSWER B (candidate): {b}\n\n"
        'Reply with JSON: {{"winner": "A"|"B"|"tie", "rationale": "<one sentence>"}}'
    )

    def score(self, example: Example, prediction: str) -> ScoreResult:
        if self.judge is None:
            raise ValueError("PairwiseComparison requires a judge client")
        baseline = example.metadata.get("baseline", "")
        question = example.input if isinstance(example.input, str) else json.dumps(example.input)
        prompt = self.prompt_template.format(question=question, a=baseline, b=prediction)
        raw = self.judge.complete(prompt=prompt, temperature=0.0, max_tokens=200)
        winner = "tie"
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            obj = json.loads(raw[start:end])
            winner = str(obj.get("winner", "tie")).lower()
        except (ValueError, json.JSONDecodeError):
            low = raw.lower()
            if '"b"' in low or "winner: b" in low:
                winner = "b"
            elif '"a"' in low or "winner: a" in low:
                winner = "a"
        score = {"b": 1.0, "tie": 0.5, "a": 0.0}.get(winner, 0.5)
        return ScoreResult(
            name=self.name,
            score=score,
            passed=score >= 0.5,
            rationale=f"winner={winner}",
            metadata={"raw": raw},
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


SCORER_REGISTRY: dict[str, type[Scorer]] = {
    "exact_match": ExactMatch,
    "contains": Contains,
    "regex": Regex,
    "numeric_tolerance": NumericTolerance,
    "llm_judge": LLMJudge,
    "pairwise": PairwiseComparison,
}


def build_scorer(spec: dict[str, Any], judge_client: Any = None) -> Scorer:
    """Build a scorer from a YAML dict like {type: exact_match, case_sensitive: false}."""
    spec = dict(spec)
    stype = spec.pop("type")
    cls = SCORER_REGISTRY.get(stype)
    if cls is None:
        raise ValueError(f"Unknown scorer type: {stype}")
    if stype in ("llm_judge", "pairwise"):
        spec.setdefault("judge", judge_client)
    return cls(**spec)
