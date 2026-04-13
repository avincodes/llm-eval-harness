"""Core abstractions for the eval harness.

The design mirrors how real eval frameworks (Braintrust, OpenAI evals, lm-eval-harness)
decompose the problem:

    Dataset  -> iterable of Examples (input + expected)
    Task     -> wraps a model call; turns Example.input into a prediction
    Scorer   -> pure function: (example, prediction) -> ScoreResult
    Experiment -> a configured run: dataset x task x scorers
    Run      -> the materialized output of executing an Experiment

Keeping these orthogonal means you can swap any piece without touching the others -
e.g. same dataset + scorers, different model; same task, different scorer set.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


# ---------------------------------------------------------------------------
# Data primitives
# ---------------------------------------------------------------------------


@dataclass
class Example:
    """A single eval datapoint.

    `input` is whatever the task consumes (usually a dict of template vars).
    `expected` is the reference answer - may be None for open-ended tasks
    that only use LLM-as-judge or pairwise scoring.
    `metadata` is free-form and is carried through to the Run output so you
    can slice results by tags like difficulty, category, etc.
    """

    id: str
    input: Any
    expected: Any = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoreResult:
    """Result of scoring a single prediction.

    `score` is normalized to [0, 1] where 1.0 == perfect. This makes
    cross-scorer aggregation sane even when the underlying metric is
    a boolean or a judge's 1-5 rating.
    """

    name: str
    score: float
    passed: bool
    rationale: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Dataset:
    """A thin wrapper over a list of Examples.

    Kept simple on purpose - if you need streaming/sharding, subclass this.
    The JSONL loader expects one Example per line with keys:
    id, input, expected, metadata (metadata optional).
    """

    def __init__(self, examples: list[Example], name: str = "dataset"):
        self.name = name
        self.examples = examples

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str | None = None) -> "Dataset":
        path = Path(path)
        examples: list[Example] = []
        with path.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                examples.append(
                    Example(
                        id=row.get("id", f"{path.stem}-{i}"),
                        input=row["input"],
                        expected=row.get("expected"),
                        metadata=row.get("metadata", {}),
                    )
                )
        return cls(examples, name=name or path.stem)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A task is a (prompt_template, model_client) pair.

    The prompt template uses Python str.format placeholders; whatever dict
    lives on Example.input gets spliced in. This is deliberately naive -
    it covers 90% of eval cases and you can always subclass `run` if you
    need tool use, multi-turn, etc.
    """

    name: str
    prompt_template: str
    client: "LLMClient"  # forward ref to avoid circular import
    system: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512

    def render(self, example: Example) -> str:
        if isinstance(example.input, dict):
            return self.prompt_template.format(**example.input)
        return self.prompt_template.format(input=example.input)

    def run(self, example: Example) -> str:
        prompt = self.render(example)
        return self.client.complete(
            prompt=prompt,
            system=self.system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


# ---------------------------------------------------------------------------
# Scorer base
# ---------------------------------------------------------------------------


class Scorer:
    """Abstract scorer. Subclasses implement `score`.

    Kept as a class (not a function) so scorers can hold config and be
    serialized/described in run metadata.
    """

    name: str = "scorer"

    def score(self, example: Example, prediction: str) -> ScoreResult:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "type": self.__class__.__name__}


# ---------------------------------------------------------------------------
# Experiment + Run
# ---------------------------------------------------------------------------


@dataclass
class ExampleResult:
    example_id: str
    input: Any
    expected: Any
    prediction: str
    scores: list[ScoreResult]
    latency_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Run:
    """The materialized output of executing an Experiment.

    Persisted as JSONL so you can diff two runs without loading them into
    a database. `config_hash` lets the regression detector know whether
    two runs are actually comparable.
    """

    run_id: str
    experiment_name: str
    timestamp: float
    git_sha: str
    config_hash: str
    config: dict
    results: list[ExampleResult]

    def aggregate(self) -> dict[str, float]:
        """Mean score per scorer across all examples."""
        totals: dict[str, list[float]] = {}
        for r in self.results:
            for s in r.scores:
                totals.setdefault(s.name, []).append(s.score)
        return {k: sum(v) / len(v) if v else 0.0 for k, v in totals.items()}

    def pass_rate(self) -> dict[str, float]:
        totals: dict[str, list[int]] = {}
        for r in self.results:
            for s in r.scores:
                totals.setdefault(s.name, []).append(1 if s.passed else 0)
        return {k: sum(v) / len(v) if v else 0.0 for k, v in totals.items()}

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "config": self.config,
            "aggregate": self.aggregate(),
            "pass_rate": self.pass_rate(),
            "results": [
                {
                    "example_id": r.example_id,
                    "input": r.input,
                    "expected": r.expected,
                    "prediction": r.prediction,
                    "latency_ms": r.latency_ms,
                    "metadata": r.metadata,
                    "scores": [asdict(s) for s in r.scores],
                }
                for r in self.results
            ],
        }

    def save(self, runs_dir: str | Path = "runs") -> Path:
        runs_dir = Path(runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"{self.run_id}.jsonl"
        with path.open("w") as f:
            # header row = metadata, subsequent rows = per-example
            header = {
                "kind": "header",
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
                "timestamp": self.timestamp,
                "git_sha": self.git_sha,
                "config_hash": self.config_hash,
                "config": self.config,
                "aggregate": self.aggregate(),
                "pass_rate": self.pass_rate(),
            }
            f.write(json.dumps(header) + "\n")
            for r in self.results:
                row = {
                    "kind": "result",
                    "example_id": r.example_id,
                    "input": r.input,
                    "expected": r.expected,
                    "prediction": r.prediction,
                    "latency_ms": r.latency_ms,
                    "metadata": r.metadata,
                    "scores": [asdict(s) for s in r.scores],
                }
                f.write(json.dumps(row, default=str) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Run":
        path = Path(path)
        results: list[ExampleResult] = []
        header: dict | None = None
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "header":
                    header = row
                elif row.get("kind") == "result":
                    results.append(
                        ExampleResult(
                            example_id=row["example_id"],
                            input=row["input"],
                            expected=row["expected"],
                            prediction=row["prediction"],
                            scores=[ScoreResult(**s) for s in row["scores"]],
                            latency_ms=row.get("latency_ms", 0.0),
                            metadata=row.get("metadata", {}),
                        )
                    )
        if header is None:
            raise ValueError(f"Run file {path} has no header row")
        return cls(
            run_id=header["run_id"],
            experiment_name=header["experiment_name"],
            timestamp=header["timestamp"],
            git_sha=header["git_sha"],
            config_hash=header["config_hash"],
            config=header["config"],
            results=results,
        )


@dataclass
class Experiment:
    """The recipe. Doesn't execute anything until `run()` is called."""

    name: str
    dataset: Dataset
    task: Task
    scorers: list[Scorer]

    def run(self, git_sha: str = "unknown", config_hash: str = "unknown",
            config: dict | None = None, progress: Callable[[int, int], None] | None = None) -> Run:
        results: list[ExampleResult] = []
        total = len(self.dataset)
        for i, ex in enumerate(self.dataset):
            t0 = time.perf_counter()
            try:
                prediction = self.task.run(ex)
            except Exception as e:  # noqa: BLE001 - surface errors as failed scores
                prediction = f"<ERROR: {type(e).__name__}: {e}>"
            latency_ms = (time.perf_counter() - t0) * 1000
            scores = [s.score(ex, prediction) for s in self.scorers]
            results.append(
                ExampleResult(
                    example_id=ex.id,
                    input=ex.input,
                    expected=ex.expected,
                    prediction=prediction,
                    scores=scores,
                    latency_ms=latency_ms,
                    metadata=ex.metadata,
                )
            )
            if progress:
                progress(i + 1, total)
        return Run(
            run_id=f"{self.name}-{int(time.time())}-{uuid.uuid4().hex[:6]}",
            experiment_name=self.name,
            timestamp=time.time(),
            git_sha=git_sha,
            config_hash=config_hash,
            config=config or {},
            results=results,
        )
