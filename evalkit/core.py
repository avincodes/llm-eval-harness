"""Core abstractions for the eval harness.

    Dataset  -> iterable of Examples (input + expected)
    Task     -> wraps a model call; turns Example.input into a prediction
    Scorer   -> pure function: (example, prediction) -> ScoreResult
    Experiment -> a configured run: dataset x task x scorers
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Data primitives
# ---------------------------------------------------------------------------


@dataclass
class Example:
    id: str
    input: Any
    expected: Any = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoreResult:
    """Result of scoring a single prediction. `score` is normalized to [0, 1]."""

    name: str
    score: float
    passed: bool
    rationale: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Dataset:
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
    name: str
    prompt_template: str
    system: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512

    def render(self, example: Example) -> str:
        if isinstance(example.input, dict):
            return self.prompt_template.format(**example.input)
        return self.prompt_template.format(input=example.input)


# ---------------------------------------------------------------------------
# Scorer base
# ---------------------------------------------------------------------------


class Scorer:
    """Abstract scorer. Subclasses implement `score`."""

    name: str = "scorer"

    def score(self, example: Example, prediction: str) -> ScoreResult:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "type": self.__class__.__name__}


# ---------------------------------------------------------------------------
# Experiment (skeleton — no run loop yet)
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    """The recipe. Doesn't actually execute until we wire up clients + Run."""

    name: str
    dataset: Dataset
    task: Task
    scorers: list[Scorer]

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dataset": self.dataset.name,
            "task": self.task.name,
            "scorers": [s.describe() for s in self.scorers],
        }
