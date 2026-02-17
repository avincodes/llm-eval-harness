"""Core abstractions for the eval harness.

Starting with the data primitives - Example, Dataset, Task - before
bolting on scoring and experiment orchestration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Example:
    """A single eval datapoint.

    `input` is whatever the task consumes (usually a dict of template vars).
    `expected` is the reference answer - may be None for open-ended tasks.
    `metadata` is free-form and is carried through to run outputs.
    """

    id: str
    input: Any
    expected: Any = None
    metadata: dict = field(default_factory=dict)


class Dataset:
    """A thin wrapper over a list of Examples.

    Kept simple on purpose - if you need streaming/sharding, subclass this.
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


@dataclass
class Task:
    """A task wraps a prompt template. Model call comes later once we have
    a client abstraction - for now `run` just renders the prompt."""

    name: str
    prompt_template: str
    system: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512

    def render(self, example: Example) -> str:
        if isinstance(example.input, dict):
            return self.prompt_template.format(**example.input)
        return self.prompt_template.format(input=example.input)
