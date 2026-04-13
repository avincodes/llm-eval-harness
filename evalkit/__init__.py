"""evalkit - a small, opinionated LLM evaluation harness.

Exports the core primitives so users can import them directly:

    from evalkit import Dataset, Task, Experiment, Run
    from evalkit.scorers import ExactMatch, Contains, LLMJudge
"""

from evalkit.core import Dataset, Task, Scorer, Experiment, Run, Example, ScoreResult

__all__ = [
    "Dataset",
    "Task",
    "Scorer",
    "Experiment",
    "Run",
    "Example",
    "ScoreResult",
]

__version__ = "0.1.0"
