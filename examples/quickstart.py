"""Quickstart: build an experiment in Python instead of YAML.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow running this file directly from the repo without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.clients import LocalClient
from evalkit.core import Dataset, Example, Experiment, Task
from evalkit.report import write_report
from evalkit.scorers import Contains, ExactMatch, LLMJudge


def main() -> None:
    dataset = Dataset(
        [
            Example(id="q1", input={"text": "I love this!"}, expected="positive"),
            Example(id="q2", input={"text": "This is terrible."}, expected="negative"),
            Example(id="q3", input={"text": "It's fine."}, expected="neutral"),
        ],
        name="quickstart",
    )

    client = LocalClient()  # dry-run; no API key needed
    task = Task(
        name="local-stub",
        prompt_template="Classify the sentiment of: {text}\nAnswer with one word.",
        client=client,
    )

    judge = LocalClient(model="judge-stub")
    experiment = Experiment(
        name="quickstart",
        dataset=dataset,
        task=task,
        scorers=[
            ExactMatch(),
            Contains(),
            LLMJudge(judge=judge),
        ],
    )

    run = experiment.run(git_sha="local", config_hash="quickstart")
    print("aggregate:", run.aggregate())
    print("pass_rate:", run.pass_rate())
    path = run.save("runs")
    print("saved:", path)
    report = write_report(run, "report.html")
    print("report:", report)


if __name__ == "__main__":
    main()
