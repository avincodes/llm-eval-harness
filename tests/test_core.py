"""End-to-end tests for Dataset/Experiment/Run + the regression detector."""

from __future__ import annotations

import json
from pathlib import Path

from evalkit.clients import LocalClient
from evalkit.compare import compare_runs
from evalkit.core import Dataset, Example, Experiment, Task
from evalkit.scorers import ExactMatch


def _tiny_dataset():
    return Dataset(
        [
            Example(id="a", input={"text": "great product"}, expected="positive"),
            Example(id="b", input={"text": "terrible experience"}, expected="negative"),
            Example(id="c", input={"text": "it works"}, expected="neutral"),
        ],
        name="tiny",
    )


def _build_experiment(name="tiny_exp", temperature=0.0):
    client = LocalClient()
    task = Task(
        name="stub",
        prompt_template="Classify sentiment of: {text}",
        client=client,
        temperature=temperature,
    )
    return Experiment(name=name, dataset=_tiny_dataset(), task=task, scorers=[ExactMatch()])


def test_experiment_runs_and_aggregates():
    run = _build_experiment().run(git_sha="abc123", config_hash="h1")
    assert len(run.results) == 3
    agg = run.aggregate()
    assert "exact_match" in agg
    assert 0.0 <= agg["exact_match"] <= 1.0


def test_run_save_and_load_roundtrip(tmp_path: Path):
    run = _build_experiment().run(git_sha="abc", config_hash="h1", config={"k": "v"})
    path = run.save(tmp_path)
    assert path.exists()

    from evalkit.core import Run
    loaded = Run.load(path)
    assert loaded.run_id == run.run_id
    assert len(loaded.results) == len(run.results)
    assert loaded.config == {"k": "v"}
    assert loaded.aggregate() == run.aggregate()


def test_dataset_from_jsonl(tmp_path: Path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps({"id": "x", "input": {"text": "hi"}, "expected": "neutral"}) + "\n"
        + json.dumps({"id": "y", "input": {"text": "bad"}, "expected": "negative"}) + "\n"
    )
    ds = Dataset.from_jsonl(p)
    assert len(ds) == 2
    assert ds.examples[0].id == "x"


def test_compare_runs_flags_regressions():
    run_a = _build_experiment("exp_a").run(git_sha="1", config_hash="H")
    # mutate run_b to have a failing score on one example
    run_b = _build_experiment("exp_b").run(git_sha="2", config_hash="H")
    # force a regression: zero out a score on run_b
    run_b.results[0].scores[0].score = 0.0
    run_b.results[0].scores[0].passed = False
    # and ensure run_a had a positive score on the same example
    run_a.results[0].scores[0].score = 1.0
    run_a.results[0].scores[0].passed = True

    report = compare_runs(run_a, run_b)
    assert report.comparable is True
    assert any(r.example_id == run_a.results[0].example_id for r in report.regressions)


def test_compare_runs_detects_config_drift():
    run_a = _build_experiment().run(git_sha="1", config_hash="X")
    run_b = _build_experiment().run(git_sha="2", config_hash="Y")
    report = compare_runs(run_a, run_b)
    assert report.comparable is False
