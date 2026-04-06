"""Regression detector: compare two Runs and flag drops.

Two runs are 'comparable' if they share the same config_hash. If not,
we warn but still diff - sometimes you want to compare e.g. two model
variants where config differs intentionally.

Drops are flagged per-example per-scorer: we emit a regression if the
score dropped by more than `threshold` (default 0.0, i.e. any drop).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evalkit.core import Run


@dataclass
class Regression:
    example_id: str
    scorer: str
    a_score: float
    b_score: float
    delta: float
    a_prediction: str
    b_prediction: str


@dataclass
class ComparisonReport:
    run_a_id: str
    run_b_id: str
    comparable: bool
    aggregate_a: dict[str, float]
    aggregate_b: dict[str, float]
    aggregate_delta: dict[str, float]
    regressions: list[Regression]
    improvements: list[Regression]
    new_examples: list[str]
    missing_examples: list[str]

    def summary(self) -> str:
        lines = [
            f"Run A: {self.run_a_id}",
            f"Run B: {self.run_b_id}",
            f"Comparable (same config_hash): {self.comparable}",
            "",
            "Aggregate deltas (B - A):",
        ]
        for k, v in sorted(self.aggregate_delta.items()):
            arrow = "+" if v >= 0 else ""
            lines.append(f"  {k}: {self.aggregate_a.get(k, 0):.3f} -> {self.aggregate_b.get(k, 0):.3f}  ({arrow}{v:.3f})")
        lines.append("")
        lines.append(f"Regressions: {len(self.regressions)}")
        for r in self.regressions[:20]:
            lines.append(f"  - {r.example_id}[{r.scorer}] {r.a_score:.2f} -> {r.b_score:.2f} (Δ{r.delta:+.2f})")
        if len(self.regressions) > 20:
            lines.append(f"  ... and {len(self.regressions) - 20} more")
        lines.append("")
        lines.append(f"Improvements: {len(self.improvements)}")
        if self.new_examples:
            lines.append(f"New examples in B: {len(self.new_examples)}")
        if self.missing_examples:
            lines.append(f"Missing in B: {len(self.missing_examples)}")
        return "\n".join(lines)


def compare_runs(run_a: Run, run_b: Run, threshold: float = 0.0) -> ComparisonReport:
    a_by_id = {r.example_id: r for r in run_a.results}
    b_by_id = {r.example_id: r for r in run_b.results}

    shared = set(a_by_id) & set(b_by_id)
    new_examples = sorted(set(b_by_id) - set(a_by_id))
    missing_examples = sorted(set(a_by_id) - set(b_by_id))

    regressions: list[Regression] = []
    improvements: list[Regression] = []

    for eid in sorted(shared):
        a = a_by_id[eid]
        b = b_by_id[eid]
        a_scores = {s.name: s.score for s in a.scores}
        b_scores = {s.name: s.score for s in b.scores}
        for scorer_name in set(a_scores) | set(b_scores):
            sa = a_scores.get(scorer_name, 0.0)
            sb = b_scores.get(scorer_name, 0.0)
            delta = sb - sa
            entry = Regression(
                example_id=eid,
                scorer=scorer_name,
                a_score=sa,
                b_score=sb,
                delta=delta,
                a_prediction=a.prediction,
                b_prediction=b.prediction,
            )
            if delta < -threshold:
                regressions.append(entry)
            elif delta > threshold:
                improvements.append(entry)

    agg_a = run_a.aggregate()
    agg_b = run_b.aggregate()
    agg_delta = {k: agg_b.get(k, 0.0) - agg_a.get(k, 0.0) for k in set(agg_a) | set(agg_b)}

    return ComparisonReport(
        run_a_id=run_a.run_id,
        run_b_id=run_b.run_id,
        comparable=run_a.config_hash == run_b.config_hash,
        aggregate_a=agg_a,
        aggregate_b=agg_b,
        aggregate_delta=agg_delta,
        regressions=sorted(regressions, key=lambda r: r.delta),
        improvements=sorted(improvements, key=lambda r: -r.delta),
        new_examples=new_examples,
        missing_examples=missing_examples,
    )


def find_run(runs_dir: str | Path, run_id_or_path: str) -> Run:
    """Accept either a full path or a partial run_id and locate the JSONL."""
    p = Path(run_id_or_path)
    if p.exists():
        return Run.load(p)
    runs_dir = Path(runs_dir)
    candidates = list(runs_dir.glob(f"*{run_id_or_path}*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No run matching {run_id_or_path} in {runs_dir}")
    if len(candidates) > 1:
        # pick the most recent
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return Run.load(candidates[0])
