"""YAML experiment config loader.

A config file defines:
    - dataset (path or inline)
    - one or more task variants (each with a provider+model+prompt)
    - scorers
    - optional judge (separate client used by llm_judge / pairwise)

Each variant produces its own Run, so a single config file can express
'gpt-4o-mini vs claude-3-5-haiku on the same dataset'.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evalkit.clients import LLMClient, build_client
from evalkit.core import Dataset, Example, Experiment, Task
from evalkit.scorers import build_scorer


@dataclass
class LoadedConfig:
    name: str
    raw: dict
    experiments: list[Experiment]
    config_hash: str


def _load_dataset(spec: dict | str, base_dir: Path) -> Dataset:
    if isinstance(spec, str):
        return Dataset.from_jsonl(base_dir / spec)
    if "path" in spec:
        path = Path(spec["path"])
        if not path.is_absolute():
            path = base_dir / path
        return Dataset.from_jsonl(path, name=spec.get("name"))
    if "examples" in spec:
        examples = [
            Example(
                id=row.get("id", f"inline-{i}"),
                input=row["input"],
                expected=row.get("expected"),
                metadata=row.get("metadata", {}),
            )
            for i, row in enumerate(spec["examples"])
        ]
        return Dataset(examples, name=spec.get("name", "inline"))
    raise ValueError(f"Bad dataset spec: {spec}")


def load_config(path: str | Path, dry_run: bool = True) -> LoadedConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    base_dir = path.parent

    dataset = _load_dataset(raw["dataset"], base_dir)

    # Judge client (optional, shared across scorers)
    judge: LLMClient | None = None
    if "judge" in raw:
        j = raw["judge"]
        judge = build_client(j["provider"], j["model"], dry_run=dry_run)

    # Scorers (shared across all task variants)
    scorers = [build_scorer(s, judge_client=judge) for s in raw.get("scorers", [])]

    experiments: list[Experiment] = []
    variants = raw.get("variants") or [raw.get("task")]
    for v in variants:
        client = build_client(v["provider"], v["model"], dry_run=dry_run)
        task = Task(
            name=v.get("name", f"{v['provider']}:{v['model']}"),
            prompt_template=v["prompt"],
            client=client,
            system=v.get("system"),
            temperature=v.get("temperature", 0.0),
            max_tokens=v.get("max_tokens", 512),
        )
        exp_name = f"{raw['name']}__{task.name}".replace("/", "_").replace(":", "_")
        experiments.append(Experiment(name=exp_name, dataset=dataset, task=task, scorers=scorers))

    # Stable hash of normalized config for run-comparability checks
    canonical = json.dumps(raw, sort_keys=True, default=str)
    config_hash = hashlib.sha256(canonical.encode()).hexdigest()[:12]

    return LoadedConfig(name=raw["name"], raw=raw, experiments=experiments, config_hash=config_hash)
