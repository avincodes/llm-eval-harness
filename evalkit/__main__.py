"""CLI entry point: `python -m evalkit <command>`.

Commands:
    run <config.yaml>           Execute an experiment config; write a Run
    compare <run_a> <run_b>     Compare two runs and print regressions
    list                        List runs under runs/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from evalkit.compare import compare_runs, find_run
from evalkit.config import load_config


def _git_sha(default: str = "unknown") -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return default


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, dry_run=args.dry_run)
    git_sha = _git_sha()
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded config '{cfg.name}' — {len(cfg.experiments)} variant(s), {len(cfg.experiments[0].dataset)} examples")
    print(f"config_hash={cfg.config_hash}  git={git_sha}  dry_run={args.dry_run}")

    for exp in cfg.experiments:
        print(f"\n=== {exp.name} ===")

        def progress(i, n):
            print(f"  [{i}/{n}] {exp.task.name}", end="\r", flush=True)

        run = exp.run(
            git_sha=git_sha,
            config_hash=cfg.config_hash,
            config=cfg.raw,
            progress=progress,
        )
        path = run.save(runs_dir)
        print(f"\n  saved -> {path}")
        print(f"  aggregate: {run.aggregate()}")
        print(f"  pass_rate: {run.pass_rate()}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    a = find_run(args.runs_dir, args.run_a)
    b = find_run(args.runs_dir, args.run_b)
    report = compare_runs(a, b, threshold=args.threshold)
    print(report.summary())
    return 0 if not report.regressions else 2  # nonzero exit for CI


def cmd_list(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"(no runs dir at {runs_dir})")
        return 0
    files = sorted(runs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("(no runs)")
        return 0
    for f in files:
        print(f.name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalkit", description="LLM eval harness")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run an experiment config")
    pr.add_argument("config", help="path to YAML config")
    pr.add_argument("--dry-run", action="store_true", default=True)
    pr.add_argument("--live", dest="dry_run", action="store_false")
    pr.add_argument("--runs-dir", default="runs")
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser("compare", help="compare two runs")
    pc.add_argument("run_a")
    pc.add_argument("run_b")
    pc.add_argument("--runs-dir", default="runs")
    pc.add_argument("--threshold", type=float, default=0.0,
                    help="min per-example score delta to count as a regression")
    pc.set_defaults(func=cmd_compare)

    pl = sub.add_parser("list", help="list saved runs")
    pl.add_argument("--runs-dir", default="runs")
    pl.set_defaults(func=cmd_list)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
