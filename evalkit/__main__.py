"""CLI entry point: `python -m evalkit <command>`."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalkit", description="LLM eval harness")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run an experiment config")
    pr.add_argument("config", help="path to YAML config")
    pr.add_argument("--dry-run", action="store_true", default=True,
                    help="use stub LLM clients (default: on)")
    pr.add_argument("--live", dest="dry_run", action="store_false",
                    help="disable dry-run; hit real APIs (requires keys)")
    pr.add_argument("--runs-dir", default="runs")
    pr.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
