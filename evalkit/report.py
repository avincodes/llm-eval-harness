"""Static HTML report generator.

Produces a single self-contained `report.html` - no JS framework, no external
CSS. You can email it, check it into a PR, or drop it into S3. The goal is
that a reviewer can open the file and immediately see: what passed, what
failed, and *why*.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from evalkit.compare import ComparisonReport
from evalkit.core import Run


_BASE_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #1a1a1a; }
h1, h2 { border-bottom: 1px solid #eaeaea; padding-bottom: .3em; }
.meta { color: #666; font-size: .9em; }
table { width: 100%; border-collapse: collapse; margin: 1em 0; font-size: .9em; }
th, td { text-align: left; padding: .5em .75em; border-bottom: 1px solid #eee; vertical-align: top; }
th { background: #fafafa; font-weight: 600; }
tr:hover { background: #fafbfc; }
.pass { color: #0a7f2e; font-weight: 600; }
.fail { color: #c62828; font-weight: 600; }
.score-bar { display: inline-block; width: 80px; height: 8px; background: #eee; border-radius: 4px; overflow: hidden; }
.score-bar > div { height: 100%; background: linear-gradient(90deg, #c62828, #f9a825, #0a7f2e); }
details { margin: .25em 0; }
summary { cursor: pointer; color: #0366d6; }
pre { background: #f6f8fa; padding: .75em; border-radius: 4px; overflow-x: auto;
      white-space: pre-wrap; word-break: break-word; font-size: .85em; }
.agg { display: flex; gap: 1.5em; flex-wrap: wrap; margin: 1em 0; }
.agg .card { background: #f6f8fa; padding: 1em 1.5em; border-radius: 6px; min-width: 140px; }
.agg .card .label { font-size: .75em; text-transform: uppercase; color: #666; letter-spacing: .05em; }
.agg .card .value { font-size: 1.6em; font-weight: 600; }
.delta-pos { color: #0a7f2e; }
.delta-neg { color: #c62828; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: .75em;
         background: #eaeaea; color: #333; }
"""


def _score_bar(score: float) -> str:
    pct = int(max(0.0, min(1.0, score)) * 100)
    return f'<span class="score-bar"><div style="width:{pct}%"></div></span>'


def _short(s, n=120):
    s = str(s)
    return s if len(s) <= n else s[:n] + "..."


def render_run(run: Run) -> str:
    agg = run.aggregate()
    pr = run.pass_rate()

    cards = "".join(
        f'<div class="card"><div class="label">{html.escape(k)}</div>'
        f'<div class="value">{v:.3f}</div>'
        f'<div class="meta">pass rate {pr.get(k, 0):.0%}</div></div>'
        for k, v in sorted(agg.items())
    )

    rows = []
    for r in run.results:
        any_fail = any(not s.passed for s in r.scores)
        cls = "fail" if any_fail else "pass"
        score_cells = "".join(
            f'<td><span class="{"pass" if s.passed else "fail"}">{s.score:.2f}</span> {_score_bar(s.score)}</td>'
            for s in r.scores
        )
        scorer_details = "".join(
            f"<li><b>{html.escape(s.name)}</b>: "
            f'<span class="{"pass" if s.passed else "fail"}">{s.score:.2f}</span> '
            f"— {html.escape(s.rationale)}</li>"
            for s in r.scores
        )
        rows.append(
            f"<tr>"
            f'<td class="{cls}">{html.escape(r.example_id)}</td>'
            f"<td>{html.escape(_short(r.input))}</td>"
            f"<td>{html.escape(_short(r.expected))}</td>"
            f"<td>{html.escape(_short(r.prediction))}</td>"
            f"{score_cells}"
            f"<td>{r.latency_ms:.0f}ms</td>"
            f"</tr>"
            f'<tr><td colspan="{5 + len(r.scores)}">'
            f"<details><summary>drilldown</summary>"
            f"<b>Full input:</b><pre>{html.escape(json.dumps(r.input, indent=2, default=str))}</pre>"
            f"<b>Expected:</b><pre>{html.escape(str(r.expected))}</pre>"
            f"<b>Prediction:</b><pre>{html.escape(str(r.prediction))}</pre>"
            f"<b>Scores:</b><ul>{scorer_details}</ul>"
            f"</details></td></tr>"
        )

    scorer_names = [s.name for s in run.results[0].scores] if run.results else []
    score_headers = "".join(f"<th>{html.escape(n)}</th>" for n in scorer_names)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>evalkit report — {html.escape(run.experiment_name)}</title>
<style>{_BASE_CSS}</style></head><body>
<h1>evalkit report</h1>
<p class="meta">
  <b>Experiment:</b> {html.escape(run.experiment_name)} &nbsp;
  <b>Run ID:</b> {html.escape(run.run_id)} &nbsp;
  <b>Git:</b> <span class="badge">{html.escape(run.git_sha)}</span> &nbsp;
  <b>Config hash:</b> <span class="badge">{html.escape(run.config_hash)}</span>
</p>
<h2>Aggregate</h2>
<div class="agg">{cards}</div>
<h2>Per-example results ({len(run.results)})</h2>
<table>
<thead><tr>
<th>ID</th><th>Input</th><th>Expected</th><th>Prediction</th>
{score_headers}<th>Latency</th>
</tr></thead>
<tbody>
{"".join(rows)}
</tbody></table>
<p class="meta">Generated by evalkit v0.1.0</p>
</body></html>
"""


def render_comparison(cmp: ComparisonReport) -> str:
    agg_rows = "".join(
        f"<tr><td>{html.escape(k)}</td>"
        f"<td>{cmp.aggregate_a.get(k, 0):.3f}</td>"
        f"<td>{cmp.aggregate_b.get(k, 0):.3f}</td>"
        f'<td class="{"delta-pos" if cmp.aggregate_delta[k] >= 0 else "delta-neg"}">{cmp.aggregate_delta[k]:+.3f}</td>'
        f"</tr>"
        for k in sorted(cmp.aggregate_delta)
    )
    reg_rows = "".join(
        f"<tr><td>{html.escape(r.example_id)}</td><td>{html.escape(r.scorer)}</td>"
        f"<td>{r.a_score:.2f}</td><td>{r.b_score:.2f}</td>"
        f'<td class="delta-neg">{r.delta:+.2f}</td>'
        f"<td><details><summary>diff</summary>"
        f"<b>A:</b><pre>{html.escape(_short(r.a_prediction, 500))}</pre>"
        f"<b>B:</b><pre>{html.escape(_short(r.b_prediction, 500))}</pre></details></td></tr>"
        for r in cmp.regressions
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>evalkit comparison</title>
<style>{_BASE_CSS}</style></head><body>
<h1>Run comparison</h1>
<p class="meta">
  <b>A:</b> {html.escape(cmp.run_a_id)}<br>
  <b>B:</b> {html.escape(cmp.run_b_id)}<br>
  <b>Comparable:</b> {"yes" if cmp.comparable else "<span class='fail'>NO — config hashes differ</span>"}
</p>
<h2>Aggregate deltas</h2>
<table><thead><tr><th>Scorer</th><th>A</th><th>B</th><th>Δ</th></tr></thead>
<tbody>{agg_rows}</tbody></table>
<h2>Regressions ({len(cmp.regressions)})</h2>
<table><thead><tr><th>Example</th><th>Scorer</th><th>A</th><th>B</th><th>Δ</th><th></th></tr></thead>
<tbody>{reg_rows or '<tr><td colspan=6>none</td></tr>'}</tbody></table>
<h2>Improvements ({len(cmp.improvements)})</h2>
<p class="meta">{len(cmp.improvements)} example/scorer pairs improved.</p>
</body></html>
"""


def write_report(run: Run, out_path: str | Path = "report.html") -> Path:
    out = Path(out_path)
    out.write_text(render_run(run))
    return out


def write_comparison_report(cmp: ComparisonReport, out_path: str | Path = "comparison.html") -> Path:
    out = Path(out_path)
    out.write_text(render_comparison(cmp))
    return out
