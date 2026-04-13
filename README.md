# evalkit — a lightweight LLM evaluation harness

A small, opinionated eval framework for LLM applications. Think of it as a
minimal Braintrust / OpenAI Evals: define a dataset, a prompt + model, and a
set of scorers; get back a versioned `Run` you can diff against any previous
run to catch regressions before they ship.

Built because every LLM project I touch ends up reinventing the same primitives
badly — a dict of "predictions vs. gold", some ad-hoc string matching, no
history, no comparison, no judge. `evalkit` is what I actually reach for.

---

## Motivation

When you iterate on prompts or swap models, you need three things and most
teams have none of them:

1. **A stable way to score output** — exact match is brittle, LLM-as-judge
   is noisy, and you usually want both plus a numeric tolerance plus a
   regex sanity check. Scorers should compose, not be hardcoded.
2. **A persistent record of runs** — so "v2 is better" is an assertion you
   can actually verify. Every run gets a git SHA, a config hash, and a
   JSONL file per example.
3. **Regression detection** — mean scores hide the truth. You need
   per-example drops, so when the new prompt fixes 3 edge cases and
   breaks 7, you see both.

`evalkit` packages those into ~1k lines of Python with zero heavy dependencies.

## Architecture

```
          +----------+     +--------+     +----------+
Dataset → |  Task    | →   |  Run   | →   | Scorers  |
          +----------+     +--------+     +----------+
              ↑                                  ↑
         LLMClient                        (optional judge
     (openai/anthropic/local)              = another LLMClient)
```

Five orthogonal primitives (`evalkit/core.py`):

- `Dataset` — iterable of `Example(id, input, expected, metadata)`, loadable from JSONL.
- `Task` — `(prompt_template, LLMClient)`; turns an example into a prediction.
- `Scorer` — pure function `(example, prediction) -> ScoreResult`. Score is
  normalized to `[0, 1]` with a separate `passed` boolean so aggregate mean
  and pass-rate both make sense.
- `Experiment` — the recipe: dataset × task × scorers. Doesn't execute
  until you call `.run()`.
- `Run` — materialized output. Persisted as JSONL with a header row
  (metadata + aggregates) followed by per-example rows.

Clients live behind a `LLMClient` protocol (`evalkit/clients.py`). Three
stubs ship in-box: `openai`, `anthropic`, `local`. Each has a real-call
code path *and* a `dry_run=True` mode that returns deterministic canned
outputs — so the whole harness (tests, examples, sample experiments) runs
offline with no API keys. Swapping in real calls is a one-line flag.

Scorers (`evalkit/scorers.py`):

| Scorer              | Use it for                                        |
|---------------------|---------------------------------------------------|
| `ExactMatch`        | classification, strict canonical answers          |
| `Contains`          | "did the model mention X"                         |
| `Regex`             | structural checks (starts with `SELECT`, etc.)    |
| `NumericTolerance`  | math / numeric extraction with abs or rel tolerance |
| `LLMJudge`          | open-ended generation; pluggable judge model      |
| `PairwiseComparison`| A/B: "did B beat A?" via a judge                  |

## How to run

```bash
# install
pip install -e .[dev]

# run the sentiment experiment (two variants, exact + contains scorers)
python -m evalkit run experiments/sentiment.yaml

# run the SQL experiment (LLM-as-judge scorer + regex)
python -m evalkit run experiments/sql.yaml

# list saved runs
python -m evalkit list

# compare two runs — flags per-example regressions, nonzero exit if any
python -m evalkit compare sentiment-<ts1> sentiment-<ts2>

# quickstart without YAML
python examples/quickstart.py
```

Every `run` command writes:

- `runs/<experiment>-<timestamp>-<uid>.jsonl` — full structured output
- `report.html` — static, self-contained HTML report with aggregates and
  per-example drilldowns (open it in any browser)

A `compare` command also writes `comparison.html` with delta tables and
diff views for every regressed example.

### Going live (real API calls)

Every stub client has a real-call path behind the `--live` flag:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
python -m evalkit run experiments/sentiment.yaml --live
```

The providers are `openai`, `anthropic`, and `local`. `local` is wired as a
stub but is the obvious place to drop in an Ollama / vLLM / llama.cpp HTTP
backend — one method, `complete(prompt, ...)`.

## Tests

```bash
pytest
```

The suite covers every scorer (including `LLMJudge` and `PairwiseComparison`
using the offline stub client), the Dataset/Run save-and-load roundtrip,
the full Experiment execution path, and the regression detector. Runs
completely offline.

## Sample report

Running `python -m evalkit run experiments/sentiment.yaml` produces
`report.html`. The report shows:

- Aggregate score and pass rate per scorer
- A sortable table of examples with per-scorer bars
- Click-to-expand drilldowns showing the full input, expected answer,
  model prediction, and per-scorer rationale

(See `report.html` after a run — intentionally not screenshotted into this
README so it's always in sync with the code.)

## Out of scope (on purpose)

- **Distributed execution.** Everything is single-process. If you need
  parallelism across examples, wrap `experiment.run` in a thread pool —
  but honestly, most eval sets are <1k examples.
- **A web UI / dashboard.** Reports are static HTML; compose them in S3,
  GitHub Pages, or a PR comment. I do not want to run a server.
- **A results database.** JSONL is good enough and diffs cleanly in git
  if you want to check in gold runs.
- **Built-in token/cost tracking.** The `LLMClient` protocol is the right
  place to add it — deliberately left open.
- **Prompt versioning / tuning.** Use git.

## Roadmap

- [ ] Caching layer: hash `(model, prompt, temp)` → response, so re-running
  an experiment only hits the API for new examples.
- [ ] Bootstrap confidence intervals on aggregates (so we stop comparing
  means of 20 examples and pretending a 2pp delta is real).
- [ ] Token + latency tracking surfaced in the HTML report.
- [ ] Ollama/vLLM HTTP backends for `local`.

## Layout

```
evalkit/
  core.py         # Dataset, Task, Scorer, Experiment, Run
  clients.py      # LLMClient protocol + openai/anthropic/local stubs
  scorers.py      # all six scorer implementations + registry
  config.py       # YAML experiment config loader
  compare.py      # regression detector
  report.py       # static HTML report generator
  __main__.py     # CLI: run / compare / list
experiments/
  sentiment.yaml
  sql.yaml
  data/*.jsonl
examples/
  quickstart.py
tests/
  test_core.py
  test_scorers.py
```

## License

MIT.
