# Rung 5's recordings

Not recorded yet. Until they are, `python -m threetears.evals examples/5-writer/writer.py` refuses before anything
runs, naming the first missing file and the key that would record it: `judge.jsonl` and `OPENROUTER_API_KEY` first,
then the writer's first turn file and `ANTHROPIC_API_KEY`.

**The one command that records them**, from `packages/evals/` (owner-approved: Sonnet for both the analyst and its
editor, effort `medium`, at most $15 for the whole run, the judge included):

```bash
ANTHROPIC_API_KEY=... OPENROUTER_API_KEY=... uv run python -m threetears.evals examples/5-writer/writer.py
```

No `RECORD=1`: a recording that is missing is made when its key is set, and one that exists is replayed. The run
writes:

| File | What it holds | Key |
|---|---|---|
| `judge.jsonl` | the scoring judge's calls, `google/gemini-3.8-flash` through OpenRouter | `OPENROUTER_API_KEY` |
| `turns/<arm>-<skill12>-claude-sonnet-5-5-medium-r<revisions>/<case>-<snapshot12>.jsonl` | one report's analyst and editor sessions, one file per arm and case | `ANTHROPIC_API_KEY` |

It writes the 7 held-out cases for each of the 2 arms: 14 reports, each capped at $1.00, each arm under its $7.50
half of `max_cost_usd`, which the engine enforces on the run's whole spend. The 4 dev cases are for tuning a change
and are not recorded by it.

A turn recording is keyed by everything it does not check: the arm, the skill's digest, the model, the effort, the
revisions and the case's snapshot. Edit `skills/lead-first.md`, or refreeze a case, and its old recordings are no
longer found: a run with no key refuses again, and one with both keys records them afresh.
