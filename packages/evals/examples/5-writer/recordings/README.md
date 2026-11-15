# Rung 5's recordings

`python -m threetears.evals examples/5-writer/writer.py` replays these with no key: no network, no cost.

| What | Recorded |
|---|---|
| On | 2026-11-03, at head `28efc1fd1`, the writer's turns; the judge's calls on 2026-11-14, over the pages built at `67c222242` |
| Reports | 14: 2 arms x 7 held-out cases, once each, about 12.4 writer turns a report |
| Writer | `claude-sonnet-5-5` at effort `medium`, analyst and editor: $6.90 |
| Judge | `google/gemini-3.8-flash`: 112 calls, the pairwise pass's 14 among them, $0.28 |
| Total | $7.18 |

The judge reads each report's page as the package lays it out, so a change to what that page prints (its words, its
figures' precision) moves every judge call's key: the judge's calls are recorded again with the `RECORD=1` command
below, and the writer's turns replay free.

**When the page text moves, the replay says so.** A call whose prompt moved is no longer found, and it is never
dropped in silence. A scoring call the recording does not hold fails that arm's run at the miss, and the eval exits 1.
The pairwise pass stops at the first call it does not hold and ends `failed`, so the run prints no share and exits 1.
Either way the message names `judge.jsonl`, the prompt's
opening, its key, and where it parts from the nearest recorded prompt. If you meant to change the page, record the
judge again with the `RECORD=1` command below, and re-pin the tests that read its shares, saying why. If you did not,
the message names the prompt that moved.

A file deleted or gone stale is refused by name, with the key that would record it: `judge.jsonl` and
`OPENROUTER_API_KEY`, or the writer's first turn file and `ANTHROPIC_API_KEY`.

**The one command that records them**, from `packages/evals/` (Sonnet for both the analyst and its editor, effort
`medium`, under the owner-approved $39 cap):

```bash
ANTHROPIC_API_KEY=... OPENROUTER_API_KEY=... uv run python -m threetears.evals examples/5-writer/writer.py
```

No `RECORD=1`: a recording that is missing is made when its key is set, and one that exists is replayed. The run
writes:

| File | What it holds | Key |
|---|---|---|
| `judge.jsonl` | the scoring judge's calls, `google/gemini-3.8-flash` through OpenRouter | `OPENROUTER_API_KEY` |
| `turns/<arm>-<skill12>-claude-sonnet-5-5-medium-r<revisions>/<case>-<snapshot12>.jsonl` | one report's analyst and editor sessions, one file per arm and case | `ANTHROPIC_API_KEY` |

The store it writes, `packages/evals/.evals/`, is ignored: commit the recordings alone.

## What it may spend

It writes the 7 held-out cases for each of the 2 arms: 14 reports. The figures are set in `writer.py`, and these are
the ones it sets today:

| | dollars |
|---|---|
| `max_cost_usd`, the owner-approved cap | 39.00 |
| the pairwise pass's reserve (`pairwise_share=0.05`) | 1.95 |
| each arm's part, half the rest | 18.52 |
| an arm's 7 reports, each capped at `budget_usd` 2.50 and 45 turns | 17.50 |
| an arm's judging: 7 cases x 7 dimensions x 2 attempts at the judge's floor | 0.99 |
| the pass: 7 pairs x 2 orders x 2 attempts at the judge's floor | 0.28 |

The judge's floor is one call with no page in it, $0.0101: its 2,048-token output cap and 2,048 prompt tokens at
OpenRouter's rates. The run is refused before any arm launches when an arm's reports and judging, or the pass, could
pass its part at that floor. The pages a call reads are priced only once they are written: a case's judging is then
priced in full and admitted against what its run has left, and a case it does not fit is set aside unjudged, so an
arm's judging stays within its part wherever the judge's price ceiling bounds what it is billed.

## When a recording run is cut short

A run stopped part-way (a 429 or 529 the retries did not outlast, a refusal, Ctrl-C) leaves files a later run cannot
finish on its own.

**A report's turn file.** The next run replays it, runs off its end, and refuses that cell, naming the file. A turn
file is never recorded again over itself, so delete it by hand (the path matches
`recordings/turns/<arm>-<skill12>-claude-sonnet-5-5-medium-r<revisions>/<case>-<snapshot12>.jsonl`), then run the
record command again: every finished report replays free, and the deleted one is written and paid for again.

**`judge.jsonl`.** Without `RECORD=1` a judge file that exists is replayed, and every call it lacks is refused. Once
the turn files are whole, record it again from the start, from `packages/evals/`:

```bash
RECORD=1 ANTHROPIC_API_KEY=... OPENROUTER_API_KEY=... uv run python -m threetears.evals examples/5-writer/writer.py
```

`RECORD=1` starts a judge file written before this run afresh and makes every judge call again, the pass's included.
It never records over a turn file, so the reports replay free. The judge's settings (`temperature`, `max_tokens`) are
part of what each call is matched on: change one after recording, and record the judge again with this command.

## What a recording is keyed by

A turn recording is keyed by everything it does not check: the arm, the skill's digest, the model, the effort, the
revisions and the case's snapshot. Edit `skills/lead-first.md`, or refreeze a case, and its old recordings are no
longer found: a run with no key refuses again, and one with both keys records them afresh.
