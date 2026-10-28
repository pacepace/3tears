# W3 acceptance: the report writer's live runs (2026-10-28)

The fourth prototype, and the first one the report writer made itself: `write_report`, run live through the dev CLI (`tests/fixtures/report_plans/analyst_cli.py write`) on `claude-opus-5-5` with the first-party key, priced at Anthropic's published rates. Each directory is one run on one fixture snapshot: Kairo round 2, and the six misfits. The analyst and the report editor are sessions of one loop, and the editor reviewed the strict page. Each run was blind: it withheld its own source note (`replay.json`'s `withhold`) and the INDEX line, as in prototype 3. `test_report_docs_accepted.py` holds the stored records to the plan's bar, and `test_report_writer_replay.py` replays every recording.

## Cost per report

| Run | Ending | Cap | Cost | Analyst | Editor | Turns | Tool calls | Rounds (analyst/editor) | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| kairo | stored | $20 | $1.60 | $1.16 | $0.45 | 19 | 35 | 1/1 | 5.1 min |
| r1 reading only bars | stored | $4 | $1.58 | $1.24 | $0.34 | 22 | 53 | 1/1 | 5.4 min |
| r1 without its gates | stored | $4 | $1.70 | $1.36 | $0.35 | 24 | 48 | 1/1 | 6.2 min |
| r2 with one gate | stored | $4 | $2.07 | $1.63 | $0.44 | 22 | 40 | 1/1 | 7.2 min |
| r2 without its gates | stored | $4 | $2.28 | $1.75 | $0.52 | 26 | 47 | 1/1 | 7.3 min |
| r2 with its gates read in each task | stored | $8 | $3.48 | $2.35 | $1.13 | 37 | 60 | 2/2 | 10.6 min |
| r3 without its configs | stored | $8 | $2.17 | $1.50 | $0.67 | 23 | 50 | 2/2 | 6.8 min |

The seven reports cost $14.88: $1.58 to $3.48 each, $2.13 on average. No draft bounced. A revision round adds $0.50 to $1.60.

Two runs replaced earlier ones. At a $4 cap, "r2 with its gates read in each task" and "r3 without its configs" each ended `over_budget` after round 1: they had spent $1.87 and $1.83, and the turn bound refused a revision turn priced at up to $2.52. The owner approved reruns at $8. Both ended `stored`, and they are the runs kept here. The two refused runs cost $3.70, so X spent $18.58 in all.

**Against W2.** W2 recorded no dollar cost. Its prototypes ran as Claude Code subagents, and the plan's W2 status records "analyst ≈130–180k subagent tokens and 10–16 min per report; editor ≈100k and ~5 min; a revision ≈150–180k cumulative context". That measures something else: the tally here counts every turn's whole prompt again. Its nearest match is each session's largest prompt: 95k to 149k for the analyst and 30k to 67k for an editor. The comparable counts are below.

| | W2 (prototypes 1 to 3) | W3 (this run) |
|---|---|---|
| Analyst tool calls per report | 52 to 86 (first), 49 to 56 (second), 41 to 56 (third) | 25 to 39 |
| Analyst time per report | 10 to 16 min | 3.5 to 6.7 min |
| Editor time per report | about 5 min | 1.4 to 4.0 min |
| Dollars per report | not recorded | $1.58 to $3.48 |

## Size of each acceptance record

`record.json` is the stored `EvalReportDoc`. The analyst and editor sessions sit inline in its `generation`, and they are most of it, as the plan's L2 ruling accepted for W3. W4 moves them to transcript storage.

| Run | record.json | Sessions in it | The document |
|---|---|---|---|
| kairo | 253 kB | 180 kB | 11 kB |
| r1 reading only bars | 255 kB | 169 kB | 10 kB |
| r1 without its gates | 265 kB | 183 kB | 12 kB |
| r2 with one gate | 299 kB | 215 kB | 13 kB |
| r2 without its gates | 312 kB | 217 kB | 12 kB |
| r2 with its gates read in each task | 444 kB | 333 kB | 13 kB |
| r3 without its configs | 304 kB | 219 kB | 9 kB |

## The byte-identical prefix

The plan's L1 note says "X runs with `prefix_mismatch_behavior: "error"` to confirm". The parameter is real: `thinking.block_binding.prefix_mismatch_behavior`, which takes `"error"` or `"drop_block"`, under the `thinking-binding-controls-2026-08-01` beta. Setting either value opts a request into the history-editing check, whatever the account's age. The writer does not send it, so **X did not run with it**, and these runs do not confirm the check from the API's side. Running one would need that field and its beta on the live path, plus a new paid run. Neither is in X.

What X does confirm:
- `test_report_writer_session.py` asserts that consecutive request bodies share a byte-identical prefix across warnings, bounces and revisions, and that every turn carries the server-side `clear_tool_uses_20250919` edit.
- Live, every turn read the previous prompt from the cache, in every session of every run: on 156 of the 157 turns after a session's first, `cache_read_tokens` is the previous turn's input less 2 to 82 tokens. The one exception is the server-side clearing itself. On "r2 with its gates read in each task", analyst turn 18 crossed the 150k trigger. The prompt fell from 149k to 108k input tokens, the cache was read up to the first cleared result (25k) and written from there, and the request was accepted with its thinking blocks. That matches clearing as the plan corrected it.

Files in each directory, as `analyst_cli.py write` leaves them:
- `turns.jsonl`: every session's recorded turns, the recording a replay serves.
- `replay.json`: how the run was made, and what a replay must reach.
- `outcome.json`: the ending, why, and the reviews.
- `generation.json`: the tally, the reviews and every session.
- `record.json`, `doc.json`: the stored record and its document.
- `page.html`, `page.md`: the page, rendered again from `record.json` through the store's path, after the kit's 390 px fix. Rendering again changed only the inlined kit in `page.html`, and no `page.md`.
