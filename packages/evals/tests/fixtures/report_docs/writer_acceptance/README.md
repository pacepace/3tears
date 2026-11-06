# Writer acceptance: the hollow skill against the shipped skill, 2026-11-03

The record of one paid run of `tests/fixtures/writer_cases/writer_acceptance.py`, kept as small JSON plus its log. The turn
recordings and the store are not kept. Files: `20261103-sonnet-hollow.json` (the result), `20261103-sonnet-hollow-run.txt` (the
printed run).

## What ran

`uv run python -m packages.evals.tests.fixtures.writer_cases.writer_acceptance --cases <11 case files> --held-out-fraction 0.99
--writer claude-sonnet-5-5 --effort medium --budget-usd 1 --turns 36 --cap-usd 32 --out <dir>` at head 6cac37f85, 2026-11-03
23:45 to 23:58 UTC. The 11 cases are rung 5's case files, `packages/evals/examples/5-writer/cases` (`--cases`); 0.99 puts every group on the held-out side, since the promotion read
reads held-out cases only. The scoring judge is `google/gemini-3.8-flash`. The command line itself was not logged; the options
are read from the log and the script's defaults.

## Result

| Reading | Figure |
|---|---|
| Primary (preferred by the judge, side by side) | not separated |
| Difference, in favour of hollow | +0.4545 |
| Interval at 0.95 | -0.327 to 0.9393 |
| Order consistency | 9 of 11 cases, 0.82 |
| Held-out cases like these that would settle the primary | 16 |

Gated readings: finished 0.27 higher for hollow (-0.35 to 0.80), must-fix findings open 0.55 lower for hollow (-2.22 to 1.13);
none separated.

## Pass criterion

FAIL: not held: the read answers keep; the primary reading reads regressed; it separated, its interval below 0. Both arms ran
every case (11 of 11), order consistency stood at 0.7 or more, and bounces did not separate.

## Pre-launch refusals and spend

Three $0 refusals preceded the run: the reserve; the judge output cap; the job budget.

| Part | Cap | Spent |
|---|---|---|
| shipped | $12.80 | $3.32 |
| hollow | $12.80 | $3.73 |
| pairwise pass (22 calls) | $6.40 | $0.19 |

## Open question

Do the skill files hurt Sonnet, or are the judge and the editor blind to them? The blind read is pending.

## The blind read (2026-11-04)

A stronger model read the eleven pairs with the arm key sealed (`20261103-blind-read.md`, `.json`; the key in
`20261103-blind-read-key.json`): shipped better in 6 cases, hollow in 5, no ties; weighted by the reader's own
confidence (sure 1.0, lean 0.6, coin flip 0.2) 3.2 to 3.4. The reader found statistical slips in both arms' pages
that the editor had passed. These are an agent's verdicts, recorded as such; no label or profile is derived from them.

## 2026-11-04: the `no-editor` variant (`20261104-sonnet-no-editor.json`, `20261104-sonnet-no-editor-run.txt`)

Same command with `--variant no-editor` (the shipped skill, `revisions=0`: the editor reviews once and never sends a
report back). The first attempt (`20261104-sonnet-no-editor-attempt1-520-run.txt`) ended after $0.67: one turn in each arm
failed with an Anthropic 520, the result's cost read as unpriced, and the cap rule stopped both runs; that is a bug,
fixed on `reset/failed-turn-cost`. The second attempt completed for $5.39 ($3.14 shipped, $2.16 no-editor, $0.09 pass).

| Reading | Figure |
|---|---|
| Primary, preferred side by side | not separated; difference -0.45, interval -1 to 0.25 |
| Cases with both pages | 7 of 11; order consistency 6 of 7 |
| Held-out cases that would settle the primary | 25 |
| Pass criterion | FAIL: the primary did not separate |

The direction is the expected one (the arm without an editor loop reads worse); the size is not enough to call it.

# Writer acceptance: no-editor against the shipped skill over 25 held-out cases, 2026-11-06

Files: `20261106-sonnet-no-editor-25.json` (the result), `20261106-sonnet-no-editor-25-run.txt` (the printed run). The turn
recordings and the store are not kept.

## What ran

`uv run python -m packages.evals.tests.fixtures.writer_cases.writer_acceptance --variant no-editor --cases <25 case files>
--held-out-fraction 0.99 --writer claude-sonnet-5-5 --effort medium --budget-usd 0.75 --turns 30 --request-s 50 --cap-usd 56
--out <dir>`, 2026-11-06 19:06 to 19:16 UTC, over rung 5's 11 case files plus the 14 held-out cases of
`tests/fixtures/writer_cases/held_out/` (the command the held-out README gives). The scoring judge is `google/gemini-3.8-flash`.

## Result

| Reading | Figure |
|---|---|
| Cases both arms completed | 25 of 25 |
| Cases with a stored page in both arms, so paired for the judge | 7 of 25 |
| Primary (preferred by the judge, side by side) | not separated |
| Interval at 0.95 | -0.044 to 0.504 |
| Candidate's share of the 7 pairs | 0.605 |
| Order consistency | 7 of 7 |
| Held-out cases like these that would settle the primary | 43 |
| Spend | shipped $2.58, no-editor $2.58, pairwise pass $0.05 |

## What the run found about itself

36 of the 50 reports ended `over_budget with no report` after 1 to 5 turns, each at about $0.10: a report's budget reserves
the next turn at its worst-case price (`the next turn can cost up to $0.67, and $0.11 of the report's $0.75 budget is spent`),
so a $0.75 budget admits a few turns and then refuses. Rung 5's finished reports cost $0.49 to $0.73 under a $2.50 budget.
The pass criterion `both arms ran every case` read yes because every cell completed, and the pairwise pass counted each
both-missing pair as a tie, so the primary read on 7 pairs. The budget rule and the criterion are filed as W6 and W7 in
the reset's backlog. The 25 cases themselves are sound (their Critic verified them before the run).

Since W6 and W7, this command is refused before anything runs: two of a Sonnet first turn's ceiling come to more than
$0.75. The criterion is now named `both arms stored a page on every case` and reads off the pass's pairs, where this
run reads shipped 7 of 25 and no-editor 9 of 25 and fails; the readout's first table leads with `Cases with two pages
to compare`, 7 of 25 here, beside 16 ties from a page missing on both sides. The JSON and the printed run above are the
record as it was and stay unchanged.

# Writer acceptance: no-editor against the shipped skill over 25 held-out cases at a $1.50 report budget, 2026-11-06

Files: `20261106-sonnet-no-editor-25-budget150.json` (the result), `20261106-sonnet-no-editor-25-budget150-run.txt` (the
printed run), `20261106-sonnet-no-editor-25-budget150-refused-cap102.txt` (the first attempt, refused before launch at $0:
the judge calls price at $0.0101 each, so the arm needed $41.05 against a $40.80 share of a $102 cap). The turn recordings
and the store are not kept.

## What ran

The same command as the $0.75 run above with `--budget-usd 1.5 --cap-usd 104`, 2026-11-06 21:03 to 21:40 UTC, over the same
25 cases (s4 and hot-path-scaled re-minted after the margin chunk). Both arms wrote a page on every case.

## Result

| Reading | Figure |
|---|---|
| Cases with two pages, paired for the judge | 25 of 25 |
| Primary (preferred by the judge, side by side) | regressed: the candidate preferred on 0 of 25 pairs |
| p | 3.1 × 10⁻⁷ |
| Order consistency | 25 of 25 |
| Gated readings regressed for no-editor | coherent_story -0.92, confounds_handled -0.92, evidence_fits_claims -1.24, sound_elision -0.56, must-fix findings open +1.48 |
| Gated readings not separated | actionable_next_steps, first_screen, honest_emphasis, bounces, report finished, the pass rate |
| Production cost | improved for no-editor (-0.21: it is cheaper) |
| Spend | shipped $10.95, no-editor $5.67, pairwise pass $0.40 |
| Verdict | PASS: the gate reads the shipped skill with no editor revisions as writing worse reports |

This is the first run on which the writer gate separated two configurations of the same writer on held-out cases, so the
pairwise instrument has signal at n=25 (design decision D4, "signal before search").
