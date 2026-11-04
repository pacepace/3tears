# Writer acceptance: the hollow skill against the shipped skill, 2026-11-03

The record of one paid run of `tests/fixtures/report_plans/writer_acceptance.py`, kept as small JSON plus its log. The turn
recordings and the store are not kept. Files: `20261103-sonnet-hollow.json` (the result), `20261103-sonnet-hollow.log` (the
printed run).

## What ran

`uv run python -m packages.evals.tests.fixtures.report_plans.writer_acceptance --cases <11 case files> --held-out-fraction 0.99
--writer claude-sonnet-5-5 --effort medium --budget-usd 1 --turns 36 --cap-usd 32 --out <dir>` at head 6cac37f85, 2026-11-03
23:45 to 23:58 UTC. The 11 cases are the case files made for this run; 0.99 puts every group on the held-out side, since the promotion read
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
