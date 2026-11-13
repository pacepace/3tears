# Writer judge calibration: flash and haiku against the Opus editor, 2026-11-03

The record of one paid run of `tests/fixtures/writer_cases/writer_calibration.py`, kept as small JSON plus its log. The turn
recordings are not kept. Files: `20261103-flash-haiku.json` (the result), `20261103-flash-haiku-run.txt` (the printed run).

## What ran

`uv run python -m packages.evals.tests.fixtures.writer_cases.writer_calibration --cap-usd 8 --out <dir>` at head 1edac5200,
2026-11-03 22:20 to 23:41 UTC, two live judges (`google/gemini-3.8-flash` and `claude-haiku-5-5`), two trials per case. The cap
of $4 is the sum of the two flash arms' caps in the log ($2.80 and $1.20); the command line itself was not logged.

## Per-criterion weighted κ

| Judge | Criterion | Cases | Parse rate | κ | Weighted κ | Weighted κ bounds | Self κ |
|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | report_writer.actionable_next_steps | 28 | 1.00 | 0.04 | 0.12 | -0.13 to 0.43 | 0.47 |
| google/gemini-3.8-flash | report_writer.actionable_next_steps | 27 | 1.00 | 0.22 | 0.35 | -0.53 to 0.79 | undetermined |
| claude-haiku-5-5 | report_writer.coherent_story | 28 | 1.00 | -0.03 | 0.17 | -0.08 to 0.47 | 0.16 |
| google/gemini-3.8-flash | report_writer.coherent_story | 25 | 1.00 | 0.48 | 0.59 | 0.19 to 0.85 | 0.52 |
| claude-haiku-5-5 | report_writer.confounds_handled | 28 | 1.00 | 0.02 | 0.23 | -0.05 to 0.54 | 0.67 |
| google/gemini-3.8-flash | report_writer.confounds_handled | 25 | 1.00 | 0.35 | 0.40 | -0.34 to 0.80 | 0.62 |
| claude-haiku-5-5 | report_writer.evidence_fits_claims | 28 | 1.00 | 0.08 | 0.20 | -0.39 to 0.63 | 0.36 |
| google/gemini-3.8-flash | report_writer.evidence_fits_claims | 25 | 1.00 | 0.21 | 0.33 | -0.06 to 0.67 | 0.71 |
| claude-haiku-5-5 | report_writer.first_screen | 28 | 1.00 | 0.04 | 0.06 | -0.15 to 0.36 | 0.64 |
| google/gemini-3.8-flash | report_writer.first_screen | 27 | 1.00 | 0.15 | 0.15 | -1.00 to 0.82 | undetermined |
| claude-haiku-5-5 | report_writer.honest_emphasis | 28 | 1.00 | 0.06 | 0.23 | -0.12 to 0.57 | 0.39 |
| google/gemini-3.8-flash | report_writer.honest_emphasis | 24 | 1.00 | 0.31 | 0.40 | 0.02 to 0.72 | 0.54 |
| claude-haiku-5-5 | report_writer.preferred | 42 | 1.00 | 0.10 | 0.59 | 0.36 to 0.78 | 0.79 |
| google/gemini-3.8-flash | report_writer.preferred | 41 | 1.00 | 0.40 | 0.68 | 0.45 to 0.85 | 0.95 |
| claude-haiku-5-5 | report_writer.sound_elision | 28 | 1.00 | -0.01 | 0.05 | -0.11 to 0.32 | 0.47 |
| google/gemini-3.8-flash | report_writer.sound_elision | 25 | 1.00 | 0.46 | 0.46 | -0.48 to 0.86 | 0.63 |

Each pair of rounds is two writer-opus-pair cases, one per order, so its bounds count each pair twice.

## Honesty rule

These are an agent's labels (the Opus editor's recorded reviews). No judge profile is recorded from them. The figures read as
agreement with the Opus editor, nothing more.

## Rounds left out

Two prototype 4 round-2 recordings no longer replay, so their rounds are out of the case sets. The two long `RecordingRefused`
lines in the log document them.

| Round left out | Why |
|---|---|
| prototype4/r2-with-its-gates-read-in-each-task, round 2 | the recording holds no price ceiling for editor turn 8 |
| prototype4/r3-without-its-configs, round 2 | the recording holds no price ceiling for editor turn 5 |

## Stopped at the cap

The flash arms stopped at their cost caps, so they hold fewer trials than the haiku arms.

| Judge | Case set | Trials done | Of |
|---|---|---|---|
| google/gemini-3.8-flash | writer-opus-abs | 256 | 392 |
| google/gemini-3.8-flash | writer-opus-pair | 76 | 84 |
| claude-haiku-5-5 | writer-opus-abs | 392 | 392 |
| claude-haiku-5-5 | writer-opus-pair | 84 | 84 |
