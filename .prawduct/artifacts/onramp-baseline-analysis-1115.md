# On-ramp baseline, run 20261115T224108Z: analysis

Run: `packages/evals/onramp/runs/20261115T224108Z/` (worktree bp-o0). Arm docs@control (wheel 3tears_evals-0.95.0 built
at f9ab6f74d), model deepseek/deepseek-v4-flash, 3 apps x 3 repeats.

Terms, defined once:

- **runs**, **measures**, **read**: the three checks in `onramp/checks.py` (`eval_runs`, `measures_the_app`,
  `read_right`).
- **door**: `python -m threetears.evals FILE`.
- **first wrong step**: the earliest turn whose mistake caused a failed check or cost 3 or more turns. Stumbles of 1 or
  2 turns that the agent recovered from are listed apart, as **minor stumbles**.
- **D, M, P, H, A**: the classes in the brief (docs, message, package design, harness, agent).
- Trial directories are named by trial index; the table gives both the repeat and the directory.

## 1. Per trial

| App | Repeat (dir) | Turns | Cost | Runs | Measures | Read | First wrong step | Class |
|---|---|---|---|---|---|---|---|---|
| recipes | 1 (recipes-2) | 7 | $0.0014 | pass | pass | fail | t7: report JSON put in `command`, `text` empty; the harness parsed no report | H |
| recipes | 2 (recipes-5) | 8 | $0.0016 | pass | fail | pass | t5: invented 2 cases, expected values taken from the buggy code; never read README or `tests/fixtures/scaling.jsonl` | A |
| recipes | 3 (recipes-1) | 14 | $0.0042 | pass | pass | fail | t12: door printed `scaling: runner passed 10 of 10 cases.` and no grader line; the grader's mean was 0.60 | M |
| holds | 1 (holds-8) | 9 | $0.0017 | pass | fail | fail | t5: invented 8 cases whose expected values contradict the docstring; never read README or the fixture | A |
| holds | 2 (holds-6) | 8 | $0.0019 | pass | fail | pass | t6: grader bug; every status-200 case fails on both app and twin | A |
| holds | 3 (holds-7) | 15 | $0.0045 | pass | pass | pass | none | none |
| glossator | 1 (glossator-3) | 13 | $0.0038 | pass | fail | fail | t8: `gloss(case)` raised `KeyError: 'id'` on every case; the package withholds `id` from the runner | P |
| glossator | 2 (glossator-4) | 35 | $0.0195 | pass | pass | fail | t14: same `KeyError: 'id'`; turns 14 to 33 ($0.0134, 69% of the trial) spent getting past it | P |
| glossator | 3 (glossator-9) | 18 | $0.0074 priced, $0.0197 bound | fail | fail | fail | t5: invented 5 cases, then rewrote them 4 times (t11 to t14) instead of reading the fixture listed at t4 | A |

### Per-trial notes

**recipes 1.** Minor stumble at t3: the reply held three tool calls, refused with `Invalid JSON: trailing characters at line 2
column 1` (H). Reading at t7: `"first_line": "runner: runner passed 2 of 4 cases.", "pass_rate": 0.5, "finished": true`.
The truth was the same except the first line, which also carries `Cases that failed a trial: case 2, case 4.` So even if
the harness had read the `command` field, the report would still fail on the cut first line (A). Check fair: yes. A
human who copied the line whole would pass.

**recipes 2.** Minor stumble at t4: it read `examples/1-triage/triage.py`, which `llms.txt` names but the wheel does not
ship (D, 1 turn). Its case `3 tsp x 2` expects `1.5 tbsp`. That is the buggy app's 4 tsp per tbsp; the docstring says
`3 teaspoons are a tablespoon`. The twin then reads worse (`check_ingredients` 1.0 to 0.0). Check fair: yes.

**recipes 3.** Minor stumbles at t2 and t3: `ls /app`, `ls /` (H: the prompt never says the working directory is `.`).
The eval used the fixture correctly. Its grader `def grader(case, out):` has no `-> bool`, so the package counted it as
a score, not a check. The door then printed `passed 10 of 10 cases.` with no measure line, while `report.md` holds
`Grader score | 0.60`. The agent concluded pass_rate 1.0; the truth was 0.60 (4 of 10 cases fail on the spoon
conversion). It misread the first line, the only figure printed. The report was also put in `command` (H). Check fair
as a measurement, but the miss belongs to the package: a human reading the console would report 1.0 too.

**holds 1.** Example expected value: `{"action": "count", "today": "2026-03-10", ..., "expected_counts_expired": 0}`.
The buggy app gives 3 and the twin 6, so the twin reads no better (0.50 on both). t6 printed `runner: runner passed 8 of
8 cases.` with grader mean 0.50 (M), which hid the bad cases. At t7 and t8 the agent sent its report as bare JSON. The
harness refused both with `Extra inputs are not permitted` (H). The t8 version (`first_line` verbatim, `pass_rate` 0.5,
`finished` true) would have passed read. At t9 it sent `done` with empty `text`. Measures check fair: yes. Read check
unfair: a correct reading was thrown away by the harness.

**holds 2.** Minor stumble at t5: `Field required`, a write missing one field (H, 1 turn). The bug is in the grader:
`if "holds" in exp and "expired" in exp:` also fires on list cases, where `out.counts` is `{}`. 4 of 9 cases pass on
both app and twin. t7 printed `run: run passed 27 of 27 trials on 9 cases.` with `correct` 0.44 (M, hid the bug). The
agent still reported 0.4444, which matches. Check fair: yes.

**holds 3.** Passed all three. It read README, found the fixture and annotated `-> bool`. Minor stumbles: t5
`tests/__init__.py` (A). At t8 and t9 a read of `llms.txt` at start 200 returned `[lines 200-199 of 114]` with ok=True
(H, 2 turns).

**glossator 1.** Minor stumble at t5: cases nested as `input` and `expected`. t6 refused with `the field grader 'headword'
reads the case's 'headword', which it lacks`. That is a good message, and it was fixed in 1 turn; but `llms.txt` shows
no case line (D). At t8, `KeyError: 'id'` hit all 3 cases. `id` is in `RESERVED_CASE_KEYS` (`quick/one_call.py:207`,
"none reaches the runner"); `llms.txt` never says so, and the message does not either. The agent gave up. Its report
(in `command`, H) matched the run exactly: the full first line, 0.0, true. The read check is unfair here. Runs passed
although no case was scored: the check is too lenient (H).

**glossator 2.** Minor stumbles at t2 to t6: `ls /app`, `ls /`, reading `tests` as a file. At t14, t17 and t27 the run
printed the `KeyError: 'id'` (P). At t23 to t28 the agent wrote a probe eval with `grader=[]`, which was refused. At
t32 it rebuilt the runner from `case["term"]` and `case["lang"]` and passed measures. Reading at t35: `"first_line":
"glossary: runner passed 24 of 36 trials on 12 cases."`, which cuts `Cases that failed a trial: t02, t03, t05, t10.`
(A). Its pass_rate of 0.6667 matched the declared first grader. The console printed `Slug score: 100%` first and
`Accuracy: 67%` second, so "first measure" was ambiguous (H). The report was in `command` (H). Check fair only on the
cut line.

**glossator 3.** Minor stumble at t6: an empty reply that used 4096 output tokens (`the reply stopped short
(max_tokens)`, H). The wall cap stopped it at 360 s on t18, at 20 s a turn, before any door run (H). The design sets
the wall cap at 15 minutes. Its eval reads `case["id"]`, so its first run would have hit the same P. Runs check fair as
scored, but the cap cut a run that was still making progress.

## 2. Counts

First wrong step, by class:

| Class | Trials | Which |
|---|---|---|
| A | 4 | recipes 2, holds 1, holds 2, glossator 3 |
| P | 2 | glossator 1, glossator 2 |
| M | 1 | recipes 3 |
| H | 1 | recipes 1 |
| D | 0 | (D appears only as minor stumbles: recipes 2 t4, glossator 1 t5) |
| none | 1 | holds 3 |

Cause of each failed check (15 failed checks):

| Check | Failed | A | P | M | H |
|---|---|---|---|---|---|
| runs | 1 | | | | 1 (wall cap; glossator 3) |
| measures | 5 | 3 (recipes 2, holds 1, holds 2) | 1 (glossator 1) | | 1 (glossator 3, no eval) |
| read | 6 | | | 1 (recipes 3) | 5 |

Of the 5 read failures marked H, two would still fail with the harness fixed, because the first line was cut (A):
recipes 1 and glossator 2.

Two patterns:

| Cases came from | Trials | Measures passed |
|---|---|---|
| The app's fixture | recipes 3, holds 3, glossator 2 | 3 of 3 |
| Invented by the agent | recipes 1, recipes 2, holds 1, holds 2, glossator 1, glossator 3 | 1 of 6 |

| Done report placed | Count of 8 done calls |
|---|---|
| In `text`, as asked | 4 |
| In `command` | 4 (recipes 1, recipes 3, glossator 1, glossator 2) |

In addition, 3 replies were the bare report JSON (holds 1 t7 and t8, glossator 1 t12).

## 3. The three changes most likely to move the result

| # | Change | Kind | Prediction (testable) |
|---|---|---|---|
| 1 | A grader that returns `bool` counts as a check whether it is annotated or not. Failing that, the quiet print shows every grader's measure line, and the first line never says `passed N of N` while a grader's mean is below 1. | M | Re-running the saved eval files of recipes 3, holds 1 and holds 2 against the candidate wheel prints `passed 6 of 10`, `4 of 8` and `4 of 9`, or a `Grader score` line. That check is free and offline. Live: trials like recipes 3 report the right pass_rate (read passes), and trials like holds 1 and holds 2 see misses on their first door run and edit their cases or grader before `done`. |
| 2 | Pass `id` through to the runner, keeping it as the case's name too. Failing that, name the withheld key in the error: "`id` names the case and is withheld from the runner; ..." and add one line to `llms.txt`. | P (+M) | Glossator trials reach a scored door run on their first or second run. glossator 2's 20-turn detour (t14 to t33, $0.013) goes away. Median glossator turns drop from 18 to 13 or fewer. |
| 3 | `llms.txt`: one case line shown whole, plus three sentences: start from the cases the app already has (its fixtures, tests, README examples); a field grader compares the runner's output field to the case's field of the same name; write expected answers from what the app is meant to do, never by running it. | D | Trials like recipes 2, holds 1 and glossator 3 load the app's fixture instead of inventing cases. Measures goes from 1 of 6 to at least 4 of 6 on those trials, since fixture trials passed 3 of 3. |

Change 1 also removes the M that hid the agent's own errors in holds 1 and holds 2. It is the highest priority because
it is a wrong number, not only a missing hint.

## 4. Harness fixes needed before the numbers mean anything (H)

| # | Problem | Evidence | Fix | Effect on the baseline |
|---|---|---|---|---|
| H1 | The report is read only from `text` | 4 of 8 done calls put it in `command`; holds 1 sent it as bare JSON | Make the act schema per tool (`done` takes only `text`), or read the JSON from `text`, then `command`, then a bare-JSON reply | Read goes from 3 to 5 of 9 (glossator 1, holds 1) |
| H2 | Refusals of a malformed act do not say the shape | `Extra inputs are not permitted`, `Field required`, `Invalid JSON: trailing characters...`: 6 turns over 5 trials | Name the missing or extra field, and restate the one-object shape | Removes 6 wasted turns |
| H3 | The working directory is not named | `ls /app`, `ls /` in recipes 3 and glossator 2 (4 turns) | "Your working directory is `.` (the app)" | 4 turns |
| H4 | A read past the end reports ok with a garbled range | `[lines 200-199 of 114]` (holds 3, t8 and t9) | Say "llms.txt has 114 lines"; mark the read as failed | 2 turns |
| H5 | Wall cap 360 s; design says 15 min | glossator 3 stopped at t18, 20 s a turn, no door run | Cap on turns and dollars; set the wall cap per design, with the cell budget raised to match | Runs is 8 of 9 partly by this |
| H6 | `max_tokens` 4096 used up by reasoning | glossator 3 t6: empty reply, 4096 tokens | Raise max_tokens or cap reasoning tokens | 1 turn per hit |
| H7 | Runs passes an eval that scored nothing | glossator 1: `3 result(s): 0 scored, 3 failed by the candidate`, runs = pass | Require at least one scored result | Runs goes from 8 to 7 of 9 |
| H8 | "first measure" is ambiguous | Console order (`Slug score` first) differs from declared order (`headword`, shown as `Accuracy`); unannotated graders print no measure at all | Define `pass_rate` as the first line's passed share, or ask the report to name its measure | Makes read gradable |
| H9 | The stuck point misses door runs that exit 0 with every case failed | glossator 2's 18-turn `KeyError` loop is not its stuck point; the console names `ls: /app` | Count a door run with 0 scored as a failed turn | The stuck list names the real obstacles |
| H10 | The spend line leaves out the cut-off call | `total spend: $0.046`; glossator 3's bound adds $0.012 (`charged_usd` 0.0197) | Print the charged bound beside the reported cost | The bound is $0.058 |
| H11 | Two intervals for one rate | Console `8 of 9 trials, 95% range 0.52 to 1.00`; report page `17 to 99.7%` (clustered on 3 cases) | Print one, and say what it covers | Readers see the honest width |

Re-scoring after H1, H7 and H8 needs no model calls: the regrade can be run again over the 9 stored trials.

## 5. Wrong numbers and misleading lines the package printed to the agent

| # | Trial, turn | Printed | Truth |
|---|---|---|---|
| W1 | recipes 3, t12 | `scaling: runner passed 10 of 10 cases.` (no measure line) | Grader mean 0.60; `report.md`: `Pass rate | 100% (10 of 10 cases)` beside `Grader score | 0.60` |
| W2 | holds 1, t6 | `runner: runner passed 8 of 8 cases.` | Grader mean 0.50 |
| W3 | holds 2, t7 | `run: run passed 27 of 27 trials on 9 cases.` | `correct` mean 0.44 |
| W4 | recipes 1, t6 | `Cases that failed a trial: case 2, case 4.` then `case 1 (repeat 1)`, `case 3 (repeat 1)` | The same two cases, numbered from 1 in one line and from 0 in the next |
| W5 | holds 3, t14 | `case 1, case 2, case 3, case 7, case 8` then `case 0 ... case 7` | Same mismatch |
| W6 | glossator 1, t8; glossator 2, t14 | `0 scored, 3 failed by the candidate` beside `Accuracy: 0%, 0 of 3 trials`; `slug` "not shown, carried by no result" | No trial was graded. Accuracy should read none, as slug does; the same failure is 0% on one measure and absent on the other |
| W7 | glossator 1 and 2 | The field grader `headword` printed as `Accuracy`, with `per label: Hello world: precision none ..., recall 0%` | Free-text headwords treated as classifier labels; the measure is not named as declared |
| W8 | glossator 1, t8; glossator 2, t14, t17, t27 | `the candidate raised KeyError: 'id'`, exit 0 | The case file holds `id`; the package withheld it. The message does not say so. Exit 0 ("Every run completed") with every case failed |
| W9 | `llms.txt` line 54 | `python -m threetears.evals 1-triage/triage.py` | Not in the wheel and unreachable offline; recipes 2 t4 tried to read it |

W1 to W3 are wrong numbers: the package's headline says every case passed while the eval's only grader says it did not.
They hit 3 of 9 trials and caused or hid 3 failed checks.

## 6. What 9 trials cannot tell, and the smallest next run

| Rate | Observed | 95% interval (Wilson) |
|---|---|---|
| All three checks | 1 of 9 | 0.02 to 0.44 |
| Runs | 8 of 9 | 0.56 to 0.98 |
| Measures | 4 of 9 | 0.19 to 0.73 |
| Read | 3 of 9 | 0.12 to 0.65 |

What 9 trials cannot tell:

- Any rate to within 30 points.
- Any difference between apps (3 trials each).
- Whether a docs fix helps less than hugely.
- Whether failures are the model's or the docs' (one tier only).
- The read rate at all until H1 and H8 are fixed: 5 of its 6 failures involve the harness.

Smallest next run, in order:

| Step | Runs | Arms | Expected cost | What it answers |
|---|---|---|---|---|
| 0 | none | none | $0 | Re-run the regrade over the 9 stored trials with H1, H7 and H8 fixed. Replay the saved eval files of recipes 3, holds 1 and holds 2 on a wheel with change 1. |
| 1 | 42 | control and candidate (changes 1 to 3, harness fixed), 3 apps x 7 repeats each, cheap tier | about $0.27 (observed $0.0051 a run, $0.0065 counting cut-off calls); bound $21 at the $0.50 cap | 21 a side detects all-three rising from 11% to 50% (two-sided 0.05, power 0.8) |
| 2 | 9 | candidate, ceiling tier (claude-haiku-5-5, direct), 3 x 3 | about $0.08 at the harness's stated rates and this run's mean of 52k input and 8.2k output tokens a run; bound $13.50 | Whether the remaining failures are the model's |

The control must be re-run under the fixed harness. The 9 trials here are not a valid control for step 1.
