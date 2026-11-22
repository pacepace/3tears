# On-ramp comparison, run 20261116T004246Z: analysis

Run: `packages/evals/onramp/runs/20261116T004246Z/` (worktree bp-o0). Model deepseek/deepseek-v4-flash. Arms:
docs@control (wheel built at f9ab6f74d) and docs@candidate (wheel built at 2e3105136), 3 apps x 7 repeats each.

Terms, defined once:

- **runs**, **measures**, **read**: the three checks in `onramp/checks.py` (`eval_runs`, `measures_the_app`,
  `read_right`).
- **twin**: the app's fixed copy (`onramp/twins/`). Planted defects: recipes `TSP_PER_TBSP = 4` (docstring: "3
  teaspoons are a tablespoon"); holds `GRACE_DAYS = 7` (docstring: expired "once its pickup date has passed");
  glossator drops French accents from the headword.
- **door**: `python -m threetears.evals FILE`.
- **first wrong step**: the earliest turn whose mistake caused a failed check. A mistake the agent recovered from is a
  **stumble**, listed apart.
- **D, M, P, H, A**: docs, message, package design, harness, agent, as in the brief.
- **invented cases**: cases the agent wrote itself instead of loading the app's `tests/fixtures/*.jsonl`.
- Repeat r1 to r7; the trial directory name is in brackets. Cost is the provider-priced cost; for a run the wall cap
  cut off, the charged bound follows it.

## 1. Per trial

### docs@control

| App | Rep (dir) | Turns | Cost | Passed | First wrong step | Class |
|---|---|---|---|---|---|---|
| recipes | r1 (recipes-12) | 7 | $0.0020 | runs, read | t4 `ls examples/ tests/` printed only `pancakes.json`: the harness lists the first path and drops the rest, so `tests/` was never seen; 5 invented cases, none crosses tsp to tbsp | H |
| recipes | r2 (recipes-21) | 13 | $0.0071 | all | none (read README, used the fixture) | none |
| recipes | r3 (recipes-13) | 7 | $0.0015 | runs, read | t5 one invented case, cups only; README and `tests/` unread | A |
| recipes | r4 (recipes-9) | 31 | $0.0332 | runs, read | t13 invented cases, expectations wrong on both apps; README unread | A |
| recipes | r5 (recipes-18) | 10 | $0.0035 | runs, read | t7 invented; expects `1.0 tbsp` for 4 tsp, the code's 4-tsp tablespoon over the docstring | A |
| recipes | r6 (recipes-20) | 8 | $0.0024 | runs, read | t6 grader recomputes the answer from the app's own `_IN_TSP` | A |
| recipes | r7 (recipes-3) | 26 | $0.0313 | runs, read | t20 grader checks `serves` only, after 14 turns reading package source (t6-t19) | A |
| holds | r1 (holds-10) | 12 | $0.0057 | runs, read | t10 invented from `data/holds.json`, expiry from the code's 7-day grace (twin 0.6 to 0.4) | A |
| holds | r2 (holds-8) | 7 | $0.0026 | runs, read | t5 same (twin 1.0 to 0.6) | A |
| holds | r3 (holds-19) | 8 | $0.0027 | runs, read | t5 same (twin 1.0 to 0.33) | A |
| holds | r4 (holds-15) | 7 | $0.0020 | runs, read | t5 same (twin 0.8 to 0.4) | A |
| holds | r5 (holds-11) | 14 | $0.0067 | runs | t6 invented; one case from the docstring, one from the code: 0.75 on both apps. Read then failed on M (§6 W1) | A |
| holds | r6 (holds-2) | 18 | $0.0105 | runs, read | t9 cases hold only `{"status": 200}` | A |
| holds | r7 (holds-6) | 15 | $0.0101 | runs, read | t13 invented, expectations wrong on both apps (0.6 both) | A |
| glossator | r1 (glossator-16) | 13 | $0.0053 | read | t7 `the candidate raised KeyError: 'id'` on every case; gave up | P |
| glossator | r2 (glossator-14) | 24 | $0.0191 | none | t11 same `KeyError: 'id'`; 13 turns, gave up; report also cut the first line (A) | P |
| glossator | r3 (glossator-17) | 20 | $0.0123 | all | none | none |
| glossator | r4 (glossator-1) | 29 | $0.0240 | none | t28 `KeyError: 'id'`; then a report naming `eval_gloss.py`, never written, with a made-up first line (A) | P |
| glossator | r5 (glossator-5) | 12 | $0.0058 (bound $0.044) | runs, measures | wall cap at t12 (37 s a turn), its fixed eval written at t10 and not yet run | H |
| glossator | r6 (glossator-4) | 16 | $0.0077 (bound $0.044) | runs | t7 `KeyError: 'id'`, 9 turns, unresolved at the wall cap | P |
| glossator | r7 (glossator-7) | 24 | $0.0198 (bound $0.047) | none | t15 `KeyError: 'id'`, 9 turns, unresolved at the wall cap | P |

### docs@candidate

| App | Rep (dir) | Turns | Cost | Passed | First wrong step | Class |
|---|---|---|---|---|---|---|
| recipes | r1 (recipes-12) | 18 | $0.0107 | runs, read | t5 invented, expectations from the code's 4-tsp tablespoon (twin 1.0 to 0.33) | A |
| recipes | r2 (recipes-21) | 4 | $0.0017 | none | t4 `done` with a made-up report; no file written | A |
| recipes | r3 (recipes-13) | 10 | $0.0047 | all | none | none |
| recipes | r4 (recipes-9) | 14 | $0.0062 | all | none | none |
| recipes | r5 (recipes-18) | 7 | $0.0023 | runs, read | t5 invented; expects `6.0 tsp` for 6 tsp (wrong on both apps); no other case crosses a unit | A |
| recipes | r6 (recipes-20) | 8 | $0.0025 | all | none | none |
| recipes | r7 (recipes-3) | 14 | $0.0046 | all | none | none |
| holds | r1 (holds-10) | 12 | $0.0050 | runs, read | t7 read `tests/fixtures`, answered `tests/fixtures: no such file` (it is a directory); t8 invented | H |
| holds | r2 (holds-8) | 8 | $0.0034 | runs, read | t6 invented, code's grace (twin 1.0 to 0.57) | A |
| holds | r3 (holds-19) | 11 | $0.0041 | runs, read | t5 same (twin 1.0 to 0.6) | A |
| holds | r4 (holds-15) | 10 | $0.0040 | runs, read | t6 same (twin 1.0 to 0.75) | A |
| holds | r5 (holds-11) | 15 | $0.0092 | runs, read | t5 read `tests`, answered `tests: no such file`; grader checks status only | H |
| holds | r6 (holds-2) | 11 | $0.0086 | runs, read | t9 invented, code's grace (twin 1.0 to 0.75) | A |
| holds | r7 (holds-6) | 6 | $0.0030 | none | t6 `done` with a made-up report; no file written | A |
| glossator | r1 (glossator-16) | 15 | $0.0063 | all | none | none |
| glossator | r2 (glossator-14) | 38 | $0.0251 (bound $0.060) | runs | t5 invented cases with the buggy headword (`"Cote d'azur"`), the fixture directory listed at t4; then `KeyError: 'id'` for 24 turns, wall cap | A |
| glossator | r3 (glossator-17) | 19 | $0.0100 | none | eval in `tests/` ran (8 of 12, report right); the regrade crashed importing the app (`Traceback`) | H |
| glossator | r4 (glossator-1) | 28 | $0.0248 | read | t11 `KeyError: 'id'`; 17 turns, gave up, reported 0.0 | P |
| glossator | r5 (glossator-5) | 24 | $0.0156 (bound $0.043) | runs | t15 `KeyError: 'id'`, unresolved at the wall cap | P |
| glossator | r6 (glossator-4) | 21 | $0.0161 (bound $0.047) | none | t5 read `tests`, answered `tests: no such file`; t6 invented | H |
| glossator | r7 (glossator-7) | 16 | $0.0088 (bound $0.034) | none | t8 `KeyError: 'id'`, unresolved at the wall cap | P |

### Stumbles (recovered, not a first wrong step)

| Arm | Trial, turn | What | Class | Turns |
|---|---|---|---|---|
| control | gloss r3 t7 | `model = Model("none")`, after llms.txt "A FIRST EVAL NEEDS 2 NAMES ... Model"; refused `'none' names no provider` | D | 6 |
| control | recipes r4 t14, r7 t21 | `an eval is named by name=, its cases file or its runner, and this one has none of them` (lambda runner) | M | 16, 3 |
| control | holds r5 t9 | `every case must be JSON: ... Object of type Reply is not JSON serializable` | M | 4 |
| control | gloss r3 t14, r5 t7 | `KeyError: 'id'`, fixed | P | 5, 3 |
| control | holds r1 t1, gloss r4 t4, gloss r7 t5 | read of a directory: `./: no such file`, `tests/: no such file` | H | 1 each |
| control | holds r5 t2, r7 t2; holds r6 t7 | `ls: -la is no directory ...`; `mkdir` refused | H | 1 each |
| candidate | recipes r1 t7 | `a scorer named match takes a measure the classifier track owns; ...`; rewrote its cases file 5 times before renaming the grader | M | 9 |
| candidate | holds r1 t9 | exit 3 traceback for a `@dataclass` in the eval file (§3b) | P | 1 |
| candidate | gloss r1 t9, r3 t13 | `KeyError: 'id'`, fixed | P | 5, 5 |
| candidate | gloss r3 t11, r5 t13 | `the cases file tests/tests/fixtures/glossary.jsonl is not there (in .)` | A | 1 each |
| candidate | recipes r1, r7; holds r1, r7; gloss r4 | `ls: -la is no directory ...` | H | 1 each |
| candidate | holds r4 t2; gloss r3 t7 | `.: no such file`; empty command refused | H | 1 each |

## 2. Class counts

| First wrong step | Control | Candidate |
|---|---|---|
| none (all three checks) | 2 | 5 |
| A | 12 | 9 |
| P | 5 | 3 |
| H | 2 | 4 |
| M | 0 | 0 |
| D | 0 | 0 |

Every P is the same one: the package withholds a case's `id` from the runner, and the app's `gloss(entry)` reads
`entry["id"]`. It hit 14 of 14 glossator trials; 4 recovered; the other 10 spent 117 turns on it (19% of all 630
turns). It was the baseline's change 2 and did not land.

The A count is dominated by invented cases (21 of 21 A-class trials invented cases or wrote none). D is 0 as a first
wrong step, but the docs leave out where cases come from (§4).

## 3. Did the package changes work in an agent's hands?

### (a) Unannotated bool graders

| | Control | Candidate |
|---|---|---|
| Trials whose door runs used an unannotated grader returning True/False | 9 (recipes r1, r3, r6, r7; holds r5, r6, r7; gloss r4, r7) | 6 (recipes r3; holds r3, r4, r5, r6; gloss r2) |
| Of those, runs where the grader failed some case | 2 | 2 |
| Headline said every case passed over a failing grader | 2 (holds r5, r7) | 0 |
| Wrong report that followed | 1 (holds r5: pass_rate 1.0, truth 0.75) | 0 |
| Failing cases listed under "top misses" | 0 | 2 |

What each arm printed for a failing unannotated grader:

| Arm | Trial, turn | Printed |
|---|---|---|
| control | holds r5 t13 | `run: run passed 12 of 12 trials on 4 cases.` / `Grade score: mean 0.75 (12 trials, min 0, max 1)` |
| control | holds r7 t14 | `runner: runner passed 15 of 15 trials on 5 cases.` / `Grader score: mean 0.6 (15 trials, min 0, max 1)` |
| candidate | recipes r3 t9 | `scaling: runner passed 18 of 30 trials on 10 cases. Cases that failed a trial: case 1, case 4, case 9, case 10.` / `Grader score: 60%, 18 of 30 trials` |
| candidate | holds r3 t8 | `eval_cases: runner passed 4 of 5 cases. Cases that failed a trial: case 2.` / `Grader score: 80%, 4 of 5 trials` |

What the candidate's agents did with the failures: recipes r3 reported 0.6 (right, and measures passed: those are
the planted defect). holds r3 found case 2 was its own mistake (a 404 case with no `expected_message`) and fixed it at
t9: a legitimate fix. Its other expectations still came from the buggy code, so measures failed anyway. The change
did its job: no wrong headline in the candidate, and no counterpart to the control's wrong report (holds r5). It moved
no measures result.

### (b) Refusals hit in the candidate

No refusal new in the candidate fired. All six that fired exist in both wheels.

| Message (first hit quoted) | Trial, turn | Outcome | Says what to do? |
|---|---|---|---|
| `a scorer named match takes a measure the classifier track owns; to grade a classifier, grade it with a field grader (grader=['label']), and name any other scorer something else` | recipes r1 t7 | flailed 9 turns (rewrote its cases file 5 times), then renamed: recovered | Partly. The fix is the last clause, after a classifier remedy that does not apply |
| `failed on an unanticipated error`, traceback ending `AttributeError: 'NoneType' object has no attribute '__dict__'` | holds r1 t9 | dropped its `@dataclass`, recovered in 1 turn | No. It is a package bug: `quick/files.py` `load_evals` runs the module without putting it in `sys.modules`, which `@dataclass` needs (both wheels) |
| `python -m threetears.evals FILE takes only --verbose; not 2>&1` | holds r4 t9 | sent done (it already had a run) | Yes |
| `the cases file tests/tests/fixtures/glossary.jsonl is not there (in .)` | gloss r3 t11, r5 t13 | recovered in 1 turn each | Yes, by the doubled path; it does not say a cases path is read beside the eval file |
| `gloss_eval.py defines no Eval at module level; assign one (my_eval = Eval(...)) to run it` | gloss r6 t18 | recovered in 1 turn | Yes |

### (c) Kind and declaration refusals at build time

None fired, in either arm. Every function grader written in 42 trials was a plain `def` (none `async`), annotated
`-> bool` (8 trials a side) or not at all (9 control, 6 candidate). No `-> None`, `-> float` or union. Every runner
took `(case)` and returned a value. Every field grader named a field its cases had. Pass counts that state ungraded
cases never showed: no grader raised. This run gives no evidence on (b) and (c); they need seeded tests, not more
trials.

### (d) Did anything candidate-only make it worse?

| Runs failures | Control (4) | Candidate (6) |
|---|---|---|
| `KeyError: 'id'` on every case (P) | 3 (gloss r1, r2, r7) | 3 (gloss r4, r6, r7) |
| Made-up report, no eval file (A) | 1 (gloss r4) | 2 (recipes r2 t4, holds r7 t6) |
| Regrade crash on an eval in `tests/` (H) | 0 | 1 (gloss r3) |

The two extra failures are not refusals. Two are made-up `done` reports sent after reading only llms.txt and the
app's source, before any package output. One is the harness. One candidate-only line misled (§6 W4): `cases: runner
ran 4 of 4 cases; no grader here says pass or fail.` (gloss r2, t14 and t29). The grader was `-> bool` and all 4 cases
raised. The agent then dropped the annotation and wrote a probe grader returning `True` (t31). The `KeyError` loop
around it is in both arms.

## 4. Why "measures the app" fails

| Cause of the measures failure | Control | Candidate | Trials |
|---|---|---|---|
| Invented cases, expected values taken from the buggy code (twin reads worse) | 5 | 6 | ctl recipes r5, holds r1-r4; cand recipes r1, holds r1-r4, r6 |
| Invented cases that cannot expose the defect, or expectations wrong on both apps | 4 | 1 | ctl recipes r1, r3, holds r5, r7; cand recipes r5 |
| Grader checks the wrong thing (structure, status, `serves`, recomputes from the app's constant, a probe returning `True`) | 4 | 2 | ctl recipes r4, r6, r7, holds r6; cand holds r5, gloss r2 |
| Runner raised on every case (`KeyError: 'id'`) | 4 | 4 | ctl gloss r1, r2, r6, r7; cand gloss r4-r7 |
| No eval to re-run | 1 | 2 | ctl gloss r4; cand recipes r2, holds r7 |
| Regrade crash (harness) | 0 | 1 | cand gloss r3 |
| Total | 18 | 16 | |

| Cases came from | Control | Candidate | Measured the app |
|---|---|---|---|
| The app's fixture | 6 | 9 | 8 of 15; 8 of 8 not stopped by `KeyError: 'id'` or the regrade crash |
| Invented | 14 | 10 | 0 of 24 |
| None written | 1 | 2 | 0 of 3 |

| Read README.md | Recipes | Holds | Glossator |
|---|---|---|---|
| Control | 1 of 7 | 0 of 7 | 2 of 7 |
| Candidate | 1 of 7 | 0 of 7 | 1 of 7 |

The arms' measures gap (3 vs 5) is all recipes: fixture use 1 of 7 vs 4 of 7 (Fisher p = 0.27). Neither llms.txt says
a word about cases, so the gap is how the agents happened to explore. The 22 lines llms.txt gained (`NEEDS A SHIPPED
EXAMPLE`, `KINDS`) are host reference material.

Fairness:

- **Recipes, glossator: fair.** Every agent that ran the fixture through a working runner passed (8 of 8).
- **Holds: fair, given the README.** The defect is invisible from the code: `GRACE_DAYS = 7` carries the comment "Days a
  hold stays on the shelf after its pickup date before the desk is told". A competent human writing cases from
  `service.py` alone could side with the constant. The 4-line README names `tests/fixtures/requests.jsonl`. No holds
  agent read it (0 of 14). Two candidate agents tried `tests/` and were told `no such file` (H).

**The single change:** put the source of cases in llms.txt, before the example. Every agent read llms.txt first (42 of
42); 5 of 42 read a README. Proposed text: "Your cases come from the app first: look in its README and `tests/` for
fixtures (`*.jsonl`) and pass that file as `cases`. Write expected values from what the app is documented to do, never
by running it or reading its constants. One case line, whole: `{...}`." Tied to trials: the 24 invented-case trials
measured nothing; the 8 fixture trials with a working runner all did. This was the baseline's change 3, and it did not
land. Second: pass `id` to the runner (the baseline's change 2): 10 glossator trials.

## 5. Harness causes, both arms

| # | Cause | Trials (ctl / cand) | Effect | Fix |
|---|---|---|---|---|
| H1 | Wall cap 450 s; glossator turns took 12 to 38 s | 3 / 4, all glossator | 7 read failures; ctl gloss r5 lost a working eval's report | Cap turns and dollars; wall cap 15 min as designed, with the cell budget (`CELL_BUDGET_S` 600) raised to hold it and the regrade |
| H2 | No warning before the wall | 7 | Agents mid-debug never sent `done` | Say "N turns / M s left; send done" at 80% |
| H3 | `read` of a directory answers `<dir>: no such file` | 3 / 4 | Consequential in cand holds r1, holds r5, gloss r6: each concluded there was no fixture | "`tests` is a directory: `ls tests`" |
| H4 | Regrade cannot import the app for an eval file in a subdirectory: `rerun.py` chdirs to the file's directory and puts it first on `sys.path`; the door keeps the app root | 0 / 2 | cand gloss r3 lost all three checks with a right eval and report | Run the regrade with the app root as cwd and on `sys.path`, as the door does |
| H5 | `ls -la` refused: `ls: -la is no directory in the app or the installed package` | 2 / 5 | 1 turn each | Accept and ignore flags, or say "ls takes paths, no flags" |
| H6 | `ls a b` lists `a` only, silently | 1 / 0 | ctl recipes r1 never saw `tests/` | List each path, or refuse the second |
| H7 | With no report, the regrade grades the last door run, a debug probe | 1 / 2 (ctl gloss r6; cand gloss r2, r5) | No check changed here (all were stuck on `KeyError`); wrong in general | Grade the eval file that declares the app's runner, or the agent's last written eval |
| H8 | `mkdir`, empty command refused | 1 / 1 | 1 turn each | Name `write` as the way to make a directory |
| H9 | Cause classes: wall-stop measures failures marked "harness" though 5 of 6 were stuck on `KeyError`; cand gloss r3's regrade `Traceback` marked "cannot tell" | 7 wall-stops, 1 crash | Harness counts in the console are wrong both ways | Class a regrade traceback as harness; class a wall-stop as harness only when its last door run scored |
| H10 | Stuck detector ignores exit-0 door runs where every case raised | 10 trials | The console names `surface.md: no such file` or a read past the end while the stuck point was `KeyError: 'id'` | Count a door run with 0 scored as a failed turn (baseline H9, still open) |
| H11 | Spend line sums priced turns only | 3 / 4 cut-off calls | Console `$0.22` / `$0.18`; charged bound `$0.32` / `$0.30`. The candidate looks cheaper because it had 4 wall-stops | Print the charged bound beside it |

Optional noise reducer: refuse a `done` whose `eval_file` does not exist (cand recipes r2, holds r7; ctl gloss r4).

## 6. Wrong numbers and misleading lines printed to an agent

| # | Printed | Trial, turn | Truth | Control | Candidate |
|---|---|---|---|---|---|
| W1 | `run: run passed 12 of 12 trials on 4 cases.` | ctl holds r5 t13 | `grade` mean 0.75; 3 trials graded False | yes | fixed |
| W2 | `runner: runner passed 15 of 15 trials on 5 cases.` | ctl holds r7 t14 | `grader` mean 0.6 | yes | fixed |
| W3 | `Cases that failed a trial: case 1, case 4, case 9, case 10.` then `case 0 (repeat 1)`, `case 3`, `case 8`, `case 9` | ctl recipes r2 t12, holds r1 t11, holds r4 t6; cand recipes r3 t9, r4 t13, r5 t6, r6 t7, r7 t13, holds r3 t8 | Same cases, numbered from 1 in one line and from 0 in the next | yes | remains |
| W4 | `cases: runner ran 4 of 4 cases; no grader here says pass or fail.` | cand gloss r2 t14, t29 | Grader declared `-> bool`; all 4 cases raised. `verdict._holds_a_check` reads the measure catalog, which is empty when nothing scored | no | new |
| W5 | `the candidate raised KeyError: 'id'` | 14 of 14 glossator trials | The case file holds `id`; the package withheld it. Nothing printed says so | yes | remains |
| W6 | Field grader `headword` printed as `Accuracy`, with `per label:` precision and recall per headword and a `confusion (expected → predicted)` table | 5 ctl, 6 cand glossator trials | A free-text field match, not a classifier. The agent must report "Accuracy" for a grader it named `headword` | yes | remains |
| W7 | `Accuracy: 0%, 0 of 12 trials` beside `not shown, carried by no result: 1 measure (slug)` | every 0-scored glossator run, both arms | Nothing was graded. One measure reads 0%, the other is absent | yes | remains |
| W8 | exit 0 ("0 Every run completed") with `0 scored, 12 failed by the candidate` | same runs | Every case raised | yes | remains |
| W9 | llms.txt "A FIRST EVAL NEEDS 2 NAMES ... Model" | ctl gloss r3 t7 | A non-model app's eval needs `Eval` only | yes | remains |

One wrong-number class is fixed (W1, W2). One is new (W4). W3, W5 to W9 remain.

## 7. What this run can and cannot conclude

| Check | Control | Candidate | Fisher p |
|---|---|---|---|
| All three | 2 of 21 (0.03 to 0.29) | 5 of 21 (0.11 to 0.45) | 0.41 |
| Runs | 17 of 21 (0.60 to 0.92) | 15 of 21 (0.50 to 0.86) | 0.72 |
| Measures | 3 of 21 (0.05 to 0.35) | 5 of 21 (0.11 to 0.45) | 0.70 |
| Read | 15 of 21 | 14 of 21 | 1.00 |

Intervals are 95% Wilson. The console's "docs@candidate passed more often" counts all three checks.

It shows that treating an unannotated bool grader as a check removes the wrong headline in an agent's hands (2
trials to 0), and that none of the candidate's new refusals was exercised. It cannot show a difference between the
arms on any check (every p at least 0.41, with 7 wall-stops and 1 regrade crash blurring them), nor any effect of the
docs, since llms.txt changed only in host reference sections. What it does show is a bottleneck both arms share:
invented cases (24 trials, 0 measured) and `KeyError: 'id'` (14 of 14 glossator trials).

Next run worth paying for, after free steps:

| Step | Runs | Arms | Expected cost | What it answers |
|---|---|---|---|---|
| 0 | none | none | $0 | Fix H1 to H4 and H9 to H11. Re-run the regrade over these 42 stored trials (cand gloss r3 becomes all three). Seed one offline test per new refusal (async grader, `-> None`, `-> bool \| float`, missing field, runner signature, returns nothing), since agents did not reach them |
| 1 | 78 | candidate vs candidate plus the two unlanded changes (llms.txt case source and one case line; `id` passed to the runner), 3 apps x 13 repeats a side, cheap tier | about $1.15 (charged mean $0.0146 a trial here); bound $8.58 at the $0.11 cap | 39 a side detects measures rising from 20% to 50% (two-sided 0.05, power 0.8), the size the fixture effect suggests (0 of 24 vs 8 of 8) |

A control-vs-candidate rerun at this size is not worth paying for: its docs do not differ where agents fail.
