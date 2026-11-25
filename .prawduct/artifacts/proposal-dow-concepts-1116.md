# Proposal: dungeons-of-wagons as the test of the package's concepts

Drafted 2026-11-16 by a read-only planner; nothing was run or changed. Not started: for the owner to accept, cut or redirect.

## Where the app is

- `/home/vscode/dungeons-of-wagons` holds only an empty `.venv` and `web/node_modules`. The real checkout is `/home/vscode/work/dow/dungeons-of-wagons`, branch `evals-ladder`, 6 commits ahead of `main`, last commit 2026-11-06.
- It pins the evals package at `27215529b`, 1,177 commits behind `evals/main`.
- It imports `Eval`, `Model`, `Judge`, `Example`, `grader`: the quick path only (rungs 0 to 3).
- Four evals: `router.toml` (rung 0); `router.py` (`router-real`, 24 of 30 pass); `forge.py` (5 cases, a judge checked against no one, `forge-to-label.csv` empty); `router_prompts.py` (router-v1 against router-v2, exits 4: the safety guardrail is undecided on 5 safety cases).
- Spend so far $0.039. Nothing has been decided with these evals.
- Rung 4 (the game master) is blocked: `Model.generate` takes one system and one user message.

## Concepts, and the experiment that could break each

| # | Concept | Experiment | Counts against | $ | Owner min |
|---|---|---|---|---|---|
| C1 | An author's mistake is refused, never scored as the app failing | In copies of `router.py`: grader `["labl"]`, a two-argument runner, a function grader with no `-> bool` | Any prints a pass rate | 0 | 0 |
| C2 | A case missed on every repeat points at the case before the model | Flip 3 labels in `router-cases.jsonl`, listed and sealed before the run | The flipped cases are not among the top misses | 0 | 0 |
| C3 | Two identical arms do not separate beyond the stated level | `router-v2` against itself: 30 cases, k=3, 100 fresh replicates on gpt-5-nano | More than 10 of 100 separate on any measure, cost included | 3.20, shared with C4, C5 | 0 |
| C4 | Pairing on cases narrows the interval | Same runs: half-width paired against unpaired | Median narrowing under 20% | 0 | 0 |
| C5 | Adding cases after looking costs error unless planned | Same replicates grown to 120 cases; look at 30, 60, 90, 120; stop at the first separation | Peeking rate within 3 points of the single-look rate | 0.01 | 0 |
| C6 | A guardrail is decided on its own; undecided is never held | 30 new safety and near-safety messages; arms `router-v2`, `router-v2` again, a planted breach | The breach does not read `breached`, or the identical arm does | 0.10 | 10 |
| C7 | A margin set before reading is the only route to "no worse" | The forge decision, margin written first | No margin the owner accepts can be met at 60 cases or fewer | in decision | 2 |
| C8 | A result says what it cannot decide and how many more cases would settle it | Record the stated N, run exactly N more, planned | It settles in fewer than half of those readings | in decision | 0 |
| C9 | A judge is trusted only as far as a blind read measures it; a model's read ranks below a person's | The forge judge and a planted lenient one; the best model reads all outputs; the owner reads 20 blind | The lenient judge is not `not_met` against both readers, or the two readers differ | about 3 | 20 |
| C10 | Hard-negative examples are a second check, never pooled with labels | The existing `on_theme` pair plus two for `fun` | They always agree with labels (redundant) or miss the lenient judge (useless) | 0.01 | 0 |
| C11 | Results from two answering models are never pooled | A planted arm alternating two models by case; count real splits across every run | The planted arm gets an arm-level rate anywhere; or zero real splits | 0.01 | 0 |

Already observed on rung 2 (2026-11-16), bearing on C9: the owner and the model reader both showed the judge wrong, and disagreed with each other on 13 questions across 10 of 20 replies. See the program file.

## The decision

Move the encounter forge from `openai/gpt-6-luna` to `deepseek/deepseek-v4-flash`? `dow.toml` states the rule: "the cheapest that does the job".

- Question and margin, written before any run: DeepSeek is no worse than luna by more than 10 points on the judge's `fun` dimension.
- Guardrail: `on_target` as `grader(guard=0.05)`.
- Cases: 40 forge cases split by party size and difficulty, briefs from `content/modules/silver-hollow.yaml`, sealed before the run; k=2.
- Exercises C1, C4, C7 to C11, and cost.

The decision the app needs most is the game master's own model. It cannot be reached without building, so it is reported as a finding, not run.

## Order of work

| Step | Work | Done when | If the concept fails |
|---|---|---|---|
| 1 | Re-pin DoW after Q1 and O1+O3 merge; replay `tests/test_evals_ladder.py`; run C1, C2 | All four rungs replay, first-line changes listed; all three planted mistakes exit non-zero naming the mistake | Recordings that no longer replay are a cost every adopter pays: report it first |
| 2 | C3 to C5 | A table of false-separation rates, single look against peeking, with binomial intervals; the half-widths | C5 fails: drop the looks caveat and `declared_look`. C4 fails: cut the pairing lesson from rung 3a |
| 3 | C6 | The breach reads `breached`, the identical arm `held`, at a stated size | Still undecided at 60 safety cases: rung 3c teaches a pass-rate floor instead |
| 4 | The decision's two arms with both judges; C9, C10 | Per dimension, agreement with bounds, both judges against both readers | Readers disagree: `reviewer=` leaves the quick path. Examples redundant: cut from rung 2 |
| 5 | Make the decision; C7, C8, C11 | The owner records a call from the code-only first screen | He had to read below the four slots: #1194's cut is off. He decided from them: the cut goes ahead |
| 6 | One table in #1177: concept, result, kept or cut | Each cut merged, measured in source lines | None |

Owner time 32 minutes. Spend about $6.50, inside the $25 campaign budget in the app's `docs/product-brief.md`.

## Not built

No multi-message recording `Model` for the game master; no planned-looks parameter on `Eval`; no shared adapter module (the 22-line `RecordedClient` is copied three times, itself a finding); no rung 5, no host, no new public name. Each gap is reported with its count.

## Risks

| Risk | Guard |
|---|---|
| The toy flatters the package | Known answers (A/A, planted breach, lenient judge, flipped labels) decided by arithmetic; planted lists committed by hash before any run |
| Cases tuned to outputs already seen | Forge cases and the margin sealed before the first run |
| A cheap classifier hides judge problems | C9 runs on the forge, which is generative |
| Every reading says "cannot decide" | C8 turns that into a measured outcome |
| A model's read passes for a person's | C9 reads both; the print names who made each judgment |

## Not determined by reading

- Whether DoW's recordings replay after the fingerprint and recording-key changes.
- Whether the quick path prints "about N more cases" (the sizing package is unmerged).
- Whether a blind read handles more than two arms.
- Whether OpenRouter reports different served models for DeepSeek's providers.
- Whether `reviewer=` accepts an Anthropic model when the app has only an OpenRouter key.
- Whether `openai/gpt-6-luna` and `openai/gpt-5-nano` are still served under those ids.
