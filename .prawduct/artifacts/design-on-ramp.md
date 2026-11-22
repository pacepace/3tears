# Design: the on-ramp (track T6)

Status: design, 2026-11-15. Read against `evals/bp-v` @8e2709c14 (main plus package V). Nothing here is built.
Paths are under `packages/evals/` unless they start with `/` or `.prawduct/`. Line counts are `wc -l` unless they
say "non-blank". Every error message quoted below came from a probe run in a scratch directory against bp-v's
source, with no API key set.

T6's done-when: a product keeps `Eval`, swaps the store and pricer, and stops there, in under 100 lines; a newcomer
path under 1,500 lines; an agent with no prior knowledge adds an eval to an unfamiliar app from the docs and error
messages alone, timed and recorded.

## 0. Today, measured

### 0.1 The ladder, per rung

| Rung | Section (ladder.md) | Lines to read | Example, non-blank lines | Names typed (cumulative) |
|---|---|---|---|---|
| 0 | README.md:10-55, building-an-inventory.md:1-120 | 46 + 120 | `triage.toml` 14 | 0 |
| 1 | :31-111 | 81 | `triage.py` 21 | 2 (`Eval`, `Model`) |
| 2 | :112-220 | 109 | `replies.py` 38 | 4 (+`Judge`, `Example`) |
| 3a-3f | :221-487 | 267 (74, 29, 58, 21, 63, 22) | 3 to 27 per file, 105 over six files | 5 (+`grader`) |
| 4a, 4b | :488-591 | 104 | `agent.py` 57, `serve.py` 13 | 8 (+`World`, `as_host`, `mount_fastmcp`) |
| 5 | :592-723 | 132 | `writer.py` 33 | 11 (+`AnalystSkill`, `Writer`, `WriterEval`) |

The rungs are short and good. The newcomer path around them is not: README 164 + building-an-inventory 570 +
ladder 738 + concepts 709 = **2,181 lines**, and the README's doc list (README.md:106-139) names 14 pages with no
order beyond "Climb / Go deeper / Background". `concepts.md` is 50 lines of six concepts (:8-57) and 580 lines of
glossary (:116-695).

### 0.2 The cliff

Past rung 3 the next documented step for a product is a host. What the courier host (the smallest reference host,
`tests/fixtures/courierhost/__init__.py`) writes: 565 lines (487 non-blank), **46 imported names from six roots**, six
parts (measures, sweepables and levers, world, `HostProfile`, a `CandidateKind` with `prepare`/`invoke`, `EvalHost`
with six fields), plus a `LaunchHost` with `launch` and `resolve_subject` to be driven from the command line. A store
of its own is the `DocumentStore` port: 8 methods (`schema/store_port.py:385-640`) proven by 43 conformance cases
(`testing/store_conformance.py`). `llms.txt:81-89` lists 18 ports a host may implement. `adopting-a-host.md` is 2,300
lines.

The step between is half-built and untaught. "Keep your runs and read them back" (`serving.md:24-89`) needs five
names from five roots (`Eval`, `ops.scope_history`, `quick.as_host`, `run.run_blocking`,
`storage.SqliteDocumentStore`) and the host's `blocking_executor`, the first time a newcomer meets an executor.

### 0.3 What is already possible, by the code

| Need | Today | Status |
|---|---|---|
| Persist results somewhere real | Every run lands in `.evals/evals.sqlite` (contract.py:17-19); `Eval.run(store=)` takes a file, `":memory:"` or any `DocumentStore` (contract.py:415-426) | Possible; a shared database means writing the 8-method port |
| Real prices, product's own client | Return `Answer(value, model=, input_tokens=, output_tokens=, cost_usd=)` (quick/answer.py:24-44) | Possible; taught only on cost-and-budgets.md:507, "off the ladder" in llms.txt:79 |
| Real prices, `Model` | The default pricer reads Anthropic's published rates, then OpenRouter's catalogue (cost-and-budgets.md:32-53). The production host uses exactly that default (`/opt/venv/discodon/evals/completion_clients.py:56`) | Possible. A negotiated rate needs a two-method `Pricer` class whose `price` half needs `usage_counts`, which lives in `providers.wire`, not a public root (the repo's own pricers import it: tests/test_providers_pricing.py:30) |
| CI gate | `python -m threetears.evals FILE` exits 4 on a regression, breach or guardrail not held (files.py:215-226, serving.md:18-22) | Possible; rung 3d's `ci.py` (9 non-blank lines) repeats what the door does |
| Custom measure | A function grader with a docstring (rung 1's `escalates`); `grader(fn, margin=, guard=, range=)` (quick/graders.py:109-159) | Possible and taught |
| Judge with a blind read | Rung 2 (`Judge(labels=)`, `blind_read.html`) | Possible and taught |
| Compare over time | `scope_history` tests each run against the run before (serving.md:79-84) | Possible, five roots deep |
| Campaign with two arms | A runner mapping plus `question=` (rung 3a) | Possible and taught |
| App that cannot be a function | `Log("outputs.jsonl")` (quick/logged.py) | Built, provisional, taught nowhere: no page, example or host uses it |

So the middle rung is mostly **teaching**. Three things need building, each a parameter or a behaviour, none a new
name (§1.2).

## 1. The middle rung: rung 4, "in your product"

### 1.1 Shape

One rung, between today's rung 3 (a comparison) and the agent. It absorbs rung 3d (CI) and today's 4b (serve), and
teaches `Answer` and `Log`. The ladder becomes:

| Rung | Idea | Example |
|---|---|---|
| 0 | No Python | `1-triage/triage.toml` |
| 1 | Your own function | `1-triage/triage.py` |
| 2 | A judge, checked against a person | `2-replies/` |
| 3 | Two versions compared: margin, guardrail | `3-invoices/` (3a-3c; the grid and the writer door move out, §3.3) |
| **4** | **In your product: its store, its prices, CI, history, a log** | **`4-quests/` (new)** |
| 5 | An agent, graded on the world it leaves | `5-booking/` (was 4) |
| 6 | Evaluate the evaluator | `6-writer/` (was 5) |

The renumbering rides the re-root's single cut-over (#1199), which rewrites every doc import anyway, so paths are
taught once.

### 1.2 The steps, in the order a product team needs them

| # | Step | What the user types | Must know | Replaces | Cost to them |
|---|---|---|---|---|---|
| 4.1 | Keep runs where the team reads them | Nothing in Python. In CI, keep `.evals/` between runs (cache keyed on the default branch) | Runs land in `.evals/evals.sqlite`; the scope is the eval's name | `serving.md`'s 40-line script | 4 lines of CI config |
| 4.2 | Its own client's prices | `return Answer(reply.text, model=reply.model, input_tokens=..., output_tokens=..., cost_usd=reply.cost)` | Missing is unknown, never $0 | Spend reported as 0 or unknown | 2 lines |
| 4.2' | `Model` at a negotiated rate | **NEW**: `Model(id, pricer=ModelRates(...))`: a rate card is a pricer, guaranteed, the engine prices each reply from its usage | The five per-million rates | A two-method `Pricer` class importing a non-public reader (about 20 lines) | 2 lines |
| 4.3 | Gate the merge | `python -m threetears.evals evals.py` as the CI step; upload `.evals/reports/` | Exit 4 is a failed gate; 1 a run that did not finish; 2 a refusal | `ci.py` | 8 lines of CI config |
| 4.4 | Measure what the product cares about | A `def` with a docstring returning a bool (taught at rung 1) | Write the grade as what you want more of | (nothing; `@measure` is removed, §2) | 4 lines a measure |
| 4.5 | A judge, read blind | Rung 2, unchanged | — | — | — |
| 4.6 | Compare over time | **NEW**: nothing to type. A run whose store holds an earlier run of the same scope prints one line below the first screen: "since the run of 2026-11-14: Accuracy regressed (over 30 cases both ran)" | A step is tested against the run before, over shared cases | The five-root `scope_history` script | 0 lines |
| 4.7 | Two arms, one question | Rung 3, unchanged: `runner={"current": ..., "new": ...}, question="..."` | — | — | — |
| 4.8 | The app is not a function | `runner=Log("quests.jsonl")`; the app (any language) appends `{"id", "output", "cost_usd", "model"}` lines | The k-th repeat is the k-th output logged; a missing one is excluded `not_logged`, never failed | Hand-written lookups that replay one output k times (a wrong k) | 1 line, plus the app's write |
| 4.9 | Share with the team | `as_host([quests], store=".evals/evals.sqlite")` and `mount_fastmcp` (today's 4b, moved) | — | — | 13 lines |

A team that needs its own database goes to the host branch for one thing only: the 8-method `DocumentStore` and its
kit, then `await ev.run(store=TheirStore())` in a six-line `main()`. That is the edge of rung 4, stated as the edge.

### 1.3 The example, `examples/4-quests/`, under 100 lines

The domain is the approved free toy, dungeons-of-wagons: a wagon-train game whose scout writes the party's next
quest. It is unlike every domain on the ladder (§5.2). The game's engine is JavaScript in the real app, which is why
step 4.8 is real here rather than staged.

```python
# quests.py: the product's own eval, rung 4
from decimal import Decimal as D

from threetears.evals import Eval, Model
from threetears.evals.providers import ModelRates  # the re-root decides this path; §7 waits for it
from game import PROMPT, PROMPT_TERSE, landmarks, write_quest  # the product's code, imported as it ships

RATES = ModelRates(base_input=D("0.10"), cache_write_5m=D("0.10"), cache_write_1h=D("0.10"),
                   cache_read=D("0.01"), output=D("0.40"))  # what the studio pays: the contract's rates
scout = Model("deepseek/deepseek-v4-flash", pricer=RATES, recording="recordings/quests.jsonl")  # NEW: pricer=rates


def on_the_map(case: dict, quest: str) -> bool:
    """Every landmark the quest names is on the party's map."""
    return all(place in case["map"] for place in landmarks(quest))


def asks_one_thing(case: dict, quest: str) -> bool:
    """The quest asks the party for one task, not a list."""
    return quest.count("?") <= 1


quests = Eval(
    "situations.jsonl",
    runner={"current": write_quest(scout, PROMPT), "terse": write_quest(scout, PROMPT_TERSE)},
    question="Does the terse prompt keep quests on the map as often as the current one?",
    margin=0.05,
    grader=[on_the_map, asks_one_thing],
    max_cost_usd=0.50,
)
```

```python
# logged.py: the same graders over what the JavaScript game logged in play (rung 4.8)
from threetears.evals import Eval, Log
from quests import asks_one_thing, on_the_map

played = Eval("situations.jsonl", runner=Log("played.jsonl"), grader=[on_the_map, asks_one_thing], k=3)
```

```yaml
# .github/workflows/evals.yml (shown on the page, not shipped as a workflow)
- uses: actions/cache@v4
  with: {path: packages/game/.evals, key: "evals-${{ github.ref_name }}", restore-keys: evals-main}
- run: python -m threetears.evals quests.py   # exit 4 fails the merge on a regression or a guardrail not held
- uses: actions/upload-artifact@v4
  with: {name: eval-report, path: packages/game/.evals/reports/}
```

Budget: `quests.py` about 30 non-blank, `logged.py` 4, `game.py` (the product stand-in, with `landmarks`) about 30,
the CI block 7: about 71 lines. `played.jsonl` is the game's own log, so `logged.py` replays offline by construction;
`quests.py` replays a recording made once on a cheap model (under $0.05).

## 2. `@measure` and `Log`

### 2.1 `@measure` (`quick.measure`, `quick.Measure`): remove

It declares a `MetricDescriptor` on the function computing it (quick/measures.py:60-122). Its parameters are host
vocabulary: `family`, `transferability_class`, `attribution_scope`, `data_type`, and every other descriptor field by
keyword. For a non-expert on the quick path it is the wrong level: a function grader already is a measure, named by
its `__name__`, described by its docstring, declared by `grader(margin=, guard=, range=)`. Two ways to declare a
quick-path measure fail criterion 3 (one way). Nothing in production or a shipped example uses it (criterion 2): the
production host does not import it, and its one user is the toy host fixture (tests/fixtures/toyhost/kind.py:70-134),
while the courier host writes `MetricDescriptor`s by hand. The one capability it alone gives the quick path, a
lower-is-better number, has no user.

Ruling: remove `measure` and `Measure` from `quick`, delete the seven `Measure` branches in `quick/one_call.py`, give
the toy host hand-written descriptors as the courier has, record both in `MIGRATIONS.toml` with their replacements
(`grader(...)` on the quick path; `MetricDescriptor` in `MeasureRegistry` on a host). If a real user needs a
lower-is-better grade, it arrives as a `grader()` parameter, never as a second declaration route.

### 2.2 `Log`: keep, teach at rung 4.8

It is the right way, for a reason that is about correctness, not convenience. The obvious hand-written alternative,
`runner=lambda case: outputs[case["id"]]`, serves one output to all k repeats, so `k=3` reports three trials that were
one: a false precision the reader cannot see. `Log` serves the k-th logged output to the k-th repeat and excludes a
missing repeat as `not_logged`, counted and named, never failed (quick/logged.py:1-15, 120-139). It also carries
spend, and a mapping of two logs is an A/B over production outputs. Its format is a JSONL line, so an app in any
language can write it: that is the case a function runner cannot reach.

Smallest real example: `examples/4-quests/logged.py` above, 4 lines, over a log the game wrote. That satisfies
criteria 2 and 4; its meaning can then be promised (criterion 5) and `provisional` is dropped when rung 4 lands.

`Answer` is taught at step 4.2 by the same move, which closes the third untaught typed name.

## 3. The docs trunk

### 3.1 The trunk, in reading order

| # | Page | One job | Budget | Branches it links out to |
|---|---|---|---|---|
| 1 | README.md | What it is, install, rung 0, and the map: the trunk, then the branches | 120 | every branch, one line each |
| 2 | ladder.md (rungs 1-4) | Climb from your own function to your product | 650: intro 25, r1 110, r2 140, r3 220, r4 155 | judges, statistics, serving, host |
| 3 | reading-what-comes-back.md (split from reading-reports.md:8-134 plus serving.md:220-231) | The first screen, the exit codes, the ten errors and what to do (§3.4) | 150 | reading reports (full) |
| 4 | concepts.md (the six concepts only, :8-57 today) | The six words every eval is made of | 80 | glossary |
| — | llms.txt | The agent's entry: the same trunk, compressed, with installed paths | 150 | the same |

Human trunk: about 1,000 lines. An agent reading `llms.txt` and then the trunk: about 1,150. Under 1,500 with room
for the first screen S5b renders.

### 3.2 Branches (read when the trunk sends you)

| Branch | Pages |
|---|---|
| Designing cases | building-an-inventory.md (whole; its rung-0 ending moves to the README) |
| Statistics | measuring-soundly.md, plus the grid (today's 3e) |
| Judges | judges-and-calibration.md |
| Agents | rung 5 in its own page with evaluating-agents.md |
| The report writer | today's 3f door and rung 6, beside the analyst sections of reading-reports.md |
| Serving and operations | serving.md, cost-and-budgets.md |
| A host of your own | adopting-a-host.md, the courier host, then the toy host |
| Reference | reference.md, surface.md, the glossary (moved out of concepts.md), design.md, open-problems.md |

### 3.3 What moves out of the trunk

| Moves | From | To |
|---|---|---|
| The 15-row "For experts" table (`create_campaign`, `campaign_record`, `EvalHost` on the front page) | README.md:140-164 | surface.md |
| "What is different" | README.md:93-104 | design.md |
| The 14-page doc list | README.md:106-139 | the trunk/branch map, 20 lines |
| The glossary | concepts.md:116-695 | reference branch |
| Building an inventory | the README's first "Climb" link | designing-cases branch |
| Rungs 3e (grid), 3f (writer door) | ladder.md:403-487 | statistics, report writer |
| `ci.py` | examples/3-invoices | deleted; the file door is the CI step |
| Host ports, "needs a shipped example", 87 actions, kinds | llms.txt:81-136 | surface.md (linked, not copied) |

**The trunk ships in the wheel.** Today only `llms.txt` does (pyproject.toml:52-54); the README itself says its links
do not resolve on PyPI (README.md:107-109), and `llms.txt` points at GitHub URLs an offline agent cannot open. The
four trunk pages go into the wheel under `threetears/evals/docs/`, and `llms.txt` names them by installed path. About
1,000 lines of Markdown; no code.

### 3.4 The errors a newcomer hits

Probed in a scratch app (`cases.jsonl` with `id`, `text`, `label`), no key set.

| # | Mistake | What it says today | Tells them what to do? | Change |
|---|---|---|---|---|
| 1 | Runs `python -m threetears.evals --help` first | Lists `run ls report bundle spend gate frontier judge-temperature`, each needing `--host MODULE:FACTORY` (cli.py:276-279); the file door is not mentioned | **No: it points away from the door** | The help's first line names `python -m threetears.evals FILE.py` and rung 1 |
| 2 | No key, no recording | `no key for openrouter: pass api_key= or set OPENROUTER_API_KEY` (providers/model.py:1207) | Partly | Add "or pass recording= and run once with the key" when no recording is named (it is said only when one is) |
| 3 | Cases file path wrong | `the cases file missing.jsonl is not there (in <dir>)` | Yes | — |
| 4 | Eval inside a function | `defines no Eval at module level; assign one (my_eval = Eval(...))` (files.py:163) | Yes | — |
| 5 | `lambda` grader | `a grader's __name__ names its measure ... write each grader as a def` | Yes | — |
| 6 | Field grader names a key the answer lacks | Scored: `answered no label it can be graded on: {"category":"greet"}`, 0%, **exit 0** (result.py:620) | **No: reads as the app failing** | When no answer carries the field, refuse the reading as the author's: "no answer has a field 'label'; answers carry: category" (exit 2) |
| 7 | Runner takes the wrong arguments | Scored: `the candidate raised TypeError: classify() takes 0 positional arguments`, 0%, **exit 0** | **No: a wiring bug read as a product failure** | Refuse at `Eval(...)` when the runner cannot take one positional argument (contract.py:187-193 checks only `callable`) |
| 8 | Grader raises | Run fails, exit 1, names scorer and exception; the same line printed 6 + 2 times for 6 trials | Yes, noisily | One line with its count (presentation: after S5b, frozen until then) |
| 9 | Two arms, no name | `name this eval (name="invoices"): ...` | Yes, with a confusing example name | Suggest the cases file's stem or `name="..."` |
| 10 | A recording that no longer covers a call | `RecordingRefused` naming the prompt and a remedy (testing/recorded_completions.py:720-740) | Yes | — |

Also good today: margin without question, a bare Claude id, an expression grader without a world, a grader returning a
string. Rows 6 and 7 are the ones that lie: both exit 0 with a measured 0%, and an agent (or a person) will read "the
app is broken" and stop.

### 3.5 What `llms.txt` must say

Under 150 lines, generated, every line about adding an eval:

1. What it is, and the three slots (keep today's :7-13).
2. **Wrapping an unfamiliar app**: the runner is any function of one case; a class method or HTTP call is a two-line
   adapter; an app that cannot be called gets `Log`. Today nothing says how to find the app's entry point.
3. **Where cases come from**: the app's own fixtures, tests or examples, one JSON object per line; `id`, `stratum`,
   `world` are reserved. Today `llms.txt` never says how to write a case.
4. Graders: field names, `def` with a docstring returning a bool, `Judge`. The first example whole (keep).
5. Running: the file door, recordings and keys, and that a deterministic app runs free.
6. Reading: the first screen and exit codes (keep :57-70), then the ten errors of §3.4 with the fix for each.
7. The trunk pages by installed path; branches by name. Host and actions: one line pointing at `surface.md`.

## 4. The agent test

### 4.1 Built on the package

The test is an eval, so it is written as one: rung 5's shape, an agent graded on the world it leaves.

```python
onramp = Eval(
    "apps.jsonl",                                   # one case per unfamiliar app; world = its sandbox
    runner={"docs@control": World(Sandbox, tools=[read, write, run])(adopter),   # today's wheel
            "docs@candidate": ...},                 # the wheel with a doc or error fix
    question="Does the candidate docs build let an agent add a working eval more often?",
    grader=[eval_runs, measures_the_app, read_right],
    k=3, max_cost_usd=6.0,
)
```

The arms are docs builds. Every stuck point fixed becomes a candidate arm, and the package's own comparison says
whether the fix helped, with the same honesty it asks of everyone else. Model tier is a second lever when wanted.

### 4.2 The unfamiliar apps

Three small apps under `packages/evals/onramp/apps/`, written for this and for nothing else: not examples, never named
in the docs, so the docs cannot have been tuned to them. Each is about 120 lines, deterministic (no key, no cost, so
every failure is the docs' or the agent's), with a **planted defect** on a known slice of inputs and a **fixed twin**:

| App | Shape | Planted defect |
|---|---|---|
| `recipes` | A class with a `scale(recipe, servings)` method returning a dict | Unit conversion wrong for spoons |
| `holds` | A small HTTP-less service module: `handle(request: dict) -> Reply` | Holds past their date are not flagged |
| `glossator` | A CLI that writes JSONL (cannot be a function) | Drops accents in one language |

They differ in entry-point shape on purpose: a function, a class, a CLI. The cases the agent needs are discoverable in
each app's own `examples/` or test fixtures, as in a real repository.

### 4.3 What the agent is given

The installed wheel (whose `llms.txt` and trunk pages are inside it), the app's directory, and one sentence: "Add an
eval to this app with 3tears-evals, run it, and report what it found as the JSON below." No repository, no network,
no examples directory. Tools: `read(path)` (the app and the installed package), `write(path, text)` (in the sandbox),
`run(command)` (allow-listed: `python`, `python -m threetears.evals`, `ls`, `cat`). Turn cap 40, wall cap 15 minutes.

Model tiers, cheapest first: `deepseek/deepseek-v4-flash` and `google/gemini-2.5-flash-lite` through OpenRouter for
every routine measurement; `claude-haiku-5-5` (`provider="anthropic"`, `ANTHROPIC_API_KEY`, no temperature) as the
ceiling, once per wave end.

### 4.4 Success, checked by code

| Grader | Passes when |
|---|---|
| `eval_runs` | The agent's eval file, re-run by the harness in a fresh process, exits 0 or 4 (never 2 or 3), and stores at least one finished run |
| `measures_the_app` | Re-run against the app's fixed twin, its reading moves the right way on at least one measure, and a do-nothing runner (returns `None`) fails it on at least half the cases: it beats doing nothing |
| `read_right` | The agent's final JSON (`eval_file`, `first_line`, `pass_rate`, `finished`) matches the stored run: the first line verbatim, the pass rate within half a point, `finished` true exactly when the run finished |

### 4.5 Recorded, per run

Wall time; turns; input and output tokens and cost (priced by `Model`, so on the package's own ledger); every command
run with its exit code and its first stderr line; every error message from §3.4's set that appeared, by id; every
doc page read and how many lines; the stuck point, defined as the longest span of consecutive turns whose commands
failed, with the message that opened it. Stored as each trial's output in the eval's own store, so the report is the
package's report.

### 4.6 Budget and cost

Caps, not estimates: $0.50 a run on the cheap tier, $1.50 on the ceiling, enforced by `max_cost_usd`. A routine
campaign (3 apps, k=3, two docs arms, cheap tier) is 18 runs, at most $9. The model's turns are recorded, so the
harness itself replays offline in CI for free; a docs change moves the prompts and needs a live re-measure, which is
correct, since the docs are what is under test.

### 4.7 Feedback

Each run's stuck point is filed once per theme, deduplicated, labelled `on-ramp`, with the message and the page that
did not help. A fix ships only as a candidate arm against the control, and is kept when the comparison shows it better
or, for an error message, when the stuck point it named disappears from the runs. T6's done-when is met when the
ceiling tier passes all three graders on all three apps in at least 2 of 3 repeats and the cheap tier's rate is
recorded beside it.

## 5. Joy

### 5.1 The three least joyful places

| # | Where | Evidence | Change |
|---|---|---|---|
| 1 | The cliff after rung 3 | The next step a page offers is a 2,300-line host page and a 46-name host; keeping and reading runs takes five roots and an executor (serving.md:24-89) | Rung 4 (§1): 71 lines, 3 more names, all already public |
| 2 | Mistakes that score as failures | A misnamed field and a wrong runner signature both print 0% and exit 0 (§3.4 rows 6, 7). The newcomer learns their app is broken when their eval is | Refuse both as the author's, naming the fix |
| 3 | The front door is a map of everything | README's experts table puts `campaign_record (floors_overridden, bars_overridden)` on page one; `--help` shows eight host commands and not the file door; `llms.txt` spends 56 of 136 lines on ports, actions and kinds | Trunk-only README, help names the file door, `llms.txt` about adding an eval |

### 5.2 Varied examples

Today's domains: support email triage, support email replies, invoice extraction, meeting booking, the report writer.
Four are back-office text and two share one inbox. Additions that show range and add no public name:

| Add | Domain | Shows |
|---|---|---|
| `4-quests/` | A game (dungeons-of-wagons): creative generation | Product rung, a non-Python app through `Log`, a rate card |
| A second rung-1 file, `1-triage/` sibling or its own dir | SQL from a question, graded by running the query on an in-memory SQLite | "Grade by running it": a function grader that executes, the strongest grader a non-expert can write |

Two, not three: every example is maintained through every architecture change (examples-follow-the-arch).

## 6. What the on-ramp subtracts

| Gone from a newcomer's path | Count |
|---|---|
| Public names | `quick.measure`, `quick.Measure` removed; the last three untaught typed names (`Answer`, `Log`, `measure`) resolved |
| Example files | `3-invoices/ci.py` |
| Trunk lines | 2,181 to about 1,000 (README 164 to 120; concepts 709 to 80; building-an-inventory and rungs 3e, 3f, 5 out) |
| `llms.txt` lines about anything but adding an eval | 56 to 1 |
| Concepts met before rung 5 | host, kind, port, executor, scope as a term, campaign as a term, action catalogue |
| Rung-3 steps | six to three |

Added: two parameters' worth of behaviour (`pricer=` taking a rate card, the history line) and two refusals. Each by
the admission criteria: no new public name; each names what it replaces (§1.2); each is used by a shipped example
(`4-quests`). The harness under `onramp/` is not shipped and declares no public name.

## 7. Build chunks

Work in flight to stay clear of: the judges chunks R1, B1, B2, D1, A2, C1, E1 own `quick/result.py`, `quick/blind.py`,
`quick/contract.py`; S5b renders the first screen; the re-root (#1199) moves import paths in one cut-over.

| Chunk | Scope | Owns | Size | Waits for | Done when |
|---|---|---|---|---|---|
| O0 Harness and baseline | §4 whole; one live baseline on today's wheel (the control arm) | `onramp/**` (new), `scripts/onramp-agent.sh` | about 900 | nothing | One recorded run replays offline in a test; a live cheap-tier campaign stored, every §4.5 field filled, stuck points filed |
| O1 Door and messages | §3.4 rows 1, 2, 9 | `quick/cli.py`, `quick/files.py`, `providers/model.py` (message only) | about 120 | nothing | A test per row asserts the new text names its fix |
| O2 Rate card | `Model(pricer=ModelRates(...))` | `providers/model.py`, `providers/pricing.py`, their tests | about 150 | O1 (same file, sequence) | A recorded call priced by a rate card equals `ModelRates.cost` on its usage; source names the card |
| O3 Mistakes refused | §3.4 rows 6, 7 | `quick/contract.py`, `quick/graders.py`, `quick/one_call.py`, wording at result.py:620 | about 200 | judges chunks on contract.py and result.py | Both probes exit 2 with the fix named; no shipped example changes reading |
| O4 `@measure` removed | §2.1 | `quick/measures.py`, `quick/__init__.py`, `quick/one_call.py` branches, toyhost `kind.py`, MIGRATIONS | net about -200 | rides the re-root's demotion list (one surface regeneration) | Admission test passes; toy host goldens unchanged |
| O5 History line | §1.2 step 4.6 | the render owner's file after S5b | about 200 | S5b; the statistical sign-off below | A second run of rung 1 in one store prints the step line; a fresh store prints none |
| O6 Rung 4 and renumber | §1.3, `Log` and `Answer` taught, 4b moved, `ci.py` deleted, dirs renumbered | `examples/**`, `docs/ladder.md`, `examples/README.md`, `tests/test_examples.py`, `tests/test_doc_blocks.py` | about 450 plus a recording under $0.05 | re-root, S5b, O2 | `4-quests` replays offline; the page's blocks are held by `test_doc_blocks.py` |
| O7 Trunk | §3.1-3.3; trunk pages into the wheel | `README.md`, `docs/*.md` splits, `pyproject.toml` wheel includes | moves about 1,200 lines | O6 | A test sums the trunk's lines (under 1,500) and checks every trunk page is in the built wheel |
| O8 `llms.txt` | §3.5 | `scripts/generate_surface.py` (llms part), `llms.txt`, its test | about 200 | O7 | Generated, under 150 lines, names installed paths only |
| O9 Measure again | O0's campaign with `docs@candidate` = the O1-O8 wheel | `onramp/` results only | $9 cap | O8 | §4.7's bar met, report stored, recorded on #1177 |

O0, O1 and O2 can start now. O3 and O5 wait for files others own. O4, O6, O7, O8 wait for the re-root so no path is
taught twice.

## 8. Decisions taken by principle

1. **One middle rung, no new public name.** Everything a product needs short of its own database is already public;
   it was untaught. The three builds are parameters and behaviour.
2. **`@measure` removed** (one way; no use). **`Log` kept** (the only correct k on pushed outputs).
3. **`pricer=` takes a `ModelRates`** rather than a new `rates=` parameter: one parameter, one way, a rate card being a
   pricer that guarantees its rates.
4. **No shipped Postgres store.** No outside user has asked (criterion 1); the 8-method port and its kit are the edge.
5. **The trunk ships in the wheel**, because an agent offline and a reader on PyPI otherwise have no docs.
6. **The agent test is an eval on the package**, arms are docs builds, so a doc fix is kept on a verdict.
7. **The history line informs; it does not gate.** The file door's gate stays the question's. Whether a run-to-run
   step on a single arm is a sound contrast when the code changed between runs (M1, same conditions) is the
   statistician's to confirm before O5 builds.
8. **Renumbering rides the re-root.**

## 9. Owner questions

1. **Which tier defines "an agent can"?** Recommendation: the bar is the ceiling (haiku 5.5) passing §4.4 on all three
   apps; the cheap tier's rate is reported as the trend to raise, not gated.
2. **Spend for the agent measurement.** Recommendation: $9 cap per routine campaign on the cheap tier, run at O0 and
   O9 and once per wave after; the ceiling tier at most once per wave end, about $14 at the cap.

## 10. Not verified

- dungeons-of-wagons: `/home/vscode/dungeons-of-wagons/web` holds only `node_modules`; that its engine is JavaScript
  and logs anything is assumed from the directory, not read. `4-quests` does not depend on it.
- That `World` tools may run subprocesses and take seconds, and that a 40-turn agent's turns replay through `World`'s
  recording as rung 4a's do.
- Token prices for `deepseek-v4-flash` and haiku 5.5; §4.6 states caps, not costs.
- Whether a runner that both calls `Model` and returns `Answer` counts spend twice (rung 4 avoids it).
- The re-root's target paths for `ModelRates`, `as_host`, `mount_fastmcp`.
- That no judges chunk touches `quick/graders.py`, `quick/one_call.py` or `quick/cli.py`.
- S5b's first-line wording, which `llms.txt` and the trunk quote.
- That `scope_history`'s run-to-run test is statistically sound when the code changed between runs (decision 7).
