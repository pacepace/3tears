# Concepts

**For** anyone stopped by a word in the README, the ladder or a guide. **Answers:** the six concepts every eval is
made of, the rung that teaches each word past them, and, below, the glossary of every term the host contract and the
reports use. Nothing here assumes you have built an eval before. Why the pieces are shaped this way is in
[Design](design.md#rationale).

## Six concepts

An eval is three slots, `Eval(cases, runner, grader)`. Run it, and it gives you a result you read; give it two
runners, and it is a comparison. That is the whole of it: [World](ladder.md#rung-4-an-agent-graded-on-the-world-it-leaves)
and `Log` are shapes of runner, `Model` is a client you may ignore, and `grader(...)` and `Answer` are optional
modifiers.

### Case (test case)

One input and what a good answer looks like: one email and the triage a person gave it. A case is a dict, or a line
of a `.jsonl` file, frozen once stored, which is what lets two runs over the same cases be compared. Three keys are
reserved and never handed to the runner: `id` names the case (the report calls it by that name), `stratum` names its
kind (a `lookalike`, say) so every reading is also split by kind, and `world` is an agent's starting state.

### Runner

The code under test: any function from one case to an answer, async or not. It is one **arm**, one version under
test. A runner may also be a `World`'s, which hosts your agent in a seeded state, or a `Log` of outputs your app
pushed when it cannot be wrapped as a function.

### Grader

What counts as right, in the `grader` list. A field name (`"category"`) grades the answer's field against the
case's: the first is read as a classifier, with a confusion matrix and each label's precision and recall, any other as
a pass rate. A function `(case, answer) -> bool | float` is a grader named by its `__name__`, and every grader is
higher-is-better. A string that is not a name is a [goal-state check](#goal-state-check) over a world. A grader that
raises excludes its trial: it is part of the measuring setup, not the runner.

### Judge

A model that grades an answer against a rubric when no code can: one written dimension per call, pass/fail by
default, with "can't tell" an answer of its own. A `Judge` is a grader, listed with the others. A judge is a model
too, so it is checked against people's labels before its scores are leaned on, and every judged reading carries the
[evidence tier](#evidence-tier) its checks reached.

### Comparison

A runner given as a mapping, `{"v1": ..., "v2": ...}`: each runner is an arm, every arm runs over the same cases, and
the first key is the **control** the others are tested against. Each reading gets a delta, an interval, and a
**verdict**, naming both arms: separated (`v2 better than v1`, `v2 worse than v1`), not separated (`v2 and v1 not
separated`: the cases could not tell the arms apart, which never means "no difference"), `equivalent` (shown inside a
margin you declared), or `untested`. A guardrail is
decided apart, `held`, `breached` or `undecided`, so a gain elsewhere cannot pay for it.

### Report

What a run tells you. Printed, it is quiet: the verdict, each arm's readings, the top misses, and where the runs and
the report are kept. Written, it is one page, result first ([Reading reports](reading-reports.md)). Never quieted:
errors, exclusions and their causes, truncation, unpriced spend, the self-judging warning, and how the judge stands
against people.

## Every other word is a rung

| Word | In one line | Taught at |
|---|---|---|
| k | how many times each case is played, because model answers vary | [rung 1](ladder.md#rung-1-your-own-function) |
| miss | a result the runner got wrong, with why; read them before the means | [rung 1](ladder.md#rung-1-your-own-function) |
| stratum | a case's kind, so each reading is split by kind | [rung 1](ladder.md#rung-1-your-own-function) |
| labels, κ, evidence tier | a person's ratings of judged answers, the judge's agreement with them, and how far its scores can be leaned on | [rung 2](ladder.md#rung-2-a-judge-checked-against-a-person) |
| arm, control, verdict | one version; the version the others are tested against; what the evidence supports | [rung 3](ladder.md#rung-3-two-prompts-compared) |
| margin, guardrail, gate | the difference too small to matter; a reading no arm may get worse on; the CI check over the verdicts | [rung 3](ladder.md#rung-3-two-prompts-compared) |
| world, goal check, did-nothing control | the state an agent acts on; a check of the state it left; the proof that a check can fail | [rung 4](ladder.md#rung-4-an-agent-graded-on-the-world-it-leaves) |
| store, scope, host | where runs are kept; the name they are found by; the value that serves them | [serving](serving.md#keep-your-runs-and-read-them-back) |
| campaign, aspect, floor | one question asked of the arms; where it is measured; a line a candidate is warned for crossing | [measuring soundly](measuring-soundly.md#declare-the-campaign-its-question-arms-and-aspects) |

## The picture underneath

An `Eval` builds all of this for you (`as_host`); a host of your own declares it
([Adopting the engine](adopting-a-host.md)). The glossary below defines each box.

```
  template  "triage a support ticket"
     |  holds or generates
     v
  cases     40 tickets, each with its expected queue          (frozen into every run that uses them)
     |
     |  a launch starts one run per arm
     v
  run       one arm = one variant x every case x k repeats
            e.g. "prompt v2 on model-x": 40 tickets x k=3 = 120 trials
     |
     v
  results   one per case x repeat (the trial loop calls each one a "cell")
            graded by  measures         -- numbers code computes: match, latency, cost, your graders
                  and  judged dimensions -- scores an LLM judge gives against a rubric (tone, say)
     |
     |  you put the runs you want compared into
     v
  campaign  one question asked of its arms, e.g. "should v2 replace v1?", in one scope
     |  measured on one or more
     v
  aspect    one template and the runs gathered on it, with its margin and design;
            its results pool into analysis cells: one variant measured under one rig
     |
     v
  analysis bundle   every number code computed about one aspect, frozen and fingerprinted
     |
     |  optionally, an analyst model investigates a frozen snapshot with tools
     |  and writes a report document, which code checks and an editor reviews
     v
  report    one page per campaign, HTML and Markdown, beside campaign.json and each
            aspect's bundle: the analyst's report, or "code-only" when none was written
```

Underneath all of it sits your **host**: the one value through which your app tells the engine what it can
change (levers), what it can see (measures), how to run your code (a kind), and where to store things.
An `Eval` builds one for you (`as_host`), which is why the README's first example needs none.

## Glossary

The words past the ladder, grouped by where you meet them. Code names are in backticks; the module or class named is
where the definition lives in the source.

### What you test

#### Case set
A named, versioned, frozen list of one template's cases (`CaseSet`, minted by `mint_case_set` or the
`case_set_mint` action). It is append-only: changing it mints the next version, and rewriting a stored
version is refused. A launch can name one (`case_set_name` and `case_set_version`), runs exactly its cases,
and records it on the run, so history reads a change of suite as `smoke v1` to `smoke v2`. It labels the
run's frozen case ids, never a second identity. *Example:* `smoke v2`, the three
tickets every release is checked on.

#### Template
The blueprint the cases belong to (`EvalTemplate`): what is being tested (its intent), how to score it
(goal-state checks and rubric dimensions), the axes cases vary along, and anything only its kind reads (its
spec). A template marked `universal=True` applies to every subject, whatever tools it requires. *Example:* "route a
support ticket to a queue", with the four labels in its spec. An `Eval` takes its
intent from the runner's docstring: see [what the engine reads from your
code](#what-the-engine-reads-from-your-code).

#### Variation axis
One direction along which a template's cases differ (`VariationAxis`): `enum` yields each listed value once,
`sample` draws from listed values, and `llm` asks a model to write new values. *Example:* an axis
`tone` with values `angry`, `polite`, `terse`, so generated tickets cover all three.

#### Stratum
The kind of case a case is, in your own words (`EvalTestCase.stratum`), so results can be broken down by it.
It is never shown to the thing under test. *Example:* the 12 tickets marked `lookalike` (a billing question
that reads like a bug report) get their own column in the report. An `Eval`'s cases carry it under the reserved
key `stratum`.

#### Subject
What is being measured, as your host names it, frozen at launch (`SubjectSnapshot`). *Example:* "the triage
service". In an `Eval`, the subject is just the runner's name, or a comparison's name for every arm.

#### Candidate
The thing that answers a case during a trial: your code, built for one cell by the kind's `prepare` and
called by its `invoke`. *Example:* the triage function with prompt v2 loaded.

#### Kind (candidate kind)
The adapter that tells the engine how to run one sort of subject (`CandidateKind`): `prepare` builds a
candidate, `invoke` runs it on a case and returns a `CandidateOutput`. Every template names its kind
(`candidate_kind`). Runs of different kinds are always different variants. *Example:* a `ticket_router`
kind whose `invoke` calls your production classifier. (Not to be confused with a *stratum*, which the
docs sometimes call "the kind of case", or a measure's *family*, "what kind of measure it is".)

#### Kind contract
Two optional Pydantic models a kind declares once, on the host profile (`KindContract`): its **overlays**
and its **spec**, plus which **seats** of the rig its runs fill. *Example:*
`KindContract("ticket_router", overlays=RouterOverlays, spec=RouterSpec)`.

#### Overlay
A knob a launch may turn for one kind's runs: a field of the kind's overlay model. Each field becomes a
lever named `<prefix>.<field>` (the prefix is the kind's name unless its contract sets `prefix`), is frozen onto the run, and enters the variant key. *Example:*
`ticket_router.prompt_version`, set to `"v2"` for one launch.

#### Spec (kind spec)
What a template of that kind declares beyond the engine's own fields (`EvalTemplate.kind_spec`), checked
against the kind's spec model when the template is written and frozen onto every run. It is part of the
measurement context, not the variant. *Example:* the label set `["billing", "bug", "account", "other"]`.

#### World
A stateful environment the subject acts in, declared on the profile (`WorldRegistry`) and handed to each
cell as a `WorldSession`. Seeded before the first turn and read back after the last. A classifier has none.
*Example:* for a support *agent* (not the classifier), a ticketing system whose open tickets it can close.
In an `Eval`, `World(State, tools=[...])` declares one from a pydantic state model, each case carries its starting
state under the reserved key `world`, and goal checks are graders (`examples/4-booking/agent.py`). Why a subject runs in a seeded
world at all: [The world model](evaluating-agents.md#why-a-world-the-world-model).

### What you vary, and what must hold still

#### Sweepable
Any input that can change a score, declared by your host with a reader that says how to find its value on a
run (`Sweepable`). Every sweepable has one of three roles: **lever**, **apparatus** or **label**. The engine
ships the ones every LLM product has (`SHARED_CORE`); you add your own.

#### Lever
A sweepable you deliberately change to see what it does (role `lever`). The engine always resolves two
itself, the candidate model (`model`) and the candidate kind (`candidate_kind`); your kind's overlays add
more. *Example:* `model` and `ticket_router.prompt_version`.
With no host of your own, an arm named as `k=v` pairs (`{"prompt=v2": runner}`) states one beside the model, as
`callable.prompt` (`callable-judged.prompt` on a judged run), and arms named `"model=m1, prompt=v2"` are keyed by
their level of both (`examples/3-invoices/extract.py`'s docstring crosses the prompt with a reasoning
effort).

#### Label (sweepable role)
A sweepable that identifies a run without determining its score (role `label`), so a difference in it is
reported but never treated as a rival explanation. Unrelated to a classifier's labels.

#### Variant and variant key
A **variant** is everything that would ship if this configuration won: the level of every lever a result
was measured at. Its **variant key** is a digest of those levels' content hashes
(`compute_variant_key`), so two runs that ran the same levers share a key however they were launched.
*Example:* `{model: model-x, candidate_kind: ticket_router, ticket_router.prompt_version: v2}`.

#### Arm
One variant as a contestant. A run measures exactly one arm (one candidate model); a launch naming three
models starts three runs. In a report, an arm is everything measured at one variant key, so re-running an
arm adds observations to the same arm rather than creating a new one. *Example:* the v1 arm and the v2 arm.

#### Apparatus, rig
The **rig** is the measuring setup around the thing under test; each of its inputs is an **apparatus**
sweepable (role `apparatus`). It is supposed to hold still, and when it moves, a comparison stops being
about the lever. The engine's own: the judge model, which judge each dimension was actually
scored by (`judge_dim_divergence`), the temperature each judge call was sent at, the request settings the judge was
asked for and those its calls were sent with (`judge_request_settings`, `judge_sent_settings`), the judge configs,
the simulator model and the request settings it was asked for and sent with (`simulator_request_settings`,
`simulator_sent_settings`). The per-run cost ceiling (`max_cost_usd`) is declared apparatus too, so a kind can seat
it, but it is no dimension of the rig (`no_rig_coordinate`): a cap decides when a run may stop, not what it measured,
so runs that differ only in their cap are one rig and the bisection states the difference as a plain fact.
*Example:* the v1 runs were judged by judge-a and the v2 runs by judge-b: the rig moved, so the report will not pool
them as one condition.

#### Apparatus class
One configuration of the rig as recorded on the runs (`ApparatusClass`). Two observations pool only when
their variant matches and their apparatus falls in the same class; an arm measured under two classes is
named with `(test setup <digest>)` in a report. These cells, and every composite, cost and latency, pool on
variant x rig alone. pass^k splits further: on the measurement context, the template revision and the model
that answered (see [Measurement context](#measurement-context)).

#### Apparatus settings
Rig values your host declares and a launch may set (`apparatus_settings`), so one template can be run at two
of them and compared. *Example:* who sits in an adjudicator's seat: `{"adjudicator_seat": "model:default"}`.

#### Seat
A place in the rig a kind's runs actually fill (`KindContract.seats`): a pinned role such as `judge` or
`simulator`, or one apparatus dimension. A dimension a kind does not seat does not apply to its runs, so
its blank there is not a confound. `None`, the default, holds the kind to every dimension. *Example:* the
callable kind an `Eval` runs seats none of the judge, simulator or spend ceiling; its
judged kind, `callable-judged`, seats the judge and nothing else.

#### Measurement context
Everything pinned around a run that is not the variant: the subject and its state, the template, the resolved
judge and simulator models, judge configs, cassette mode and corpus, tool bound, scope, world seed, apparatus
settings (`compute_context_components`). Its digest is the context key, stamped once at launch. The frozen
case set is not in it: more cases, or more repeats, under one context are more samples of one condition.

A contestant's pass^k pools per case over the runs of one pass^k cell (`PassHatKCell`): one context (read at
today's identity version), one template revision (the template's id and the digest of what it measured that
the run froze, `EvalRun.template_digest`), one apparatus provenance and one set of served models, whatever
cases each run covered and whenever each ran. Its headline figure, interval, bar and verdict read the headline
cell alone, chosen in the one order `PassHatKCellReading` states (a recorded, resolved context first, then a cell
that pools, then most runs, most results and the first run id); every other cell stands beside it with its own
pass^k and the reason it is not the headline (`FrontierPoint.pass_hat_k_cells`), named on the frontier's text, the report page and
the memo writer's view. A run kept out of the headline is kept out of the frontier's pass^k, bar and verdict
only: the arm's tables, its cells (see [Apparatus class](#apparatus-class)), composite, cost and latency still
count it. A run that froze no template digest, and a run with a candidate call whose response named no model, is
a cell of its own: a silence is never evidence of a match, so it pools with nothing and is named with what it did
not record, the headline too when every run is silent and others are kept out beside it. A run whose candidate made no model call recorded
that, and pools on everything else.

### Running

#### Host
The one value your app builds to adopt the engine (`EvalHost`): its profile, its storage, a factory for
completion clients, its failure describer, tracing, executor and cell timeout. There is no default or
global host. An app that starts runs wraps it in a `LaunchHost` (launch settings, a registry of launchable
kinds, a job timeout).

#### Host profile
Your vocabulary (`HostProfile`): the sweepables you declare, the measures you record, your bars, your
world, your kinds and your style. The engine knows a host only through it.

#### Scope
One opaque string (`scope_id`) every stored document carries: your tenant, project or environment. The
engine never interprets or defaults it in your store, and every read names one. A campaign and its runs live in one
scope. *Example:* `"dev"`.

#### Run
One execution of a template's frozen case set by one arm (`EvalRun`): one candidate model, `k` repeats per
case, a status (`completed`, `failed`, `budget_stopped`, ...). The case ids are frozen at start, so cases
generated later never change what a run scored against. *Example:* the v2 arm over the 40 tickets, `k=3`.

#### Launch, launch group, launcher
A **launch** asks for one or more arms of one template; it is always a **launch group** (`LaunchGroup`),
whose runs are prepared together and started together, or none of them are. A **launcher** is the per-kind
code your host supplies: it receives a `LaunchRequest` and returns `launch_run(host, request, KindWiring(...))`.

#### Latency under test (`measure_latency`)
The one declaration that latency is being measured, on a launch (`start_run`, `Eval(measure_latency=True)`, CLI
`--measure-latency`) and on a campaign's design. Declared, a run executes its cells one at a time and a
launch's arms one after another, so their latency is read clean; not declared (the default), cells run
several at once — as many as the kind declares it is safe at (`CandidateKind.max_concurrent_cells`; a kind that
declares nothing runs serially), never more than the host's ceiling — and any latency recorded is marked read
under concurrency (`execution_mode`
`concurrent`) and left out of every comparison, bar and ranking. A design that asks about latency (a
bar, a question or a ranking on it) without declaring it is refused. Each run records both
(`measure_latency`, `cell_concurrency`).

#### Arm schedule (`arm_schedule`)
How the arms of one launch share time and provider capacity, on `run_launch`, `sweep_launch` and `start_run`.
Running every arm side by side at its own width (`concurrent`) loads one provider with every arm at once, and a
provider rationing one arm's calls shrinks that arm's sample and not the other's: the comparison reads host capacity
as a difference between arms. Running them one after another (`sequential`) shares no load. By default the arms are
`interleaved`: every arm runs at once, their cells taken in turn, case by case, under one width shared by the launch,
and every arm plays its cases in one order. It is the default because it costs nothing and absorbs a provider's bad
minute across every arm. It does not make a comparison more of one: arms measured minutes, hours or days apart are one
experiment while nothing that decides their result changed. The shared width holds within one launch: separate
launches are separate widths, so arms launched apart still load their providers side by side. Latency under test runs the arms `sequential`, the
only schedule it takes. Each run records the schedule it ran under (`arm_schedule`), and a campaign whose arms were
run side by side at their own widths says so beside its readings, since co-loading changes a provider's capacity per
arm; when the arms ran is never said. It is a disclosure, never part of a run's identity.

#### k (repeats)
How many times each case is played in a run (`k_runs`, `Eval(k=...)`, CLI `--k`; default 3). LLM
answers vary, so one play per case under-reports that variance. *Example:* 40 tickets, `k=3`, 120 trials.

#### Depth
How many times an arm actually ran each case: its cases, its usable trials, and its repeats per case, fewest,
median and most (`Depth`, on `ArmScoreboard.depth`, each reading's `control_depth` / `arm_depth` and each
`FrontierPoint.depth`). `k` is what a run planned; depth is what the arm delivered, pooled over every run of it.
Arms differ in depth when old runs pool forward or a run stops part-way. A comparison of pass^k reads the
**common depth** (`SubjectFrontier.common_depth`): the fewest runs any arm gave a case every arm shares, over
every shared case, each arm read there over all its runs by `C(c, k) / C(n, k)`, exactly. A depth some arm never
reached is unmeasured for it (`unmeasured_for`), and at depth 1 the comparison is of pass rates, not reliability.
Extra repeats tighten an arm's own estimate, not a comparison much: they shrink the noise within cases, and the
spread between cases is a floor only more cases lower (a reading's `precision`). The pass-rate interval weighs each
shared case's share equally whatever its depth, while the passed-of-measured counts pool every trial. Above an
arm's depth nothing is measured; a host may ask for a **model-based** pass^k there (`HostProfile.model_pass_hat_k`,
off by default): a beta-binomial over the per-case pass chance, labelled as a model with what it assumes, never
ranked or gated on.

#### Cell
The docs use this word in two senses. **While running**, a cell is one trial: one case at one repeat,
producing one result (a candidate that raises "fails its cell"; a broken rig "costs one cell").
**In analysis**, a cell is every observation sharing one variant and one apparatus class, `(variant_key,
apparatus_class_id)`; a re-run of an arm adds observations to the same analysis cell. Strata, the decision
surface and report tables read analysis cells.

#### Result (observation)
One trial's record (`EvalResult`): one case, one model, one repeat, with its goal-state outcomes, rubric
scores, cost and what the kind reported. Its turn-by-turn trace is stored beside it (`EvalTrace`).

#### Evidence core
The stored documents a run produced or binds (`CORE_DOC_TYPES`): runs, results and their traces, out-of-run
spend, campaigns, sweeps, analysis attempts, generation slots, and the records of every other job (launches, second
judge passes, judge temperature comparisons, host action jobs). Read from the core baseline on: an older one is
upgraded as it is read, and one a newer minor wrote (an additive change) is read with the fields this build does
not know kept and written back. A core baseline reset deletes it (`reset_core`). What people author is not in it:
see [durable](#kept-durable-and-regenerable).
Analyses, insights, judge
profiles and cassettes are outside it: a release that changes their format refuses them, and they are made again
([what is kept](adopting-a-host.md#stored-data-what-is-kept)). A cassette's format has a version of its own, so a
change to the analyses never refuses a corpus.

#### Cassette
A recording of what a candidate's tools answered, kept by a host's own runs. A run in `cassette_mode="capture"`
records one; a run in `"replay"` is served that recording instead of calling the tools live, so two arms can face
exactly the same tool answers. It records the tools only, never the candidate. An `Eval` keeps its own the same
way in a file: a world of lookups with `recording=` (`World(tools=[search], recording="search.jsonl")`) records
its answers once and replays them after. Its stored format has
its own version (`CASSETTE_SCHEMA_VERSION`), apart from the analyses': a corpus is refused only when the cassette's
own format changes, and is then re-captured.

#### Simulator
The engine's simulated user: the other side of a conversation a conversing kind holds
(`ConversationSpec`). A classifier template has none.

#### Job
Work started by an operation because it is long or calls a model (a launch, a sweep, an analysis generation, a
second judge's pass, a judge temperature comparison, a host action's draft), answered by `job_poll` and
`job_cancel`: the operation's call prices and admits it and returns at once, so no call waits on a model. Its id names the durable record the work writes, so it survives a restart and reads the same in
every process. The record carries the running process's lease; a job whose lease lapses before it ends is
`lost`, which the first reader writes on the record. A run's, a sweep's or a launch's job that ended before its work
did (lost, cancelled, or stopped by a budget) is taken up again as the same job by `job_resume`, and never on its
own; `jobs_resumable` lists what it would take up. See [Operations and jobs](serving.md#operations-and-jobs).

#### Lifetime (holding)
One process's holding of a run, from when it took the run up to how that holding ended. A run that is resumed spans
several, each an entry of `EvalRun.lifetimes` (`RunLifetime`): its holder and lease epoch, how it ended, the cells
stored by then, the cost cap it ran under, and the host's labels when it began. A deploy shows here, and is not a
change in what the run measures. A sweep records its own (`SweepLifetime`). A run never resumed has none: the run
itself describes its one holding.

### Grading

#### Measure
A number code computes about a result, declared in your host's measure registry with its unit, direction
(is higher better?) and family. An `Eval` makes one per function grader, named by its `__name__`. A
classifier lands two core measures, `match` and `confusion_cell`, and the analysis derives `accuracy` from
`match`. A host may not declare a measure named like a core one (`score`, `f1`, `cost_usd` and the rest), whose
readings would pool with the engine's own; the runner likewise refuses a kind that lands a core-named key on
`host_measures` (other than the classifier's `match` and `confusion_cell`), and a result stored before a
release made its key engine-owned has the key dropped on read and named as unreported. A result's covariates are held to the same rule. *Example:* `match` is 1 when ticket 17 went to `billing`, else 0.

#### Scorer
A function grader's older name, still the word for the measure it reports. See [grader](#grader).

#### Goal-state check
A code check over a cell's end state (`state.<dimension>`), the calls the candidate made (`calls(...)`) and
what fired in the world (`fired(...)`), written in the goal-state language. Objective, so no judge. Its pass
rate measures the behaviour only when a control proves the check beats doing nothing; otherwise every surface
marks it `unproven` or `refuted`. A case parameter (`variation.<name>`) is one string, as the case stores it:
compare it or look for it (`contains(state.tags, variation.category)`), and write a set of values as a list
literal (`intersects(state.tags, ["toys", "games"])`). Reading a parameter as a collection is refused.

#### Judged dimension (rubric dimension)
One subjective quality a judge scores (`RubricDim`): a name, a description and a scoring guide. *Example:*
"does the auto-reply sound polite", for a variant that also drafts a reply. A **judge config**
(`JudgeConfig`) is a versioned prompt, model and settings for one dimension; name its model by a concrete model id,
never an alias, since the served model is part of the judge's calibration key and an alias re-pointed under the same
name merges two judges' evidence. Every judge call is requested at temperature 0 (`DEFAULT_JUDGE_TEMPERATURE`)
unless a config states another, whether or not the dimension has a config; a model that refuses a temperature is sent
none, and each score records what was sent.

#### Guardrail
Something the candidate must not do: leak data, take a destructive action, break policy. A judged dimension
on the `boundary` axis (`RubricDim.axis`, which the judge stamps on every score and every "can't tell") or a
measure the host declares `guardrail=True` is one; on the quick path, `grader(fn, guard=0.02)` declares a
function's and `Judge(..., guard={"dimension": 0.02})` a dimension's, each with its margin; every grade is
higher-is-better, so no direction is stated. Guardrails never join the composite, pass^k or a comparison
family, so a "can't tell" on one leaves the trial in both; the bundle decides each one for every arm against
the control as `held`, `breached` or `undecided`. A catalog dim's `axis` is its embedded dim's, so copying a
boundary catalog dim into a template keeps it a guardrail. *Example:*
`boundary.correct`, "declined the unsafe ask", held at no change while a new prompt raises task success.

#### Evidence tier
How far a judged score can be leaned on, decided by code from how reliable the judge was measured to be:
`calibrated` (agrees with people), `separation` (agrees with itself), `incidental` (measured, and shown
below both bars), `undetermined` (not shown either way). Each is decided on confidence bounds for the
agreement, never its point estimate, and an undecided one says how many more results it needs. See [reading reports](reading-reports.md#how-far-a-judged-score-can-be-leaned-on-evidence-tiers).

### Comparing and reading

#### Campaign
What a person set out to learn, and the runs gathered to learn it (`EvalCampaign`): **one question** asked of its
**arms**, measured on one or more **aspects**, with the **floors** a candidate is warned for crossing, in one scope.
Membership is chosen, not queried: a run joins one aspect of a campaign, and may sit in several campaigns. The
hierarchy is campaign → aspect → run → cell. A campaign is authored, so it is durable: no core baseline reset
deletes it ([Stored data](adopting-a-host.md#stored-data-what-is-kept)). A campaign none of whose aspects declares
a design is **exploratory**, and its creator says so (`exploratory=True`); that is read off the missing designs,
never stored. One that declares a design but asks no question asks nothing, so its readings are exploratory too.
*Example:* "Should v2 replace v1?", control v1, measured on two aspects, routing and replies. Declaring one:
[Measuring soundly](measuring-soundly.md#declare-the-campaign-its-question-arms-and-aspects).

#### Aspect
One place a campaign's question is measured (`Aspect`): a template, the runs gathered on it, its **margin** (how
far, in points of the pass rate, a candidate may fall below the control there before it is shown worse) and its
**design** (the axes it varies, what it held fixed, the merit axes its readings are tested on, its bars). Each
aspect is read on its own, against the same control, and keeps its own pre-registration time (`asked_at`), which
also dates its bars and a margin or floor declared with it. A criterion set after the aspect's first stored result
reads as one set before it, and every page names it once with its time (`reading_scope.late_criteria`); when the
runs ran never counts. A campaign adopting another's aspect (`aspects_from`) keeps those times, so it reads as the
campaign declared before the runs did; whatever it adds is dated when it is added. (Known gap: an aspect adopted from
a campaign that asked no question keeps that time when the adopting campaign first asks it one, so being put to the
question after its readings is not named.) A campaign stored before aspects
existed reads as one aspect labelled with its behavior.

#### Floor
A line a candidate is warned for crossing on one aspect (`Floor`): a typed clause, `RateDrop` (no pass rate drops
by more than `max_points`) or `CostLower` (the candidate costs less per reply), never words the engine parses. A
floor reads `broken`, `held` or `unclear`, on its own interval at the nominal level. It only **warns**, loudly: a
broken floor is the first line of the page. It never decides, blocks or writes the verdict; code reports, the
writer recommends, and a person decides. Held on every floor of every asked aspect, each declaring one, a candidate is shown no worse,
which licenses the writer to recommend it as a separated gain does: a floor is how a campaign asks "no worse?".

#### Campaign kind
What a campaign can decide, read off its arms and never declared (`campaign_kind`):

| Kind | Arms | Code reports, per aspect | The writer |
|---|---|---|---|
| measurement | one | each measure with its interval, and each bar met or missed | a summary and findings; no recommendation |
| choice | two or more, no control | every pair's difference with its interval, and which arm is shown ahead | picks an arm, or says no clear winner |
| change | a control and candidates | each candidate against the control: shown better, shown worse, no worse than the margin or unclear | recommends a switch or keeping the control |

A margin and a relative floor read a candidate against a control, so a campaign naming none is refused either,
with the remedy to declare a bar instead. A bar is absolute, so every kind is held to its bars.

#### Recommendation and call
**Code reports, the analyst recommends, a person decides.** A report names the arms it recommends, as data, with
the claims and confidence that carry the call; a tradeoff may recommend none and state its lean in prose. When it
names exactly one, that arm is the campaign's recommended variant. The person's decision is the **call**
(`CampaignCall`, recorded by `campaign_record`): the arm, why, and each broken floor and missed bar it overrides by
name. The call is read against code's readings, never the report, and is matched by variant key, since a derived
arm's label can shift when a variant joins.

#### Control
The arm every other arm in a campaign is tested against: usually what runs today. It is a variant, not a run,
so any run of that configuration counts toward it. An `Eval` comparison sets it from its runner mapping's first
key; elsewhere it is the campaign's first arm, with role `control` (`CampaignArm`), addressed from a run: an arm's
`from_run_ids` at creation, or `set_campaign_control` (`control_from_run_id` on the `campaign_update` action)
afterwards. The server records the run's variant key beside its lever map and identity version, as a run records
its own; nobody types those. A control named after an aspect's first result is named on that aspect's pages as a
criterion set after its readings were visible.
Every read resolves it through `resolve_control_arm` to the
keys it matches: the stored key, which the cells stamped under its identity version keep, and, under a later
version, also the key recomputed from that lever map, which new runs of the arm carry. That recomputation follows a
change to how a lever map is digested, not a renamed lever, kind or overlay field. So a control keeps matching its
old cells and is never stale while they are in the campaign. The control ARM (`design.control_arm` in the bundle),
which every reading names and every contrast is read against, is the cells under the current identity version
wherever any run carries the recomputed key; the cells under the earlier version are then the control under that
version (`design.earlier_control`), named on every surface, disclosed once, in no contrast or adoption check, no contestant on the frontier,
and not in the control's numbers. A key typed by hand records no lever map; when no cell carries it under a later
identity version it is **stale**: the bundle says it cannot be resolved (`control_excluded`), and the remedy is to
address it again from a run. A candidate arm nobody named is read off the runs: each other variant among them is
one. *Example:* the v1 arm.

#### Delta, interval and adjusted p
What a comparison reports for each arm against the control, per reading. The **delta** (the report's **Change**
column) is the arm's mean minus the control's, over the cases the test read: in points for a rate, in the reading's
own unit otherwise. Its **interval** is where the true difference plausibly lies, widened so that every interval
tested together holds at once, 95% of the time. The **adjusted p** (Holm's, in the folded Statistics table) is the
p-value corrected for every comparison in its family. A separation needs that p below 0.05 and an interval that
excludes zero. See [reading a comparison](reading-reports.md#reading-a-comparison).

#### Verdict
What a comparison's evidence supports, per arm and reading: **separated** (printed `v2 better than v1` or `v2 worse
than v1`, each arm by its label), **not separated** (the cases could not tell the arms apart, which never means "no
difference"), **equivalent** (shown inside a margin the measure declares, and the only verdict that says "good
enough"), or **untested** (no test could decide). A report names, under Not tested, each measure that declares no
margin and so can never read equivalent. A guardrail is decided apart, as `held`, `breached` or `undecided`, and a bar as `cleared`, `missed` or
`undecided`. Each is also a typed value a program reads (`Result.verdicts()`), which the printed words are
rendered from. See [reading a comparison](reading-reports.md#reading-a-comparison).

#### Miss
A result the runner got wrong: a wrong label, a grader on the wrong side of 0 (0 or less where higher is better,
above 0 where lower is better, as `leaked` is; never for a measure with no direction), a failed goal check, a failed
pass/fail judgement, or a result the runner failed outright (it raised, say). An excluded result is never a
miss. `summary.misses()` lists them, each with its reason; [rung 1](ladder.md#rung-1-your-own-function) says how
to read them.

#### Analysis bundle
Everything code computed about one aspect of a campaign, assembled once and fingerprinted
(`AnalysisContextBundle`, its `aspect` naming which), so two reads of the same fingerprint read the same evidence. A
campaign of several aspects is assembled one aspect at a time (`assemble_context_bundle(..., aspect=...)`). The CLI's
`bundle` prints it inside a `BundleInspection` wrapper (campaign, scope, fingerprint, and the bundle as its `bundle`
field); save the bundle alone with `bundle.to_json()`.

#### Report snapshot
A campaign's bundles, built into the report input and frozen as one stored document the analyst's tools read. It is
addressed by its content, so the same evidence frozen twice names one snapshot, and every tool is a function of the
snapshot and its own parameters alone: the analyst, its editor and a replay of either read the same figures, and
runs added later change nothing under a report.

#### Analysis
The campaign's written report (`EvalAnalysis`, linked to the **report document** it stores). An **analyst** model
investigates the campaign's snapshot with read-only tools, guided by a skill the package ships, and writes the
document: a headline, the arms it recommends, its confidence and caveats, claims in reading order each with the
evidence blocks that show it, what it left out and why, next steps and method. Every number in it is a **figure key**
code resolves; the analyst types none. Code checks the draft mechanically and sends a failing one back; an
**editor**, the same model in a fresh context, then reviews the page as a reader would before it is stored. Optional,
and it costs money; the verdict and every count are code's either way. Older analyses were a memo or a planned
report; a memo still renders as its page, and a planned report reads as the code-only page with a line to regenerate.
See [the analyst's report](reading-reports.md#the-analysts-report-written-over-frozen-evidence).

#### Confidence
How sure a report's call is, in one of three forms: high, high if the caveats below aren't material, or low. It is
read from how the campaign was measured (its design, criteria set after the results, the judge's standing,
confounds, variance), not from how the result came out. Each open caveat says what would make it matter; one code
proves cannot change the call is printed as code's checked line instead. A lead's risk of reading the wrong way is
stated beside the lead, and never lowers the form.

#### Trial
One case run once on one arm — one stored result. It is passed, failed (the candidate's failure: its model, its
budget, a check or a judged dimension) or lost (the test setup's: the harness faulted, the judge could not tell, or
the judge's reply was cut off at its cap). A lost trial says nothing about the candidate.

#### Decision surface
The campaign's measured analysis cells (one `CellFacts` each) and the bars they were held to, frozen when
an analysis is generated (`DecisionSurface`), so runs added later cannot change the numbers under it.

#### Disclosure
A block code adds to a report to say what a reader must know, whoever wrote the rest: that no analysis
was generated, say, or that a stratum holds too few cases to read alone.

#### Out-of-run spend
Model calls the engine makes outside any run (generating cases, proposing a rubric, repeating judge scores):
priced before they are made and ledgered as `OutOfRunSpend`. Writing a report is held to a budget of its own, which
the host sets. See [cost and budgets](cost-and-budgets.md).

### Building a host

The words [Adopting the engine](adopting-a-host.md) uses for what a host author writes or wires.

#### Bar
A standard one measure must reach for one behavior (`Bar`, registered on the profile's `BarRegistry`, with
`PassThreshold` for the 1–5 level an attempt must reach to pass). It is decided on a cell's interval, never its
mean: `cleared`, `missed` or `undecided`. A campaign's design may declare a bar tighter than the registered one,
never looser, and the frontier takes one as `bar=`. Either kind of bar can cover judged behaviour: a campaign's
`BarOverride.measure_id` may name a rubric dimension of its template, and a registered `Bar` with `judged=True` names
one that each campaign of the behavior resolves against its own template, refused at authoring when the template does
not declare it. *Example:* routing accuracy of at least 0.9 for the `routing` behavior. See
[Reading reports: Methods](reading-reports.md#methods).

#### Conformance kit
A set of checks you run in your own test suite against what you implemented, so a gap fails there and not in a
measurement: `STORE_CONFORMANCE_CASES` for a store, `check_completion_conformance` for a completion client,
`jobs_conformance_cases` for the action catalogue you mount (your own actions among them),
and `check_world_conformance` for a world. A world check the kit
cannot run is reported `unavailable` with its reason, never passed. See
[the store's kit](adopting-a-host.md#the-store-and-the-conformance-kit-that-proves-it) and
[the world's](evaluating-agents.md#step-5-run-the-conformance-kit-against-your-world).

#### Executor and the event-loop rule
The store is synchronous and the engine is async, so every store call runs on the host's `blocking_executor`
(a thread pool, or `None` for the loop's default), never on the event loop. A call made on a thread running a
loop is refused (`StoreCalledOnEventLoop`, naming the call site). Your own async code follows the same rule: a
synchronous function that takes the storage or the host is called as
`await run_blocking(host.blocking_executor, fn, ...)`.
See [the store](adopting-a-host.md#the-store-and-the-conformance-kit-that-proves-it).

#### Fold
Two uses. **A completion's usage is folded** into a usage ledger: added to the role's running totals, never
replacing them (`RoleUsageLedger.add_llm_result`, or a conversation driver's `fold_usage`). **A resolved lever
folds into its knob** (`ResolvesInto`): when every level of the knob carried one level of the lever it is
written into, the lever gets no row or confound of its own and the arm is named by the knob. A fold no two arms
could test is disclosed as `unverified_fold:<lever>`. See [the host](adopting-a-host.md#the-host) and
[the kind](adopting-a-host.md#the-kind-what-you-are-evaluating).

#### Kept, durable and regenerable
Every stored type's **lifecycle** (`DOC_TYPE_FACTS`). A **kept** type is the [evidence core](#evidence-core):
read from the core baseline on, and deleted by a core baseline reset. A **durable** type is what people author
(`DURABLE_DOC_TYPES`: templates, rubric dims, judge configs and their tombstones, case sets, test cases, calibration
ratings): read forever in a version space of its own, changed only additively or through an upgrader, and never
touched by a core baseline reset. A **regenerable** type (analyses, insights, judge
profiles, cassettes) is refused after its format changes and made again: an analysis, insight or profile over the
kept runs, a cassette by capturing it anew. A store can route each lifecycle to its own backend
(`DocTypeRoute(by="lifecycle")`). See [Stored data: what is kept](adopting-a-host.md#stored-data-what-is-kept).

#### Lease
A running [job](#job)'s claim on its record: the process running it renews it on a schedule (`JobLeases`,
`LaunchHost.job_lease_s`), stamped in the store's own time when the store offers a `StoreClock`. A job whose lease
lapses before it ends reads `lost` everywhere, and no renewal revives it. Each lease carries an **epoch**, one higher
each time the job is resumed, and each result row records the epoch it was written under, so a process holding an
older epoch writes nothing more into the job. See [Operations and jobs](serving.md#operations-and-jobs).

#### Seam
A place the engine calls code your host supplies, through a protocol the engine declares: the kind seam
(`prepare` and `invoke`, under a `KindContract`), the cassette seams a kind wires (`CassetteSeams`). See [the kind](adopting-a-host.md#the-kind-what-you-are-evaluating) and
[cassettes](adopting-a-host.md#cassettes-recording-and-replaying-tools).

#### Sink and reservation
A **sink** is what code reports into rather than returns: a cell's `CellSink` keeps its spend by source,
`TraceSink` is your tracing, and a `RegressionWatch` awaits a sink of yours with each alert. A **reservation**
is the handle of a paid call a cell is making (`sink.reserve(source, ceiling)`, then `settle(cost)` or
`release()`), which lands the call's cost on the cell's spend; it holds nothing back, because the run's cost cap is
read between cells and refuses no call inside one. `CappedCompletion` does it for every call. See [Caps](cost-and-budgets.md#caps).

#### Store, `DocumentStore` and port
A **port** is a protocol the engine declares and your host implements: `DocumentStore`, `CompletionClient`,
`TraceSink`. The **store** is your `DocumentStore` (or an `AsyncDocumentStore` behind `bridge_async_store`): one
collection keyed `(scope_id, doc_type, id)`, which the engine reads and writes through `EvalStorage`. The engine
ships three (`InMemoryDocumentStore`, `SqliteDocumentStore`, `RoutedDocumentStore`). See
[the store](adopting-a-host.md#the-store-and-the-conformance-kit-that-proves-it).

#### Surface
A place something is shown. A **world's surface** is one place the agent perceives state: each dimension names
every surface that shows it (`perceived_by`), and the conformance kit checks that each moves with its dimension
alone. A **reporting surface** is one place a reading is shown to a person or a program: the run summary, the
report, the bundle, an action's reply. "Said on every surface" means each of them says it. The
[decision surface](#decision-surface) is its own term. See
[Evaluating a tool-using agent](evaluating-agents.md#step-1-declare-the-world).

## What the engine reads from your code

An `Eval` takes plain functions, so a few things you might think of as documentation are read as data. Everything they read is below, with the way to state it outright instead.

| What | Which part | Who reads it | To state it instead |
|---|---|---|---|
| A runner's docstring | its first line | The [template](#template)'s intent, which a [judge](#judge) reads beside every answer, so rewording it can move judged scores. With several arms, it is read only when every arm's docstring shares that first line. | Reword the docstring; a judged run's `summary.render()` prints the intent and where it came from. |
| A runner's `__name__` | the whole name | The arm's label: the run's candidate model, keyed into its variant, and the eval's default name. | Name the arm: a runner mapping of one (`{"arm-a": runner}`). A comparison names each arm by its key on the `candidate` lever, every arm at one model; keys named `"model=m1"` are models. |
| A function grader's `__name__` | the whole name | The [measure](#measure)'s name, in the summary, reports and the analysis bundle. | Rename the function. |
| A function grader's docstring | its first line | The measure's description, in the analysis bundle's `measure_catalog`, which a model writing an [analysis](#analysis) reads for what the measure means. | Declare the measure, with its description, on the function (`@measure(...)`), or on a [host](#host) of your own (`host=`). |
| A world tool's function: `__name__`, docstring and type hints | the name, the docstring's first line, and each parameter's type | The tool's name, description and parameter schema, which the model is shown, as in any tool-use API. | None: the function's name, docstring and types are the tool's. |

With no docstring, a function grader is described as "The score the `<name>` function gave the candidate's answer.",
a tool as "The `<name>` tool.", and the intent is a generic sentence the summary labels as such.
