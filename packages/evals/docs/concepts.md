# Concepts: the nouns of 3tears-evals, and how they fit together

**For** anyone stopped by a word in the README or a guide. **Answers:** how the pieces relate (one picture),
and what every term of art means, in plain words with one running example. Nothing here assumes you have
built an eval before. Why the pieces are shaped this way is in
[Design rationale](design-rationale.md) and [The world model](world-model.md).

**The running example.** You own a support-ticket triage classifier: it reads a ticket's subject and body
and picks one queue, `billing`, `bug`, `account` or `other`. You have a prompt in production (call it v1),
a rewrite you think is better (v2), and 40 hand-written tickets whose right queue you know. You want to
know whether v2 routes tickets better than v1, by how much, at what cost, and whether the evidence can
tell at all. (The same classifier appears in [Designing a classifier eval set](designing-classifier-evals.md).)

## The picture

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
            graded by  measures         -- numbers code computes: match, latency, cost, your scorers
                  and  judged dimensions -- scores an LLM judge gives against a rubric (tone, say)
     |
     |  you put the runs you want compared into
     v
  campaign  e.g. "v1 vs v2", in one scope; its results pool into
            analysis cells: one variant measured under one rig
     |
     v
  analysis bundle   every number code computed about the campaign, frozen and fingerprinted
     |
     |  optionally, a model reads the bundle and writes findings: an analysis
     v
  report    Markdown, HTML or JSON; "code-only" when no analysis was generated
```

Underneath all of it sits your **host**: the one value through which your app tells the engine what it can
change (levers), what it can see (measures), how to run your code (a kind), and where to store things.
`run_eval` builds a throwaway host for you, which is why the README's first example needs none.

## Terms to learn first

The one short list: the README links here rather than keeping its own.

| Term | In one line |
|---|---|
| [case](#case-test-case) | One input and what a good answer looks like: one ticket and its right queue. |
| [candidate](#candidate) | The code under test, which answers each case. |
| [scorer](#scorer) | A function that grades an answer with a number, when code can check it. |
| [measure](#measure) | A number code computes about a result: did it match, how long it took, what it cost. |
| [judge](#judge) | A model that grades an answer against a rubric, when code cannot. |
| [judged dimension](#judged-dimension-rubric-dimension) | One written quality of a rubric, which the judge scores. |
| [k](#k-repeats) | How many times each case is played, because model answers vary. |
| [run](#run) | One batch of trials: one arm, played over every case, `k` times each. |
| [result](#result-observation) | One trial: one case, one repeat, with every grade it got. |
| [arm](#arm) | One version under test, run over every case. |
| [control](#control) | The arm every other arm is tested against. |
| [campaign](#campaign) | The runs you want compared, grouped so they can be analysed together. |
| [verdict](#verdict) | Separated, not separated or equivalent: what the evidence supports. |
| [interval and p](#delta-interval-and-adjusted-p) | Where the true difference plausibly lies, and the Holm-adjusted p a verdict is decided on. |
| [report](#report) | The one document you read: tables, charts, and (if generated) findings in words. |
| [world](#world) | The state an agent acts on, set for each case and read back after. |

## Glossary

Terms are grouped by where you meet them. Code names are in backticks; the module or class named is
where the definition lives in the source.

### What you test

#### Case (test case)
One concrete input, frozen once written (`EvalTestCase`). A case's content never changes after it is
stored, which is what lets two runs over the same cases be compared. *Example:* ticket 17, "I was charged
twice this month", expected queue `billing`.

#### Case set
A named, versioned, frozen list of one template's cases (`CaseSet`, minted by `mint_case_set` or the
`case_set_mint` action). It is append-only: changing it mints the next version, and rewriting a stored
version is refused. A launch can name one (`case_set_name` and `case_set_version`), runs exactly its cases,
and records it on the run, so history reads a change of suite as `smoke v1` to `smoke v2`. It labels the
run's frozen case ids, never a second identity. Not a [battery](#battery). *Example:* `smoke v2`, the three
tickets every release is checked on.

#### Template
The blueprint the cases belong to (`EvalTemplate`): what is being tested (its intent), how to score it
(goal-state checks and rubric dimensions), the axes cases vary along, and anything only its kind reads (its
spec). *Example:* "route a support ticket to a queue", with the four labels in its spec. `run_eval` takes its
intent from `intent=` or a candidate's docstring: see [what the engine reads from your
code](#what-the-engine-reads-from-your-code).

#### Variation axis
One direction along which a template's cases differ (`VariationAxis`): `enum` yields each listed value once,
`sample` draws from listed values, and `llm` asks a model to write new values. *Example:* an axis
`tone` with values `angry`, `polite`, `terse`, so generated tickets cover all three.

#### Stratum
The kind of case a case is, in your own words (`EvalTestCase.stratum`), so results can be broken down by it.
It is never shown to the thing under test. *Example:* the 12 tickets marked `lookalike` (a billing question
that reads like a bug report) get their own column in the report.

#### Subject
What is being measured, as your host names it, frozen at launch (`SubjectSnapshot`). *Example:* "the triage
service". In `run_eval`, the subject is just the candidate's name.

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
In `run_eval`, `world=`, `seed=` and `goal_checks=` (`examples/world.py`). Why a subject runs in a seeded
world at all: [The world model](world-model.md).

### What you vary, and what must hold still

#### Sweepable
Any input that can change a score, declared by your host with a reader that says how to find its value on a
run (`Sweepable`). Every sweepable has one of three roles: **lever**, **apparatus** or **label**. The engine
ships the ones every LLM product has (`SHARED_CORE`); you add your own.

#### Lever
A sweepable you deliberately change to see what it does (role `lever`). The engine always resolves two
itself, the candidate model (`model`) and the candidate kind (`candidate_kind`); your kind's overlays add
more. *Example:* `model` and `ticket_router.prompt_version`.
With no host of your own, `run_eval(..., levers={"prompt": "v2"})` states one beside the model, as
`callable.prompt` (`callable-judged.prompt` on a judged run), and `compare(..., factors=("model", "prompt"))`
keys each arm by its level of both (`examples/prompt_x_model.py`).

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
about the lever. The engine's own: the judge model and its settings, which judge each dimension was actually
scored by (`judge_dim_divergence`), the temperature each judge call was sent at, the judge configs, the simulator model and its settings, and the per-run cost ceiling (`max_cost_usd`). *Example:* the v1 runs were judged by
judge-a and the v2 runs by judge-b: the rig moved, so the report will not pool them as one condition.

#### Apparatus class
One configuration of the rig as recorded on the runs (`ApparatusClass`). Two observations pool only when
their variant matches and their apparatus falls in the same class; an arm measured under two classes is
named with `@ rig <digest>` in a report.

#### Apparatus settings
Rig values your host declares and a launch may set (`apparatus_settings`), so one template can be run at two
of them and compared. *Example:* who sits in an adjudicator's seat: `{"adjudicator_seat": "model:default"}`.

#### Seat
A place in the rig a kind's runs actually fill (`KindContract.seats`): a pinned role such as `judge` or
`simulator`, or one apparatus dimension. A dimension a kind does not seat does not apply to its runs, so
its blank there is not a confound. `None`, the default, holds the kind to every dimension. *Example:* the
callable kind `run_eval` builds seats none of the judge, simulator or spend ceiling (`CALLABLE_UNSEATED`); its
judged kind, `callable-judged`, seats the judge and nothing else (`JUDGED_CALLABLE_UNSEATED`).

#### Measurement context
Everything pinned around a run that is not the variant: the subject and its state, the frozen case set, the
template, the resolved judge and simulator models, judge configs, cassette mode and corpus, tool bound,
scope, world seed, apparatus settings (`compute_context_components`). Its digest is the context key,
stamped once at launch.

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
The one declaration that latency is being measured, on a launch (`start_run`, `run_eval`, `compare`, CLI
`--measure-latency`) and on a campaign's design. Declared, a run executes its cells one at a time and a
launch's arms one after another, so their latency is read clean; not declared (the default), cells run
several at once, and any latency recorded is marked read under concurrency (`execution_mode`
`concurrent`) and left out of every comparison, bar and ranking. A design that asks about latency (a
bar, a question or a ranking on it) without declaring it is refused. Each run records both
(`measure_latency`, `cell_concurrency`).

#### k (repeats)
How many times each case is played in a run (`k_runs`, `run_eval(k=...)`, CLI `--k`; default 3). LLM
answers vary, so one play per case under-reports that variance. *Example:* 40 tickets, `k=3`, 120 trials.

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
The stored documents a later release always reads (`CORE_DOC_TYPES`): cases, runs, results and their traces,
calibration ratings, out-of-run spend, case sets, and the templates, judge configs and rubric dims they name.
An older one is upgraded as it is read. Campaigns, analyses, insights, sweeps and cassettes are outside it:
a release that changes their format refuses them, and they are made again
([what is kept](adopting-a-host.md#stored-data-what-is-kept)).

#### Battery
The universal templates, run as one pre-flighted set (`start_universal_battery`): templates marked
`universal=True` apply to every subject.

#### Cassette
A recording of what a candidate's tools answered. A run in `cassette_mode="capture"` records one; a run in
`"replay"` is served that recording instead of calling the tools live, so two arms can face exactly the
same tool answers. It records the tools only, never the candidate (`examples/cassettes.py`).

#### Simulator
The engine's simulated user: the other side of a conversation a conversing kind holds
(`ConversationSpec`). A classifier template has none.

#### Job
Long work started by an operation (a launch, an analysis generation), answered by `job_poll` and
`job_cancel`. Its id names the durable record the work writes, so it survives a restart.

### Grading

#### Measure
A number code computes about a result, declared in your host's measure registry with its unit, direction
(is higher better?) and family. `run_eval` makes one per scorer, named by the scorer's `__name__`. A
classifier lands two core measures, `match` and `confusion_cell`, and the analysis derives `accuracy` from
`match`. A host may not declare a measure named like a core one (`score`, `f1`, `cost_usd` and the rest), whose
readings would pool with the engine's own; the runner likewise refuses a kind that lands a core-named key on
`host_measures` (other than the classifier's `match` and `confusion_cell`), and a result stored before that
refusal has the key dropped on read. A result's covariates are held to the same rule. *Example:* `match` is 1 when ticket 17 went to `billing`, else 0.

#### Scorer
In `run_eval`, a plain function `(case, answer) -> bool | number` that becomes one measure, named by its
`__name__` (never a core measure's name) and described by its docstring's first line. A scorer that raises
excludes the cell: it is part of the rig, not the candidate.

#### Goal-state check
A code check over a cell's end state (`state.<dimension>`), the calls the candidate made (`calls(...)`) and
what fired in the world (`fired(...)`), written in the goal-state language. Objective, so no judge. Its pass
rate measures the behaviour only when a control proves the check beats doing nothing; otherwise every surface
marks it `unproven` or `refuted`. A case parameter (`variation.<name>`) is one string, as the case stores it:
compare it or look for it (`contains(state.tags, variation.category)`), and write a set of values as a list
literal (`intersects(state.tags, ["toys", "games"])`). Reading a parameter as a collection is refused.

#### Judge
A model the engine asks to score a result against a rubric, reading only the evidence the kind rendered
(`JudgeEvidence`). Each dimension is scored 1-5 (`ordinal`) or `pass_fail`; "can't tell" is an answer.

#### Judged dimension (rubric dimension)
One subjective quality a judge scores (`RubricDim`): a name, a description and a scoring guide. *Example:*
"does the auto-reply sound polite", for a variant that also drafts a reply. A **judge config**
(`JudgeConfig`) is a versioned prompt, model and settings for one dimension. Every judge call is requested at
temperature 0 (`DEFAULT_JUDGE_TEMPERATURE`) unless a config states another, whether or not the dimension has a
config; a model that refuses a temperature is sent none, and each score records what was sent.

#### Guardrail
Something the candidate must not do: leak data, take a destructive action, break policy. A judged dimension
on the `boundary` axis (`RubricDim.axis`, which the judge stamps on every score and every "can't tell") or a
measure the host declares `guardrail=True` is one; on the quick path, `compare(guardrails=...)` declares either,
with its margin and direction (`Guardrail`). Guardrails never join the composite, pass^k or a comparison
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
A curated set of runs under one subject and behaviour, the hub an analysis attaches to (`EvalCampaign`).
Membership is chosen, not queried; a run may sit in several campaigns. Its declared design names a
**control**, which is a variant key, not a run, and what it **held fixed** (`held_fixed`: the stimulus,
controlled or not, and the apparatus, commissioned or witnessed). *Example:* "triage v1 vs v2", control =
the v1 variant, held fixed = one case battery on a commissioned rig. Which design to declare for a question:
[Choosing a campaign design](choosing-a-design.md).

#### Control
The arm every other arm in a campaign is tested against: usually what runs today. It is a variant, not a run,
so any run of that configuration counts toward it. `compare` sets it from `control=`. *Example:* the v1 arm.

#### Delta, interval and adjusted p
What a comparison reports for each arm against the control, per reading. The **delta** is the arm's mean minus
the control's, over the cases the test read. Its **interval** is where the true difference plausibly lies, widened so
that every interval tested together holds at once, 95% of the time. The **p (Holm-adjusted)** is the p-value
corrected for every comparison in its family. A separation needs that p below 0.05 and an interval that excludes
zero. See
[reading a comparison](reading-reports.md#reading-a-comparison).

#### Verdict
What a comparison's evidence supports, per arm and reading: **separated** (printed `improved` or `regressed`
on the control), **not separated** (the cases could not tell the arms apart, which never means "no
difference"), **equivalent** (shown inside a margin the measure declares, and the only verdict that says "good
enough"), or **untested** (no test could decide). A report from `compare` names, in a line above its contrasts
table, each measure that declares no margin and so can never read equivalent. A guardrail is decided apart, as `held`, `breached` or `undecided`, and a bar as `cleared`, `missed` or
`undecided`. Each is also a typed value a program reads (`Comparison.verdicts()`), which the printed words are
rendered from. See [reading a comparison](reading-reports.md#reading-a-comparison).

#### Miss
A result the candidate got wrong: a wrong label, a scorer on the wrong side of 0 (0 or less where higher is better,
above 0 where lower is better, as `leaked` is; never for a measure with no direction), a failed goal check, a failed
pass/fail judgement, or a result the candidate failed outright (it raised, say). An excluded result is never a
miss. `summary.misses()` lists them, each with its reason; the [tutorial](tutorial.md#3-read-your-misses) says how
to read them.

#### Analysis bundle
Everything code computed about a campaign, assembled once and fingerprinted (`AnalysisContextBundle`), so
two analysis prompts run over the same fingerprint are comparable. Nothing is fetched while an analysis is
written: the bundle is the whole context. The CLI's `bundle` prints it inside a `BundleInspection` wrapper
(campaign, scope, fingerprint, and the bundle as its `bundle` field); save the bundle alone with
`bundle.to_json()`.

#### Analysis
Findings a model wrote over a bundle (`EvalAnalysis`): a headline, findings, decisions and next steps, with
every number in its prose replaced by the value code read. Optional, and it costs money.

#### Report
The one document a campaign is read through (`Report`): ordered text, table, chart and disclosure blocks,
serialised as Markdown, HTML or canonical JSON. Without an analysis it is a **code-only report**
(`basis="code_only"`): the evidence, with no author's words.

#### Decision surface
The campaign's measured analysis cells (one `CellFacts` each) and the bars they were held to, frozen when
an analysis is generated (`DecisionSurface`), so runs added later cannot change the numbers under it.

#### Disclosure
A block code adds to a report to say what a reader must know, whoever wrote the rest: that no analysis
was generated, say, or that a stratum holds too few cases to read alone.

#### Out-of-run spend
Model calls the engine makes outside any run (generating cases, proposing a rubric, writing an analysis,
repeating judge scores): priced before they are made and ledgered as `OutOfRunSpend`. See
[cost and budgets](cost-and-budgets.md).

## What the engine reads from your code

`run_eval` and `compare` take plain functions, so a few things you might think of as documentation are
read as data. Everything they read is below, with the way to state it outright instead.

| What | Which part | Who reads it | To state it instead |
|---|---|---|---|
| A candidate's docstring | its first line | The [template](#template)'s intent, which a [judge](#judge) reads beside every answer, so rewording it can move judged scores. With several arms, it is read only when every arm's docstring shares that first line. | `intent=` on `run_eval` or `compare`; a judged run's `summary.render()` prints the intent and where it came from. |
| A candidate's `__name__` | the whole name | The arm's label: the run's candidate model, keyed into its variant. | `model=` on `run_eval`. `compare` names each arm by its key on the `candidate` lever, every arm at one model; with `factors=("model",)` the key is the model. |
| A scorer's `__name__` | the whole name | The [measure](#measure)'s name, in the summary, reports and the analysis bundle. | Rename the function. |
| A scorer's docstring | its first line | The measure's description, in the analysis bundle's `measure_catalog`, which a model writing an [analysis](#analysis) reads for what the measure means. | Declare the measure, with its description, on a [host](#host) of your own (`host=`). |
| A `WorldTool`'s function: `__name__` and docstring | the name, and the docstring's first line | The tool's name and description, which the model is shown, as in any tool-use API. | None: the function's name and docstring are the tool's. |

With no docstring, a scorer is described as "The score the `<name>` function gave the candidate's answer.",
a tool as "The `<name>` tool.", and the intent is a generic sentence the summary labels as such.
