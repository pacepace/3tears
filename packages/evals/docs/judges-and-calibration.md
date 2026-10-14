# Judges and calibration

**For** someone grading output that no code can check (is the answer grounded, is the tone right, did it refuse
what it should) who wants to know whether to trust the grades. **Answers:** how to write a rubric dimension, run
an LLM judge, collect person ratings, and read whether the judge agrees with people and with itself, how many
ratings that takes, and how the ratings correct the judge's mean. Grade with code wherever code can decide; a judge is for judgment
([principles](principles.md)).

The snippets run top to bottom as one script, in one async context (`python -m asyncio`, or inside
`async def main()`). A store call blocks, so a snippet makes it off the event loop with
`await run_blocking(host.blocking_executor, fn, ...)`. `client` is any `CompletionClient` (an async
`generate(system=, user=, response_format=)` and an `aclose()`); `Model` from `threetears.evals.providers` is the
shipped one, which `examples/2-replies/replies.py` uses. Every judge call sends its reply's shape as `response_format`, a strict `json_schema`
directive (the criterion's answers, "can't tell" among them); pass it to a provider that enforces structured output,
or send the nearest mode yours has. The reply is read and refused on return either way. Every enum in it is a string
enum, the one form every provider's strict mode enforces (a Gemini model accepted an integer one and left the score
out), so a 1-5 score is asked for as the string `"1"` to `"5"` and read back to its integer; a bare integer is read too.
`strict_response_format` (`threetears.evals.kernel`), the projection the engine's calls go through and yours can,
refuses a schema enumerating anything but strings with `UnportableSchema`: declare the values as strings and convert
them on read.

## Step 1: choose a scale, and prefer pass/fail

Each rubric dimension is answered on one scale: `pass_fail` (stored as 1 or 0, so its mean is a pass rate) or
`ordinal`, 1 to 5. **Prefer `pass_fail` for a new criterion.** It forces you to say what failure looks like,
it is easier for people to label and to agree on, and its agreement with them is plain kappa. Use 1–5 only for
a quality with real degrees. Its agreement is quadratic-weighted kappa, which counts a 4 against a 5 as a near
miss. Do not fold a rule into a broad 1–5 dimension, where everything else outweighs it: give it a narrow
pass/fail dimension of its own ([measuring soundly](measuring-soundly.md)).

## Step 2: write a rubric dimension

A dimension is one question, named `<context>.<name>`. Write it from failures you have seen in real outputs. The
judge is asked one dimension per call, gives its reasoning before its score, and may answer "can't tell", which
excludes that cell from the dimension. It reads only the answer and what `case_material` renders for the case,
so give it what the candidate answered from. With no `case_material` it sees the case as JSON, and on a
classifier run (a field grader) every field holding the expected label is left out, so it never grades against the
answer key by accident. To grade against a reference answer, render the reference through `case_material`.

```python
from threetears.evals.schema import RubricDim
from threetears.evals import Judge

POLICY = "Unworn items can be returned within 30 days of delivery for a full refund. Sale items are exchange only."

GROUNDED = RubricDim(
    name="answer.grounded",
    description="Every claim in the answer is stated in the store policy.",
    scale="pass_fail",
    scoring_guide={"pass": "Each claim can be pointed to in the policy.", "fail": "Any claim the policy lacks."},
)
NO_PROMISES = RubricDim(
    name="answer.no_promises",
    description="The answer promises nothing the policy does not allow: no exceptions, credits or extensions.",
    scale="pass_fail",
    axis="boundary",  # a guardrail: step 3
)
judge = Judge(  # `client` names the model it calls ("judge-model"), as client.model_name
    client, [GROUNDED, NO_PROMISES],
    case_material=lambda case: f"Store policy:\n{POLICY}\n\nCustomer question: {case['question']}",
)
```

Name the judge's model by a concrete model id, never an alias: the served model is part of the judge's calibration
key, so an alias re-pointed to a new model under the same name merges two judges' evidence (Step 7).

A scoring guide's levels must be the scale's own (`pass`/`fail`, or `1`–`5`). For a quick rubric, pass a mapping
of name to description, each scored pass/fail: `Judge(client, {"helpful": "..."})` names the dimension
`answer.helpful`. A dimension on another scale is a `RubricDim` stating it.

**A dimension declares its judge.** `RubricDim.judge` says which judge scores it: `"builtin"`, the engine's own
prompt for its scale, or `"configured"`, the scope's active `JudgeConfig` for the dimension (created with
`create_judge_config` in `threetears.evals.run`, `rubric_dim_id` set to the dimension's name). On the quick path a
`Judge` is the declaration: every dimension it builds, or is handed without one, is declared `"builtin"`, so the
examples above state nothing more. On a host, where a template's rubric is registered apart from its judges (`create_template`, a seed corpus,
`update_template` writing `rubric`), every judged dimension must carry it, and registration refuses one that does
not, naming every such dimension at once. Registration and every launch also refuse `"configured"` with no active
config, and `"builtin"` beside an active (or launch-selected) one, so no dimension is ever scored by the built-in
prompt in its config's place, or the reverse. A template stored before the declaration existed reads, and its
launch is refused until `update_template` declares its dimensions. The declaration is not part of the criterion
(`fingerprint_criterion`), so labels keep pairing across it. The two reserved axes (transcript and outcome) are
declared by the kind, not by a rubric, and keep taking a config when one is active.

## Step 3: guardrails are not capabilities

A **capability** dimension (`axis="capability"`, the default) is something the candidate should do well, and it is
read into the composite and pass^k. A **guardrail** (`axis="boundary"`) is something it must not do: leak, comply
with an unsafe ask, promise what policy does not allow. A guardrail never joins the composite, pass^k or a
comparison family. In a comparison each one is decided for each arm against the control as `held`, `breached` or
`undecided`, and a breach blocks adopting that arm whatever it gained. An undecided guardrail is never read as
safe ([reading the guardrails](reading-reports.md#reading-the-guardrails)). `Judge(guard={"no_promises": 0.1})`
puts a rubric dimension on the boundary axis with the margin it is held to. A single run's summary shows a guardrail's
mean like any other dimension.

## Step 4: run the judge

```python
from threetears.evals import Eval

POLICY_CASES = [
    {"question": "Can I return worn shoes?"}, {"question": "How long do I have to return a jacket?"},
    {"question": "Do I get a refund on sale items?"}, {"question": "Do you gift wrap?"},
]


async def answer(case):
    """Answer a customer's question from the store policy."""  # the judge reads this line as the intent
    return "Unworn items can be returned within 30 days for a full refund."


result = await Eval(POLICY_CASES, {"candidate-model": answer}, [judge], k=2).run(store=":memory:", scope="support")
host, summary = result.host, result.summary  # kept: the host's in-memory store holds what the next steps read
print(summary.render())
```

**Use a different model from the candidate's.** A model grading its own output tends to favour it. The quick path
does not flag this. On a full host, when a launch names no judge and the default judge is one of its candidates,
`LaunchSettings.judge_alternate_model` scores instead, and a run judged on a candidate's model says so wherever its
judges are listed.

**Temperature.** Every judge call is requested at `DEFAULT_JUDGE_TEMPERATURE` (0) unless a `JudgeConfig` for the
dimension states another. Each score records what its call was actually sent at (`RubricScore.judge_temperature`;
`model_default` when none was sent), and that is part of the judge's identity: agreement and tiers are kept
separately per temperature, and a repeat at another temperature is not paired. On the quick path your client
builds the request, so send temperature 0 and report it as `CompletionResult.temperature`. How much temperature
moves your judge's scores is measured, not assumed: see [Step 10](#step-10-measure-what-temperature-does-to-the-judge).
On a judge that honours it, 0 halves the spread: re-judging 50 borderline cases 5 times each, mean score variance
was 0.27 at 0 against 0.55 at a provider default that samples, a paired difference of -0.28 (95% interval -0.45 to
-0.12) and about five times the same-setting noise floor (#633, [the valid
measurement](#the-2026-10-10-measurement-on-a-judge-that-honours-temperature)). 0 does not make a reasoning judge
deterministic, though: 20 of the 50 cases still moved at 0. An earlier run on a judge that takes no temperature is
void for the question ([the void run](#the-2026-10-10-run-is-void)). On a judge whose model accepts no temperature,
0 is requested but never applied: the run stamps `judge_temperature=0` (the request) while each score records
`model_default` (what was sent).

**Request settings.** The judge asks for `JUDGE_REQUEST_SETTINGS` (`threetears.evals.run`): a reasoning budget of
8,192 tokens (`reasoning_max_tokens`, `JUDGE_REASONING_HEADROOM_TOKENS`) under an output cap of 10,240, which is the
budget plus the 2,048 the reply's schema needs (`JUDGE_ANSWER_BUDGET_TOKENS`: one reasoning paragraph and one score),
and strict output. Reasoning is asked as a bound because a word bounds no tokens: a model that read `high` as licence
to think on reasoned past the cap and was cut off with no answer written. A model that takes a budget (Anthropic's
`thinking.budget_tokens`, OpenRouter's `reasoning.max_tokens`) is held to it; a model that takes effort words only
(OpenAI's chat completions, or one that refused a budget) is sent the word the budget's share of the cap reads as,
`high` (0.80, `BUDGET_SHARES`), its sent settings naming the budget that went unsent
(`unsent_reasoning_max_tokens`), and there the cap stays the only hard stop. Reasoning stays on for every scale, pass/fail included: it is part of the instrument. Your client
builder applies the settings to the judge's client (the shipped `Clients` does for a `judge` role given by id), and a
judged run records them as its `judge_asked_settings`. What a call was actually sent with is the client's to say, as
the optional `sent_settings` on its completion: a client may send a model the closest setting it accepts (an effort
word in place of a budget, a lower effort word, no enforcement where a router could not place the call). Each score records it
(`RubricScore.judge_sent_settings`; `None` when the client did not say, which is unverified, never refused), and
that is part of the judge's identity too, the sixth component of `JudgeKey`: agreement, evidence tiers, judge
profiles and drift are kept apart per sent settings, so a judge sent `high` and the same model sent a reasoning budget
are two judges, and an unverified judge is a judge of its own. A repeat (Step 8) whose call was sent other settings
than the first score's, or where only one of the two reported them, is unpaired (`settings_changed`). Each tier's
sentence names what its judge was sent, or says `request settings unconfirmed`, and a run whose judge calls were not
shown to be sent as asked says so in one sentence on its summary
([reading reports](reading-reports.md#how-the-judge-and-the-simulated-user-were-sent)).

**A reply cut off at the cap is excluded as `truncated`, never re-bought.** When the provider stops a judge reply at
its output cap (or early, for a cause it does not name), the dimension is left unscored, the call is not retried,
and the result records the cut as a fact beside its `judge_error` line: `EvalResult.judge_cut_short[dim]` holds the
stop reason, the settings the call was sent with (whose `max_tokens` is the cap it ran into; `None` when the client
did not say) and its token counts. The trial is excluded with cause `truncated` rather than `infra`
(`trial_exclusion`), and the analysis bundle counts each arm's exclusions by cause with the caps the cut calls were
sent ([reading reports](reading-reports.md#excluded-trials-by-cause)). A reply the provider filtered, or that came
back as an error, is recorded the same way and stays a fault of the rig.

**The built-in prompt is a judge too.** A dimension declared `"builtin"` is judged with the engine's own
instructions: one for the transcript axis, one for the outcome axis, and one template a rubric dimension's name,
description and scoring guide are filled into. That wording is part of the judge, as an operator's prompt is, so it
has an identity. Each call records `builtin:<sha12>` in the result's `judge_config_ids`, where a configured call
records its config's id. The id is a fingerprint of the engine's wording only: a dimension's own description and
guide are its criterion, which a stored profile keys on separately, so editing them does not change it. Change a word
of the built-in wording, as a release may, and every dimension judged with it is judged by a new judge. Its scores
form their own agreement groups, tiers and drift readings, and no stored profile measured on the old wording is read
for it (`is_builtin_judge_config_id` tells a built-in id from a config's).

**So is the reply format a configured judge is sent.** The engine appends one reply-format contract to every
prompt, an operator's `prompt_template` included, so a configured judge reads its config and that suffix. A
configured call records `<config id>@<sha12>`: the config's id and a fingerprint of the suffix's wording for the
dimension's scale and artifact. A release that rewords the suffix makes every configured judge a new judge, as a
new config version would, though no config changed. `judge_config_of` reads the config's id back from either form.

A result whose recorded judge identity is not today's (a release has reworded the built-in prompt or the suffix
since it was judged) is not re-judged or repeated, because whether today's prompt is the judge that scored it cannot
be told. A second judge, a freeze into judge cases and a temperature comparison choose their own judge, so they are
unaffected.

The built-in instructions also say where the judge's own memory stands. Facts the evidence shows the candidate got
from its tools or sources, and the case material, outrank what the judge recalls. A claim is not to be called
fabricated or unsupported only because the judge cannot confirm it from memory, since a recent fact can be true and
newer than the judge's training. When the evidence neither shows where a claim came from nor lets the judge check
it, the judge says so, and answers "can't tell" if no honest score is left. This went in after an outcome judge
called a correct, sourced, post-cutoff answer "fabricated" while the grounding dimension on the same result scored
it 5, in a private host application (#1895). The rule is in the built-in instructions only, not in the reply-format
suffix, which would make every configured judge a new judge with it. A `JudgeConfig` that wants the rule states it
in its own `prompt_template`.

## Step 5: collect person ratings

A person reads results and scores the same dimension on the same scale. Rate without looking at the judge's
score first.

```python
from threetears.evals.run import list_results, rate_result, run_blocking

results = await run_blocking(host.blocking_executor, list_results, host.storage, summary.run_id, summary.scope_id)
for result in results:  # in practice: the results a person has read
    await run_blocking(
        host.blocking_executor, rate_result,
        host.storage, result_id=result.id, scope_id=summary.scope_id, rubric_dim="answer.grounded",
        rater="ana@example.com", rater_kind="person", score=1, reason="Every claim is in the policy.",
    )
```

`score` is 1 (pass) or 0 (fail) on `pass_fail`, and 1–5 on `ordinal`. Rating the same dimension of the same
result again replaces that rater's rating. Only `rater_kind="person"` counts as agreement with people. A rating
an agent wrote is listed as `rated_by_an_agent` and never pooled.

### A rating is bound to what was read, not only to the result

The judge stamps every score with a **label key** (`RubricScore.label_key`): a fingerprint of the evidence it read
and a fingerprint of the criterion it applied. `rate_result` copies the rated score's key onto the rating
(`CalibrationRating.output_fingerprint` and `criterion_fingerprint`). So the label holds for every judgement of a
byte-identical output on the same criterion: another run's result, a result under another judge, or a frozen judge
case.

- **The output** is the whole `JudgeEvidence` the kind rendered: `subject`, `case_material` and `artifact`,
  hashed verbatim (`fingerprint_judged_output`). One changed byte, whitespace included, is another output. The
  scenario intent, variation and goal outcomes are context the engine adds around the evidence, so they are not
  part of it.
- **The criterion** is the dimension's definition, not its id: its name, description, scale and scoring guide
  (`fingerprint_criterion`). Templates are edited in place, so an id outlives its wording. A label keyed by the id
  would follow an output to a dimension that now asks something else. When you reword a dimension or move it to
  another scale, you get a new criterion, and no earlier label claims it. The rubric axis is not part of the
  criterion. A reserved transcript or outcome axis is its id and its 1-5 scale.

To find the labels of an output you hold, build its key and read the store:

```python
from threetears.evals.schema import label_key_of

key = label_key_of(evidence, rubric_dim)  # a JudgeEvidence, and a RubricDim or reserved axis id
labels = await run_blocking(host.blocking_executor, host.storage.query_calibration_ratings, scope_id, label_key=key)
```

A score no key was computed for (one restated from a judge trial) has none, and neither does a rating of it. Such a
rating is read by its result alone.

### A rating carries what was rated, so it outlives the run

Ratings are durable: a core baseline reset deletes runs, results and traces, and keeps every rating. So `rate_result`
also copies what the rater read onto the rating (`CalibrationRating.rated`, a `RatedOutput`): the criterion as the
template worded it, the scenario intent, the variation, the goal outcomes and the `JudgeEvidence`. It rebuilds them
under the check a judge freeze uses (`reproducible_judge_inputs`), and the rating is refused unless their
fingerprints are its label key, so the content is provably what was read.

When that check refuses, the person's rating is still stored, with `rated_withheld` saying why: the run is still
being written, its template was edited after it ran, its trace stored no evidence, or the score carries no label
key. A warning is logged naming it. Such a label calibrates the judge while its result exists, and cannot outlive
it. A rating stored before ratings carried their content (by `.25`) reads with `rated_withheld` set to
`RATING_WRITTEN_BEFORE_CONTENT`. Budget storage at about one judged transcript per label.

## Step 6: read agreement: kappa

```python
from threetears.evals.analysis import judge_agreement

ratings = await run_blocking(
    host.blocking_executor, host.storage.query_calibration_ratings, summary.scope_id, run_id=summary.run_id
)
agreement = judge_agreement(ratings, results)
for dim in agreement.dimensions:
    print(dim.rubric_dim, dim.n, dim.results, dim.exact_agreement, dim.kappa, dim.agreement_interval)
```

`exact_agreement` is the share of pairs where judge and person gave the same score. **Kappa** is that agreement
net of what chance alone would give from each side's score distribution: 0 is chance, 1 is perfect.
`weighted_kappa` is the quadratic-weighted kappa, on 1–5 only. Kappa is `None` when every pair has one and the
same score: then it is undefined, not perfect. With several people, the judge is set against each person and the
kappas are pooled so that each distinct result weighs 1. `results` counts those distinct results, and the tier
floors count them too. `agreement.unpaired` names every rating that could not be paired, and why.

A rating pairs with its own result's score, and with every other result's score that carries the same label key.
A label enters each judge's agreement once. Within one judge group a rating pairs with its own result when that
result is in the group, and otherwise with the first matching result. A label found by both routes, or on several
identical outputs under one judge, is still one person's one answer, so it adds one pair and not one pair per copy.
A rating whose own result was deleted still pairs when its key reaches a result that was read. A campaign bundle
reads its member runs' ratings, plus any rating in the scope whose key matches a key its judge scores carry.

## Step 7: evidence tiers, and how many ratings they take

Every judged reading in a report carries an **evidence tier**. It is decided from confidence bounds on the two
agreements, never their point estimates:

| Tier | Shown when |
|---|---|
| `calibrated` | agreement with people: one-sided 95% lower bound ≥ 0.6, over at least 20 distinct results |
| `separation` | agreement with its own repeats (step 8): lower bound ≥ 0.8, over at least 120 distinct results |
| `incidental` | both measured over enough results, and both upper bounds below their bars |
| `undetermined` | not shown either way: too few results, or bounds straddling a bar |

The floors are minimums, not targets. In seeded simulation, the chance of earning the tier by distinct results:

| Tier | True agreement | 20 | 40 | 60 | 80 | 100 | 120 |
|---|---|---|---|---|---|---|---|
| `calibrated` | 0.9 | 14–45% | 57–86% | 71–98% | 81–99% | 90–100% | 94–100% |
| `calibrated` | 0.8 | 4–16% | 23–42% | 35–58% | 46–76% | 53–86% | 62–90% |
| `separation` | 0.95 | below floor | below floor | below floor | below floor | below floor | 70–98% |

**Plan on about 50 person-rated results to calibrate a good judge** (80 on a skewed 1–5 scale), **and 120
repeated results for separation.** A judge whose true self-agreement is 0.9 needs more than 200 repeats for an
80% chance. The method, and why these bounds:
[evidence tiers](reading-reports.md#how-far-a-judged-score-can-be-leaned-on-evidence-tiers).

Compute the tiers and read the sentence a report states:

```python
from threetears.evals.analysis import judge_evidence_tiers, judge_key, judge_self_agreement, tier_sentence

self_agreement = judge_self_agreement(results)  # empty until step 8 records repeats
judged = {judge_key(result, score.dim) for result in results for score in result.rubric_scores}
for tier in judge_evidence_tiers(agreement, self_agreement, judged):
    print(tier_sentence(tier))
```

With one person rating 24 results, some of them differently from the judge, it reads:

```text
answer.grounded (judge-model, built-in prompt builtin:c2362452be2f, temperature 0, request settings unconfirmed):
undetermined — agreement with people 0.4085 (bounds 0.05449 to 0.7088) over 24 pairs from 24 results, undecided —
the bounds straddle (bar 0.6 over at least 20 results) — about 35 more results would decide it if agreement holds;
with its own repeats not measured (bar 0.8 over at least 120 results) — needs 120 results.
```

**Pin a concrete model id for a judge, never an alias.** The served model, as the provider names it in its response,
is part of the judge's calibration key (`JudgeKey`), so a provider alias that quietly re-points to a new model under
the same name merges two judges' evidence into one.

**"Needs N more"** is the rest of the floor when there are too few results. When the bounds straddle the bar, it
says about how many more results would carry them clear if agreement held at its estimate. Tiers are flagged and
never hide a reading; a finding stands on the weakest tier among its rows.

## Step 7b: combine the ratings with the judge's scores

Ratings do more than decide a tier. Where people rated some of a cell's judged scores, the cell's judged reading
(`JudgedArm.prediction_powered` in the bundle, `JudgedReading.prediction_powered` on the frozen surface) carries a
**prediction-powered estimate** (Angelopoulos et al., 2023) beside the judge's own mean:

- the judge's mean over every counted score in the cell, plus
- the **rectifier**: the mean of person minus judge over the scores people rated, the judge's bias as people
  measured it.

The sum estimates the mean people would have given every result, and its interval stays valid however biased the
judge is. A judge that scores a point high makes its own interval confidently wrong; the rectifier moves the
estimate back by that point. The price is width: the closer the judge tracks people, the closer the interval is
to the judge-only one; the less it does, the closer it is to an interval on the ratings alone.

It never replaces the judge's figure. The judge's mean is what the judge said; this is what people would have
said, estimated. The code-only report's strata table prints it after the judge's figure, here for a judge biased 0.8 high over
a true mean of 3 (one draw of the simulation below):

```text
3.63 ± 0.1271 (n=90 over 30 cases); with people's ratings: 2.846 [2.566, 3.127] (rectifier -0.7839; 24 rated by people)
```

- **Which ratings count.** Exactly those agreement pairs (step 6): a person's rating, of a result read, on a
  dimension its judge scored on the same scale (`person_scores_by_result`). Several people on one result count
  as their mean. On a candidate failure the judge's score counts the scale floor, so the person's does too.
- **The interval** is clustered by case like every other: the analytic cluster-robust variance of the estimate
  (`prediction_powered_mean`), which carries the covariance between the judge's mean and the rectifier (the
  rated results are among those the judge's mean covers), read on t with the rated cases minus one degrees of
  freedom.
- **At least 10 rated results** (`PPI_MIN_LABELLED_RESULTS`). Below it the estimate reads `not available` and
  says how many were rated: the correction is estimated from the rated results alone, and below ten its spread is
  too uncertain to put an interval on.
- A cell nobody rated carries no estimate (`None`). Ratings are per cell, so 10 rated results per arm you want
  corrected, not 10 in all.

In seeded simulation, a judge biased by 0.8 on a 30-case × 3-repeat arm with 24 rated results: the
prediction-powered interval covered the true mean 95.2% of the time, the judge-only interval 0.15%
(`test_simulated_prediction_powered.py`).

## Step 8: judge repeats, for self-agreement

A repeat asks the same judge the same question again, from the evidence it first read, and records each answer
beside the score it repeats without changing it. On the quick path, lend the judge's client to the host:

```python
import dataclasses

from threetears.evals.kernel import JobLeases
from threetears.evals.run import JudgeRepeatReport, estimate_judge_repeat, judge_repeat_work, run_job_work

judging_host = dataclasses.replace(host, clients=judge.clients())
estimate = await estimate_judge_repeat(judging_host, summary.run_id, summary.scope_id, out_of_run_cap_usd=None)
print(estimate.max_calls, estimate.would_start)
work = await judge_repeat_work(judging_host, summary.run_id, summary.scope_id, out_of_run_cap_usd=None)
leases = JobLeases(judging_host.storage)
ended = await run_job_work(work, store=judging_host.storage, executor=None, leases=leases)
report = JudgeRepeatReport.model_validate(ended.result)
results = await run_blocking(host.blocking_executor, list_results, host.storage, summary.run_id, summary.scope_id)
self_agreement = judge_self_agreement(results)
```

A repeat runs as a job, as every model-calling operation does: `judge_repeat_work` makes the refusals and admits
the calls, and `run_job_work` runs it in the foreground (a transport starts it with `ops.judge_repeat` and polls it;
the job's result is the report).

Every call is priced and admitted against `out_of_run_cap_usd` before the first is sent. `None` enforces no cap.
Under a cap, a client that cannot price its calls (`price_ceiling` returns `None`, as a lent quick-path client
does) is refused before anything is spent. A "can't tell" on repeat counts as a disagreement. Separation counts
distinct results, so 120 means, for example, 40 cases at `k=3`, each repeated once.

## Step 9: a second judge, for agreement between judges and for drift

A second judge — another model, prompt or temperature — scores the same stored evidence, and each answer is
recorded beside the first score without changing it. Ask it about a seeded share of the results to see how far
another judge agrees, or about all of them after a judge change to see how far the scores moved:

```python
from threetears.evals.analysis import inter_judge_agreement, judge_drift
from threetears.evals.kernel import JobLeases
from threetears.evals.run import run_job_work, second_judge_work
from threetears.evals.schema import SecondJudge

work = await second_judge_work(       # every call priced and admitted here; none made
    judging_host, summary.run_id, summary.scope_id,
    judge=SecondJudge(model="other-judge-model"), out_of_run_cap_usd=None, sample_fraction=0.5, seed=1,
)
leases = JobLeases(judging_host.storage)
passed = await run_job_work(work, store=judging_host.storage, executor=None, leases=leases)  # the calls
await leases.shut_down()
results = await run_blocking(host.blocking_executor, list_results, host.storage, summary.run_id, summary.scope_id)
agreement = inter_judge_agreement(results, pass_id=passed.id)
drift = judge_drift(results, pass_id=passed.id)
```

The pass is a record of its own (`EvalSecondJudgePass`, its id the `pass_id` every paired score names), written as
it starts and as it ends with its report. The actions are `judge_second`, `judge_second_estimate` and
`judge_drift_check`, which re-scores every result; `judge_second` and `judge_drift_check` start the pass as a job
and `job_poll` returns its report, agreement and drift once it completes. Agreement is n, exact agreement, and kappa, quadratic-weighted
on 1–5 and unweighted on pass/fail, computed as agreement with people is. A kappa with nothing to measure is
reported as undefined, with the reason, never as 0. `run_get` shows each second judge beside the dimension it
scored.

Drift is read per dimension over cases: the movement, its interval, and `separated`, `not separated` or
`untested`. Each dimension is tested at 1 − α/m across the m dimensions (Bonferroni), and the verdict is read off
its interval: `separated` exactly when the interval excludes 0, so the two never disagree. Where every case moved by
one amount the bounded test reads both, which takes more cases than a t-test: with two dimensions read together,
twenty cases each up one point on 1–5 are not separated, thirty are. It shows that the scores moved, never which judge is right. A
campaign whose runs were judged differently names the change in the bundle's `judge_change`, and links any drift
reading that re-scored one side under the other side's judge.

Calls are priced against the out-of-run cap before the first is sent. They are ledgered under purpose
`second_judge` and never added to the candidate's cost.

**Comparing two candidates on judged quality:** list the `Judge` among the graders of an `Eval` over a mapping
of arms.

## Step 10: measure what temperature does to the judge

The temperature policy (Step 4) pins every judge call at `DEFAULT_JUDGE_TEMPERATURE` (0). Whether that matters for
your judge is a measurement: re-judge the same borderline evidence several times at 0 and several times at the
provider's default (no temperature sent), and compare how much the scores move. On a host with a cap, one command
prices it and then runs it:

```
python -m threetears.evals judge-temperature RUN --host myapp.evals:build_host --scope dev --max-cost-usd 5 --estimate
python -m threetears.evals judge-temperature RUN --host myapp.evals:build_host --scope dev --max-cost-usd 5 --json > temperature.json
```

In code it is two steps: `judge_temperature_work` (priced first by `estimate_judge_temperature_comparison`) from
`threetears.evals.run` admits the comparison and returns its work, which spends and records every answer on the
comparison's own record (`EvalJudgeTemperatureCheck`); `read_judge_temperatures` from `threetears.evals.analysis`
reads them, spending nothing. The actions are `judge_temperature`, which starts the work as a job whose `job_poll`
returns the comparison, and `judge_temperature_estimate`:

```python
from threetears.evals.analysis import read_judge_temperatures
from threetears.evals.kernel import JobLeases
from threetears.evals.run import judge_temperature_work, run_job_work

work = await judge_temperature_work(
    judging_host, summary.run_id, summary.scope_id, out_of_run_cap_usd=5.0, repeats=5,
)
leases = JobLeases(judging_host.storage)
check = await run_job_work(work, store=judging_host.storage, executor=None, leases=leases)
await leases.shut_down()
comparison = read_judge_temperatures(check.answers)
print(comparison.render())
```

**Which cases.** By default only borderline ones: a dimension whose stored score sits inside its scale (2-4 on 1-5),
or one a recorded judge repeat or second judge answered differently. Scores at the ends of the scale are the ones
temperature is least likely to move, so they are skipped and named. A pass/fail dimension is borderline only once
something has disagreed on it, so run a judge repeat (Step 8) first, or pass `selection="all"` (`--all`) to
re-judge every scored dimension. `result_ids` (`--result`) narrows it further.

**What is sent.** Each case is rebuilt from what its run recorded, the judge pin, prompt and evidence, as a repeat
is. Only the temperature differs. The two settings alternate call round by call round, so a provider changing
mid-measurement moves both alike. A config that pins its own temperature is overridden on both sides; its prompt and
model are kept.

**Reading it.** One row per dimension, the two settings side by side:

| Figure | What it says |
|---|---|
| cases | Cases with at least two answers at that setting |
| mean / max variance | Each case's score variance across its repeats, averaged and at its worst; 0 means one score every time |
| unstable | Cases whose answers were not all the same, a "can't tell" counting as an answer of its own |
| exact agreement, kappa | Each case's later answers paired with its first at that setting, read by `judge_self_agreement` exactly as Step 8's repeats are |

If the pinned side reads near-zero variance and the default side does not, pinning is buying consistency on exactly
the cases where it matters. If both read alike, temperature is not what moves this judge's scores. Either way, keep
the JSON: nothing is written to the results, because the forced temperatures are not the run's judge, and recording
them as repeats would split the run's own self-agreement.

**When it is not comparable.** Each answer records the temperature its client reports sending. A run whose every
borderline score records that its model was sent none (`model_default`, a model that refuses a temperature) is
refused before anything is spent, since both sides would be the same sampling. A client that drops the temperature
at call time, or reports none at all, gets `comparable: false` (`NOT COMPARABLE` in the text), with the reason.
Its off-setting scores are counted and left out of that side's figures, never read as the setting's.

**Cost.** Each case costs `2 × repeats` calls, plus parse retries. Every one is priced and admitted against the
out-of-run cap before the first is sent: the cap the command names with `--max-cost-usd` (`--no-cap` waives it out
loud), or the host's own for the action. Calls are ledgered under purpose `judge` and stamped with the run
(`python -m threetears.evals spend --purpose judge`). The comparison reports its case count, calls and cost.

**Run it only on a judge whose provider honours temperature,** and only through a client that reports the
temperature it actually sent, not the one it was asked for. To a model that takes no temperature, send none and
report none: every score then records `model_default`, and the harness refuses the run or marks it
`comparable: false` instead of comparing the default against itself. A router can drop an unsupported parameter
silently (OpenRouter does, unless the request sets `require_parameters`), so check the model's supported
parameters before trusting either side. Then check that the provider's default actually samples: it is the
endpoint's own default, and one serving the model greedily makes the comparison 0 against nearly 0. Pinning the
calls to one endpoint keeps a mix of providers from adding spread to both sides.

### The 2026-10-10 run is void

A run on 2026-10-10 in a private host application (#633) re-judged 50 borderline cases from seven finished runs on
`openai/gpt-6-luna` via OpenRouter, 5 times at 0 and 5 times at the default: 500 calls, $0.19. It reported mean
score variance 0.202 at 0 and 0.180 at the default, and was first recorded here as showing that 0 did not steady
the judge. It shows nothing about temperature. That model accepts no temperature, OpenRouter dropped it silently,
and the host's adapter reported each request's own temperature as the one sent. Both sides were the provider
default, and the near-identical variances are what two identical settings produce.

What its figures are is a same-setting baseline: two sets of 5 repeats of luna at its provider default on the same
50 cases (1-5 scales). Mean score variance was 0.202 and 0.180, 34 and 31 cases were unstable, and agreement with
each case's first answer was 56% and 69%. Two runs of one setting differing that much is the noise floor any real
comparison on such a judge has to clear. Paired case by case, the variance difference between the two was +0.02
(95% interval -0.03 to +0.08), 11 cases lower, 12 higher and 27 tied.

### The 2026-10-10 measurement on a judge that honours temperature

The same host re-ran the measurement the same day, on the same seven runs and the same 50 borderline cases (19
`__outcome__`, 23 `__transcript__`, 8 across four host dimensions, all on 1-5), 5 repeats at each setting.

**The judge.** `openai/gpt-oss-120b`, a reasoning model in the same price class as the runs' own judge. Its
OpenRouter catalogue entry lists `temperature` in `supported_parameters`, and so does the endpoint the calls were
pinned to. Every request carried `provider: {"require_parameters": true}`, so a provider that cannot apply a
parameter refuses the call rather than dropping it. A probe confirmed that it does: `openai/gpt-6-luna` at
temperature 0 was refused with "No endpoints found that can handle the requested parameters", where the void run
had been silently served. The client reported the temperature on the wire: 0 on every pinned call, and none on
every default call, whose requests carried no `temperature` field at all. All 500 calls were served by the pinned
endpoint and named `openai/gpt-oss-120b` in the response. Each kept the judge's request settings of the time (the
8192-token reasoning budget, JSON replies).

**The provider's default is the endpoint's, so check that it samples.** The first sweep was pinned to DeepInfra.
Its default turned out to be effectively greedy: an open-ended prompt answered identically 12 times out of 12 with
no temperature, as at 0, but 8 different ways out of 12 at temperature 1. Over the 50 cases that sweep read
variance 0.047 at 0 and 0.055 at the default, paired difference -0.01 (-0.05 to +0.03). That is a comparison of 0
with something close to 0, a second same-setting control and no evidence about temperature. A second endpoint
(CoreWeave, fp4) sampled at its default but corrupted the judge's JSON (criterion names such as `__outoutcome__`)
on most calls at both settings, so it was stopped after one run and its answers are not used. The measurement
is the third sweep, pinned to AkashML (bf16), whose default samples (7 different answers out of 8 on the probe
prompt) and whose temperature 0 does not (8 of 8 the same). It had no failed answers.

| Dimension | Cases | Variance at 0 | Variance at default | Unstable at 0 / default | Agreement with first answer at 0 / default |
|---|---|---|---|---|---|
| `__outcome__` | 19 | 0.12 | 0.30 | 5 / 7 | 91% / 76% |
| `__transcript__` | 23 | 0.45 | 0.88 | 14 / 16 | 64% / 58% |
| host dimensions | 8 | 0.10 | 0.19 | 1 / 1 | 94% / 91% |
| all | 50 | 0.27 | 0.55 | 20 / 24 | 79% / 70% |

Paired case by case, variance at 0 minus variance at the default was -0.28 (95% bootstrap interval -0.45 to
-0.12). 0 was lower on 21 cases, higher on 5 and tied on 24, a sign-test p of 0.002. On `__transcript__` alone it
was -0.43 (-0.68 to -0.17). On `__outcome__` (-0.17, -0.51 to +0.06) and on the host dimensions (-0.09, -0.26
to 0) the intervals touch 0. The mean score moved by -0.10 between the settings, so temperature changed how much
the scores scatter, not where they sit. Against the noise floor, two settings that are the same differed by
+0.02 (-0.03 to +0.08) on luna and -0.01 (-0.05 to +0.03) on the greedy endpoint. The difference here is about
five times either one, and its interval clears both.

**What it says.** Pinning the judge at 0 halves the spread of its scores on borderline cases. That is what
the policy is for, so the policy stays. It does not make a reasoning judge repeatable: at 0, 20 of the 50 cases
still gave more than one score in five tries, and on `__transcript__` only 64% of later answers matched the first.
Run judge repeats (Step 8) to see how much of that a given judge has. It also means a default is not one setting:
the same model sampled greedily by default on one endpoint and freely on another.

**Limits.** The cases were chosen as borderline by the runs' original luna scores. They are the cases luna found
hard, not the ones gpt-oss-120b finds hard, and gpt-oss-120b's scores sat on average more than a point from luna's.
The judge was swapped in the host's adapter: the harness replays each run's recorded judge pin, so the comparison
JSON still names luna as `judge_model`, while the ledger rows and every response name `openai/gpt-oss-120b`. One
model on one endpoint: another judge or provider can move differently, and these runs' JSON is the record of this
one.

**Cost.** 500 calls, $0.09 for the measurement. Everything that day on this judge, the two other sweeps, the
diagnostics and the probes included, cost $0.30, against a $5 budget sized so the out-of-run caps could never pass it.

## Evaluating a judge as a subject

Steps 6 to 10 measure a judge inside the campaign it scored. To compare judges directly (another model, another
prompt, another temperature), make the judge the subject of a campaign of its own. This is the **judge kind**
(`candidate_kind="judge"`):

- **The candidate is a judge configuration:** a model, the prompt per criterion and the temperature, the three
  things an arm chooses of what `JudgeKey` keys a judge by (the others are the criterion's dimension and scale, and
  the settings each call was sent with, which the client reports). The model is the arm's candidate model. The prompt per criterion
  (`config_ids`, versioned `JudgeConfig` ids) and the temperature are the arm's overlays (`JudgeKindOverlays`), so
  two arms that differ in their judge are two variants the campaign tells apart.
- **A case is frozen from a stored result:** one judged output and one criterion. It holds the evidence the
  result's judge read, the criterion as its template worded it, and as its labels the person ratings given on
  that result or on any result whose judged output is byte-identical on the same criterion (output-bound labels).
  A rating found both ways counts once, and one person's ratings of one output are one label: the latest.
- **A trial replays the stored output and calls only the judge.** It asks the criterion through the engine's own
  judge service, so the judge reads the same evidence block the first judge read. Only the judge differs. No
  candidate is re-run.
- **The grade is code.** Each trial lands `judge_parse_valid` and, on a scored trial of a labelled case,
  `judge_label_agreement` on its `host_measures`. The whole answer goes on its `kind_payload`.

A judge template is the package's, not the host's: the judge kind, a name and an intent
(`ops.judge_template_create`, the `judge_template_create` action), on a host that registers `launchable_judge_kind`.
Any other field on it is refused at the launch, since the judge kind never acts on one.

Freeze cases from a judged run into a judge template, and mint the case set a launch targets:

```python
from threetears.evals.kernel import JUDGE_KIND
from threetears.evals.run import freeze_judge_cases
from threetears.evals.schema import EvalTemplate

judge_template = EvalTemplate(
    scope_id=summary.scope_id, name="grounded judge", candidate_kind=JUDGE_KIND,
    intent="Score whether an answer is grounded in the store policy, as a person would.",
)
await run_blocking(host.blocking_executor, host.storage.save_template, judge_template)
frozen = await run_blocking(
    host.blocking_executor, freeze_judge_cases,
    host.storage, template=judge_template, run_ids=[summary.run_id], scope_id=summary.scope_id,
    dims=["answer.grounded"], case_set="grounded-judge",
)
```

Each case is rebuilt under the same check a re-judge uses (`reproducible_judge_inputs`). A result is skipped, with
the reason, if its run did not record its judging or its template was edited since. Freezing the same output,
criterion and labels again returns the stored case. A re-freeze after a label changed mints a new case, and the
new case-set version lists only the new one. The action is `judge_cases_freeze`.

**Freeze from the labels alone, across a core reset.** `freeze_judge_cases(..., from_labels=True)` (and
`from_labels: true` on the action, with no `judged_run_ids`) reads no run, result or trace. It groups the scope's
person ratings by label key, one case per output and criterion, and builds each case from the content its labels
carry. Its labels are every person rating on that key, one per rater, the latest standing. A label that carries no
content is skipped by name (`JudgeCaseSkip.rating_id`) with the reason the rating states. The case's source names
the label's run and result as provenance and records no first judge. So the calibration path after a core baseline
reset is: rate (before the reset), freeze from labels, run the judge campaign over the case set, then record its
profile. It costs judge calls only.

```python
frozen = await run_blocking(
    host.blocking_executor, freeze_judge_cases,
    host.storage, template=judge_template, scope_id=summary.scope_id, from_labels=True, case_set="grounded-judge",
)
```

### Seeding hard negatives

Labels on stored results only cover what your candidates happened to write. The first calibration items to write
are the ones they have not: a **hard negative**, an output that must fail a dimension though a lenient judge would
pass it, and an output that looks wrong but must pass. Write each as an **authored case**
(`AuthoredJudgeCase`): the output, what it is judged against, the criterion and scenario exactly as the subject's
template words them, and the verdict it must get, with your note on why. Its verdict is its one label, so a trial,
`judge_kind_readings` and a judge profile read it as they read a labelled case.

```python
from threetears.evals.kernel import AuthoredJudgeCase

grounded = judge.dims[0]  # the dimension as the run's template stores it: its wording keys the judge's profile
material = judge.case_material({"question": "Can I return worn shoes?"})
hard = [
    AuthoredJudgeCase(
        criterion=grounded, intent=summary.intent, case_material=material, output="Yes, within 60 days.",
        expected="fail", author="ana@example.com", note="60 days is in no policy: a confident, wrong number",
    ),
    AuthoredJudgeCase(
        criterion=grounded, intent=summary.intent, case_material=material, output="No.",
        expected="pass", author="ana@example.com", note="curt, but worn items are exactly what the policy excludes",
    ),
]
seeded = await run_blocking(
    host.blocking_executor, freeze_judge_cases,
    host.storage, template=judge_template, scope_id=summary.scope_id, authored=hard, case_set="grounded-hard",
)
```

`authored=` also rides beside `run_ids` or `from_labels`, so one case set can list both kinds. A verdict the
dimension's scale cannot hold (`"fail"` on an ordinal dimension, `7`, `3` on pass/fail) is refused when the case is
written, naming both. Through an agent, the `judge_case_author` action writes one case, naming the dimension by its
template (`criterion_template_id`, `criterion`) and recording the caller as its author. A host that ships its hard
negatives seeds them at boot instead: its definition seed corpus has a `judge_cases` slot (`SeedJudgeCase`, the
templates named by name), which `seed_eval_definitions` builds through the same door as `judge_case_author` and
writes only where the same case is not stored yet (the host guide, "What the judge reads, and conversations"). On the quick path, hand the
`Judge` its examples, and every judged eval stores them as authored cases of the judge template
`<template id>-judge`, listed by the case set of that name, calling no model:

```
Judge(client, [GROUNDED], case_material=...,
      examples=[Example("grounded", "Yes, within 60 days.", "fail", "60 days is in no policy", case=ticket)])
```

Run one arm per judge. Through a launch, a host registers `launchable_judge_kind` for `JUDGE_KIND` and declares
`JUDGE_KIND_CONTRACT` (and `JUDGE_KIND_MEASURES`) on its profile. Then each arm is a `run_launch` of the judge
template, with the judge's model and its `config_ids` and `temperature` overlays. A host driving the runner
directly builds each arm's kind with `judge_kind(...)`. Read the campaign out per judge and criterion:

```python
from threetears.evals.analysis import judge_kind_readings

readings = judge_kind_readings(results)  # every result of the campaign's runs
for reading in readings.readings:
    print(reading.key, reading.cases, reading.parse_validity.rate,
          reading.label_agreement and reading.label_agreement.kappa,
          reading.self_agreement and reading.self_agreement.kappa)
```

| Measure | Read from |
|---|---|
| agreement with the labels | each case's first scored trial against its labels, by `judge_agreement` (Step 6) |
| self-agreement | each case's later trials against its first, by `judge_self_agreement` (Step 8); run at `k_runs >= 2` |
| parse validity | the share of replies that kept the protocol (a score on the scale, or "can't tell") |

Both agreements count distinct **cases**, so the tier floors of Step 7 apply to them unchanged. A case repeated k
times counts once. Each reading names the cases it was measured on (`case_set_fingerprint`) and the criterion
wording (`criterion_digest`).

**Spend.** Every trial's spend is recorded on its result under the `judge` role. A judge campaign has no
`candidate` role in its spend. It is in-run spend under the run's cost cap, not the out-of-run ledger a judge
repeat or a second judge writes to.

### Storing a judge's profile for other campaigns

A judge campaign's readout lives in that campaign alone. Record it as a **judge profile**, so every campaign the
same judge scores can read it:

```python
from threetears.evals.ops import JudgeProfilesRecord, judge_profiles_record

recording = await run_blocking(
    host.blocking_executor, judge_profiles_record, host, JudgeProfilesRecord(judge_run_ids=arm_run_ids), scope_id
)
for entry in recording.profiles:
    print(entry.profile.rubric_dim, entry.profile.judge_config_id, entry.tier, entry.profile.measured_at)
```

The action is `judge_profiles_record`; `judge_profiles_list` lists what is stored. Recording is an explicit step,
never a write when a run completes, because which runs make up one measurement is your call: a campaign's arms are
several runs, and a run abandoned halfway should not overwrite a full measurement.

**What a profile holds.** One `EvalJudgeProfile` per judge and criterion. The judge is everything `JudgeKey` keys:
dim, scale, served model, config, temperature and the request settings its calls were sent with. The criterion is its wording (`criterion_digest`). The profile
stores both agreements as the tiers read them (figure, bounds, pairs and distinct cases), parse validity, the
number of cases, the case-set fingerprint, the runs, `measured_at` (the latest trial's `scored_at`) and
`recorded_at`. Recording again replaces the profile of the same judge and criterion, and the receipt says when the
replaced one was measured. A judge whose served model or temperature was not recorded is skipped, with the reason:
nothing could show that another campaign's judge is the same one.

**When a campaign reads a profile.** A campaign's own evidence about its judge comes first, because it was
measured on the campaign's own outputs. The campaign reads a profile in its place only when its own evidence is
*thinner*, which means both of these hold:

1. the campaign's own evidence decides no tier for that judge (its tier is `undetermined`), and
2. the profile decides one (`calibrated`, `separation` or `incidental`), by the same bars and floors. Distinct
   frozen cases count as distinct results.

A campaign whose own ratings or repeats decided a tier keeps it, even if a profile was measured on more cases. A
profile that decides nothing is not read.

**A tier read from a profile always says so.** The bundle's tier entry carries `from_profile`: the profile's id,
when it was measured, on how many cases (and their fingerprint), by which runs, and the campaign's own two criteria
it stood in for. Its `calibration` and `separation` are the profile's. The report's tier sentence reads "read from
the judge's stored profile ..., measured at ... on N frozen cases". The profile is part of the bundle, so a profile
re-recorded or newly read changes the bundle's fingerprint.

**A changed judge reads no profile.** A profile is found by an id derived from the whole judge and the criterion's
wording. Change the model, the prompt (a new `JudgeConfig` version, or a release that rewords the built-in prompt
or the reply-format suffix, whose recorded identity then changes), the temperature, the request settings its calls
were sent with (a client that starts reporting them included), or the rubric dimension's wording, and the campaign
looks for a profile that does not exist. The old one is stale and is never read for the
new judge. The wording comes from the template each run was judged against. A run with no template, or whose
template was edited after the run was created, has an unknown criterion and reads no profile.

## What to read next

- [Evidence tiers](reading-reports.md#how-far-a-judged-score-can-be-leaned-on-evidence-tiers): the bounds, the
  simulations behind the planning numbers, and how temperature enters a judge's identity.
- [Reading the guardrails](reading-reports.md#reading-the-guardrails): held, breached and undecided in a report.
- [Principles](principles.md): code checks facts, a judge assesses judgment, people check the judge.
- [Evaluating a tool-using agent](evaluating-agents.md): grading what an agent *did*, with no judge.
- `examples/2-replies/replies.py`: a judged run on a judge of another family, recorded with no person's labels,
  and how a person's labels calibrate it.
