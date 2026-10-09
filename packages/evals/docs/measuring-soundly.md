# Measuring soundly

Read this when you are about to design a campaign, choose k and a case count, pick a judge, or decide
whether to believe a surprising result. It collects what running evals measured while the engine was built
against one production host, July to October 2026: a conversational agent with tools, its three-label
classifier, a retrieval sub-agent it calls, and the engine's own analysis writer.

Each finding is a rule, its reason and an *Evidence:* line; "replicated" means two or more independent
campaigns or arm pairs showed it. Terms are as [Concepts](concepts.md) defines them.
**pass^k** is the share of cases that pass on every one of their k repeats.

## Variance, k and how many cases

**Measure the noise floor before reading any difference.** Re-run an unchanged arm and see how far its own
readings move. A smaller difference is not a finding. Stability tracks how mechanical a judged dimension is:
one anchored on an observable order reproduced exactly, and one judging "the right artefact" moved most.
*Evidence:* agent with tools and retrieval sub-agent, re-runs at n=12 per dimension and at k=1 then k=3, 2 campaigns, 2026-08 and 2026-09, 4 of 6 dimensions within 0.2 and 2 moving 0.5–0.7, latency moving 12%, replicated.

**Never rank on k=1.** The engine defaults to k=3.
*Evidence:* retrieval sub-agent, 6 models, k=1 then k=3 over 3 cases, 2026-07, one model's two leading dimensions fell from 3.3 to 1.4, single campaign.

**Add cases before repeats.** k measures consistency and cases measure coverage, so five cases at k=3 are not
fifteen independent draws. The engine's interval is computed over observations, so it is too narrow when
they cluster by case. It says so (`ResolvedReading.dispersion`), and you should read the case count. At n=5 a
paired test resolves only d_z ≈ 1.25, so "not significant" there says almost nothing. Size the bank from a
measured effect: for d_z ≈ 0.74 about 16 cases reach 80% power, and 32 cases detect about 0.50.
*Evidence:* agent with tools, two independent arm pairs at n=5, 2026-08, pass^k 0.2 → 0.8 gave p=0.174 and p=0.629, replicated; in simulation (2026-09) a 95% interval over observations from 5 cases × k=3 covered the truth 70.3% of the time, against 94.3% over case means.

**A small bank flatters.** Widening it lowers the score and improves the measurement. See
[How many cases](designing-classifier-evals.md#4-how-many-cases).
*Evidence:* three-label classifier, k=3, 8 then 20 cases, 2026-09, pass^k 0.875 → 0.750, the wider bank exposing a 5-of-5 failure class, single campaign.

**Pair by case.** Between-case variance dominates small banks, so compare per-case differences, as the engine
does where levels share cases. Two standard errors over five pairs is still loose: t with 4 degrees of
freedom needs about 2.8.
*Evidence:* retrieval sub-agent, 2 arms, n=5, k=3, 2 campaigns, 2026-09, pooled means showed nothing while paired differences gave −8.1 s at 2.9 se and then −8.7 s at 5.7 se; replicated for latency, though quality signs flipped between the campaigns.

**A bar on a mean over mixed cases measures the case mix.** Set bars per
[stratum](reading-reports.md#results-by-kind-of-case-strata) or over a homogeneous population.
*Evidence:* retrieval sub-agent, 5 cases, every arm, 2026-09, two cases scored 4–5 and three scored 3 under every configuration, so no arm cleared 4.0, single campaign.

**A bar decides on the interval, against a margin.** Seeded at the incumbent's mean and read on a cell's
mean, a bar fails the unchanged incumbent half the time. The engine seeds at the permissive end of the
incumbent's interval and misses a cell only when its whole interval falls short by more than the measure's
materiality threshold. A bar on a handful of cases then catches only a gross regression; read the interval
beside the verdict.
*Evidence:* seeded simulation (`tests/test_sim_bars_and_change.py`), σ = 1, margin 0.1σ, one observation per case, 2026-10: the mean rule missed an unchanged incumbent 49–50% of the time at n = 3, 6 and 15; the interval rule cleared it 99.7% of the time or more, and missed a candidate 1.6σ worse 86% of the time at n = 15, 26% at n = 6 and 4% at n = 3.

## Order and time

**Launch compared arms together, and never let running order follow the lever.** Throughput and load
drift over hours. The engine shuffles cell order within a run and discloses comparisons whose runs did not
overlap in time (`measurement_window_disclosure`).
*Evidence:* agent with tools, a 5-hour sweep run in lever order, 2026-08, Spearman ρ = 0.87 between running position and latency; retrieval sub-agent, 2026-09, throughput drifting 9–25 tok/s across an afternoon explained a ~205 s tail credited to a lever; replicated.

## Metrics that mislead

**Keep the pass^k conjunction small and reliable.** A trial passes only when every goal-state check passes
and every judged dimension clears the bar (`compute_pass_k`). So pass^k collapses to its weakest member, and
judge noise counts as candidate unreliability. Conjoin checks a competent candidate clears and dimensions
whose retest agreement you have measured. Ranking and diagnosis are two jobs.
*Evidence:* agent with tools, 2 campaigns, 2026-07 and 2026-08, one never-clearing dimension held pass^k to 0.0 or 0.2 for every arm, and a strict check set pinned it at 0 for every arm; in simulation (2026-09) a perfect candidate with five dimensions, each falsely failing 5% of the time, shows pass^3 ≈ 0.46.

**Pair quality with coverage and latency with a delivery rate, and read the floor.** An arm delivering
almost nothing has little to get wrong; one can be fast by declining the work. Rank latency on p95.
*Evidence:* retrieval sub-agent, 2026-07, an arm delivering almost nothing scored grounding and honesty 5.0, and one with a 69 s p95 declined 60% of hard cases; agent with tools, 2026-08, an incumbent averaging 4.3 had a minimum of 1 against a challenger's 5; 2 campaigns.

**Price per token is not cost.** Reasoning tokens bill at the output rate. Compare measured cost per
observation ([how a result's cost is counted](cost-and-budgets.md#how-a-results-cost-is-counted)).
*Evidence:* retrieval sub-agent, 2026-08, a model 25% cheaper per output token was 1.6× costlier and 1.8× slower per observation, single campaign.

**Rank on the measure at the lever's scope.** An end-to-end measure moves with everything, not only the lever.
Each measure carries an `attribution_scope` (`end_to_end` or `subsystem`). Report both and rank on the
subsystem's.
*Evidence:* agent with tools, 2026-07, a written analysis credited a sub-agent setting with halving end-to-end p95 (98.3 → 48.5 s) while the sub-agent's own p95 was flat (36.6 → 37.6 s), and recommended an arm failing 13% against 0%, single campaign.

## Did the lever move

**Assert the lever moved before reading the outcome.** A lever that never took effect reads like one that
changed nothing. Declare what it acts on
([mechanism checks](reading-reports.md#did-a-lever-take-effect-mechanism-checks-and-observed-mechanisms)).
*Evidence:* retrieval sub-agent, 3 arms, 2026-09, a reasoning effort was sent on the wire and the model returned 0 reasoning tokens in every arm, single campaign.

**A reasoning effort is a label, not a bound. Bound reasoning rather than raising the cap.** Each vendor maps
the word to its own budget, and the engine names a confound when two arms' reasoning shares differ by
`REASONING_SHARE_DIVERGENCE` (0.20). A provider refusing to turn reasoning off (HTTP 400) is a permanent rig
result, not a low score; never retry it as transient.
*Evidence:* three subjects, 2026-08 to 2026-10, replicated. A classifier parsed 3 of 6 at a 32-token cap with reasoning on, 6 of 6 with it off, and 6 of 6 at a 512 cap for 10–90× the tokens. At effort "low" one model spent 0.28–0.48 of its completion reasoning and another 0.68–1.00, losing cases to truncation. Turning reasoning off cut the agent's p95 time to first action from 163 s to 37 s on one model, while doubling the cap barely moved latency.

**Change one thing at a time; a judge change voids a before/after.** The engine never pools observations
across [apparatus classes](concepts.md#apparatus-class).
*Evidence:* agent with tools, 2026-07, a dimension rose 2.7 → 4.4 when a fix and a judge swap landed together, while the four dimensions judged identically moved between −0.3 and +0.5; ruled unanswerable, single campaign.

**Give every arm the same inputs by construction.** A key-wise overlay leaves keys it omits as they were. The
variation generator's default random source is unseeded, so subsets drawn per arm differ. Generate cases
once and run every arm over the stored cases.
*Evidence:* retrieval sub-agent, 2026-09, single campaign.

## Rig failures that score as candidate failures

Each defect below produced plausible numbers, some for months. A rig fault is
excluded, never scored ([Rig failures](adopting-a-host.md#rig-failures-a-broken-rig-costs-one-cell-never-the-run)).
The converse matters as much: **a swept lever's own failure is the outcome.** Excluding a swept sub-model's
timeouts as harness failures dropped 10 of 15 cells and left pass^k at 1.0 over 2 cases. The engine charges a
candidate for a turn truncated at its output cap, and never excludes a trial a failed check already decided.

- **The candidate was told something the seed contradicted.** A "you were just restarted" notice fired at
  every cell start, so for about four months quality numbers measured which model ignores a contradiction.
  Anything the candidate perceives must be seeded world.
- **Templates assumed a state the world did not show.** Six of seven described a system mid-task while the
  seed showed it idle; they measured the confound.
- **Tools were never started under eval.** Every call returned "not started", the judge scored grounding
  1/5, and a cassette recorded the failure as the tool's answer. An earlier probe had confirmed "third-party
  calls go uncounted", true only because none was made. When a probe confirms a predicted absence, demand one
  positive observation of the path running.
- **Muted actions succeeded where production refuses them** (2 of 199 failures). The fix: production's
  handlers on seeded production state, replacing only what leaves the process.
- **Goal checks counted refused calls.** One trial passed 7 of 7 checks with every call refused. The call
  ledger now records succeeded calls only.
- **An account limit failed candidate and judge alike,** and negative checks (`not any(...)`,
  `call_count(...) <= 1`) passed vacuously on empty conversations; 54 runs were void. An account refusal is
  now a rig failure, and a failed candidate fails every check.
- **The judge could not see results delivered asynchronously,** so it scored grounding 1 every time.
- **The eval measured a stub,** and the classifier scored 32.8% against a 33.3% chance floor. See
  [Feed the classifier exactly what production feeds it](designing-classifier-evals.md#5-feed-the-classifier-exactly-what-production-feeds-it).

*Evidence:* agent with tools and its classifier, 2026-05 to 2026-10, eight separate defects, each reproduced live.

**Prove a check can fail before trusting it passing.** A goal check giving the same verdict whether the
candidate acted or did nothing grades nothing. Authoring requires a control end state for every goal check
(`goal_check_controls`) and refuses a check that gives the same verdict on it and on the untouched seed; a
template saved straight to the store can lack controls, and its checks then read as unproven. Calibrate any script that reads verdicts on a known pass and a known fail, and classify on
reported counts, not exit codes: a run that never happened and a run that failed both exit non-zero.
*Evidence:* agent with tools, 2026-10, an audit found a hold check reading a parameter rather than its effect, and goal checks reading the static seed rather than the end state, single audit.

**Test failure paths, not more samples.** The defects that change verdicts live in the code deciding what an
incomplete run may contribute. The engine now discloses cells short of their intended repetitions, and gives
no cost band below 3 observations.
*Evidence:* agent with tools, 3 rounds, 2026-08, defects found rose 4 → 7 → 10; the failure-path round found a budget-stopped run flipping a frontier ranking and a ±1.5% cost band from 2 observations missing the actual by 13%; single program.

## Judges

**Choose a judge by self-consistency first, then agreement with people, and check the people's labels can
beat a constant.** Lenient judges compress toward the ceiling, where a high exact retest is not reliability.
The engine labels judged readings with an
[evidence tier](reading-reports.md#how-far-a-judged-score-can-be-leaned-on-evidence-tiers).
*Evidence:* analysis writer, 6 judges × 5 memos × k=3 × 7 dimensions, 2026-09: two judges ordered memos repeatably (self-ρ 0.80, 0.86), four were lenient (self-ρ −0.12 to 0.39), and a constant 3 landed in band on 29 of 32 human labels. Agent with tools, 2026-10: judges agreed with themselves 66% and 83% exact, and the more consistent sided with the code check at a fifth of the cost. The same judge won on both subjects; replicated.

**Give the judge the evidence the candidate acted on, and let code grade what code can check.**
*Evidence:* retrieval sub-agent, 2026-07, the judge called a tool-grounded answer "fabrication" while scoring its grounding 5, and its "parse errors" vanished once it saw the trace, single campaign.

**Calibrating a judge mostly repairs the apparatus.** Render what the judge reads as a reader sees it. A rule
folded into a composite 1–5 scale is outweighed by everything else; give it a narrow pass/fail dimension.
*Evidence:* analysis writer, 5 frozen cases, 3 calibration rounds, 2026-09: the judge scored the layout's field names, then charged re-assembly artifacts to the memo, then truncated as a reasoning judge and was re-billed; single campaign.

## The analysis writer

**The writer model, not the prompt's form, limits groundedness, and judge noise limits what you can tune.**
Subtract rubric criteria before adding them. Blind human ratings are a separate measurement: raters preferred a
cheap writer to an expensive one, scoring the expensive one too long.
*Evidence:* analysis writer, a latitude A/B with a pre-stated rule, a 2×2 form × input study and a blind rating by 3 raters over 4 campaigns, 2026-09: groundedness near 2 in every condition, the latitude arm failing its rule, judge test–retest noise (0.42–0.46) exceeding every effect ranked, and raters preferring the cheaper writer 6–2 with 1 tie; single campaign.

**Code renders every number, and the model's prose references them.** A deterministic gate over prose never
converges ([What the schema checks](reading-reports.md#what-the-schema-checks-and-what-only-the-model-does)).
Never build a prompt's worked example from real output: it fossilises that output's errors.
*Evidence:* analysis writer, 2026-07 to 2026-09, 10 of 10 checkable claims in two production memos were false, a prose fact gate grew to 4,099 lines in 33 days and was deleted, and an example taken from a real memo had its false claim reproduced near-verbatim; single program.

**Runs are cheap; the memo is the bill.** A refused generation is billed and stores nothing. Price analyses
like runs ([Spend outside any run](cost-and-budgets.md#spend-outside-any-run)).
*Evidence:* three campaigns, ≈ $34 in total, 2026-09: over 99% of the classifier campaign's spend was memo generation and ≈ $9.4 bought refused or discarded generations; only the multi-turn agent campaign's runs dominated (≈ $11 of $17).

## How an eval program drifts

From an audit of one eval program, each mechanism with the counter-measure adopted:

- **Every defect was answered with an addition** (2.74 lines inserted per line deleted), and removals were
  written down but never built. Name what each new mechanism replaces; delete at the third rework.
- **Locally right rules** ("fix what you find", "prove every guard") made every finding mandatory machinery,
  with no fixed point. Fix an apparatus defect now only if it would change the current decision or corrupt
  stored evidence, and cap review at two rounds.
- **The product was defined as prose,** the part that cannot converge. Code checks structure; an eval
  measures prose.
- **Generality was built for absent consumers.** The identity version moved 15 times and the bundle schema 24
  times in two months, each bump shortening stored evidence's life. Build for a consumer you can name.
- **Campaigns were scheduled to produce repairs, and "trusted enough" had no test.** No product decision
  reached production in five weeks while about 150 apparatus defects surfaced. Declare the decision, bar and
  inconclusive default before launch.
- **Nothing had a budget.** Set size ceilings that only fall, and spend and repair budgets per campaign.

*Evidence:* one eval program's commit history and rulings, about 1,900 commits over six months to 2026-09, single program.
