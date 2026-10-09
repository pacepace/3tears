# Principles

**For** anyone about to design a campaign, extend the engine, or argue with one of its refusals. **Answers:**
which rules the engine keeps, and why. Where the package does not yet keep a
rule in full, the entry says so and points at [open problems](open-problems.md). Terms are as
[concepts](concepts.md) defines them.

## Purpose and design

**Every eval declares its purpose before it runs: to decide, to learn, or to watch.**
A decision names what may change and what settles it. A campaign with no declared question is passive,
and its report leads with what the evidence taught. A watch re-runs a fixed suite against an earlier
measurement; its value is the regression it catches, so most of its runs change nothing.

**What nobody asked about is a lead, not an answer.**
A reading no declared question names is labelled exploratory wherever a reader meets it, so a report
cannot lead with it as if it were confirmed; a campaign that declares no question says once that every
finding is exploratory, rather than on every row, where the label would be skipped.

**A report shows tradeoffs, not a winner.**
Every arm is placed on quality, reliability, cost and latency, and called best only against a declared
bar or question: "best" with no axis named is a preference passed off as a finding.

**Test the real system, intercept only what would reach the outside world, and simulate only what cannot
be real.**
A stand-in tool measures the stand-in. A kind calls your production code; look-ups run live or replay
from a [cassette](adopting-a-host.md#cassettes-recording-and-replaying-tools).

**Average only what is provably comparable.**
An arm's identity is a digest of every setting it ran with, defaults included, so a shared name never
merges two configurations. Observations pool only within one [cell](concepts.md#cell): one arm under one
rig. Observations whose rig was set beforehand (commissioned) never pool with ones whose rig was recorded
as found (witnessed), because only the first can show cause.

**Whatever the subject perceives is seeded world, so a case reads the same every run.**
A seed carries every field production shows, with production's types. State the rig cannot seed is
recorded as witnessed rather than left to chance. See [world model](world-model.md).

## Grading

**Code checks facts, a judge assesses judgment, and people check the judge; the three never swap.**
Code never grades a model's prose: a deterministic gate over free text never converges, because each
generation finds a phrasing it does not parse. A judge never grades a count or order code can check, and a
prompt never coaches toward a grader's words, which measures obedience.

**Criteria come from observed failures, and judges follow standard practice.**
Write criteria from failures seen in real transcripts, test each behaviour in both directions, and prefer
pass/fail where a behaviour allows it: a binary criterion is easier for people to label and to agree on.
The engine asks one criterion per call, takes the judge's reasoning before its score, and treats "can't
tell" as an answer that excludes the trial from that criterion. Pairwise judging is not adopted for now
([open problems](open-problems.md#pairwise-judging-declined-for-now)). Several judges per criterion is not
adopted either: the error reduction claimed for it rests on a single study, and it would multiply judge
spend and change how a trial's identity is computed.

**What the subject must never do is held, never traded.**
A guardrail (a boundary judged dimension, or a measure declared one) stays out of every composite and
comparison family and is decided per arm against the control: a breached one keeps the arm from adoption
whatever it gained, and an undecided one is never read as safe.

## Numbers

**A failure of the rig is not a measurement.**
A rig fault excludes its cell and is counted beside the value, never scored as the subject failing; a
failure the candidate caused is the candidate's. Unknown is never shown as zero, and a cost nobody could
price is unpriced, not free. See [rig failures](adopting-a-host.md#rig-failures-a-broken-rig-costs-one-cell-never-the-run).

**Count test cases, not attempts.**
Five cases run three times are five pieces of evidence, not fifteen: repeats of one case are correlated. So
comparisons run on per-case means, every interval is computed over cases, and a reading of one case repeated
has no interval.

**A verdict comes from a corrected test, and "not separated" never means "no difference".**
Each contrast against the control is `improved`, `regressed`, `equivalent`, `not_separated` or
`untested`, read off a Holm-adjusted p within one family: one per declared question, or one campaign-wide
over every reading on a merit axis (the rig's own readings, such as the judge's time and spend, are on none).
`equivalent` takes an equivalence test (TOST) against the measure's declared margin, corrected in the same
family; the run-history read follows the same rule. Twenty uncorrected tests find a chance "winner" more often
than not. False-discovery control across campaigns and sequential testing stay out. The methods are named in
[reading reports](reading-reports.md#methods).

**Trust in a judge is measured, not asserted, and nothing waits for calibration.**
Code assigns every judged reading an [evidence tier](reading-reports.md#how-far-a-judged-score-can-be-leaned-on-evidence-tiers)
from its judge's measured agreement with people and with itself. A comparison between arms can stand on a
lower tier, since bias mostly cancels across arms; an absolute claim needs agreement with people, since
self-consistency measures precision, not accuracy.

**Bars start where the incumbent performs ("never ship worse than what runs today") and only tighten.**
A proposed bar is never adopted
automatically, and one looser than the registered bar is refused. A bar decides on the interval against the
measure's declared margin, never the mean: cleared, missed, or undecided when the interval straddles the line,
which is neither a pass nor a failure. A bar at the mean, read on a cell's mean, failed an unchanged incumbent
about half the time.

**What an arm would cost in production and what it cost to measure are kept apart.**
Spend is recorded per role: the candidate's calls and the work they start are what production would pay;
judging, simulation and spend outside any run are measurement ([cost and budgets](cost-and-budgets.md)). A
total over several roles says which.

## Reports

**Code states the facts, the model interprets, and an eval grades the interpretation.**
The writer cites figures as references that code resolves against the frozen bundle, and confidence is a
closed tier, so no field can hold a model's probability. No check reads the prose itself, so a figure typed
into a sentence still passes: this rule is kept for references, not for every number. The schema
checks structure and nothing else ([what the schema checks](reading-reports.md#what-the-schema-checks-and-what-only-the-model-does)).
Whether the prose is accurate and useful is measured by evaluating the analysis writer as a subject. "We
could not answer this, because X, and fixing X costs Y" is a legitimate bottom line.

## The engine itself

**One generic engine; the host owns the vocabulary.**
The engine never branches on one host's shapes or names; a kind is dispatched at exactly one site, and
tests hold host words in engine source to a ceiling that only falls. Each concept has one name and one
implementation, so two screens cannot disagree about a p95.

**Engine code must be needed by something the engine does, and a fix removes the cause before it adds
anything.**
A defect a campaign exposes arrives in one host's vocabulary, so a fix lands in the engine only with a
failing host-neutral test. A new caveat, shim or exemption is not a fix, and a part reworked three times
starts its next plan from what to delete.

**Contracts are strict, and a change to one is a version bump.**
Unknown fields are refused on read and construction, and identity changes bump a version, because tolerant
readers hide drift until it corrupts a comparison.

**Statistical rigour, not enterprise plumbing.**
Sized for one operator: rigour earns its cost by changing decisions, while multi-tenant access control
would govern users that do not exist.

**The eval system is meant to evaluate itself.**
Its model-driven parts are subjects, and every chain of evaluation should end in a code check or a human
label. The analysis writer has a kind, and a judge's self-agreement is measured by repeating its scores. The statistics
are tested against seeded data with a known truth at the sample sizes the engine sees (2–15 cases, 1–5 repeats):
the `test_simulated_*` and `test_sim_*` files in `tests/`, on the generators and reference answers in
`tests/simulation_support.py`. They check interval coverage, false-positive rates, power, family-wise error and
estimator bias, and each property the engine misses stays in the suite as a strict `xfail` stating the measured rate
against the nominal.

**A run's memory should scale with its matrix, never with how much a cell produced.**
How talkative a candidate is should not decide whether a run survives. Nothing measures this yet.
