# Principles

Read this when you want the rules the engine keeps and why: before you design a campaign, extend the
engine, or argue with one of its refusals. Where the package does not yet keep a
rule in full, the entry says so and points at [open problems](open-problems.md). Terms are as
[concepts](concepts.md) defines them.

## Purpose and design

**Every eval declares its purpose before it runs: to decide, to learn, or to watch.**
A decision names what may change and what settles it. A campaign with no declared question is passive,
and its report leads with what the evidence taught. A watch re-runs a fixed suite against an earlier
measurement; its value is the regression it catches, so most of its runs change nothing.

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

## Numbers

**A failure of the rig is not a measurement.**
A rig fault excludes its cell and is counted beside the value, never scored as the subject failing; a
failure the candidate caused is the candidate's. Unknown is never shown as zero, and a cost nobody could
price is unpriced, not free. See [rig failures](adopting-a-host.md#rig-failures-a-broken-rig-costs-one-cell-never-the-run).

**Count test cases, not attempts.**
Five cases run three times are five pieces of evidence, not fifteen: repeats of one case are correlated.
Comparisons against the control run on per-case means, paired when both arms ran the same cases. A single
reading's interval is still computed over trials and says "interval too narrow" instead
([open problems](open-problems.md)).

**A verdict comes from a corrected test, and "not separated" never means "no difference".**
Each contrast against the control is `improved`, `regressed`, `not_separated` or `untested`, read off a
Holm-adjusted p within one family: one per declared question, or one campaign-wide. Twenty uncorrected
tests find a chance "winner" more often than not. False-discovery control across a history of campaigns and
sequential testing stay out. The two-run change helper still calls a sub-threshold move `flat`
([open problems](open-problems.md)); the figures behind these rules are in
[measuring soundly](measuring-soundly.md).

**Trust in a judge is measured, not asserted, and nothing waits for calibration.**
Code assigns every judged reading an [evidence tier](reading-reports.md#how-far-a-judged-score-can-be-leaned-on-evidence-tiers)
from its judge's measured agreement with people and with itself. A comparison between arms can stand on a
lower tier, since bias mostly cancels across arms; an absolute claim needs agreement with people, since
self-consistency measures precision, not accuracy.

**Bars start where the incumbent performs ("never ship worse than what runs today") and only tighten.**
A proposed bar is never adopted
automatically, and one looser than the registered bar is refused. The engine still seeds a proposal from
the incumbent's mean and clears a bar on a cell's mean; the intended rule is the interval against a
declared margin, since a bar at the mean fails an unchanged incumbent about half the time
([open problems](open-problems.md)).

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
label. The analysis writer has a kind, and a judge's self-agreement is measured by repeating its scores. Testing the statistics
against simulated data with known answers is not yet in the suite ([open problems](open-problems.md)).

**A run's memory should scale with its matrix, never with how much a cell produced.**
How talkative a candidate is should not decide whether a run survives. Nothing measures this yet.
