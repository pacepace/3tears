# Design: principles, rationale and prior art

**For** anyone about to extend the engine, argue with one of its refusals, or place it among other tools. **Answers:**
which rules the engine keeps and why ([principles](#principles)); for each measurement concept, the problem that
forced its shape, the choice, and what was rejected ([rationale](#rationale)); and which published practice each part
follows or departs from, with the industry's words mapped onto the engine's ([prior art](#prior-art)). Where the
package does not yet keep a rule in full, the entry says so and points at [open problems](open-problems.md). Terms
are as [Concepts](concepts.md) defines them. The rationale and the Evidence lines are practice notes from building
against one private host; the campaigns they summarise are not published.

## Principles

### Purpose and design

**Every eval declares its purpose before it runs: to decide, to learn, or to watch.**
A decision names what may change and what settles it, the latter as typed clauses (`Question.decision`): a rule
read out of words is settled by whatever a parser made of them. A campaign with no declared question is passive,
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
recorded as witnessed rather than left to chance. See [the world model](evaluating-agents.md#why-a-world-the-world-model).

### Grading

**Code checks facts, a judge assesses judgment, and people check the judge; the three never swap.**
Code never grades a model's prose: a deterministic gate over free text never converges, because each
generation finds a phrasing it does not parse. The verdict and every count on a report are code's, a recommendation
names the findings it rests on, and a null claim in a finding's title is checked against that finding's evidence
structurally; only whether a title uses one of a few null phrasings is read off its words
([Having a model write the analysis](reading-reports.md#having-a-model-write-the-analysis-over-frozen-evidence)).
A judge never grades a count or order code can check, and a prompt never coaches toward a grader's words, which
measures obedience.

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

### Numbers

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
automatically (the `bars_propose` action proposes them from a single-cell baseline campaign and registers
nothing; a person writes the bar into the host's registrations), and one looser than the registered bar is refused. A bar decides on the interval against the
measure's declared margin, never the mean: cleared, missed, or undecided when the interval straddles the line,
which is neither a pass nor a failure. A bar at the mean, read on a cell's mean, failed an unchanged incumbent
about half the time.

**What an arm would cost in production and what it cost to measure are kept apart.**
Spend is recorded per role: the candidate's calls and the work they start are what production would pay;
judging, simulation and spend outside any run are measurement ([cost and budgets](cost-and-budgets.md)). A
total over several roles says which.

### Reports

**Code states the facts, the model interprets, and an eval grades the interpretation.**
The writer cites figures as references that code resolves against the frozen bundle, and confidence is a
closed tier, so no field can hold a model's probability. No check reads the prose itself, so a figure typed
into a sentence still passes: this rule is kept for references, not for every number. The schema
checks structure and nothing else ([what the schema checks](reading-reports.md#having-a-model-write-the-analysis-over-frozen-evidence)).
Whether the prose is accurate and useful is measured by evaluating the analysis writer as a subject. "We
could not answer this, because X, and fixing X costs Y" is a legitimate bottom line.

**The first screen carries the result and every fact that changes how it reads; the rest is one step down.**
The title, the declared question quoted whole, the design, the verdict and the scoreboard come first; contrasts
follow as one table, and cases, trials, what was not tested, the methods and the glossary start folded. A figure
is stated in the reader's terms (a rate as a percent, its change in points) with its interval beside it, and the
only picture besides the outcome bar is each interval drawn in the row that states it
([the page](reading-reports.md#the-report-one-page-result-first)).

### The engine itself

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
readers hide drift until it corrupts a comparison. The one tolerance is scoped to where a tested contract makes
the unknown safe: a newer minor of the evidence core, additive by rule, whose fields are kept untouched and
written back rather than read.

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

## Rationale

### Who owns the vocabulary: levers, apparatus and labels

**Problem.** The engine began inside its first host, and an import-boundary test kept it from importing
host code. That test stayed green while the host's object model shipped inside the engine as
*vocabulary*: of 19 registered inputs, six named host concepts, so a second host would inherit six
permanently empty entries. Import coupling and concept coupling are different failures, and only one had
a test.

**Choice.** The engine keeps the roles ([lever](concepts.md#lever),
[apparatus](concepts.md#apparatus-rig), [label](concepts.md#label-sweepable-role)), the
same/differs/unknown algebra and the confound scan; the host registers its own
[sweepables](concepts.md#sweepable). What stays shared passes one test: would a second LLM product have
this concept? Every product has a model, a judge, a simulator and a cost ceiling, so those are
`SHARED_CORE`. Every product has something that shapes generation, and no two share its shape. The slot is
shared; the shape is the host's.

**Rejected: opaque inputs.** An engine whose inputs are bare hashes writes analyses saying "component 3
moved". So a registration carries prose, and apparatus without `confounds` prose is refused: "a bare name
is a label". The same rule applies one level out: an engine-owned field meaning repeats to one consumer
and a retrieval window to another became `intended_repetitions`, and the engine's caveat kinds are open
to host additions.

### Identity is content, with a rendering

**Problem.** Every component of the [variant key](concepts.md#variant-and-variant-key) was hashed as
content except one, a map of slot to preset *name*. Operators could edit presets, so two runs naming one
preset across an edit shared a key over different text: a wrong merge, the one outcome nothing downstream
can undo.

**Choice.** A swept value is identified by the bytes that entered the run, never by a key into a store the
engine cannot see. **Rejected:** widening the name-keyed path to reach more components, which would add a
second host-shaped mechanism. A second product, built independently, had made the same split: content-hashed
prompt blocks, everything else by name.

**What changed.** A hash is a correct identity and an unreadable label, so a level became
`SweepableValue{content_hash, display, scale, raw?}` with `display` always present: only a value built from
opaque bytes with no name given falls back to its digest's prefix (`sha256:<12>`), a poor label but an honest one.
`scale` is nominal, ordinal or interval, not a bare rank, because a lever swept at 0.1 / 0.4 / 0.85 drawn
as ranks 1 / 2 / 3 puts the knee in the wrong place.

Rules that came with it (`kernel/identity.py`):

- **Hash resolved config, never the request.** A launch parameter of `None` meaning "role default" would
  compare equal across a change to that default. Whether a value was named or inherited is recorded, never
  hashed.
- **Never hash one input into both keys.** The variant key holds the contestant and the
  [context key](concepts.md#measurement-context) the conditions; a lever in the context key would report a
  deliberate sweep as not comparable.
- **A predicate change is a version bump.** `IDENTITY_VERSION` is stored beside every key, golden-vector
  tests pin the digests, and each key is stored with its pre-image, so a bump costs comparability and
  never the record of what was measured.
- **Reads never backfill.** A key recovered from an old run comes back marked derived. Tokens, cost or
  covariates not captured when a trial ran are lost: filling them in later would be fabrication. So
  capture is designed first, from the questions the data must answer.

### The arm is the unit

**Problem.** One launch could put several models in one run, and the analysis keyed cells on arms but its
design lenses on runs. So the writer read a balanced design as confounded, recommended re-running finished
work, and recorded the false premise as an insight. The cleanest design was the one it narrated worst.

**Choice (2026-09).** An [arm](concepts.md#arm) is the unit of comparison and a [run](concepts.md#run)
carries exactly one. A multi-model launch expands into one run per model in one group; runs of one arm
pool into its repeats; the analysis refuses a stored run carrying several models. The candidate kind is a
core lever, because two kinds with no overlays set had pooled into one arm.

**Deferred: interleaving arms inside one run.** Arms launched together at `k=3` share drift and already
pair by their frozen cases; interleaving waits for an analysis showing drift that overlap cannot answer.
Within a run, cell order is shuffled from the run's id, because running order was otherwise fully
confounded with configuration.

*Evidence:* agent with tools, 5-hour sweep run in lever order while host performance degraded, latency vs running position Spearman ρ = 0.87, latency doubled first arm to last, 2026-08, single campaign.

### What pools with what

**Problem.** Apparatus varies *within* a run (prompt-cache state alone moved cost up to tenfold between a
run's first and second repeat), so a run-level record of the rig is false for some of its trials.

**Choice.** Apparatus provenance attaches to the observation, and an analysis
[cell](concepts.md#cell) is `(variant_key, apparatus_class_id)`. Each merge rule refuses in the safe
direction (`analysis/cells.py`):

- **Recorded at two values: no merge.** The dimension is a rival explanation; the reason is stated.
- **Recorded on one side only: no merge.** `unknown` is neither agreement nor difference. The refusal says
  what recording would unblock it and how many observations it would pool.
- **Commissioned and witnessed never pool.** A rig set before the fact is an experiment; one found after
  it is a log. Pooling them reports an experimental `k` over partly observational evidence.

**Seats.** A host with no judge or simulator left four core dimensions blank, and every bundle reported
them as undecided confounds and split cells over them. Sentinel values were impossible, since the host had
no reader for them. The answer was declared inapplicability, per kind ([seats](concepts.md#seat)),
because one host grades one kind with a judge and another with code. The default, `None`, holds a kind to
every dimension: declaring a real gap inapplicable quietly switches a detector off.

### Confounds qualify, never suppress

A confounded comparison is often still worth reading; it may not credit the movement to the named lever
alone. Rules told to the writer in prose were broken (inferring "no effect" from overlapping intervals was
forbidden in capitals and done anyway), so each rule below is computed into the bundle.

- **A lever seen twice is not a confound.** A knob and the resolved surface it lands in move together, so
  every sweep of the knob read as confounded by itself. A knob now names its surface (`ResolvesInto`), and the
  engine folds the surface into the knob where the runs show it moved with the knob alone. Where one arm ran each
  knob level, a host reader of the surface with the knob taken out checks the fold; a fold neither two arms nor
  such a reader could test is disclosed as `unverified_fold`.
- **Factors that moved together are one group.** Factors that split the runs identically are reported once,
  as `aliased_factors` ("these four move together across all 22 runs; no comparison separates them"). Every
  pair of varying factors is checked, and a pair that co-varies outside a group gets a pivot whose unrun
  combinations read `not_run`; `factor_pairs` says how many pairs were examined. Aliasing with an
  interaction (C = A⊕B) is not checked, and the bundle says so.
- **Assert the knob moved before reading the outcome.** A lever names the measure it acts on, and the
  bundle reports it `moved`, `inert` or `unchecked`: see
  [Did a lever take effect](reading-reports.md#did-a-lever-take-effect-mechanism-checks-and-observed-mechanisms).

  *Evidence:* agent with tools, reasoning-effort sweep, setting confirmed sent, 0 reasoning tokens in every arm, 2026-09, single campaign.
- **Significance has three states:** separated, `not_separated`, and `untested`: no test could decide
  (fewer than two cases a side, or an arm that took no turn). The frontier's dominance and cheapest-pick read
  the same three, and so does the history's step-to-step change flag. A gap with zero spread (every case moved
  by one amount) is read by the bounded test on the measure's declared range; with no range it is
  `not_separated` and says to declare one, since the exact sign-flip test tests symmetry, not the mean. Zero
  spread is decided on each value's written decimal, so a float residue never passes for a spread.
  Collapsing the last two retires a lever nobody tested. A fourth, `equivalent`, is a claim of its own: an equivalence test against a declared margin.
- **Subtract a part from a whole only under declared containment** (`MetricDescriptor.contained_by`) and
  when the declared parts exhaust it. One remainder of about 95 seconds described no wall-clock at all,
  because the part ran as detached background work.
- **Rank on the scope under test; report both.** A subsystem change was credited with halving an
  end-to-end latency its own time did not touch, and the losing arm was recommended. Measures carry a
  scope; the bundle pairs end-to-end and subsystem measures by unit and tests whether the whole moved by
  a different amount than the part: each case's whole minus its part, compared between the levels
  (paired over shared cases, Holm-corrected within the lever). Grading each movement alone and reporting
  where the verdicts differed published a divergence that did not exist 11–33% of the time, because "the
  whole moved, the part did not" is not a test of the difference. Each movement still reads `improved`,
  `regressed`, `not_separated` or, against a declared margin, `equivalent`; none reads "did not move".

  *Evidence:* agent with tools, turn p95 98.3 s → 48.5 s against subsystem p95 36.6 s → 37.6 s, 2026-07, single campaign.

### Guardrails are a pillar apart from capability

A single score lets a capability gain pay for a guardrail loss, and the sum reads as progress. Online
experimentation keeps guardrail metrics apart from the criterion it optimizes (Kohavi, Tang & Xu, 2020),
Ng calls them satisficing against optimizing metrics, and Anthropic's agent-eval guidance tests where a
behaviour should and should not occur ([prior art](#methodology-and-statistics)). So a
boundary dimension or a guardrail measure enters no composite and no comparison family, and is decided per
arm against the control by the bar rule read on the difference; a breach refuses the adoption, and an
undecided guardrail is stated on it rather than blocking it, because at a few cases with no margin almost
every guardrail is undecided.

### A failure is charged to whoever caused it

Five steps, each fixing the last:

1. An errored trial scored 0, so a candidate could not hide failures by erroring; excluding it was
   rejected because it diverges denominators.
2. That floored candidates for the rig's faults (judge, simulator, factory). Now an infra failure is
   excluded from both the pass^k denominator and the mean: excluded is not pass and not fail.
3. A provider refusing the *calling account* (auth, payment) was charged to every candidate, and negative
   goal checks ("no call of this kind") passed vacuously on empty conversations. Account refusals are now
   the rig's, and a candidate failure counts every goal-state check as failed.
4. A turn ended by the output cap fails, and a trial a failed check already decided is never excluded
   because the judge could not tell. When the swept lever is a model, its own timeouts are the outcome.
5. An arm whose every call was refused showed the refusals' round trip as its latency and their empty bill as
   its cost. A failure still counts against its arm in every rate, accuracy included, but latency and cost
   read only over results that delivered a turn, and an arm with none reads "no successful results". A kind
   reports how many turns it delivered, so a conversation that failed on its sixth turn keeps the first five.

*Evidence:* agent with tools, account spend limit reached mid-campaign, 54 runs voided, 2026-10, single campaign.

### The subject key, and evaluability

**Subject key.** Identity keyed on a display name split a renamed subject and merged two same-named ones. So a
[subject](concepts.md#subject)'s key and label are separate required fields, the engine warns on one key under
two labels or the reverse, and a pooled score states its dimension basis. "Never pool across subjects" was
rejected, because it would forbid comparing candidates that share one declared rubric.

**Evaluability.** One campaign hit four independent failures, each found by spending a run: a scenario the
world could not represent, a fix target no run could vary, an eval calling a stub instead of the
production path, and a decision logged where nothing ingested it.

- **Derive the map; never maintain a second list.** The sweepables registry is the controllability map,
  and campaign authoring refuses an unregistered axis. The world registry is
  the representability map.
- **The unit is a precondition, never an area.** The scenario archived as unrepresentable was later
  measured another way; "this area: unsupported" would have discouraged that probe.
- **Fidelity is a test, never a map entry.** One constructor builds the payload for production and eval,
  a `FidelityContract` checks each caller's source reaches it, and a boundary-equality test asserts both
  payloads are identical. That test is the load-bearing part; a registry is a checklist.
- **Observability is derived, never checked** (2026-09): the measures registry is the observability map,
  but its coverage check was deleted, because rubric dimensions and goal checks mint measure names no
  registry enumerates, so the check called every honest name "uncovered".

### The analysis: from gated prose to an evaluated document

**Prose gates did not converge.** A deterministic gate checking the numbers in the
[analysis](concepts.md#analysis)'s sentences grew to 4,099 lines in 33 days and was deleted, because every real
generation found a claim shape it could not parse. The writer's prompt grew 12× in 23 days, mostly defence
against past incidents, and a test asserting prompt text passed while the writer did what the text forbade.

**Code renders the numbers.** The writer is asked to cite every figure as a reference,
`{{c1|<measure>|<reading>|<stat>}}`, and code substitutes the value from the decision surface
(`analysis/prose_refs.py`). A reference that does not resolve is refused, with one repair round naming
what exists; the words around it are never read, so a figure typed into a sentence passes unchecked. The grammar
is appended from the parser's constants, because a copy in a stored prompt went stale.

**Typed shape, freeform content** (2026-09-25). A strict schema derived from the stored analysis was
refused by some providers as a grammar too large. The authored document (`kernel/authored.py`) types
only its parts (findings, decisions, question answers, next steps), their order, links by position and
the readings code fills, under a measured slot budget; everything said is prose. The evidence tier is read
off the readings a finding names, never chosen by the writer.

**The writer is graded by its own eval.** The reporter is a candidate kind (`analysis/reporter_kind.py`):
a case is a frozen campaign bundle, and the judge reads what the writer read and the memo as written. Code
checks the structure of model output and never its prose; a judge never grades a fact code can check.

**The first screen is code's, not the memo's** (2026-10). A headline said halving a token budget "does not change"
latency, cost or delivery, over a contrast that did not separate at five cases; another memo headlined "keep the
current model — adopted" over a run whose judge had been cut off at its output cap on ten of the challenger's twelve
trials, while the word "truncated" appeared nowhere a reader would look. Most of what a reader needs is countable, so
the report's first screen — the basis, the verdict and each arm's census of passed, failed and lost trials, by cause
— is written by code from the bundle, and the memo authors findings and recommendations only, each recommendation
pinned to its findings and each finding to its trials. The null check moved onto finding titles: a title in one of a
short list of null phrasings (`analysis/null_claims.py`) is refused when its finding cites no reading the bundle
established alike. Only "does this title say the arms are alike" is read off the words; nothing is parsed, so a
reworded null passes, and the judge measures the rest.

The judge's noise exceeds the effects being ranked, so rubric criteria are removed before any are added.

*Evidence:* analysis writer, 2026-09, a prompt-latitude A/B (5 cases, k=3) and a 4-condition form × input study (4 cases, k=2): effects of −0.27 and −0.8 against within-cell SD 0.33–0.68 in the first (variant not adopted), and every gap in the second (≤ 0.33) inside the judge's re-score noise (62% exact, 0.46 mean absolute difference); single campaign each.

## Prior art

Sources were checked against their own text, and verdicts against the package code, in 2026-10. **Adopted** means
taken as recommended, **adapted** taken with a stated change, **not adopted** declined on purpose, and **not built**
accepted but absent ([open problems](open-problems.md)).

### Methodology and statistics

| Source | What it recommends | Here | Why |
|---|---|---|---|
| [Miller, "Adding Error Bars to Evals"](https://arxiv.org/abs/2411.00640) (2024) | Report the standard error, clustered when questions come in related groups (about 3× the naive value on one benchmark). Resample each question K times; compare models on paired differences; size an eval by power analysis. | Adapted | Repeats are averaged within a case; arms are tested paired on shared cases (Welch when they share fewer than two). Standard errors are Miller's cluster-robust form over cases on `n_cases − 1` df, and a rate's Wilson interval uses the effective sample size. A launch estimate sizes the eval by power: the smallest difference its paired test finds at 80%, from earlier runs' variance. |
| [Anthropic, "A statistical approach to model evaluations"](https://www.anthropic.com/research/statistical-approach-to-model-evals) (2024-11) | The Miller paper's five recommendations for practitioners: a standard error with every score, clustered standard errors, resampling within questions, paired differences, power analysis. | Adapted | As in the Miller row, with t rather than the normal at small n. |
| [Bowyer, Aitchison & Ivanova](https://arxiv.org/abs/2503.01747) (ICML 2025); [Brown, Cai & DasGupta](https://doi.org/10.1214/ss/1009213286) (2001) | Central-limit intervals undercover below a few hundred datapoints; use exact or Bayesian methods, and Wilson or Jeffreys intervals for proportions. | Adapted | Every 0/1 measure takes a Wilson interval, so 3 of 3 keeps a width. Means use t, clipped to the measure's scale. A gap with zero spread is read by the bounded test on the measure's declared range on every surface, and with no range is `not_separated`: the exact sign-flip test it once read tests symmetry, not the mean. |
| [Holm](https://www.jstor.org/stable/4615733) (1979); [Benjamini & Hochberg](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) (1995) | Control error across a family of tests: the family-wise rate (Holm) or the false discovery rate (BH). | Adapted | Holm runs within each declared question's family, or campaign-wide when the campaign declares no question, and a separation needs the adjusted p below α and its interval clear of zero. Only measures on a merit axis join a family (judge latency and judge-inclusive cost stay out), and each contrast's interval is Bonferroni-widened to 1 − α/m. There is no FDR across campaigns and no sequential testing. The run-history lens tests each adjacent pair of runs uncorrected. |
| [Lakens, equivalence testing (TOST)](https://doi.org/10.1177/1948550617697177) (2017) | Claim "no difference" only when the interval sits inside a declared equivalence margin. | Adapted | A comparison that misses significance reads `not_separated`, which claims no null. A campaign comparison reads `equivalent` only when a paired TOST against the measure's declared margin (`materiality_threshold`) passes after Holm correction; judged dimensions and unpaired tests run no equivalence test. The run-history read applies the same rule per step. The same threshold is the margin a bar's verdict allows. The history read covers engine-core measures, which declare no threshold and on which a host cannot declare one, so no history step reads `equivalent` yet. |
| [Waudby-Smith & Ramdas, estimating means of bounded random variables by betting](https://doi.org/10.1093/jrsssb/qkad009) (JRSS-B 2024) | A test of a mean that holds its error rate for any distribution on a known range, at any sample size. | Adopted | Each one-sided test of the TOST on a measure with a declared range (the predictable plug-in bet, its stake capped at 0.9 of the most it could stake without ruin). A t-test on coarse scores claimed `equivalent` at up to fifteen times its rate where a regression broke a few cases outright. |
| [Gelman & Stern](https://doi.org/10.1198/000313006X152649) (2006) | Test a difference directly. "A moved and B did not" is not evidence that A and B differ. | Adopted | Each contrast is tested against the control itself, and a scope divergence tests the whole's movement minus the part's; neither is inferred from two separate verdicts. |
| [τ-bench, Yao et al.](https://arxiv.org/abs/2406.12045) (2024); [Chen et al.](https://arxiv.org/abs/2107.03374) (2021) | pass^k is the chance that all k trials succeed, estimated without bias as E[C(c,k)/C(n,k)] from n ≥ k trials. pass@k is "at least one of k". | Adopted | Per case C(c,k)/C(n,k), averaged over the cases with n ≥ k (`compute_pass_hat_k`), reported as the pass^1..pass^K curve with each point's case count. The frontier pools a case's trials across runs of one variant under one measurement context, so a repeat run adds trials rather than a second copy of each case. The field is `pass_hat_k`, never a pass@k name. |
| [Inspect](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html) (UK AISI) | Epoch reducers, including `pass_at` and `pass_k`; `stderr(cluster=<metadata key>)`; bootstrap stderr; LLM roles named and resolved late. | Adapted | Inspect serves as the vocabulary reference. The judge and simulator models are resolved at launch and recorded in the [measurement context](concepts.md#measurement-context). Stderr is clustered by case on every reading; bootstrap stderr is not built. |
| [Kohavi, Henne & Sommerfield](https://ai.stanford.edu/~ronnyk/2007GuideControlledExperiments.pdf) (2007); [NIST/SEMATECH e-Handbook](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm) | Factors and levels; control against treatments; one overall evaluation criterion; blocking on nuisance factors; A/A tests to measure the system's own noise. | Adapted | A [lever](concepts.md#lever) is a factor, and the [rig](concepts.md#apparatus-rig) is the held-constant nuisance factors. Arms launched together share drift, which is blocking. The composite stays secondary to per-dimension results. A/A runs are not a built practice. |
| [Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments*](https://doi.org/10.1017/9781108653985) (2020); [Ng, *Machine Learning Yearning*](https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf) (2018), "optimizing and satisficing metrics" | Keep guardrail metrics, which must not degrade, apart from the criterion you optimize (Kohavi); a "satisficing" metric need only clear a threshold while one "optimizing" metric is maximized (Ng). | Adapted | A boundary judged dimension or a measure declared `guardrail` joins no composite and no comparison family. Each is decided per arm against the control as non-inferiority against its margin: `held`, `breached` or `undecided`. A breach refuses the adoption; an undecided guardrail is stated on the adopted decision rather than blocking it, since at a few cases almost every guardrail is undecided. |
| [Kapoor et al., "AI Agents That Matter"](https://arxiv.org/abs/2407.01502) (2024) | Optimise accuracy and cost jointly, and hold out data against overfitting. | Adapted | The frontier ranks on what shipping costs, and the cost of measuring is kept apart ([cost and budgets](cost-and-budgets.md)). A contestant is flagged dominated only when another is shown better on every axis by paired tests, Holm-adjusted across the subject's pairs; otherwise it reads `not_separated`, or `untested` where no test could decide. There is no held-out split. |
| [Center for Open Science, preregistration](https://www.cos.io/initiatives/prereg) | State the plan before the study; keep confirmatory work apart from exploratory work. | Adapted | A [campaign](concepts.md#campaign) declares its design, and the bundle compares it with the design actually realised. A reading no live question names is labelled exploratory in the bundle, the report and the writer's view, and a finding resting only on such readings carries the label; a campaign declaring no question says once that every finding is exploratory. Questions can be added after the evidence is seen, so the label separates asked from unasked, not pre-registered from post hoc. |
| [Messing, "Hidden Measurement Error in LLM Pipelines"](https://arxiv.org/abs/2604.11581) (2026); [Kotawala, "Resolution Diagnostics for Paired LLM Evaluation"](https://arxiv.org/abs/2605.30315) (2026) | Naive standard errors omit judge and prompt variance. Invert the paired test to find the smallest detectable effect; many published gaps do not resolve. | Adapted | One judge per dimension; its noise sets the evidence tier, not the interval. A launch estimate states the smallest difference its paired test detects at 80% power, from earlier runs' repeat and case variance ([cost and budgets](cost-and-budgets.md)); judge and prompt variance are not separated in it. |

### Judging and graders

| Source | What it recommends | Here | Why |
|---|---|---|---|
| [Anthropic, "Demystifying evals for AI agents"](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) (2026-01) | Use code graders where possible; grade the outcome, not the path; one judge per dimension, with an "Unknown" way out; separate capability from regression suites; start from 20–50 real failures; test where a behaviour should and should not occur; give each task a reference solution. | Adapted | [Goal-state checks](concepts.md#goal-state-check) are code, each dimension is one judge call, and "can't tell" excludes the trial and is counted. Reference solutions are not built: a goal-check control proves the check tells two outcomes apart, not that the task is solvable. No suite tags or saturation tracking. "Should not occur" is the `boundary` axis, decided apart from capability. |
| [Zheng et al., MT-Bench](https://arxiv.org/abs/2306.05685) (2023) | Judges show position, verbosity and self-enhancement bias. Strong judges agree with humans over 80% of the time, as often as humans agree with each other. | Adapted | Pointwise scoring removes position bias. Verbosity bias has no control. Self-enhancement is handled in the next row. |
| [Panickssery, Bowman & Feng](https://arxiv.org/abs/2404.13076) (2024) | LLM evaluators recognise and favour their own generations. | Adopted (disclosure) | When the default judge is one of the candidate models, a configured alternate judge substitutes. Without one, every surface that lists judges says so. |
| [Husain, "Using LLM-as-a-Judge for Evaluation"](https://hamel.dev/blog/posts/llm-judge/) (2024) | Use binary pass/fail with a written critique. Validate the judge on a train/dev/test split of expert labels, and report true-positive and true-negative rates. | Adapted | `pass_fail` ships beside the 1–5 scale and every dimension states its scale; prefer it for new criteria. Agreement with people is Cohen's κ, not TPR/TNR, and labels are not split. |
| [Husain & Shankar, evals FAQ](https://hamel.dev/blog/posts/evals-faq/) | Do error analysis first: open coding, then axial coding into a failure taxonomy, then about 100 traces to saturation. Generic metrics create false confidence. | Not built | No error-analysis workflow ships. The rubric proposer drafts criteria, and reviewing them against observed failures is left to the author. |
| [Yan, "Evaluating the Effectiveness of LLM-Evaluators"](https://eugeneyan.com/writing/llm-evaluators/) (2024) | Use binary outputs where possible and pairwise comparison for subjective qualities. Measure agreement with κ, τ or ρ. | Adapted | κ is adopted. Pairwise judging is declined for now ([open problem](open-problems.md#pairwise-judging-declined-for-now)). |
| [Shankar et al., "Who Validates the Validators?"](https://arxiv.org/abs/2404.12272) (2024) | Criteria drift: grading outputs changes the criteria you grade them by. | Adapted | Judge configs are versioned and pinned to each run, and a changed prompt counts as a different judge. Re-scoring stored evidence measures self-agreement, and re-scoring it under a changed judge measures the drift (`judge_drift_check`, [step 9](judges-and-calibration.md#step-9-a-second-judge-for-agreement-between-judges-and-for-drift)). It runs when asked, not on every config change. |
| [Dev et al., Judge Reliability Harness](https://arxiv.org/abs/2603.05399) (2026) | No judge is uniformly reliable across benchmarks. | Adopted | Reliability is measured per dimension, scale, served model, judge config, temperature and the request settings the judge was sent with, and never pooled. The claim, sometimes attributed to this paper, that ordinal scoring is more fragile than binary is not in its abstract. |
| [Norman et al., "Reliability without Validity"](https://arxiv.org/abs/2606.19544) (2026); [Cohen, weighted κ](https://doi.org/10.1037/h0026256) (1968) | Exact-match agreement overstates reliability, so use a chance-corrected statistic. High test–retest reliability can coexist with severe bias. | Adopted | Both judge measurements use κ, quadratic-weighted on 1–5. The self-agreement tier is named as precision, never accuracy ([evidence tiers](reading-reports.md#how-far-a-judged-score-can-be-leaned-on-evidence-tiers)). |
| [Angelopoulos et al., prediction-powered inference](https://arxiv.org/abs/2301.09633) (2023) | Combine a small human-labelled set with many model predictions to get valid intervals. | Adopted | Where people rated some of a cell's judged scores, the judged reading carries a prediction-powered estimate beside the judge's mean: the judge's mean plus the rectifier, with a case-clustered interval, stated from 10 rated results ([judges and calibration](judges-and-calibration.md#step-7b-combine-the-ratings-with-the-judges-scores)). |
| [Ding, AdaRubric](https://arxiv.org/abs/2603.21362) (2026, single-author preprint) | Task-adapted rubrics track humans better than a static rubric (reported r≈0.79 against ≈0.46). | Adapted | Judged dimensions belong to a [template](concepts.md#template), not to a global fixed set. |
| [Seshadri et al., "Lost in Simulation"](https://arxiv.org/abs/2601.17087) (2026) | Swapping the simulated user's model moves agent success by up to 9 points, and simulators are miscalibrated against humans. | Adopted | The [simulator](concepts.md#simulator)'s model and settings are rig, so a run under a different simulator never pools with one under the first. |
| [Zhang et al., Who&When](https://arxiv.org/abs/2505.00212) (ICML 2025) | The best automated failure attribution names the responsible agent 53.5% of the time and the decisive step 14.2%. | Adopted | No judge is asked to assign blame among the participants of a transcript. |
| [Langfuse score configs](https://langfuse.com/docs/evaluation/scores/data-model) | Register each score's name, type and range once. | Adopted, extended | Every measure declares unit, direction and family in a `MetricDescriptor`. |

### Agent environments and worlds

The [world model](evaluating-agents.md#how-it-got-here) gives the design lineage. This table records the verdicts.

Most of what follows is established practice; what the engine adds is narrower
([what is new here](evaluating-agents.md#what-is-new-here)).

| Source | What it recommends | Here | Why |
|---|---|---|---|
| [OSWorld setup configs](https://raw.githubusercontent.com/xlang-ai/OSWorld/main/desktop_env/controllers/setup.py) | Each task declares a list of setup steps (download, launch, execute, open…) that a controller runs, in order, before the agent starts; a failed step stops the task. | Adapted | The starting state is declared too, but per world dimension, each with a schema, a read handle and the surfaces that show it, and seeded through production's own write paths. |
| [τ²-bench §3.2](https://arxiv.org/html/2506.07982) (2025) | Initialization functions set each task's starting state; the authors verify that a task is not resolved until all its solution functions are applied. | Adapted | The not-resolved check, at the level of each goal check: one that passes on the untouched seed as on its authored end state is refused. There are no solution functions, so nothing shows a task is solvable. |
| [AgentDojo task suites](https://raw.githubusercontent.com/ethz-spylab/agentdojo/main/src/agentdojo/task_suite/task_suite.py) | Injection vectors with defaults are filled into a declared environment; a suite check runs each task's ground truth and confirms a canary placed in every vector reaches a tool output. | Adapted | The canary idea is the conformance kit's perception check: each named surface must move when its dimension does. Payload injection is not an engine feature; a host builds it into its own tools. |
| [PRISM (D'Oro et al., "Computer Use at the Edge of the Statistical Precipice")](https://arxiv.org/html/2605.08261) (2026) | Environments reset deterministically to any specified initial state; an integrity check drops a configuration whose task is already complete in its initial state. | Adapted | The integrity check is the goal-check control on the untouched seed. The engine also asserts preconditions after seeding, per cell, and excludes a trial that failed one (`precondition_failed`). |
| [Harbor `nop` and `oracle` agents](https://docs.harborframework.com/agents/pre-integrated-agents) | `nop` does nothing and `oracle` runs the task's reference solution, so a sound task fails the first and passes the second. | Adapted | The do-nothing control is `nop`'s. There is no `oracle`: a control is an authored end state no run reaches, so it is not a reference solution and proves the check, not the task. |
| [Inspect error handling](https://inspect.aisi.org.uk/handling-errors.html) | A sample that raises is an error, not scored by default; `fail_on_error` sets how many the eval tolerates and `retry_on_error` retries them. | Adapted | The same split, decided by whose code failed: a rig fault excludes the cell and is counted, while the candidate's own raise is a scored failure. Nothing retries a cell. |
| [Turner-Trauring, verified fakes](https://pythonspeed.com/articles/verified-fakes) | Run the same contract tests against the real service and its fake, so the fake is proven to match. | Adopted | The store kit (`STORE_CONFORMANCE_CASES`) runs against the in-memory store and any host's store alike, and the world kit runs against the host's own world. |
| [vcrpy](https://vcrpy.readthedocs.io/en/latest/usage.html) | Record HTTP interactions to a cassette on the first run and replay them afterwards. | Adapted | [Cassettes](concepts.md#cassette) record what a tool answered per case. A replay asked something its capture never recorded is a rig failure that excludes the cell; it is never sent live. |
| [Gymnasium `Env`](https://gymnasium.farama.org/api/env/); [OpenEnv core API](https://huggingface.co/docs/openenv/main/en/reference/core.md) | Type the action and observation spaces. `reset` takes a seed and an untyped options dict (Gymnasium) or untyped keyword arguments (OpenEnv). | Not adopted | Neither declares what a harness may set before the agent starts. The world registry declares exactly that. |
| [Model Context Protocol](https://modelcontextprotocol.io/specification) | A server declares tools (`inputSchema`, `outputSchema`) and resources for a client that does not know it in advance. | Adapted (precedent) | It is the precedent for "the app declares, a generic engine consumes". It lacks "can a harness set this?" and "does the subject perceive this?", which the world contract adds. |
| [METR Task Standard](https://github.com/METR/task-standard) | Define a task as a class with imperative `install`, `start` and `score`. METR has since moved to Inspect. | Not adopted | Imperative setup cannot be introspected. |
| [τ-bench](https://arxiv.org/abs/2406.12045); [AppWorld](https://arxiv.org/abs/2407.18901) | Score the final environment state against the goal; AppWorld's state-based tests also catch collateral damage. | Adopted | Goal checks read the end state, the calls made and the world events that fired. |
| [Zhu et al., Agentic Benchmark Checklist](https://arxiv.org/abs/2507.02825) (2025) | Flaws in task setup and reward design misestimate performance by up to 100% in relative terms. On τ-bench's airline split, an agent that does nothing scores 38%. | Adopted as justification | Goal-check controls refuse a check that cannot tell doing nothing from doing the task. Preconditions exclude a trial whose world did not start where the case says. |
| [Meta ARE / Gaia2](https://arxiv.org/abs/2509.17158) (2025) | Seed initial state, and let events arrive independently of the agent. | Adapted | Triggered world dimensions fire during a cell, and goal checks read them with `fired()`. |
| [Java Technology Compatibility Kit](https://jcp.org/en/resources/guide-tck) | The specification owner ships tests that every implementation runs. | Adopted | The conformance kits a host runs against its own code: the [store kit](adopting-a-host.md#the-store-and-the-conformance-kit-that-proves-it) and the world kit, `check_world_conformance` ([world model](evaluating-agents.md#the-rules-the-contract-enforces)). |
| [PDDL](https://planning.wiki/ref/pddl) | The domain declares predicates, problems use them, and the planner's language never changes. | Adopted | The goal language has a closed grammar over an open host vocabulary. |
| [OpenTelemetry GenAI conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md), including `gen_ai.evaluation.result` | Trace model calls as spans, and emit eval scores as events parented to the evaluated span. | Not adopted | Spans are a latency source, never a scoring substrate: prompt content is opt-in and off by default, sampling and attribute limits drop data silently, and the conventions are at Development status. The engine scores stored outputs, the call ledger and the world's end state, and reads spans and latency only through a host's `TraceSink`. |
| Online evaluation, e.g. [Langfuse](https://langfuse.com/docs/evaluation/overview) | Score live production traces continuously, beside offline experiments. | Not built | Scope: the engine evaluates commissioned runs offline and does not monitor production. |

### Reporting and communicating uncertainty

| Source | What it recommends | Here | Why |
|---|---|---|---|
| [Mitchell et al., Model Cards](https://arxiv.org/abs/1810.03993) (2019) | Use fixed sections that end in caveats, and report results disaggregated by subgroup. | Adapted | A caveat attaches to the finding it qualifies, not to a closing section. [Strata](reading-reports.md#results-by-kind-of-case-strata) give the disaggregation. |
| [HELM, Liang et al.](https://arxiv.org/abs/2211.09110) (2022) | Never report a single number: measure many metrics, expose trade-offs, and name what is not covered. | Adopted | Reports give a frontier and bars, never an unqualified winner, and disclosures name what was not measured. |
| [van der Bles et al.](https://doi.org/10.1098/rsos.181870) (2019) | Keep direct uncertainty (the range) apart from indirect uncertainty (the quality of the evidence). | Adopted | Intervals and [evidence tiers](concepts.md#evidence-tier) render as separate axes. |
| [Padilla et al.](https://doi.org/10.3389/fpsyg.2020.579267) (2021) | Show verbal confidence alongside quantified uncertainty, never instead of it. | Adapted | Tier words sit beside n and intervals. Confidence is a closed tier, so no field can carry a probability a model made up; nothing calibrates one. |
| [Frans et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10623599/) (2023) | Quantile dot plots were read more accurately than error bars, with no clear effect on decision quality. | Not adopted | An interval is drawn as a small glyph in the table row that states its figure, so rows compare at a glance, with every value also written as text. Quantile dot plots are not built. |
| [NeurIPS paper checklist](https://neurips.cc/Conferences/2022/PaperInformation/PaperChecklist), statistical significance item | Put error bars on main claims, say what variability they capture, and draw no impossible bounds. | Adopted | Intervals are computed over cases, never repeats; each column's header states its level (`Interval (99%)`), and intervals are clipped to the measure's scale. |
| [GRADE](https://book.gradepro.org/guideline/overview-of-the-grade-approach) | Rate certainty high, moderate, low or very low, downgrading for bias, imprecision and inconsistency. | Adapted | Evidence tiers are GRADE-like, but they grade the judge (`calibrated`, `separation`, `incidental`, `undetermined`), and code assigns them. |
| [Uncertainty Toolkit for Analysts in Government](https://analystsuncertaintytoolkit.github.io/UncertaintyWeb/) (UK) | Give ranges over point estimates, anchor verbal probability to numbers, give each caveat its reason, and quantify nothing rather than imply false precision. | Adopted | Every reading carries its interval or says why it has none, tier words sit beside numbers, and a caveat attaches to the finding it qualifies. |
| [Vega-Lite](https://vega.github.io/vega-lite/) | A declarative grammar in which a chart is data. | Rejected | A report draws two marks, the outcome bar and the interval glyph a table row draws beside its figure, as inline SVG of its own with every value written as text. A scriptless page cannot run vega-embed, and a pinned rasteriser wheel for two marks cost more than the marks; a host that wants its own pictures reads the report's `bundle.json`. |

### Where this engine departs from common practice

The principle behind each is in [principles](#principles).

- **Comparability is decided by a key, not by memory:** variant and context keys are content digests, and a
  report names which component differs.
- **Declared and witnessed evidence never pool:** observational data cannot license a causal claim.
- **The judge is part of the instrument:** its config is versioned and pinned to each run, so a changed judge is
  a different instrument.
- **Shipping cost and measuring cost are separate:** the frontier ranks on production cost.
- **A rig failure excludes, and a candidate failure fails:** Inspect, PRISM and Harbor draw the same line; here it
  is decided by whose code failed.
- **The denominator is part of the number:** a run short of its promised trials says so on every surface.
- **Evidence tiers instead of "calibrate first":** every judged reading carries the tier its judge's measured
  agreement reached, so judged scores stay usable while their standing stays visible.
- **Code renders every number in a written analysis, and an eval grades its prose**
  ([reading reports](reading-reports.md#having-a-model-write-the-analysis-over-frozen-evidence)).
- **"Best" is never unqualified:** a quality–cost–latency frontier plus bars (satisficing thresholds); the
  cheapest contestant above the bar is named only when a test shows it cheaper than each other one, otherwise
  the set the data cannot order (`cost_decision`, `tied_with`).

### Vocabulary map

| Industry term (where from) | This engine | Note |
|---|---|---|
| System under test (ISTQB); agent (Anthropic) | [subject](concepts.md#subject); candidate | The candidate is the subject built for one cell. |
| SUT adapter; solver (Inspect); provider (promptfoo) | [kind](concepts.md#kind-candidate-kind) | |
| Task, test case (Anthropic); sample (Inspect) | [case](concepts.md#case-test-case) | Avoid "task": Inspect uses it for the whole eval, and Braintrust uses it for the system under test. |
| Scenario (HELM); task definition | [template](concepts.md#template) | |
| Case dimension (Husain & Shankar) | [variation axis](concepts.md#variation-axis) | |
| Suite, dataset | [case set](concepts.md#case-set); a template's frozen cases | |
| Trial (Anthropic); epoch (Inspect); repetition (LangSmith) | [result (observation)](concepts.md#result-observation); a cell while running | |
| Trials per case; epochs; replicates (DOE) | [k](concepts.md#k-repeats) (`k_runs`) | `k_runs` counts trials, not runs. |
| Eval run (Inspect); experiment (Braintrust, LangSmith); job (Harbor) | [run](concepts.md#run) | In DOE and W&B a "run" is one trial. |
| Factor, with levels (Kohavi; NIST) | [lever](concepts.md#lever) | |
| Nuisance or held-constant factor; evaluation harness | [apparatus, rig](concepts.md#apparatus-rig) | |
| Variant, treatment (Kohavi); arm (clinical trials) | [variant](concepts.md#variant-and-variant-key), [arm](concepts.md#arm) | |
| Experiment, study; preregistered design | [campaign](concepts.md#campaign) | Its control is a variant, contemporaneous, never a temporal "baseline". |
| Grader, scorer, evaluator | [scorer](concepts.md#scorer); [goal-state check](concepts.md#goal-state-check); [judge](concepts.md#judge) | |
| Rubric criterion | [judged dimension](concepts.md#judged-dimension-rubric-dimension) | |
| `NOANSWER` (Inspect); "Unknown" (Anthropic) | "can't tell" (`cannot_tell`) | It excludes the trial on that dimension, and it is counted. |
| Certainty of evidence (GRADE) | [evidence tier](concepts.md#evidence-tier) | It grades the judge. |
| pass^k (τ-bench; Inspect `pass_k`) | the field **`pass_hat_k`**, with its curve `pass_hat_k_curve` | All of k pass. Not pass@k (at least one of k, Inspect `pass_at`), which the engine does not report. |
