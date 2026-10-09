# Design rationale

Read this when you want to know why the engine's measurement concepts are shaped as they are. Each
section gives the problem that forced the design, the choice, what was rejected, and what changed later.
Terms are defined in [Concepts](concepts.md); the world has its own page, [The world model](world-model.md).

## Who owns the vocabulary: levers, apparatus and labels

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

## Identity is content, with a rendering

**Problem.** Every component of the [variant key](concepts.md#variant-and-variant-key) was hashed as
content except one, a map of slot to preset *name*. Operators could edit presets, so two runs naming one
preset across an edit shared a key over different text: a wrong merge, the one outcome nothing downstream
can undo.

**Choice.** A swept value is identified by the bytes that entered the run, never by a key into a store the
engine cannot see. **Rejected:** widening the name-keyed path to reach more components, which would add a
second host-shaped mechanism. A second product, built independently, had made the same split: content-hashed
prompt blocks, everything else by name.

**What changed.** A hash is a correct identity and an unreadable label, so a level became
`SweepableValue{content_hash, display, scale, raw?}`, `display` required and never derived from the hash.
`scale` is nominal, ordinal or interval, not a bare rank, because a lever swept at 0.1 / 0.4 / 0.85 drawn
as ranks 1 / 2 / 3 puts the knee in the wrong place.

Rules that came with it (`contracts/identity.py`):

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

## The arm is the unit

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

## What pools with what

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

## Confounds qualify, never suppress

A confounded comparison is often still worth reading; it may not credit the movement to the named lever
alone. Rules told to the writer in prose were broken (inferring "no effect" from overlapping intervals was
forbidden in capitals and done anyway), so each rule below is computed into the bundle.

- **A lever seen twice is not a confound.** A knob and the resolved surface it lands in move together, so
  the analysis reported every sweep of the knob as confounded by itself. A knob now names its surface
  (`ResolvesInto` on an overlay field, `Sweepable.resolves_into` underneath), and the engine folds the surface
  into the knob where the runs show it moved with the knob alone: every level of the knob carries one level
  of the surface. For a map overlay, the surface is compared with the swept entries removed. An unreadable
  surface folds nothing. The surface is part of the variant key, so only two *arms* at one knob level can
  show it moving on its own; where no level has two, the fold is applied and disclosed as an
  `unverified_fold` confound ([open problems](open-problems.md#a-fold-with-one-arm-per-knob-level-is-untested)).
- **Assert the knob moved before reading the outcome.** A lever names the measure it acts on, and the
  bundle reports it `moved`, `inert` or `unchecked`: see
  [Did a lever take effect](reading-reports.md#did-a-lever-take-effect-mechanism-checks-and-observed-mechanisms).

  *Evidence:* agent with tools, reasoning-effort sweep, setting confirmed sent, 0 reasoning tokens in every arm, 2026-09, single campaign.
- **Significance has three states:** separated, `not_separated`, and `untested` (fewer than two cases a
  side). Collapsing the last two retires a lever nobody tested.
- **Subtract a part from a whole only under declared containment** (`MetricDescriptor.contained_by`) and
  when the declared parts exhaust it. One remainder of about 95 seconds described no wall-clock at all,
  because the part ran as detached background work.
- **Rank on the scope under test; report both.** A subsystem change was credited with halving an
  end-to-end latency its own time did not touch, and the losing arm was recommended. Measures carry a
  scope; the bundle pairs end-to-end and subsystem measures by unit and reports where their directions
  differ. A movement counts only past 2 × the standard error of the difference (each side's SEM in
  quadrature), a bar that adapts per measure instead of being tuned. Inside it the movement is `flat`, and
  `flat` is a direction: "the whole moved and the part did not" is what this exists to catch.

  *Evidence:* agent with tools, turn p95 98.3 s → 48.5 s against subsystem p95 36.6 s → 37.6 s, 2026-07, single campaign.

## A failure is charged to whoever caused it

Four steps, each fixing the last:

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

## The subject key, and evaluability

**Subject key.** Identity keyed on a display name split a renamed subject and merged two same-named ones.
"Never pool across subjects" was rejected: it is phrased on who the subject is while its reason is where
the rubric comes from, and it forbids comparing candidates that share one declared rubric. So a
[subject](concepts.md#subject)'s key and label are separate required fields, the engine warns at
population level on one key under two labels or the reverse, and a pooled score states its dimension
basis. That disclosure had to exist before any pooling ban was relaxed.

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
- **Observability was deleted** (2026-09): rubric dimensions mint measure names no registry enumerates, so
  a coverage check called every honest name "uncovered".

## The analysis: from gated prose to an evaluated document

**Prose gates did not converge.** The [analysis](concepts.md#analysis) was ruled the product, with a
deterministic gate checking the numbers in its sentences. Committed 83 minutes after the ruling, the gate
grew to 4,099 lines in 33 days and was deleted, because every real generation found a claim shape it
could not parse. The writer's prompt grew 12× in 23 days, roughly 95% of it defence against past
incidents, and one of 44 tests asserting prompt text passed while the writer did what the text forbade.

**Code renders the numbers.** The writer never types a figure: it writes a reference,
`{{c1|<measure>|<reading>|<stat>}}`, and code substitutes the value from the decision surface
(`analysis/prose_refs.py`). A reference is structure: one that does not resolve is refused, with one
repair round naming what exists, and the words around it are never read. The grammar is appended from the
parser's constants, because a copy in a stored prompt went stale and every generation was refused at full
cost.

**Typed shape, freeform content** (2026-09-25). A strict schema derived from the stored analysis was
refused by some providers as a grammar too large. The authored document (`contracts/authored.py`) types
only its parts (findings, decisions, question answers, next steps), their order, links by position and
the readings code fills, under a measured slot budget; everything said is prose. The evidence tier is read
off the readings a finding names, never chosen by the writer.

**The writer is graded by its own eval.** The reporter is a candidate kind (`analysis/reporter_kind.py`):
a case is a frozen campaign bundle, and the judge reads what the writer read and the memo as written. Code
checks the structure of model output and never its prose; a judge never grades a fact code can check.

*Evidence:* analysis writer, prompt-latitude A/B, 5 cases, k=3, 2 arms, 2026-09, insight −0.27 and decision shape −0.8 against within-cell SD 0.33–0.68, variant not adopted, single campaign.

*Evidence:* analysis writer, 4 form × input conditions, 4 cases, k=2, 2026-09, judge re-score 62% exact / 0.46 mean absolute difference, every gap ≤ 0.33 inside it, single campaign.

The judge's noise exceeds the effects being ranked, so rubric criteria are removed before any are added.
