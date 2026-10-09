# The world model

**For** anyone deciding whether their subject needs a world, designing one, or wondering why the world contract
refuses something they wrote. **Answers:** why an eval runs its subject in a seeded world, what is new about the
engine's world contract, what was tried first, and the reasoning behind each rule. [Concepts](concepts.md#world)
defines the world and [Adopting the engine](adopting-a-host.md#a-world-through-the-cells-session) shows how to
wire one.

## The problem

A scenario asked an agent to act on the item it was currently serving. The world had no dimension holding that
item, so nothing could put one there. The agent behaved correctly for the world it was given, the rubric scored it
at the floor, and the run was cancelled. It read as a catastrophic behavioural failure.
At the time the engine's world slot held a bare list of dimension names, a label list rather than a map. The field
calls this a break in *task validity*: the task could not be solved for reasons unrelated to the subject, so the
score measured the harness. The Agentic Benchmark Checklist (Zhu et al., arXiv 2507.02825) found misestimates of up
to 100% relative in shipped benchmarks from this class of defect, and requires that "each task is verified to be
solvable".

*Evidence:* conversational agent with tools, one template, 2026-08, single campaign.

## What is new here

The engine evaluates an application's own agent in a world that the host application declares to it.

- Each dimension of that world has a JSON Schema, opaque seed and read handles that use production's own write
  paths, and the list of surfaces that show it to the agent.
- Every run records each dimension as seeded-and-perceived, seeded-only (judge-only), perceived-but-not-seeded
  (witnessed) or out of play. Witnessed state is a confound that is disclosed and never pooled with seeded state.
- A conformance kit the host runs against its own world checks that seeds read back, that each named surface moves
  when its dimension does and unnamed ones hold still, and that dimensions are independent. A check it cannot run
  is reported as unavailable, with its reason, never waived.

We found no prior system that combines the host-declared contract, the per-run record and the host-run kit.

What is *not* new: declaring a run's starting state (OSWorld, τ²-bench, AgentDojo, Google ADK and PRISM all do it),
the do-nothing control, excluding a trial whose world did not start where the case says, and splitting rig from
subject failures (τ²-bench, PRISM, Harbor's `nop` and `oracle` agents, Inspect). The verdicts and sources are in
[Prior art](prior-art.md#agent-environments-and-worlds).

## How it got here

**Static mocks.** The first eval returned canned tool results, the same five items for any search, so
multi-step flows were graded against state that did not exist.

**A synthetic tool world.** Test doubles over a shared state object, with a goal language reading it, were built
and dropped before production: open-ended cases needed a hand-curated catalogue per template, the doubles could
not see drift in the real services, and definitions became per-subject code. The replacement ran the real subject
with *muted actuation* (outbound side effects suppressed, read-only tools live, [cassettes](concepts.md#cassette)
for replay). Two lessons from this phase still hold:

- **The judge must not see what the candidate did not.** The seed lived in a side store while the candidate's
  prompt was rendered from the empty production object, so the judge saw a queue the candidate never perceived.
  The invariant since: **one resolution path**, so candidate, judge and report read the same resolved world,
  frozen on the run as provenance.
- **Cassettes cannot hold an open-ended case.** A cassette replays identical requests, and an open-ended case is
  valuable because the model chooses its own query. The first host answered by delivering a seeded payload to
  *whatever* call the subject makes; that is a technique a host builds into its own tools, not an engine feature.
  From this came the split between a rig failure (a replay miss, a malformed seed:
  [`ApparatusError`](adopting-a-host.md#rig-failures-a-broken-rig-costs-one-cell-never-the-run), excluded) and an
  in-world failure (a rate limit, an item not found: shown to the subject, which may be scored on how it copes).

**Production handlers on seeded production state.** A muted call reported success, so it skipped every refusal
production gives (an empty message, an unknown target, resuming what was not paused). So the seed is now written
into real production state through production's own write paths, production handlers run against it, and only
what leaves the process is replaced, at its lowest seam. Once actions were real, eval-only renderers that still
read the seed contradicted them, so there is **one render path**, and every reader reads production state.
Seeding settles before the first turn (the registry's `settle` handle), leaves no perceptions behind, and
replaces rather than appends on re-seed.

*Evidence:* conversational agent with tools, 2 of 199 failures across two stages of one campaign traced to muted
refusals, 2026-10, single campaign.

## The rules the contract enforces

**The world is a registry of dimensions, and the engine never reads a name.** A `WorldDimension` declares a JSON
Schema, `seed` and `read` handles, `perceived_by` surfaces, a `carrier`, `evidence` (`machine` or `labeled`), timing
and required `matters` prose. Handles are opaque strings, possibly awaitable, so an in-memory world and a device
reached over RPC use one contract. *Rejected:* an engine operation vocabulary over paths. It presumes a
path-addressable document, and one surveyed consumer's world is a display device behind async RPC that can be
asleep.

**The registry is the seeding path.** If `seed` named a second path written for the contract, the conformance kit
would prove a path no run takes. Because declaring is the only way to seed, the declaration cannot decay into
optional paperwork, and "this host registers no world" becomes a claim a run record can contradict.

**Seeded and perceived are derived from the run, never declared.** At registration, seedable × perceivable gives
`representable`, `judge_only` or `witnessed`; the fourth combination is refused as a field, not a dimension. At run
time the engine computes the same two bits (`WorldRegistry.place`) from what this run seeded and which carriers
this subject attached, and records them on the run as `world_placements`. Those two inputs are not read off the
cells: the host's `world_placements` callable on its `LaunchHost` supplies them for the assembled run, and the
engine does not check them against what the cells seeded. "Derived from the run" holds as far as that callable is
right. *Rejected:* a per-run mode flag, which can disagree with the run and cannot express a run
that seeds some dimensions and witnesses others. Walking another application's code showed the need: its agent's
memory is seedable in a commissioned run and only witnessed in production traffic, on one host. Witnessed state is
a confound to disclose, and [commissioned and witnessed observations never pool](concepts.md#apparatus-class). A
run with no placements recorded reads as undecided, which blocks a merge rather than faking one.

**Anything the subject perceives must be seeded, including time and history.** One dimension of the first host was
declared permanently witnessed, so every mid-operation template ran in an idle system, where an ordinary message
read as a different event entirely. The template measured the confound, not the model. The dimension was made
seedable and the templates re-issued. The same reasoning moved the clock, the turn number and recent history into
the world, so a case reads the same on every run. The clock design is worth copying: cell time is the seeded instant
plus elapsed wall time, and a test seeds a clock far from any wall time and fails if the real date reaches a
prompt.

*Evidence:* conversational agent with tools, 6 of 7 templates affected, 1 campaign, 2026-10, single campaign.

**A seed carries every field production shows, present and correctly typed, and is never checked for
plausibility.** A seed missing a field tests a world production never presents. A seed setting every time to 1970 is
the author's to answer for, as with any unrealistic fixture.

**Conformance obligations are derived from shape, with no waivers.** A declaration cannot be derived from unknown
host code, so the engine ships a kit the host runs against its own code (a TCK, as in Java and MCP). Six checks:
round trip, perception A/B (each named surface moves across a generated pair and the schema's boundary values),
perception stillness (unnamed surfaces hold still), ambient isolation, independence, vocabulary completeness. Each
reports `passed`, `failed` or `unavailable` with a reason from a closed list. *Rejected:* one mandate with waivers,
because the hosts the contract exists for, the least like an in-memory dict, would live in the waivers. Building
the kit added three rules:

- `unavailable` comes from the declared shape (no perturbation binding, a trigger only a person fires) or from a
  check's own setup not landing (`seeding_did_not_take`, a defect the round trip already names as failed), never
  from a handle that throws; catching exceptions would launder a broken host into a permanent disclosed gap. An engine
  gap raises `WorldConformanceError` instead of blaming the host.
- One defect gets one finding. A dead seeder once failed three checks, two blaming innocent code; checks now confirm
  setup landed (`seeding_did_not_take`), and round trip seeds a value different from what the world holds, since a
  dead seeder passes on the default.
- The kit moves the world through the host's handles and does not restore it: run it against a rig.

**Coupled is not dependent.** A queue over an idle output starts its head; a document with no scan cannot be skewed.
A host names a `base_world` every check composes over and a `coherence(world)` handle saying which composed worlds it
holds, and the kit draws only held values. A coupling that starves a check is `failed`, and no `unavailable` reason
names a coupling, so `coherence` cannot become a waiver.

**Preconditions are checked, and a failed one is excluded rather than scored.** The goal language has a closed
grammar and an open host vocabulary, PDDL's split between domain and problem. A precondition carries `presumes`
prose so an exclusion says what was presumed. The engine asserts it at t=0 for every cell, after the kind's
`prepare` and before the first turn, against the world its session read back once the seed settled. A host's
kind need not assert it, and one that does gets the same answer. A failed presumption, or a cell whose kind never
seeded through its session, is excluded as `precondition_failed` and counted, never scored. The same static check
catches a goal-check typo that would otherwise read a missing value and silently score the subject down.

**A goal check must beat a do-nothing control.** On τ-bench's airline split an agent that does nothing scores 38%
([Zhu et al., 2025](https://arxiv.org/abs/2507.02825)): wherever success means leaving the state unchanged, doing
nothing passes. A correct refusal is a legitimate probe; the defect is a check that cannot tell it from paralysis.
Each goal check declares `act` or `hold` and names an authored end state where the behaviour happened, and the engine refuses at authoring
any check that gives the same verdict there and on the untouched seed (`GoalCheckControls`). A control proves only
that the check can tell those two outcomes apart. It is not a reference solution and does not show the task can be
solved: the authored end state is stated, never reached by a run. A control states its case parameters as a case
stores them, each one string, so no check is proven on a type no case can carry. A check launched without a proving control (a
template saved past authoring, or a quick run) is recorded as unproven on its run and marked wherever its pass
rate is shown. A run stamps the proof rules it was proven under; a `proven` recorded before controls were read as
a case stores its parameters reads as unproven, counted as needing re-proof, because the controls were not frozen
with the run and cannot be re-proven for the template it graded. A stored template can carry a check a later
grammar rule refuses: the launch records it with the reason, grades every cell on the other checks, and the run
summary, the bundle and the report name it as refused under the current grammar.

**A refused call is not an action.** The [call ledger](concepts.md#goal-state-check) records only calls that
succeeded; the trace keeps refusals for the judge. Stored runs re-grade from their ledgers (`recheck_goal_states`) rather
than re-running.

*Evidence:* conversational agent with tools, 96 runs, 1 campaign, 2026-10, one result passed 7 of 7 action checks
with every call refused, single campaign.

**The world is stimulus, not variant.** It belongs to the [measurement context](concepts.md#measurement-context), so
observations under different worlds never pool. A name registered as both a sweepable and a world dimension is
refused. *Rejected:* detecting that through a shared binding table, since a sweepable's after-the-fact reader and a
live world handle are different callables that never collide.

## Review

An independent review found two outside citations in the first draft fabricated or misquoted, now corrected.
Walking the design against three other applications produced both contract-level changes.

## What the contract still cannot see

- Vocabulary completeness checks that a path names a declared dimension, not the shape or content beneath it. A
  `labeled` read proves plumbing, never truth.
- Undeclared perception is the residual risk, and ambient isolation, which finds it, is permanently `unavailable` on
  hosts whose world moves on its own.
- A dimension whose meaning changes under the same name passes every check.
- Deriving a detection eval's answer key from the planted fact is designed but deferred; host predicates in the goal
  language are specified but not built.
- Every fidelity defect the first host found in its October audit came from running and reading cells, none from a
  gate: an unseeded count the subject cited, carried state leaking into cells, a check reading a parameter instead of
  its effect, and others.

*Evidence:* conversational agent with tools, k=1, 6 templates, 2 passes of 22 results, 2026-10, not replicated.

## What a host author should take from it

Seed state through the code production uses and replace only what leaves the process; treat anything the subject
sees that no dimension declares as a defect; give every goal check a control and read effects, not parameters;
and run a few cells and read them before trusting a template, since the gates came after the audits.
