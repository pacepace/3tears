# Evaluating a tool-using agent

**For** someone whose agent calls tools that change things (closes a ticket, moves money, switches a light) and
who wants to know whether a change made it better or worse, judged by what it did rather than what it said,
without mocking every tool. **Answers:** how to declare a world, seed each case, write goal checks that measure
something, prove your world's wiring, and read the results. Why the engine works this way:
[the world model](#why-a-world-the-world-model), at the end of this page.

The code blocks run in order, in one async context (`python -m asyncio`, or inside `async def main()`), each using
names an earlier one defined. A store call blocks, so a block makes it off the event loop with
`await run_blocking(host.blocking_executor, fn, ...)`.

The **quick path** (an `Eval` whose runner is a `World`'s: `ROOM(agent)`) wires everything for you. A **full host**
([adopting a host](adopting-a-host.md#a-world-through-the-cells-session)) is needed when your world is production
state behind your own write paths, when a template must state preconditions or prove its checks, when lookups
need cassettes in the same run as a world, or when the agent sees the world through more than one surface.

## Step 1: declare the world

A world is a few named pieces of state, each with a JSON Schema and a sentence on why it matters, plus the tools
that change it. On the quick path the state is a pydantic model, each field a dimension and its `description` why
it matters, and a tool is a plain function of the state and its typed parameters. The schema of each parameter,
read from its type, is what the model is shown, and a call that breaks it changes nothing, is not recorded, and
tells the agent why.

```python
from typing import Literal

from pydantic import BaseModel, Field

from threetears.evals import Eval, World

class Room(BaseModel):
    light: Literal["on", "off"] = Field(description="The lamp the assistant controls.")
    daylight: Literal["dark", "bright"] = Field(description="Whether the room needs the lamp at all.")

def switch_light(room: Room, to: Literal["on", "off"]) -> str:
    """Turn the room's light on or off."""  # the model sees this line as the tool's description
    room.light = to
    return f"The light is now {to}."

ROOM = World(Room, tools=[switch_light])  # named for its state: calls are "room.switch_light"
```

`ROOM.registry` is the engine's declaration that this builds: each dimension gets a `seed` and a `read` handle and
is `perceived_by` the one surface the agent sees (`await room.view()` inside the agent). On a full host you
write that declaration yourself. The handles must be production's own write and read paths, and `perceived_by`
must name **every** surface that shows the dimension to the agent:

```python
from threetears.evals.kernel.host import WorldDimension, WorldRegistry

room = {"light": "off", "daylight": "dark"}  # stands in for your production state

def render(*, surfaces):  # what the agent sees, one entry per surface asked for
    shown = {"lamp_icon": f"lamp: {room['light']}", "window": f"outside: {room['daylight']}"}
    return {surface: shown[surface] for surface in surfaces}

REGISTRY = WorldRegistry(
    [
        WorldDimension(
            name="light", schema={"enum": ["on", "off"]}, matters="The lamp the assistant controls.",
            carrier="room", seed="room.set_light", read="room.get_light", perceived_by=("lamp_icon",),
        ),
        WorldDimension(
            name="daylight", schema={"enum": ["dark", "bright"]}, matters="Whether the room needs the lamp.",
            carrier="room", seed="room.set_daylight", read="room.get_daylight", perceived_by=("window",),
        ),
    ],
    bindings={  # handle -> your code; seed and read through production's own paths
        "room.set_light": lambda v: room.update(light=v), "room.get_light": lambda: room["light"],
        "room.set_daylight": lambda v: room.update(daylight=v), "room.get_daylight": lambda: room["daylight"],
        "room.render": render,
    },
    subject_view="room.render",
)
```

## Step 2: seed each case's starting state

Each case states where its world starts, under its reserved `world` key. A field it leaves out takes the state
model's default; a starting state that leaves a field with no default unset, or breaks its schema, is refused
before anything runs, and every cell gets a fresh world. Goal checks (step 3) are listed among the graders.

```python
async def agent(case, room):  # your agent; examples/4-booking/agent.py runs a model in a tool loop here
    seen = await room.view()
    wanted = "on" if seen["daylight"] == "dark" else "off"
    if seen["light"] != wanted:
        await room["switch_light"](to=wanted)
    return "Done."

STARTS = [(light, daylight) for light in ("on", "off") for daylight in ("dark", "bright")]
CASES = [{"world": {"light": light, "daylight": daylight}} for light, daylight in STARTS]
LIGHT_MATCHES_DAYLIGHT = '(state.light == "on") == (state.daylight == "dark")'

result = await Eval(CASES, ROOM(agent), [LIGHT_MATCHES_DAYLIGHT], k=2).run(store=":memory:", scope="lights")
host, summary = result.host, result.summary  # kept, so the results can be read back in step 6
print(summary.render())
```

On a full host, a template also states **preconditions**: what it presumes the world holds at the first turn, each
an expression and the sentence it presumes (step 3 shows one). The engine asserts them for every cell against the
world read back once the seed settles. A cell that fails one is excluded as `precondition_failed` and counted,
never scored. The quick path states none: each case's seed already sets every dimension.

## Step 3: write goal checks

A goal check is an expression over the world the agent left (`state.<dimension>`), the calls it made
(`calls("<world>.<tool>")`, `call_count`, `called_before`, `called_after`, `last_call_was`), the calls a tool
refused (`refused`, `refused_count`), whether it deliberately passed (`passed()`), what fired (`fired`), and the case's
own parameters (`variation.<name>`). It also has `any`/`all` generators, `contains`,
`intersects` and `length`. A path that holds nothing is *not established*, and a check resting on one fails. The
full grammar is in the `threetears.evals.kernel.dsl` docstring.

**Read effects, not parameters.** `state.light == "on"` grades the room. `calls("room.switch_light")[0].to == "on"`
grades what the agent asked for, and passes one that asked for the right thing and left the room wrong.

**Every check must beat doing nothing.** A check is `act` (doing nothing fails it) or `hold` (the agent must not
do something, so doing nothing passes it). It is **proven** only when it gives opposite verdicts on the
do-nothing control (the seed untouched, no calls), which the engine derives, and on an end state you author where
the behaviour happened. The quick path authors none, so its checks are `unproven`, and the summary says in how
many cases an agent that did nothing would pass; a check that passes in every case is marked `NOT A MEASUREMENT`.
A `hold` check passes doing nothing by design, so only a control can prove it. A control shows only that the
check tells the two outcomes apart: it is not a reference solution and does not show the task can be solved.

To prove checks, state `goal_check_controls` on a template. `create_template` refuses a check that gives the same
verdict on both, so you can try your controls on the quick host from step 2 before building your own. Launching
the template is a host's job:

```python
from threetears.evals.schema import (
    ControlEndState,
    GoalCheckControl,
    GoalCheckControls,
    Precondition,
    RecordedCall,
    WorldSeed,
)
from threetears.evals.run import create_template, run_blocking

LIGHT_ON = 'state.light == "on"'
AT_MOST_ONE_SWITCH = 'call_count("room.switch_light") <= 1'
accept = lambda *_: None  # stand-ins for a host's own tool-catalog and seed checks
switch = lambda to: RecordedCall(tool="room", action="switch_light", params={"to": to})

template = await run_blocking(  # a store write, so it runs off the event loop
    host.blocking_executor,
    create_template,
    host,
    {
        "candidate_kind": "callable",  # the kind an eval's runs are of
        "name": "dark room", "intent": "Light a dark room for reading.",
        "world_seed": WorldSeed(namespaces={"room": {"light": "off", "daylight": "dark"}}),
        "preconditions": [Precondition(expression='state.light == "off"', presumes="The lamp starts off.")],
        "goal_state_checks": [LIGHT_ON, AT_MOST_ONE_SWITCH],
        "goal_check_controls": GoalCheckControls(
            checks=[
                GoalCheckControl(check=LIGHT_ON, intent="act", control="switched-on"),
                GoalCheckControl(check=AT_MOST_ONE_SWITCH, intent="hold", control="flicked"),
            ],
            end_states={  # the world and calls where the behaviour happened; the seed fills the rest
                "switched-on": ControlEndState(
                    describes="Turned the lamp on once.", world={"room": {"light": "on"}}, calls=[switch("on")]
                ),
                "flicked": ControlEndState(describes="Flicked it off and on.", calls=[switch("off"), switch("on")]),
            },
        ),
    },
    scope_id="lights",
    require_known_tools_allowed=accept, refuse_undeclared_world_seed=accept, refuse_undeliverable_template=accept,
)
```

Seed the same template with the lamp already on, and `LIGHT_ON` is refused: it "passes (True) when the
candidate did nothing and passes (True) on control" — the same verdict on both.

Pitfalls:

- `called_before(a, b)` and `called_after(a, b)` are **false when either action never happened**. "Never
  order before searching" is `call_count("shop.order") == 0 or called_before("shop.search", "shop.order")`.
  `last_call_was` reads only the cell's final call.
- **A deliberate pass is `passed()`, not a host's spelling.** Your kind records a pass with
  `CallLedger.record_pass()`, and `passed()` holds when the cell recorded a pass and no call. Acting and then
  passing is not a pass. The call builtins never see the pass entry.
- **The call builtins count only calls that succeeded.** A call the tool refused is recorded apart, and only
  `refused()` and `refused_count()` read it, so an agent whose every `queue_track` was refused passes
  `call_count("player.queue_track") == 0` exactly as one that never tried. Pair an action check with
  `refused_count(...) == 0` where trying and being refused is a failure of its own. A host kind records its tools'
  refusals with `CallLedger.record_refusal()` (or starts its ledger with `CallLedger(refused=[])`); against a ledger
  that keeps no refusal record, a refusal check cannot be graded rather than reading "none refused".
  `passed()` does not read refusals: it is an end-state check (a deliberate pass and no call that took effect),
  and a refused call is its own outcome, which `refused()` and `refused_count()` exist to grade.
- A string comparison over a call parameter is allowed only where the tool's own schema closes the value
  (`enum`, `const` or `pattern`). Anything else compares text the model wrote, which belongs to a judge. The
  authoring gate refuses it; an eval does not check this today, so hold yourself to it.
- A `variation.*` parameter is one string, never a list: `intersects(state.tags, variation.tags)` is refused.
  Write the set as a list literal, or use `contains(state.tags, variation.tag)`.

## Step 4: cassettes for tool results

World tools always run, because replaying a recording in their place would leave the world unmoved and the end
state graded wrong. A cassette is for **lookups** whose answers drift or cost money: a search, a price feed. On
the quick path these are a world with no state, `World(tools=[...], recording=...)`. Record once at `k=1`, then
replay:

```python
import random
from pathlib import Path
from tempfile import mkdtemp

def search(query: str) -> list[str]:  # a live index whose ranking drifts
    return random.sample(["14 days", "30 days", "90 days"], k=2)

async def answers_from_search(case, tools):
    return (await tools["search"](query=case["query"]))[0]

QUERIES = [{"query": "refund window", "expected": "14 days"}, {"query": "log retention", "expected": "90 days"}]
LOOKUPS = World(tools=[search], recording=Path(mkdtemp()) / "search.jsonl")  # a file of your own, in practice
capture = await Eval(QUERIES, LOOKUPS(answers_from_search), ["expected"], k=1).run(store=":memory:", scope="search")
replay = await Eval(QUERIES, LOOKUPS(answers_from_search), ["expected"], k=3).run(store=":memory:", scope="search")
```

A replay asked something the recording does not hold excludes that cell as a rig failure; it never runs live.
`RECORD=1` records the file again. A world with state takes no recording. A host's own kind can wire cassettes for its
lookups next to a world ([cassettes](adopting-a-host.md#cassettes-recording-and-replaying-tools)).

## Step 5: run the conformance kit against your world

A declaration is a claim. The kit tests it through your own handles. It **moves the world and does not put it
back**, so run it against a rig, in your test suite:

```python
from threetears.evals.kernel.host import check_world_conformance

report = await check_world_conformance(REGISTRY, expressions=[LIGHT_ON, 'state.light == "off"'])
for result in report.results:
    print(result.check, result.dimension, result.outcome, result.qualification or "", "|", result.detail)
assert not report.failures
```

Hand it every goal check and precondition your templates use. Each result is `passed`, `failed` or
`unavailable`, with a `detail` sentence. `result.proved` is true only for a clean pass.

| Check | Proves | `failed` means |
|---|---|---|
| `round_trip` | a value you seed reads back | the seed handle reaches nothing, or not the world the read sees |
| `perception_ab` | each surface in `perceived_by` changes when the dimension does | a named surface never shows it: the renderer was removed or reads elsewhere |
| `perception_stillness` | surfaces *not* named hold still | a surface leaks a dimension it is not declared to show |
| `independence` | seeding one dimension leaves another intact | seeding B undoes A, so a case presuming both is false |
| `ambient_isolation` | moving undeclared state leaves the agent's view alone | the agent perceives state no dimension declares |
| `vocabulary_completeness` | every `state.` path in your expressions names a declared dimension | a typo or a missing dimension, which would otherwise score the agent down |

`unavailable` is never a pass and never a waiver: it names what your declared shape cannot supply. Common reasons:
`no_perturbation_binding` (no `perturb_ambient` handle to move undeclared state), `nothing_to_observe` (no other
surface to watch), `schema_admits_too_few_values`, `nothing_to_resolve` (no expressions given) and
`seeding_did_not_take` (the check's setup did not land; `round_trip` names that defect). The registry above
passes everything but `ambient_isolation`, which is unavailable. Make the lamp icon also print the daylight and
`perception_stillness` fails for `daylight`. A `WorldConformanceError` is a gap in the engine, not your host.

## Step 6: read the results

**The summary.** Each goal check prints `passed x/n` and, unless it is proven, why it is not a measurement yet
(step 3). Excluded cells (a seed the world refused, a tool that raised, a failed precondition) are counted apart
from the agent's own failures.

**Each cell.** Read back what the agent did and where the world ended. These are store reads, which block: from
async code, run them off the event loop with `await run_blocking(host.blocking_executor, fn, ...)`.

```python
from threetears.evals.run import get_result_trace, get_run, list_results

results = await run_blocking(host.blocking_executor, list_results, host.storage, summary.run_id, summary.scope_id)
for result in results:
    trace = await run_blocking(host.blocking_executor, get_result_trace, host.storage, result)
    failed = [outcome.expression for outcome in result.goal_state_outcomes if not outcome.passed]
    print(result.test_case_id, trace.call_ledger.calls, trace.end_state, failed or "ok")
```

**A change, better or worse.** `Eval(CASES, {"old": ROOM(old_agent), "new": ROOM(new_agent)}, [checks...])`, the
first key the control, runs both agents on the same seeded cases. Each goal check becomes a
reading named `goal_state:<check>` in the contrasts table, with its interval and verdict, and an unproven check is
disclosed under Methods. [Reading a comparison](reading-reports.md#reading-a-comparison) explains the verdicts.

**Placements.** Every run records what it did with each dimension:

```python
run = await run_blocking(host.blocking_executor, get_run, host.storage, summary.run_id, summary.scope_id)
print(run.world_placements)
print(REGISTRY.place(seeded={"light"}, carriers={"room"}))  # {'light': 'representable', 'daylight': 'witnessed'}
```

| Placement | Seeded | Perceived | Means |
|---|---|---|---|
| `representable` | yes | yes | the case controlled what the agent saw |
| `judge_only` | yes | no | set for the judge and the checks; the agent cannot see it |
| `witnessed` | no | yes | the agent saw state no case set: a **confound** |
| `out_of_play` | no | no | not something this run could presume |

A quick run seeds every dimension, so all of them are `representable`. On a full host the two inputs come from
your `LaunchHost.world_placements` callable, which the engine does not check against what the cells seeded.
Witnessed state is disclosed, never controlled: when the runs being compared placed a dimension differently,
the comparison names it as the confound `world:<dimension>`, and a run that recorded no placements reads as
`undecided`.

## Why a world: the world model

Why an eval runs its subject in a seeded world, what is new about the engine's world contract, what was tried first,
and the reasoning behind each rule: read this when deciding whether your subject needs a world, designing one, or
wondering why the contract refuses something you wrote.

### The problem

A scenario asked an agent to act on the item it was currently serving. The world had no dimension holding that
item, so nothing could put one there. The agent behaved correctly for the world it was given, the rubric scored it
at the floor, and the run was cancelled. It read as a catastrophic behavioural failure.
At the time the engine's world slot held a bare list of dimension names, a label list rather than a map. The field
calls this a break in *task validity*: the task could not be solved for reasons unrelated to the subject, so the
score measured the harness. The Agentic Benchmark Checklist (Zhu et al., arXiv 2507.02825) found misestimates of up
to 100% relative in shipped benchmarks from this class of defect, and requires that "each task is verified to be
solvable".

*Evidence:* conversational agent with tools, one template, 2026-08, single campaign.

### What is new here

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
[Prior art](design.md#agent-environments-and-worlds).

### How it got here

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

### The rules the contract enforces

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
rate is shown. A run stamps the proof rules it was proven under; a `proven` recorded under older proof rules
reads as unproven, counted as needing re-proof, because the controls were not frozen
with the run and cannot be re-proven for the template it graded. A stored template can carry a check a later
grammar rule refuses: the launch records it with the reason, grades every cell on the other checks, and the run
summary, the bundle and the report name it as refused under the current grammar.

**A refused call is not an action, and it is still recorded.** The [call ledger](concepts.md#goal-state-check)'s
calls are the ones that succeeded, and every call builtin counts only those. The calls a tool refused sit apart on the
same ledger (`CallLedger.record_refusal`: the tool, the parameters, the reason and where it fell among the calls), and
`refused("<world>.<tool>")` and `refused_count(...)` read them, so a candidate whose every call was refused no longer
reads as one that never tried. The quick world records its refusals; a host kind records its tools' own. A ledger that
keeps no refusal record is never read as "none refused": a refusal check against it cannot be graded.

*Evidence:* conversational agent with tools, 96 runs, 1 campaign, 2026-10, one result passed 7 of 7 action checks
with every call refused, single campaign.

**The world is stimulus, not variant.** It belongs to the [measurement context](concepts.md#measurement-context), so
observations under different worlds never pool. A name registered as both a sweepable and a world dimension is
refused. *Rejected:* detecting that through a shared binding table, since a sweepable's after-the-fact reader and a
live world handle are different callables that never collide.

### Review

An independent review found two outside citations in the first draft fabricated or misquoted, now corrected.
Walking the design against three other applications produced both contract-level changes.

### What the contract still cannot see

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

### What a host author should take from it

Seed state through the code production uses and replace only what leaves the process; treat anything the subject
sees that no dimension declares as a defect; give every goal check a control and read effects, not parameters;
and run a few cells and read them before trusting a template, since the gates came after the audits.

## What to read next

- [Adopting a host](adopting-a-host.md#a-world-through-the-cells-session): wiring a world through the cell's
  session, per-cell worlds and witnessed cells.
- [Judges and calibration](judges-and-calibration.md): grading what the agent *said*, beside what it did.
- [`examples/4-booking/agent.py`](../examples/4-booking/agent.py): the quick path with a model in a tool loop.
