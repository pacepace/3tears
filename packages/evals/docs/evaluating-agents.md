# Evaluating a tool-using agent

**For** someone whose agent calls tools that change things (closes a ticket, moves money, switches a light) and
who wants to know whether a change made it better or worse, judged by what it did rather than what it said,
without mocking every tool. **Answers:** how to declare a world, seed each case, write goal checks that measure
something, prove your world's wiring, and read the results. Why the engine works this way:
[the world model](world-model.md).

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

## What to read next

- [The world model](world-model.md): why the contract has these rules, and what it still cannot see.
- [Adopting a host](adopting-a-host.md#a-world-through-the-cells-session): wiring a world through the cell's
  session, per-cell worlds and witnessed cells.
- [Judges and calibration](judges-and-calibration.md): grading what the agent *said*, beside what it did.
- [`examples/4-booking/agent.py`](../examples/4-booking/agent.py): the quick path with a model in a tool loop.
