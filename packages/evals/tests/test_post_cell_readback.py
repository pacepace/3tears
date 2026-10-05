"""Post-cell readback: the runner reads the world a cell LEFT and stores it, whatever the kind did.

After the kind's ``invoke`` returns, the runner reads every dimension of every carrier the cell
attached through its ``read`` handle (the cell's :class:`~threetears.evals.contracts.WorldSession`)
and stores it on the cell's trace as ``end_state``. Here rather than in each kind, so a kind that
never thought to read its world still stores what its candidate left behind — and none can store the
seed in its place, which is the founding defect of a world read.

The kind below moves its world during ``invoke`` (it fires the payment hold the seed armed) and never
reads the end state itself, so the only thing that can put the post-fire world on the trace is the
runner's readback. Skipping it leaves the end state unstored, and the first test goes red.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from threetears.evals.contracts import (
    CandidateOutput,
    CellSink,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    WorldEvent,
    WorldSeed,
    WorldSession,
)
from threetears.evals.contracts.host import ApparatusError, WorldRegistry
from threetears.evals.contracts.identity import IDENTITY_VERSION, DerivedVariantIdentity, compute_variant_key
from threetears.evals.contracts.host.values import SweepableValue
from threetears.evals.run.runner import RunnerOptions, run_one_result
from packages.evals.tests.factories import make_eval_result, make_eval_trace, memory_storage
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.world import PAYMENT_HOLD_CONDITION, PAYMENT_HOLD_EVENT, toyhost_world

_KIND = "readback-probe"
_MODEL_ONLY = {"model": SweepableValue.of("test/model", display="test/model")}
_CARRIERS = ("page_reader", "console")

#: Sets a dimension at t=0 and arms the event trigger the kind fires during ``invoke``.
_SEED = WorldSeed(namespaces={"page_reader": {"document_language": "de"}, "console": {"payment_hold": "held"}})


class _MovesItsWorldKind:
    """Seeds through its cell's session, fires the armed hold in ``invoke``, and never reads the end state."""

    judged_artifact = JudgedArtifact.UNJUDGED

    def __init__(self, *, seeds: bool = True, fault_after_firing: bool = False) -> None:
        self._seeds = seeds
        self._fault_after_firing = fault_after_firing
        self.handed: WorldSession | None = None

    async def prepare(self, *, world: WorldSession | None, world_seed: WorldSeed, **_: Any) -> WorldSession | None:
        self.handed = world
        if self._seeds and world is not None:
            await world.seed(world_seed, attached=_CARRIERS)
        return world

    async def invoke(self, instance: WorldSession | None, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        if instance is not None and instance.opened:
            await instance.fire("payment_hold", turn=1)
            if self._fault_after_firing:
                raise ApparatusError("the posting rig fell over after the hold took effect")
        return CandidateOutput(output=[{"posted": True}])


def _template() -> EvalTemplate:
    return EvalTemplate(
        scope_id="uni-1", name="readback", intent="move the world", candidate_kind=_KIND, world_seed=_SEED
    )


async def _cell(kind: _MovesItsWorldKind, *, registry: WorldRegistry | None = None, worldless: bool = False) -> Any:
    profile = toyhost_profile()
    if worldless:
        profile = replace(profile, world=None)
    elif registry is not None:
        profile = replace(profile, world=registry)
    template = _template()
    return await run_one_result(
        toyhost_host(profile=profile),
        template=template,
        test_case=EvalTestCase(template_id=template.id, scope_id="u"),
        subject_id="subj-1",
        model="test/model",
        k_iteration=1,
        eval_run_id="run-1",
        scope_id="u",
        judge_service=None,
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}),
        variant=DerivedVariantIdentity(
            variant_key=compute_variant_key(_MODEL_ONLY), identity_version=IDENTITY_VERSION, levers=_MODEL_ONLY
        ),
    )


async def test_the_runner_stores_the_world_the_cell_left_though_the_kind_never_read_it() -> None:
    kind = _MovesItsWorldKind()

    result, trace = await _cell(kind)

    assert result.termination == "completed"
    assert trace.end_state is not None, "the runner did not read the cell's world back after invoke"
    # The post-fire world, not the seed: the hold was armed at t=0 and took effect during invoke.
    assert trace.end_state["payment_hold"] == "held"
    assert trace.end_state["document_language"] == "de"
    assert set(trace.end_state) == {declared.name for declared in toyhost_profile().world.declarations}  # type: ignore[union-attr]
    assert result.world_events == [
        WorldEvent(
            kind="event",
            dimension="payment_hold",
            condition=PAYMENT_HOLD_CONDITION,
            caused_by="rig",
            event=PAYMENT_HOLD_EVENT,
            armed=True,
            turn=1,
        )
    ]
    assert kind.handed is not None and kind.handed.end_state_read == trace.end_state


async def test_a_kind_that_opens_no_world_stores_no_end_state_and_no_world_events() -> None:
    result, trace = await _cell(_MovesItsWorldKind(seeds=False))

    assert result.termination == "completed"
    assert trace.end_state is None
    assert result.world_events is None, "nobody opened the world: None, never an empty record"


async def test_a_host_with_no_world_hands_the_kind_no_session() -> None:
    kind = _MovesItsWorldKind()

    result, trace = await _cell(kind, worldless=True)

    assert kind.handed is None
    assert (trace.end_state, result.world_events) == (None, None)


async def test_a_readback_the_rig_fails_excludes_that_cell_as_an_apparatus_fault() -> None:
    registry, _state = toyhost_world()

    def broken_read() -> str:
        raise ApparatusError("the console's status feed is down")

    failing = WorldRegistry(
        registry.declarations,
        bindings={**registry.bindings, "toy.read_payment_hold": broken_read},
        subject_view=registry.subject_view,
        perturb_ambient=registry.perturb_ambient,
        base_world=registry.base_world,
        coherence=registry.coherence,
    )

    result, trace = await _cell(_MovesItsWorldKind(), registry=failing)

    assert result.termination == "apparatus_failed"
    assert result.infra_error is not None and "status feed is down" in result.infra_error
    assert trace.end_state is None, "a readback that failed stores no end state"
    assert result.world_events is not None and [event.dimension for event in result.world_events] == ["payment_hold"]


async def test_a_cell_cut_short_after_its_world_moved_keeps_what_fired() -> None:
    result, trace = await _cell(_MovesItsWorldKind(fault_after_firing=True))

    assert result.termination == "apparatus_failed"
    # The session is the engine's, so what fired before the fault survives the unwind that ended the cell.
    assert result.world_events is not None and [event.dimension for event in result.world_events] == ["payment_hold"]
    assert trace.end_state is None, "the end state is read after invoke returns, which this cell never did"


def test_a_trace_carrying_only_an_end_state_is_stored() -> None:
    """A candidate that produced no output still left a world behind, and a re-check reads it back."""
    storage, _ = memory_storage()
    result = make_eval_result()
    storage.save_eval_result(result, make_eval_trace(result_id=result.id, trace=[], end_state={"payment_hold": "held"}))

    stored = storage.load_eval_result(result.id, result.scope_id)
    trace = storage.load_eval_trace(result.id, result.scope_id)
    assert stored is not None and stored.has_trace is True
    assert trace is not None and trace.end_state == {"payment_hold": "held"}
