"""Per-cell world binding: each cell's session calls into that cell's own world, and no other.

The profile's :class:`~threetears.evals.contracts.host.WorldRegistry` is the declaration every reader
reads. A host whose world is real per-cell state declares ``binds_per_cell=True``, and each cell's kind
hands its own handle table to :meth:`~threetears.evals.contracts.WorldSession.bind` before seeding. The
session then calls through a session-local registry over that table, so two cells cannot cross-write:
neither holds a path to the other's world.

Driven over the toy host's real world (``fixtures/toyhost/world.py``), whose handles move an actual
object per call to ``toyhost_world()`` — so a fresh call is a fresh per-cell world, and a write that
landed in the wrong one is observable on the object it landed in.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.contracts import (
    CandidateOutput,
    CellSink,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    WorldSeed,
    WorldSession,
    WorldSessionError,
)
from threetears.evals.contracts.host import WorldRegistry, check_world_conformance
from threetears.evals.contracts.host.values import SweepableValue
from threetears.evals.contracts.host.world import WorldRegistrationError
from threetears.evals.contracts.identity import IDENTITY_VERSION, DerivedVariantIdentity, compute_variant_key
from threetears.evals.run.runner import RunnerOptions, run_one_result
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_ID, toyhost_profile
from packages.evals.tests.fixtures.toyhost.world import ToyWorld, toyhost_world

_KIND = "per-cell-world"
_MODEL_ONLY = {"model": SweepableValue.of("test/model", display="test/model")}
_CARRIERS = ("page_reader", "console")


def _per_cell() -> tuple[WorldRegistry, ToyWorld]:
    """The toy world declared as binding per cell, and the state its OWN table moves (the conformance world)."""
    registry, state = toyhost_world()
    return registry.extend((), binds_per_cell=True), state


def _seed(language: str) -> WorldSeed:
    return WorldSeed(namespaces={"page_reader": {"document_language": language}})


class _FreshWorldPerCellKind:
    """Builds a fresh toy world in every cell's ``prepare`` and binds the cell's session to it.

    The two cells meet at a barrier after binding and again before returning, so their seeds, reads and
    end-state readbacks interleave rather than running one cell to completion first.
    """

    judged_artifact = JudgedArtifact.UNJUDGED

    def __init__(self, *, binds: bool = True, cells: int = 2) -> None:
        self._binds = binds
        self._barrier = asyncio.Barrier(cells)
        self.worlds: dict[str, ToyWorld] = {}
        self.reads: dict[str, str] = {}

    async def prepare(self, *, world: WorldSession | None, world_seed: WorldSeed, **_: Any) -> tuple[WorldSession, str]:
        assert world is not None
        language = world_seed.namespaces["page_reader"]["document_language"]
        if self._binds:
            cell_registry, state = toyhost_world()
            world.bind(cell_registry.bindings)
            self.worlds[language] = state
        await self._barrier.wait()
        await world.seed(world_seed, attached=_CARRIERS)
        return world, language

    async def invoke(
        self, instance: tuple[WorldSession, str], test_case: EvalTestCase, sink: CellSink
    ) -> CandidateOutput:
        session, language = instance
        await self._barrier.wait()
        # A kind calling a handle itself calls it on the session's registry, and so reads its own cell.
        self.reads[language] = await session.registry.call("toy.read_language")
        return CandidateOutput(output=[{"posted": True}])


async def _cell(kind: Any, registry: WorldRegistry, *, language: str, run_id: str) -> Any:
    profile = replace(toyhost_profile(), world=registry)
    template = EvalTemplate(
        scope_id="uni-1",
        name=f"bind-{language}",
        intent="seed my own world",
        candidate_kind=_KIND,
        world_seed=_seed(language),
    )
    return await run_one_result(
        toyhost_host(profile=profile),
        template=template,
        test_case=EvalTestCase(template_id=template.id, scope_id="u"),
        subject_id="subj-1",
        model="test/model",
        k_iteration=1,
        eval_run_id=run_id,
        scope_id="u",
        judge_service=None,
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}),
        variant=DerivedVariantIdentity(
            variant_key=compute_variant_key(_MODEL_ONLY), identity_version=IDENTITY_VERSION, levers=_MODEL_ONLY
        ),
    )


# =============================================================================
# Two cells of two runs, concurrently, each in its own world
# =============================================================================


async def test_two_concurrent_cells_of_two_runs_each_leave_only_their_own_seed() -> None:
    registry, conformance_world = _per_cell()
    kind = _FreshWorldPerCellKind()

    (result_de, trace_de), (result_fr, trace_fr) = await asyncio.gather(
        _cell(kind, registry, language="de", run_id="run-de"),
        _cell(kind, registry, language="fr", run_id="run-fr"),
    )

    assert (result_de.termination, result_fr.termination) == ("completed", "completed")
    # The stored end state is the cell's own world, read back through its own table.
    assert trace_de.end_state is not None and trace_de.end_state["document_language"] == "de"
    assert trace_fr.end_state is not None and trace_fr.end_state["document_language"] == "fr"
    # So is what the kind read through the session mid-cell.
    assert kind.reads == {"de": "de", "fr": "fr"}
    # And each world object holds only its own seed: no write landed in the other, or in the
    # profile's table (the conformance world), which no cell ever touched.
    assert kind.worlds["de"].document_language == "de"
    assert kind.worlds["fr"].document_language == "fr"
    assert conformance_world.document_language == "en"


# =============================================================================
# The forgotten bind
# =============================================================================


class TestTheForgottenBind:
    async def test_a_session_over_a_per_cell_world_refuses_to_seed_unbound(self) -> None:
        registry, conformance_world = _per_cell()
        session = WorldSession(registry)

        with pytest.raises(WorldSessionError, match="binds per cell"):
            await session.seed(_seed("de"), attached=_CARRIERS)

        assert not session.opened
        assert conformance_world.document_language == "en", "the refusal came before any write"

    async def test_a_kind_that_forgets_to_bind_ends_the_run_through_the_runner(self) -> None:
        registry, conformance_world = _per_cell()

        with pytest.raises(WorldSessionError, match="never bound"):
            await _cell(_FreshWorldPerCellKind(binds=False, cells=1), registry, language="de", run_id="run-1")

        assert conformance_world.document_language == "en"

    async def test_a_world_that_does_not_bind_per_cell_seeds_unbound_through_the_profiles_table(self) -> None:
        registry, state = toyhost_world()
        session = WorldSession(registry)

        await session.seed(_seed("de"), attached=_CARRIERS)

        assert state.document_language == "de"
        assert session.registry is registry and not session.bound

    def test_extending_never_turns_per_cell_binding_off(self) -> None:
        registry, _state = _per_cell()

        assert registry.binds_per_cell
        assert registry.extend(()).binds_per_cell
        assert not toyhost_world()[0].binds_per_cell


# =============================================================================
# A bind is held to the declaration's own rules
# =============================================================================


class TestABindIsHeldToTheDeclaration:
    def test_a_table_missing_a_declared_handle_is_refused(self) -> None:
        registry, _state = _per_cell()
        table = dict(toyhost_world()[0].bindings)
        del table["toy.read_language"]

        with pytest.raises(WorldRegistrationError, match=r"missing 'toy\.read_language'"):
            WorldSession(registry).bind(table)

    def test_a_table_binding_a_handle_the_declaration_does_not_is_refused(self) -> None:
        registry, _state = _per_cell()
        table = {**toyhost_world()[0].bindings, "toy.read_the_other_cell": lambda: "fr"}

        with pytest.raises(WorldRegistrationError, match=r"not declared 'toy\.read_the_other_cell'"):
            WorldSession(registry).bind(table)

    def test_a_handle_bound_in_a_shape_its_role_cannot_call_is_refused(self) -> None:
        registry, _state = _per_cell()
        # A seed handle is called with the value; this one takes none.
        table = {**toyhost_world()[0].bindings, "toy.seed_language": lambda: None}

        with pytest.raises(WorldRegistrationError, match=r"seed handle 'toy\.seed_language'"):
            WorldSession(registry).bind(table)

    def test_a_non_callable_binding_is_refused(self) -> None:
        registry, _state = _per_cell()
        table = {**toyhost_world()[0].bindings, "toy.read_language": "en"}

        with pytest.raises(WorldRegistrationError, match="not callable"):
            WorldSession(registry).bind(table)

    async def test_a_session_binds_once_and_only_before_it_seeds(self) -> None:
        registry, _state = _per_cell()
        bound = WorldSession(registry)
        bound.bind(toyhost_world()[0].bindings)
        with pytest.raises(WorldSessionError, match="already bound"):
            bound.bind(toyhost_world()[0].bindings)

        plain, _state = toyhost_world()
        seeded = WorldSession(plain)
        await seeded.seed(_seed("de"), attached=_CARRIERS)
        with pytest.raises(WorldSessionError, match="already seeded"):
            seeded.bind(toyhost_world()[0].bindings)

    def test_the_bound_registry_carries_the_declaration_and_the_host(self) -> None:
        registry, _state = _per_cell()
        profile = replace(toyhost_profile(), world=registry)
        assert profile.world is registry

        bound = WorldSession(registry).bind(toyhost_world()[0].bindings)

        assert bound is not registry
        assert bound.declarations == registry.declarations
        assert (bound.subject_view, bound.perturb_ambient, bound.coherence) == (
            registry.subject_view,
            registry.perturb_ambient,
            registry.coherence,
        )
        assert (dict(bound.base_world), dict(bound.settle)) == (dict(registry.base_world), dict(registry.settle))
        assert bound.binds_per_cell
        assert bound.address("page_reader", "document_language") == "document_language"
        with pytest.raises(ValueError, match=f"host {TOYHOST_ID!r}"):
            bound.named({"page_reader": {"no_such_key": 1}})


# =============================================================================
# Conformance still proves the profile's own table
# =============================================================================


async def test_conformance_over_a_per_cell_world_proves_the_profiles_table_unchanged() -> None:
    per_cell, _state = _per_cell()
    plain, _plain_state = toyhost_world()

    def verdicts(results: Any) -> list[tuple[Any, ...]]:
        return [(result.check, result.dimension, result.outcome) for result in results]

    per_cell_report = await check_world_conformance(per_cell)
    plain_report = await check_world_conformance(plain)

    assert verdicts(per_cell_report.results) == verdicts(plain_report.results)
