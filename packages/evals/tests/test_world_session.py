"""The world session: one cell's handle on the host's world — seed, settle, triggers, perturbation, end state.

Driven over the toy host's real world (``fixtures/toyhost/world.py``), whose handles move an actual
object, so a write that did not land and a read that read the seed are both observable. The last
section drives the toy host's run path: its kind fires an event trigger at run time through its cell's
session, and the stored result records the firing.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.contracts import (
    EvalStorage,
    Firings,
    ValidationFailedError,
    WorldEvent,
    WorldSeed,
    WorldSession,
    WorldSessionError,
)
from threetears.evals.contracts.host import SeedRefused, WorldRegistry
from threetears.evals.contracts.host.world import WorldRegistrationError
from threetears.evals.contracts.identity import resolve_context_identity
from threetears.evals.run import start_run
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from threetears.evals.run.authoring import refuse_unsupplied_world
from threetears.evals.run.check_controls import refuse_non_discriminating_checks
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import PAYMENT_HOLD
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import (
    HOLD_FIRED,
    HOLD_IN_FORCE,
    RUN_MODELS,
    execute_toyhost_run,
    toyhost_arming_template,
    toyhost_template,
)
from packages.evals.tests.fixtures.toyhost.world import (
    PAYMENT_HOLD_CONDITION,
    PAYMENT_HOLD_EVENT,
    SUPERVISOR_REVIEW_EVENT,
    ToyWorld,
    ToyWorldFaults,
    toyhost_world,
)

_CARRIERS = ("page_reader", "console")

#: The toy seed's two halves: a value set at t=0, and a triggered dimension it arms.
_ARMING_SEED = WorldSeed(
    namespaces={"page_reader": {"document_language": "de"}, "console": {"payment_hold": "held"}},
)


async def _opened(seed: WorldSeed = _ARMING_SEED, **world: Any) -> tuple[WorldSession, ToyWorld]:
    registry, state = toyhost_world(**world)
    session = WorldSession(registry, provenance="commissioned")
    await session.seed(seed, attached=_CARRIERS)
    return session, state


# =============================================================================
# Seeding: the seed walk, the dimensions' own handles, and settling
# =============================================================================


class TestSeeding:
    async def test_a_seed_writes_through_the_dimensions_handles_and_arms_a_trigger_without_firing_it(self) -> None:
        session, state = await _opened()

        assert session.opened
        assert session.attached == ("console", "page_reader")
        assert session.seeded == ("document_language", "payment_hold")
        assert state.document_language == "de"
        # Armed, not fired: the hold is staged and the world still holds the default.
        assert (state.armed_payment_hold, state.payment_hold) == ("held", "released")
        assert session.events == ()
        # The event the seed handle armed, by the identity it returned — set-at-t=0 dimensions arm nothing.
        assert session.armed_events == {"payment_hold": PAYMENT_HOLD_EVENT}

    async def test_a_triggered_seed_handle_that_names_no_event_is_refused(self) -> None:
        registry, _state = toyhost_world(faults=ToyWorldFaults(arming_payment_hold_names_no_event=True))

        with pytest.raises(WorldSessionError, match=r"payment_hold is triggered, so its seed handle arms an event"):
            await WorldSession(registry, provenance="commissioned").seed(_ARMING_SEED, attached=_CARRIERS)
        # The same world, seeding nothing triggered, is fine: only an arming owes an identity.
        await WorldSession(registry, provenance="commissioned").seed(
            WorldSeed(namespaces={"page_reader": {"document_language": "de"}}), attached=_CARRIERS
        )

    async def test_a_refused_seed_writes_nothing_and_leaves_the_world_unopened(self) -> None:
        registry, state = toyhost_world()
        session = WorldSession(registry, provenance="commissioned")
        refused = WorldSeed(namespaces={"page_reader": {"document_language": "de", "documnet_language": "fr"}})

        with pytest.raises(SeedRefused, match="documnet_language"):
            await session.seed(refused, attached=_CARRIERS)

        assert state.document_language == "en"
        assert not session.opened

    async def test_a_session_seeds_once(self) -> None:
        session, _state = await _opened()

        with pytest.raises(WorldSessionError, match="already seeded"):
            await session.seed(WorldSeed(), attached=_CARRIERS)

    async def test_a_carrier_no_dimension_names_is_refused(self) -> None:
        registry, _state = toyhost_world()

        with pytest.raises(WorldSessionError, match="'page_raeder'"):
            await WorldSession(registry, provenance="commissioned").seed(WorldSeed(), attached=("page_raeder",))

    async def test_scheduled_perturbation_on_a_world_with_no_handle_for_it_is_refused(self) -> None:
        registry, _state = toyhost_world(optional_capabilities=False)
        scheduled = WorldSeed(ambient_perturbation_turns=[2])

        with pytest.raises(WorldSessionError, match="no perturb_ambient handle"):
            await WorldSession(registry, provenance="commissioned").seed(scheduled, attached=_CARRIERS)

    async def test_each_attached_carrier_settles_after_every_write_and_an_unattached_one_does_not(self) -> None:
        registry, state = toyhost_world()
        settled: list[tuple[str, str]] = []
        settling = registry.extend(
            (),
            bindings={
                "toy.settle_page": lambda: settled.append(("page_reader", state.document_language)),
                "toy.settle_console": lambda: settled.append(("console", state.armed_payment_hold)),
            },
            settle={"page_reader": "toy.settle_page", "console": "toy.settle_console"},
        )

        page_only = WorldSeed(namespaces={"page_reader": {"document_language": "de"}})
        await WorldSession(settling, provenance="commissioned").seed(page_only, attached=("page_reader",))

        # Settled once, AFTER the write it settles, and only for the carrier this cell attached.
        assert settled == [("page_reader", "de")]

        settled.clear()
        await WorldSession(settling, provenance="commissioned").seed(_ARMING_SEED, attached=_CARRIERS)
        assert settled == [("console", "held"), ("page_reader", "de")]


class TestSettleRegistration:
    """A settle declaration nothing could await is refused where it is written."""

    def test_a_settle_naming_no_declared_carrier_is_refused(self) -> None:
        registry, _state = toyhost_world()

        with pytest.raises(WorldRegistrationError, match="settle names carrier 'printer'"):
            registry.extend((), bindings={"toy.settle": lambda: None}, settle={"printer": "toy.settle"})

    def test_an_unresolvable_settle_handle_is_refused(self) -> None:
        registry, _state = toyhost_world()

        with pytest.raises(WorldRegistrationError, match="settle handle 'toy.nothing' is not in"):
            registry.extend((), settle={"console": "toy.nothing"})

    def test_a_settle_handle_the_engine_cannot_call_is_refused(self) -> None:
        registry, _state = toyhost_world()

        with pytest.raises(WorldRegistrationError, match=r"settle handle 'toy.settle' is bound to \(carrier\)"):
            registry.extend((), bindings={"toy.settle": lambda carrier: None}, settle={"console": "toy.settle"})

    def test_extending_cannot_replace_a_carrier_s_settle_handle(self) -> None:
        registry, _state = toyhost_world()
        settling = registry.extend(
            (), bindings={"toy.a": lambda: None, "toy.b": lambda: None}, settle={"console": "toy.a"}
        )

        with pytest.raises(WorldRegistrationError, match="would change the settle handle of 'console'"):
            settling.extend((), settle={"console": "toy.b"})
        # Re-supplying the same handle displaces nothing, and is admitted.
        assert settling.extend((), settle={"console": "toy.a"}).settle == {"console": "toy.a"}


# =============================================================================
# Triggers: fired by the rig, or observed happening in the world
# =============================================================================


class TestFiring:
    async def test_firing_calls_the_host_s_fire_handle_and_records_the_rig_caused_it(self) -> None:
        session, state = await _opened()

        event = await session.fire(PAYMENT_HOLD, turn=2)

        assert state.payment_hold == "held"
        assert event == WorldEvent(
            kind="event",
            dimension=PAYMENT_HOLD,
            condition=PAYMENT_HOLD_CONDITION,
            caused_by="rig",
            event=PAYMENT_HOLD_EVENT,
            armed=True,
            turn=2,
        )
        assert session.events == (event,)
        assert session.fired == Firings(dimensions=frozenset({PAYMENT_HOLD}), armed=frozenset({PAYMENT_HOLD}))

    async def test_an_unarmed_trigger_is_not_fired(self) -> None:
        session, state = await _opened(WorldSeed())

        with pytest.raises(WorldSessionError, match="not armed by this cell's seed"):
            await session.fire(PAYMENT_HOLD)
        assert state.payment_hold == "released" and session.events == ()

    async def test_a_human_trigger_has_no_fire_handle_and_the_refusal_says_to_observe_it(self) -> None:
        seed = WorldSeed(namespaces={"console": {"supervisor_signoff": "approved"}})
        session, _state = await _opened(seed)

        with pytest.raises(
            WorldSessionError, match=r"only a person brings a human trigger about; record it with observe"
        ):
            await session.fire("supervisor_signoff")

    async def test_a_trigger_whose_host_binds_no_fire_handle_is_not_fired(self) -> None:
        session, _state = await _opened(optional_capabilities=False)

        with pytest.raises(WorldSessionError, match="payment_hold has no fire handle"):
            await session.fire(PAYMENT_HOLD)

    @pytest.mark.parametrize(
        ("dimension", "attached", "refusal"),
        [
            ("payment_hlod", _CARRIERS, "declares no such dimension"),
            ("document_language", _CARRIERS, "set at t=0"),
            (PAYMENT_HOLD, ("page_reader",), "carrier 'console' is not attached"),
        ],
    )
    async def test_firing_or_observing_what_cannot_fire_here_is_refused(
        self, dimension: str, attached: tuple[str, ...], refusal: str
    ) -> None:
        registry, _state = toyhost_world()
        session = WorldSession(registry, provenance="commissioned")
        await session.seed(WorldSeed(), attached=attached)

        with pytest.raises(WorldSessionError, match=refusal):
            await session.fire(dimension)
        with pytest.raises(WorldSessionError, match=refusal):
            session.observe(dimension, event="anything")

    async def test_nothing_moves_a_world_that_was_never_seeded(self) -> None:
        registry, _state = toyhost_world()
        session = WorldSession(registry, provenance="commissioned")

        with pytest.raises(WorldSessionError, match="never seeded"):
            await session.fire(PAYMENT_HOLD)
        with pytest.raises(WorldSessionError, match="never seeded"):
            session.observe(PAYMENT_HOLD, event=PAYMENT_HOLD_EVENT)
        with pytest.raises(WorldSessionError, match="never seeded"):
            await session.at_turn(1)
        with pytest.raises(WorldSessionError, match="never seeded"):
            await session.end_state()

    async def test_an_observed_firing_calls_nothing_and_records_the_world_caused_it(self) -> None:
        session, state = await _opened(WorldSeed())

        signoff = session.observe("supervisor_signoff", event="desk-review", turn=3)
        hold = session.observe(PAYMENT_HOLD, event="nightly-audit")

        assert state.supervisor_signoff == "pending", "observing a firing must not make it happen"
        assert (signoff.kind, signoff.caused_by, signoff.armed, signoff.turn) == ("human", "world", False, 3)
        assert (hold.kind, hold.caused_by, hold.event, hold.turn) == ("event", "world", "nightly-audit", None)
        assert session.fired == Firings(dimensions=frozenset({"supervisor_signoff", PAYMENT_HOLD}))

    async def test_provenance_is_the_event_s_not_the_dimension_s(self) -> None:
        """The seed armed the hold; the world also fires a hold of its own. Only the seed's event is armed."""
        session, _state = await _opened()

        own = session.observe(PAYMENT_HOLD, event="nightly-audit", turn=1)
        assert own.armed is False, "the dimension being armed says nothing about which event fired"
        assert session.fired == Firings(dimensions=frozenset({PAYMENT_HOLD}))

        seeds = session.observe(PAYMENT_HOLD, event=PAYMENT_HOLD_EVENT, turn=2)
        assert seeds.armed is True
        assert session.fired == Firings(dimensions=frozenset({PAYMENT_HOLD}), armed=frozenset({PAYMENT_HOLD}))

    async def test_a_witnessed_session_reads_its_firings_as_a_witnessed_cell_s(self) -> None:
        """Constructed ``witnessed``, the same observations establish what fired and not which were armed.

        The commissioned session over the same world and the same observations is the control: it names the
        seed's event armed, so the difference is the provenance the session was built under and nothing else.
        """
        observed: dict[str, Firings] = {}
        for provenance in ("commissioned", "witnessed"):
            registry, _state = toyhost_world()
            session = WorldSession(registry, provenance=provenance)
            await session.seed(_ARMING_SEED, attached=_CARRIERS)
            session.observe(PAYMENT_HOLD, event=PAYMENT_HOLD_EVENT, turn=1)
            assert session.provenance == provenance
            observed[provenance] = session.fired

        assert observed["commissioned"] == Firings(
            dimensions=frozenset({PAYMENT_HOLD}), armed=frozenset({PAYMENT_HOLD})
        )
        assert observed["witnessed"] == Firings(dimensions=frozenset({PAYMENT_HOLD}), armed_known=False)

    async def test_an_armed_event_observed_under_another_dimension_it_moves_is_armed(self) -> None:
        """One event can move two dimensions; the seed armed the event, so its firing on either is the seed's."""
        seed = WorldSeed(namespaces={"console": {"payment_hold": "held", "supervisor_signoff": "approved"}})
        session, _state = await _opened(seed)

        assert session.observe("supervisor_signoff", event=PAYMENT_HOLD_EVENT).armed is True
        assert session.observe("supervisor_signoff", event=SUPERVISOR_REVIEW_EVENT).armed is True
        assert session.observe("supervisor_signoff", event="desk-review").armed is False

    @pytest.mark.parametrize("event", ["", None, 7])
    async def test_an_observed_firing_naming_no_event_is_refused(self, event: Any) -> None:
        session, _state = await _opened()

        with pytest.raises(WorldSessionError, match="names the host's identity of the event that fired"):
            session.observe(PAYMENT_HOLD, event=event)
        assert session.events == ()


class TestAmbientPerturbation:
    async def test_the_scheduled_turn_perturbs_once_and_no_other_turn_does(self) -> None:
        seed = WorldSeed(ambient_perturbation_turns=[2])
        session, state = await _opened(seed)

        assert await session.at_turn(1) is None
        assert state.processing_shift == "day"
        event = await session.at_turn(2)
        assert state.processing_shift == "night"
        assert await session.at_turn(2) is None, "a turn announced twice perturbs once"
        assert state.processing_shift == "night"

        # The toy rig reports nothing about what it moved, which is not the same as moving nothing.
        assert event == WorldEvent(kind="ambient", caused_by="rig", turn=2, moved=None)
        assert session.events == (event,)
        assert session.fired == Firings(), "ambient perturbation fires no dimension"

    async def test_a_schedule_the_kind_never_announced_a_turn_for_is_refused(self) -> None:
        session, _state = await _opened(WorldSeed(ambient_perturbation_turns=[2]))

        with pytest.raises(WorldSessionError, match="announced no turn"):
            session.require_schedule_announced(ran_its_course=True)

    async def test_a_cell_that_did_not_run_its_course_may_have_announced_nothing(self) -> None:
        """A cell that ended early — a fault, the cap, every actor gone — may have ended before turn 1."""
        session, _state = await _opened(WorldSeed(ambient_perturbation_turns=[2]))

        session.require_schedule_announced(ran_its_course=False)
        assert session.events == ()

    async def test_a_gap_is_refused_however_the_cell_ended(self) -> None:
        """Turns a kind did announce are its turns: ending early excuses announcing none, never skipping one."""
        session, _state = await _opened(WorldSeed(ambient_perturbation_turns=[2]))
        await session.at_turn(1)
        await session.at_turn(3)

        with pytest.raises(WorldSessionError, match=r"skipped \[2\]"):
            session.require_schedule_announced(ran_its_course=False)

    async def test_announced_turns_with_a_gap_are_refused(self) -> None:
        """Announcing turn 3 and not turn 2 skips the perturbation due before turn 2 while having taken it."""
        session, state = await _opened(WorldSeed(ambient_perturbation_turns=[2]))
        await session.at_turn(1)
        await session.at_turn(3)

        assert state.processing_shift == "day", "turn 2 was never announced, so nothing perturbed"
        with pytest.raises(WorldSessionError, match=r"skipped \[2\]"):
            session.require_schedule_announced(ran_its_course=True)

    async def test_a_scheduled_turn_the_cell_never_reached_is_honest(self) -> None:
        """A cell that ended after turn 1 never reached turn 2: no refusal, and no perturbation recorded."""
        session, _state = await _opened(WorldSeed(ambient_perturbation_turns=[2]))
        await session.at_turn(1)

        session.require_schedule_announced(ran_its_course=True)
        assert session.announced_turns == (1,)
        assert session.events == ()

    async def test_no_schedule_asks_nothing_of_the_kind(self) -> None:
        session, _state = await _opened(WorldSeed())

        session.require_schedule_announced(ran_its_course=True)

    async def test_what_the_rig_reports_moving_is_recorded(self) -> None:
        registry, _state = toyhost_world()
        reporting = WorldRegistry(
            registry.declarations,
            bindings={**registry.bindings, "toy.perturb_reporting": lambda: ["processing_shift"]},
            subject_view=registry.subject_view,
            perturb_ambient="toy.perturb_reporting",
            base_world=registry.base_world,
            coherence=registry.coherence,
        )
        session = WorldSession(reporting, provenance="commissioned")
        await session.seed(WorldSeed(ambient_perturbation_turns=[1]), attached=_CARRIERS)

        event = await session.at_turn(1)

        assert event is not None and event.moved == ["processing_shift"]


# =============================================================================
# The end state: every attached dimension, read once, and the world closed after it
# =============================================================================


class TestEndState:
    async def test_the_end_state_reads_every_attached_dimension_after_what_moved_it(self) -> None:
        session, _state = await _opened()
        await session.fire(PAYMENT_HOLD)

        end_state = await session.end_state()

        assert end_state["payment_hold"] == "held", "the end state reads the world as it was left, not the seed"
        assert end_state["document_language"] == "de"
        assert set(end_state) == {declared.name for declared in session.registry.declarations}

    async def test_only_the_attached_carriers_are_read(self) -> None:
        registry, _state = toyhost_world()
        session = WorldSession(registry, provenance="commissioned")
        await session.seed(WorldSeed(), attached=("page_reader",))

        end_state = await session.end_state()

        assert set(end_state) == {"document_language", "scan_quality", "handwriting_present", "vendor_template"}

    async def test_the_end_state_is_read_once_and_closes_the_world(self) -> None:
        session, state = await _opened()
        first = await session.end_state()
        state.document_language = "fr"

        assert await session.end_state() == first, "a second reader gets the one reading, not a fresh one"
        assert session.end_state_read == first
        with pytest.raises(WorldSessionError, match="end state was already read"):
            await session.fire(PAYMENT_HOLD)
        with pytest.raises(WorldSessionError, match="end state was already read"):
            session.observe(PAYMENT_HOLD, event=PAYMENT_HOLD_EVENT)
        with pytest.raises(WorldSessionError, match="end state was already read"):
            await session.at_turn(1)

    async def test_a_read_storage_cannot_hold_is_refused_naming_the_cause(self) -> None:
        registry, _state = toyhost_world()
        unstorable = WorldRegistry(
            registry.declarations,
            bindings={**registry.bindings, "toy.read_language": lambda: {"set", "of", "words"}},
            subject_view=registry.subject_view,
            perturb_ambient=registry.perturb_ambient,
            base_world=registry.base_world,
            coherence=registry.coherence,
        )
        session = WorldSession(unstorable, provenance="commissioned")
        await session.seed(WorldSeed(), attached=_CARRIERS)

        with pytest.raises(WorldSessionError, match="cannot store as JSON"):
            await session.end_state()


# =============================================================================
# The record's own shape
# =============================================================================


@pytest.mark.parametrize(
    ("fields", "refusal"),
    [
        ({"kind": "ambient", "dimension": "x", "condition": "c", "caused_by": "rig"}, "names no dimension"),
        ({"kind": "ambient", "caused_by": "rig", "event": "e"}, "names no dimension, condition or event"),
        ({"kind": "event", "dimension": "x", "condition": "c", "caused_by": "world"}, "its condition and the event"),
        ({"kind": "ambient", "caused_by": "world"}, "the rig's act"),
        ({"kind": "ambient", "caused_by": "rig", "armed": True}, "the rig's act"),
        ({"kind": "event", "caused_by": "world"}, "names the dimension that fired"),
        (
            {"kind": "event", "dimension": "x", "condition": "c", "event": "e", "caused_by": "world", "moved": []},
            "only ambient",
        ),
        (
            {"kind": "human", "dimension": "x", "condition": "c", "event": "e", "caused_by": "rig", "armed": True},
            "only a person",
        ),
        (
            {"kind": "turn", "dimension": "x", "condition": "c", "event": "e", "caused_by": "rig"},
            "only what the cell's seed armed",
        ),
        (
            {"kind": "turn", "dimension": "x", "condition": "c", "event": "e", "caused_by": "world", "turn": 0},
            "greater than or equal",
        ),
    ],
)
def test_a_world_event_whose_fields_contradict_its_kind_is_refused(fields: dict[str, Any], refusal: str) -> None:
    with pytest.raises(ValidationError, match=refusal):
        WorldEvent.model_validate(fields)


def test_a_seed_naming_a_perturbation_turn_twice_is_refused() -> None:
    with pytest.raises(ValidationError, match="names a turn twice"):
        WorldSeed(ambient_perturbation_turns=[2, 2])
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        WorldSeed(ambient_perturbation_turns=[0])


# =============================================================================
# The run path: the toy kind fires an event trigger at run time, and the result records it
# =============================================================================


async def test_the_toy_kind_fires_an_event_trigger_at_run_time_and_every_result_records_it() -> None:
    template = toyhost_arming_template()
    # Proven where it is written, under the rule a run grades by: the fired check and the end-state
    # check each fail when the candidate did nothing and pass on their control.
    refuse_unsupplied_world(template, profile=toyhost_profile())
    refuse_non_discriminating_checks(template, profile=toyhost_profile())

    path = await execute_toyhost_run(host=toyhost_host(), template=template)

    assert path.results, "the run stored no results"
    for result in path.results:
        assert result.termination == "completed"
        assert result.world_events == [
            WorldEvent(
                kind="event",
                dimension=PAYMENT_HOLD,
                condition=PAYMENT_HOLD_CONDITION,
                caused_by="rig",
                event=PAYMENT_HOLD_EVENT,
                armed=True,
                turn=1,
            )
        ]
        verdicts = {outcome.expression: outcome.passed for outcome in result.goal_state_outcomes}
        assert verdicts[HOLD_FIRED] is True and verdicts[HOLD_IN_FORCE] is True
        trace = path.trace(result)
        assert trace is not None and trace.end_state is not None
        assert trace.end_state[PAYMENT_HOLD] == "held"


async def test_each_toy_cell_seeds_a_world_of_its_own_so_one_cell_s_firing_never_reaches_the_next() -> None:
    """A cell that fires the hold leaves it in force in ITS world; a later cell on the same host starts clean.

    One world shared by every cell would carry the fired hold into the next cell, whose end state would
    then record the previous cell's world.
    """
    host = toyhost_host()
    armed = await execute_toyhost_run(host=host, template=toyhost_arming_template())
    armed_ids = {result.id for result in armed.results}
    for result in armed.results:
        trace = armed.trace(result)
        assert trace is not None and trace.end_state is not None and trace.end_state[PAYMENT_HOLD] == "held"

    # A toy run's id derives from its arm, so the plain drive's runs are the armed drive's: its own cells are
    # the ones the armed drive did not produce.
    plain = await execute_toyhost_run(host=host, template=toyhost_template())
    later = [result for result in plain.results if result.id not in armed_ids]
    assert later
    for result in later:
        trace = plain.trace(result)
        assert trace is not None and trace.end_state is not None
        assert trace.end_state[PAYMENT_HOLD] == "released", "a previous cell's fired hold leaked into this cell"


async def test_a_run_whose_seed_arms_nothing_records_an_opened_world_in_which_nothing_moved() -> None:
    path = await execute_toyhost_run(host=toyhost_host())

    for result in path.results:
        assert result.world_events == [], "the kind opened the world and nothing moved it: [] and not None"
        trace = path.trace(result)
        assert trace is not None and trace.end_state is not None
        assert trace.end_state[PAYMENT_HOLD] == "released"


# =============================================================================
# Authoring: a template asking for a world event no run of this host could produce
# =============================================================================


def test_a_fired_check_naming_no_triggered_dimension_is_refused_at_authoring() -> None:
    template = toyhost_template()
    for name, reason in (
        ("payment_hlod", "names no dimension this host's world declares"),
        ("document_language", "set at t=0"),
    ):
        typo = template.model_copy(update={"goal_state_checks": [f'fired("{name}")']})
        with pytest.raises(ValidationFailedError, match=reason):
            refuse_unsupplied_world(typo, profile=toyhost_profile())


def test_scheduled_perturbation_on_a_host_with_no_handle_is_refused_at_authoring() -> None:
    registry, _state = toyhost_world(optional_capabilities=False)
    profile = replace(toyhost_profile(), world=registry)
    template = toyhost_template()
    scheduled = template.model_copy(
        update={"world_seed": template.world_seed.model_copy(update={"ambient_perturbation_turns": [1]})}
    )

    with pytest.raises(ValidationFailedError, match="declares no perturb_ambient handle"):
        refuse_unsupplied_world(scheduled, profile=profile)
    # The same template on the host that has one is admitted.
    refuse_unsupplied_world(scheduled, profile=toyhost_profile())


def test_a_control_stating_a_dimension_that_cannot_fire_is_refused() -> None:
    template = toyhost_arming_template()
    assert template.goal_check_controls is not None
    bad = template.goal_check_controls.end_states["hold-posted"].model_copy(update={"fired": ["document_language"]})
    controls = template.goal_check_controls.model_copy(
        update={"end_states": {**template.goal_check_controls.end_states, "hold-posted": bad}}
    )

    with pytest.raises(
        ValidationFailedError, match=r"control 'hold-posted': fired\('document_language'\) names a dimension set at t=0"
    ):
        refuse_non_discriminating_checks(
            template.model_copy(update={"goal_check_controls": controls}), profile=toyhost_profile()
        )


def test_a_fired_check_whose_control_states_nothing_fired_does_not_discriminate() -> None:
    template = toyhost_arming_template()
    assert template.goal_check_controls is not None
    silent = template.goal_check_controls.end_states["hold-posted"].model_copy(update={"fired": []})
    controls = template.goal_check_controls.model_copy(
        update={"end_states": {**template.goal_check_controls.end_states, "hold-posted": silent}}
    )

    with pytest.raises(ValidationFailedError, match="do not discriminate"):
        refuse_non_discriminating_checks(
            template.model_copy(update={"goal_check_controls": controls}), profile=toyhost_profile()
        )


# =============================================================================
# Perturbation through the launch: frozen on the run, hashed into its context, applied in every cell
# =============================================================================


async def test_a_launched_run_freezes_its_perturbation_schedule_and_every_cell_records_the_perturbation() -> None:
    template = toyhost_template()
    scheduled = template.model_copy(
        update={"world_seed": template.world_seed.model_copy(update={"ambient_perturbation_turns": [1]})}
    )
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(scheduled)
    host, _client = toyhost_launch_host(storage=storage)

    runs = await start_run(
        host,
        template_id=scheduled.id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
    )
    async with asyncio.timeout(10):
        while any(host.job_manager.is_active(run.id) for run in runs):
            await asyncio.sleep(0.01)

    (run,) = runs
    stored = storage.load_eval_run(run.id, run.scope_id)
    assert stored is not None and stored.resolved_ambient_perturbation_turns == [1]
    results = storage.query_eval_results_by_run(run.id, run.scope_id)
    assert results
    for result in results:
        assert result.world_events == [WorldEvent(kind="ambient", caused_by="rig", turn=1, moved=None)]


def test_two_runs_differing_only_in_their_perturbation_schedule_are_two_conditions() -> None:
    profile = toyhost_profile()
    held = make_eval_run()
    perturbed = held.model_copy(update={"resolved_ambient_perturbation_turns": [2]})

    held_identity = resolve_context_identity(held, profile)
    perturbed_identity = resolve_context_identity(perturbed, profile)

    assert held_identity.context_components.seeded_world != perturbed_identity.context_components.seeded_world
    assert held_identity.context_key != perturbed_identity.context_key
