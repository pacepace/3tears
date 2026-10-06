"""The conformance kit — one verdict per obligation, and no unproved claim rendering as proved.

Every test here runs the kit over a world built through the **real** registration path, whose
handles move an actual object. That is what makes a verdict a statement about behaviour: a
hand-assembled report would test the reader, and a hand-assembled world would test the kit against
a dict somebody typed to match it.

**Every failure verdict is asserted in both directions on one fixture.** A check that fires when a
renderer is deleted proves nothing on its own — an implementation that always reports ``failed``
passes that half. So each fault case is paired with the sound world it was derived from, and both
halves are asserted, because a guard over a directed relation can be exactly inverted.
"""

from __future__ import annotations

from typing import Any, get_args

import dataclasses

import pytest

from threetears.evals.contracts.host.world import WorldDimension, WorldRegistry
from threetears.evals.contracts.host.world_conformance import (
    CheckName,
    ConformanceResult,
    ObligationRow,
    Outcome,
    Qualification,
    WorldConformanceError,
    # Private, and tested directly: routing every schema shape through check_world_conformance
    # would need a bound host per shape, and a generator emitting a value the schema forbids is a
    # false green nothing downstream can detect — so it is asserted where it is produced.
    check_world_conformance,
    obligation_rows,
    obligations,
)
from threetears.evals.contracts.host.world_schema import json_equal
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.courierhost import courier_world
from packages.evals.tests.fixtures.toyhost.world import (
    TOY_JUDGE_ONLY_EXPRESSION,
    TOY_RESOLVABLE_EXPRESSIONS,
    TOY_UNRESOLVABLE_EXPRESSION,
    ToyWorldFaults,
    toyhost_world,
)


def _verdict(results: tuple[ConformanceResult, ...], check: CheckName, dimension: str | None) -> ConformanceResult:
    """The one result for a check and dimension, so a test names what it is asserting on.

    Args:
        results: A report's results.
        check: The check.
        dimension: The dimension, or None for a registry-wide check.

    Returns:
        The single matching result.
    """
    matches = [result for result in results if result.check == check and result.dimension == dimension]
    assert len(matches) == 1, f"expected exactly one {check} verdict for {dimension}, got {matches}"
    return matches[0]


async def _report(**kwargs: Any) -> tuple[ConformanceResult, ...]:
    """Run the kit over a freshly built toy world.

    Args:
        **kwargs: Passed to :func:`~packages.evals.tests.fixtures.toyhost.world.toyhost_world`.

    Returns:
        The report's results.
    """
    registry, _state = toyhost_world(**kwargs)
    return (await check_world_conformance(registry)).results


class TestEveryObligationRowHasAFixture:
    """A row nobody has a fixture for is a row nobody has run, while the report still says
    "conformance"."""

    def test_the_toy_host_covers_every_row_of_the_obligations_table(self) -> None:
        """The coverage matrix is generated from the same table the kit derives obligations from.

        Hand-listing the rows beside the kit is how a new shape silently gets zero checks: the
        list stops matching and nothing says so. Reading both from ``ObligationRow`` means adding
        a row without a fixture fails here.
        """
        registry, _state = toyhost_world()

        covered = {row for declared in registry.declarations for row in obligation_rows(declared)}

        assert covered == set(get_args(ObligationRow))

    def test_every_declared_dimension_lands_on_a_row(self) -> None:
        """A shape on no row would owe nothing, and would render as conformant for that reason."""
        registry, _state = toyhost_world()

        assert all(obligation_rows(declared) for declared in registry.declarations)

    async def test_every_registrable_quadrant_gets_a_verdict(self) -> None:
        """The kit's acceptance, made mechanical rather than read off a report by eye.

        Three quadrants are registrable and the fourth is refused at registration, so a kit that
        silently produced nothing for one of the three would look identical to a clean run. The
        witnessed quadrant is the one that matters here: it is what the founding incident was, and
        a kit deriving obligations from ``seed`` alone would give it nothing to say.
        """
        registry, _state = toyhost_world()
        report = await check_world_conformance(registry)

        spoken_for = {declared.capability for declared in registry.declarations if report.for_dimension(declared.name)}

        assert spoken_for == {"representable", "judge_only", "witnessed"}

    def test_obligations_are_derived_from_shape_rather_than_declared(self) -> None:
        """The kit's one rule. A witnessed dimension owes perception and not a round trip; a
        judge-only one owes the round trip and not A/B; both owe stillness, which every dimension
        does; neither says so about itself."""
        registry, _state = toyhost_world()
        witnessed = registry.get("ingest_backlog")
        judge_only = registry.get("vendor_template")
        assert witnessed is not None and judge_only is not None

        assert obligations(witnessed) == ("perception_ab", "perception_stillness")
        assert obligations(judge_only) == ("round_trip", "perception_stillness", "independence")


class TestRoundTrip:
    """Does the seeding path reach the world, and does anything say so when it does not."""

    async def test_a_seeded_dimension_reads_back(self) -> None:
        """The plain case: seed and read are the same object, so the trip closes."""
        assert _verdict(await _report(), "round_trip", "scan_quality").outcome == "passed"

    async def test_a_seeder_wired_to_nothing_fails_and_names_both_values(self) -> None:
        """The founding defect in its purest form — an instantiation nobody verified took."""
        sound = _verdict(await _report(), "round_trip", "vendor_template")
        broken = _verdict(
            await _report(faults=ToyWorldFaults(seeding_vendor_template_does_nothing=True)),
            "round_trip",
            "vendor_template",
        )

        assert sound.outcome == "passed"
        assert broken.outcome == "failed"
        assert "reads back 'peppol-einvoice'" in broken.detail

    async def test_a_triggered_seed_that_names_no_event_fails_and_the_sound_one_passes(self) -> None:
        """A cell refuses a triggered seed with no event identity; the kit is where that is found first."""
        sound = _verdict(await _report(), "round_trip", "payment_hold")
        broken = _verdict(
            await _report(faults=ToyWorldFaults(arming_payment_hold_names_no_event=True)),
            "round_trip",
            "payment_hold",
        )

        assert sound.outcome == "passed"
        assert broken.outcome == "failed"
        assert "returned None rather than the identity of the event it armed" in broken.detail

    async def test_a_seeder_that_does_nothing_is_caught_even_at_the_world_s_own_default(self) -> None:
        """The value seeded is chosen to DIFFER from what is there, and that is load-bearing.

        Synthesizing whatever the schema offers first would let a dead seeder pass whenever that
        value happened to be the world's default — and an empty list or a zero is exactly what a
        schema offers first. This asserts the choice, by pinning that the value the check reports
        seeding is not the value the world started at.
        """
        broken = _verdict(
            await _report(faults=ToyWorldFaults(seeding_vendor_template_does_nothing=True)),
            "round_trip",
            "vendor_template",
        )

        assert "seeded 'peppol-einvoice'" not in broken.detail

    async def test_a_labeled_read_renders_as_plumbing_and_never_as_proved(self) -> None:
        """A labeled read cannot speak for the world holding the property.

        Collapsing this into a plain pass would launder an annotator's claim into a machine fact,
        which is the one thing a required classification exists to prevent.
        """
        result = _verdict(await _report(), "round_trip", "handwriting_present")

        assert result.outcome == "passed"
        assert result.qualification == "plumbing_only"
        assert not result.proved

    async def test_a_triggered_dimension_completes_through_a_bound_fire_operation(self) -> None:
        """Arm, fire, read — the whole trip, when the host can fire its own condition."""
        assert _verdict(await _report(), "round_trip", "operator_corrections").outcome == "passed"

    async def test_the_same_dimension_records_arming_only_where_no_fire_is_bound(self) -> None:
        """The hosts-without half of the row, on the same declaration.

        The difference between the two profiles is one binding, and the difference in the verdict
        is a permanent, visible gap rather than a silence.
        """
        result = _verdict(await _report(optional_capabilities=False), "round_trip", "operator_corrections")

        assert result.outcome == "unavailable"
        assert result.qualification == "arming_only"
        assert "operator_reviews_extraction" in result.detail

    async def test_a_human_trigger_records_an_answer_rather_than_a_failure(self) -> None:
        """Not instantiable in an unattended run is a fact about representability.

        Reporting it ``failed`` would tell a host to fix a declaration that is correct, and the
        remedy for a correct declaration is to rewrite the scenario.
        """
        result = _verdict(await _report(), "round_trip", "supervisor_signoff")

        assert result.outcome == "unavailable"
        assert result.qualification == "not_instantiable_unattended"
        assert "supervisor_reviews_batch" in result.detail


class TestPerceptionAB:
    """The check that survives a refactor: delete the renderer and the declaration fails."""

    async def test_a_perceived_dimension_moves_the_surface_that_carries_it(self) -> None:
        """Two schema-valid values, one surface, and a rendering that differs between them."""
        result = _verdict(await _report(), "perception_ab", "document_language")

        assert result.outcome == "passed"
        assert "document_header" in result.detail

    async def test_deleting_the_renderer_makes_the_declaration_fail(self) -> None:
        """The survives-a-refactor property, asserted in both directions on one fixture.

        Renderer present, the claim holds; renderer gone, it fails the next day rather than going
        on saying "supported" — which is what an optional read does, because an optional read
        cannot fail. One half alone would pass under an implementation that always reports failed.
        """
        sound = _verdict(await _report(), "perception_ab", "document_language")
        refactored = _verdict(
            await _report(faults=ToyWorldFaults(document_header_drops_language=True)),
            "perception_ab",
            "document_language",
        )

        assert sound.outcome == "passed"
        assert refactored.outcome == "failed"
        assert "nothing on that surface carries this dimension" in refactored.detail

    async def test_deleting_one_renderer_does_not_condemn_the_surface_s_other_dimensions(self) -> None:
        """The unit is a dimension, never a surface.

        A check that failed every dimension on a damaged surface would tell an author the area is
        unevaluable, and an author told that abandons a probe another dimension could answer.
        """
        results = await _report(faults=ToyWorldFaults(document_header_drops_language=True))

        assert _verdict(results, "perception_ab", "scan_quality").outcome == "passed"
        assert _verdict(results, "perception_ab", "handwriting_present").outcome == "passed"

    async def test_a_witnessed_dimension_is_proved_through_the_host_s_perturbation_binding(self) -> None:
        """No run controls it, and a host with a rig can still show the subject perceives it."""
        result = _verdict(await _report(), "perception_ab", "ingest_backlog")

        assert result.outcome == "passed"
        assert result.proved

    async def test_a_witnessed_dimension_is_unproved_where_the_host_cannot_perturb_it(self) -> None:
        """samsung-frame-art-loader will carry this one forever, and that is the point.

        A brightness sensor, a heartbeat and a human with a remote all move its world. The limit
        is a proven property of a real consumer, so it must be recorded rather than waived — and
        it must never look like the proved case above.
        """
        result = _verdict(await _report(optional_capabilities=False), "perception_ab", "ingest_backlog")

        assert result.outcome == "unavailable"
        assert result.qualification == "no_perturbation_binding"
        assert not result.proved

    async def test_a_seed_that_never_lands_is_not_reported_as_a_missing_renderer(self) -> None:
        """One defect, one finding, and it names the right code.

        Without the setup check this reads as a deleted renderer: the two values were never in the
        world, so the surface rendered identically both times. Blaming the renderer for the
        seeder's defect is how a report starts sending people to fix correct code.
        """
        registry, state = toyhost_world()
        state.faults = ToyWorldFaults(seeding_vendor_template_does_nothing=True)
        perceived_but_dead = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="vendor_template",
                    schema={"type": "string"},
                    matters="the adjudicated template a goal check grades extracted fields against",
                    seed="toy.seed_vendor_template",
                    read="toy.read_vendor_template",
                    perceived_by=("document_header",),
                ),
            ),
            bindings=dict(registry.bindings),
            subject_view="toy.subject_view",
        )

        result = _verdict(
            (await check_world_conformance(perceived_but_dead)).results, "perception_ab", "vendor_template"
        )

        assert result.outcome == "unavailable"
        assert result.qualification == "seeding_did_not_take"
        assert "round-trip verdict" in result.detail


class TestAmbientIsolation:
    """Hold every declaration fixed, move the surroundings, and see whether the subject followed."""

    async def test_a_world_whose_subject_sees_only_declared_state_passes(self) -> None:
        """The sound case, and the baseline the failure below is asserted against."""
        result = _verdict(await _report(), "ambient_isolation", None)

        assert result.outcome == "passed"

    async def test_a_subject_perceiving_undeclared_state_is_caught_without_spending_a_run(self) -> None:
        """The founding incident, found mechanically.

        Every declared dimension was pinned, so the only thing that moved was state no dimension
        speaks for. A view that moves with it is widening the variance of every run in the
        campaign, and nothing else in this contract would notice.
        """
        sound = _verdict(await _report(), "ambient_isolation", None)
        leaking = _verdict(
            await _report(faults=ToyWorldFaults(operator_context_reads_the_processing_shift=True)),
            "ambient_isolation",
            None,
        )

        assert sound.outcome == "passed"
        assert leaking.outcome == "failed"
        assert "no dimension here declares" in leaking.detail

    async def test_a_host_that_cannot_move_its_own_surroundings_records_it_and_never_passes(self) -> None:
        """``unproved`` where the host cannot perturb — non-fatal, and never a pass."""
        result = _verdict(await _report(optional_capabilities=False), "ambient_isolation", None)

        assert result.outcome == "unavailable"
        assert result.qualification == "no_perturbation_binding"
        assert not result.proved

    async def test_a_perceived_dimension_that_cannot_be_held_fixed_makes_the_check_unavailable(self) -> None:
        """Movement that cannot be attributed proves nothing, so it is recorded rather than passed.

        With a perceived dimension loose, a view that moved could be the surroundings leaking or
        that dimension drifting, and a check that guessed between them would be worse than one
        that declined.
        """
        loose = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="ingest_backlog",
                    schema={"type": "integer", "minimum": 0},
                    matters="the extractor trades thoroughness for speed when the queue behind it is long",
                    read="toy.read",
                    perceived_by=("operator_context",),
                ),
            ),
            bindings={
                "toy.read": lambda: 0,
                "toy.view": lambda **_: {},
                "toy.shift": lambda: None,
            },
            subject_view="toy.view",
            perturb_ambient="toy.shift",
        )

        result = _verdict((await check_world_conformance(loose)).results, "ambient_isolation", None)

        assert result.outcome == "unavailable"
        assert result.qualification == "no_perturbation_binding"
        assert "ingest_backlog" in result.detail

    async def test_a_world_no_subject_perceives_has_nothing_to_watch_and_says_so(self) -> None:
        """A check with nothing to look at must not report a pass it did not earn."""
        unwatched = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="vendor_template",
                    schema={"type": "string"},
                    matters="the adjudicated template a goal check grades extracted fields against",
                    seed="toy.seed",
                    read="toy.read",
                ),
            ),
            bindings={
                "toy.seed": lambda value: None,
                "toy.read": lambda: "unknown",
                "toy.view": lambda **_: {},
                "toy.shift": lambda: None,
            },
            subject_view="toy.view",
            perturb_ambient="toy.shift",
        )

        result = _verdict((await check_world_conformance(unwatched)).results, "ambient_isolation", None)

        assert result.outcome == "unavailable"
        assert result.qualification == "nothing_to_observe"


class TestIndependence:
    """A scenario presumes several preconditions at once, so seeding one must not undo another."""

    async def test_a_dimension_survives_every_sibling_being_seeded(self) -> None:
        """The sound case: composing these preconditions is not a lie."""
        result = _verdict(await _report(), "independence", "document_language")

        assert result.outcome == "passed"
        assert "scan_quality" in result.detail

    async def test_a_sibling_clobbering_a_dimension_is_caught_and_the_sibling_is_named(self) -> None:
        """The incident nobody has had yet, because nothing today composes preconditions.

        Named siblings matter: a failure saying only "something clobbered it" sends a host reading
        every seeder it owns.
        """
        sound = _verdict(await _report(), "independence", "document_language")
        shared = _verdict(
            await _report(faults=ToyWorldFaults(seeding_scan_quality_resets_language=True)),
            "independence",
            "document_language",
        )

        assert sound.outcome == "passed"
        assert shared.outcome == "failed"
        assert "setting scan_quality left it" in shared.detail

    async def test_a_dead_seeder_is_not_reported_as_an_innocent_sibling_clobbering(self) -> None:
        """One defect, one finding. The sibling here is blameless and must not be named."""
        result = _verdict(
            await _report(faults=ToyWorldFaults(seeding_vendor_template_does_nothing=True)),
            "independence",
            "vendor_template",
        )

        assert result.outcome == "unavailable"
        assert result.qualification == "seeding_did_not_take"

    async def test_a_dimension_nothing_can_instantiate_has_no_value_to_defend(self) -> None:
        """The human-trigger row, on the check that would otherwise silently pass it."""
        result = _verdict(await _report(), "independence", "supervisor_signoff")

        assert result.outcome == "unavailable"
        assert result.qualification == "not_instantiable_unattended"

    async def test_the_only_settable_dimension_has_nothing_to_compose_with(self) -> None:
        """Passing here is honest — there is no pair — and the detail says why, rather than nothing.

        A bare pass over an empty sibling set reads identically to a pass over a hundred siblings,
        and a reader deciding whether preconditions compose needs to know which they are holding.
        """
        held = {"value": 0}
        alone = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="shelf_stock",
                    schema={"type": "integer", "minimum": 0},
                    matters="restocking scenarios presume a depleted shelf and prove nothing against a full one",
                    seed="h.seed",
                    read="h.read",
                ),
            ),
            bindings={"h.seed": lambda value: held.__setitem__("value", value), "h.read": lambda: held["value"]},
        )

        result = _verdict((await check_world_conformance(alone)).results, "independence", "shelf_stock")

        assert result.outcome == "passed"
        assert "only settable dimension" in result.detail


class TestIndependenceNamesOnlyWhatItMoved:
    """A pass is a claim about the siblings it names, so it may only name siblings it moved."""

    async def test_a_sibling_that_could_not_be_moved_is_not_claimed_as_composed_with(self) -> None:
        """Naming it would assert a composition nothing tested.

        The seeding proved nothing about the pair, because the sibling never left the value it
        already held — and the verdict said "survived setting" anyway.
        """
        held = {"stock": 0, "pinned": 0}
        mixed = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="shelf_stock",
                    schema={"type": "integer", "minimum": 0},
                    matters="restocking scenarios presume a depleted shelf and prove nothing against a full one",
                    seed="h.seed_stock",
                    read="h.read_stock",
                ),
                WorldDimension(
                    carrier="rig",
                    name="aisle_number",
                    schema={"const": 0},
                    matters="pick-path scenarios presume the aisle the picker was routed to",
                    seed="h.seed_pinned",
                    read="h.read_pinned",
                ),
            ),
            bindings={
                "h.seed_stock": lambda value: held.__setitem__("stock", value),
                "h.read_stock": lambda: held["stock"],
                "h.seed_pinned": lambda value: held.__setitem__("pinned", value),
                "h.read_pinned": lambda: held["pinned"],
            },
        )

        result = _verdict((await check_world_conformance(mixed)).results, "independence", "shelf_stock")

        assert result.outcome == "passed"
        assert "survived setting" not in result.detail
        assert "aisle_number" in result.detail


class TestNothingUnprovedRendersAsProved:
    """The kit's one rendering rule, enforced in one place so no surface can reimplement it."""

    async def test_proved_excludes_every_qualified_pass_and_every_unavailable(self) -> None:
        """Asserted over the rendered report rather than over a check's return value.

        Disclosing this rule in prose is not enforcing it, and the failure mode is a green suite
        above a coverage surface calling a claim a fact.
        """
        registry, _state = toyhost_world(optional_capabilities=False)
        report = await check_world_conformance(registry)

        assert report.qualified, "the fixture must produce at least one qualified pass"
        assert report.unavailable, "the fixture must produce at least one unavailable"
        assert not {result.qualification for result in report.proved} - {None}
        assert set(report.proved) & (set(report.qualified) | set(report.unavailable)) == set()

    def test_a_qualified_pass_is_not_proved(self) -> None:
        """The property is the rule; a renderer reading ``outcome`` alone would launder it."""
        qualified = ConformanceResult(
            check="round_trip",
            outcome="passed",
            detail="the plumbing carried the value",
            dimension="handwriting_present",
            qualification="plumbing_only",
        )

        assert qualified.outcome == "passed"
        assert not qualified.proved

    async def test_an_absent_capability_produces_a_record_rather_than_a_missing_row(self) -> None:
        """Absent must never mean skip, because a skipped row and a green row look identical.

        The two profiles differ only in optional bindings, so every obligation present in one is
        present in the other — what changes is the verdict, never whether there is one.
        """
        with_bindings = await _report()
        without = await _report(optional_capabilities=False)

        assert {(result.check, result.dimension) for result in with_bindings} == {
            (result.check, result.dimension) for result in without
        }

    async def test_every_verdict_carries_prose_a_host_can_act_on(self) -> None:
        """A verdict with no sentence behind it is a label, for the reason a bare dimension is."""
        assert all(result.detail.strip() for result in await _report(optional_capabilities=False))


class TestTheKitRunsOffAHostProfile:
    """The kit's input is a registry, and a host hands it one through its profile.

    Two profiles differing only in optional capabilities are what the *hosts-without* rows of the
    obligations table are proven on, so both are exercised here rather than one being a passthrough
    nothing reads.
    """

    async def test_both_toy_profiles_carry_a_world_the_kit_can_run(self) -> None:
        """A profile is how a host presents its registry, and the kit must need nothing else."""
        for optional_capabilities in (True, False):
            profile = toyhost_profile(optional_capabilities=optional_capabilities)
            assert profile.world is not None

            report = await check_world_conformance(profile.world)

            assert not report.failures
            assert report.results

    async def test_the_profile_without_optional_capabilities_proves_strictly_less(self) -> None:
        """The whole point of keeping both: the gap is visible, permanent, and countable.

        A host that cannot perturb its own world is not failing — it is proving less, and a report
        that rendered the two alike would let the difference disappear the moment somebody reads a
        pass rate instead of a report.
        """
        richer = toyhost_profile().world
        poorer = toyhost_profile(optional_capabilities=False).world
        assert richer is not None and poorer is not None

        with_capabilities = await check_world_conformance(richer)
        without = await check_world_conformance(poorer)

        assert len(without.proved) < len(with_capabilities.proved)
        assert len(without.unavailable) > len(with_capabilities.unavailable)


class TestTheAwaitablePath:
    """One call path covers a world in memory and a world across a service boundary."""

    async def test_an_async_seed_read_and_fire_are_indistinguishable_from_synchronous_ones(self) -> None:
        """The toy host binds an async write, an async read and an async fire.

        If the kit awaited only where it happened to be tested, a coroutine object would reach a
        comparison and report as the host's world being wrong — a defect that would surface as a
        conformance failure against correct host code.
        """
        results = await _report()

        assert _verdict(results, "round_trip", "vendor_template").outcome == "passed"
        assert _verdict(results, "perception_ab", "ingest_backlog").outcome == "passed"
        assert _verdict(results, "round_trip", "operator_corrections").outcome == "passed"


class TestAHandleThatCannotAnswer:
    """A raising handle aborts the report, with its own type and the handle it came from.

    The kit does not translate an exception into ``unavailable``. Doing so would need
    the host to declare which of its exceptions mean "asleep" rather than "broken", and no
    consumer has pulled on that yet — while catching everything would launder a broken host into a
    permanent, disclosed, never-fixed capability gap. Pinned as a test rather than left accidental,
    so the day a consumer needs the other behaviour this is what they are changing.
    """

    async def test_the_exception_type_survives_and_the_handle_is_named(self) -> None:
        """The type is load-bearing: a runner branches on it to tell a broken rig from a real
        failure, and wrapping would erase the distinction."""

        class PanelAsleep(RuntimeError):
            """What a host raises when its world cannot be reached right now."""

        async def read_art() -> str:
            raise PanelAsleep("the panel is asleep")

        registry = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="display.current_art",
                    schema={"type": "string"},
                    matters="Selection scenarios presume what is on the panel, and no run can put it there.",
                    read="tv.get_current_art",
                    perturb="tv.force_art",
                    perceived_by=("frame_status",),
                ),
            ),
            bindings={
                "tv.get_current_art": read_art,
                "tv.force_art": lambda value: None,
                "tv.view": lambda **_: {},
            },
            subject_view="tv.view",
        )

        with pytest.raises(PanelAsleep) as raised:
            await check_world_conformance(registry)

        assert "raised by world handle 'tv.get_current_art'" in raised.value.__notes__


class TestSynthesizingValuesFromASchema:
    """The kit's own limits are the kit's, and are raised where somebody can extend them."""

    async def test_a_schema_shape_the_kit_cannot_read_raises_rather_than_reporting_unavailable(self) -> None:
        """An engine gap recorded as ``unavailable`` becomes a permanent property of the host.

        Disclosed, never fixed, because nothing says it is fixable. Raised instead, naming the
        dimension and saying whose gap it is.
        """
        structured = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="display.current_art",
                    schema={"type": "colour"},
                    matters="Selection scenarios presume what is on the panel, and no run can put it there.",
                    seed="tv.seed",
                    read="tv.read",
                ),
            ),
            bindings={"tv.seed": lambda value: None, "tv.read": dict},
        )

        with pytest.raises(WorldConformanceError, match="gap in the kit, not a property of the host"):
            await check_world_conformance(structured)

    async def test_an_unhandled_constraint_keyword_is_as_loud_as_an_unhandled_type(self) -> None:
        """An unhandled constraint and an unhandled type are one rule, and both raise.

        Ignoring a constraint seeds a value the host's own schema forbids and then passes a round
        trip over it — a false green nothing downstream can detect.
        """
        constrained = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="shelf_stock",
                    schema={"type": "integer", "minimum": 0, "exclusiveMaximum": 4},
                    matters="restocking scenarios presume a depleted shelf and prove nothing against a full one",
                    seed="h.seed",
                    read="h.read",
                ),
            ),
            bindings={"h.seed": lambda value: None, "h.read": lambda: 0},
        )

        with pytest.raises(WorldConformanceError, match="'exclusiveMaximum'"):
            await check_world_conformance(constrained)

    async def test_the_keyword_audit_covers_the_enum_and_const_paths_too(self) -> None:
        """The audit has to know which generator will ACTUALLY run, and enum short-circuits.

        The first fix put the audit in the *type* generator, which ``enum`` and ``const`` never
        reach — so ``{"enum": [1, 2, 3], "minimum": 2}`` still yielded 1, a value the host's own
        schema forbids, and a round trip passed over it. The rule was written down twice and
        obeyed in one branch; only running the audit at the single point every path goes through
        makes it true.
        """
        for schema in ({"enum": [1, 2, 3], "minimum": 2}, {"const": 0, "minimum": 1}):
            enumerated = WorldRegistry(
                (
                    WorldDimension(
                        carrier="rig",
                        name="shelf_stock",
                        schema=schema,
                        matters="restocking scenarios presume a depleted shelf and prove nothing against a full one",
                        seed="h.seed",
                        read="h.read",
                    ),
                ),
                bindings={"h.seed": lambda value: None, "h.read": lambda: 0},
            )

            with pytest.raises(WorldConformanceError, match="'minimum'"):
                await check_world_conformance(enumerated)

    async def test_a_union_type_is_not_silently_taken_as_its_first_member(self) -> None:
        """The keyword rule one level up: picking one of several types is assuming the rest away.

        ``{"type": ["string", "null"]}`` would generate strings and never say ``null`` was dropped,
        so the values would speak for only part of what the host declared.
        """
        union = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="vendor_template",
                    schema={"type": ["string", "null"]},
                    matters="the adjudicated template a goal check grades extracted fields against",
                    seed="h.seed",
                    read="h.read",
                ),
            ),
            bindings={"h.seed": lambda value: None, "h.read": lambda: None},
        )

        with pytest.raises(WorldConformanceError, match="type is a union of"):
            await check_world_conformance(union)

    @pytest.mark.parametrize(("stores", "outcome"), [(lambda value: value, "passed"), (int, "failed")])
    async def test_a_world_that_holds_a_boolean_as_its_number_has_not_held_it(self, stores: Any, outcome: str) -> None:
        """The kit compares what landed with the seed check's JSON equality, where ``False`` is not ``0``.

        Python's ``==`` let a seeder that stored ``int(value)`` pass: the world began at ``0``, the kit
        skipped ``False`` as "already held", seeded ``True``, read back ``1`` and called ``1 == True`` a
        round trip. Both halves on one fixture, so a kit that failed every boolean would not pass.
        """
        held: dict[str, Any] = {"value": 0}
        flag = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="door_open",
                    schema={"type": "boolean"},
                    matters="a cold-chain alarm scenario presumes the door is open when the alarm fires",
                    seed="h.seed",
                    read="h.read",
                ),
            ),
            bindings={
                "h.seed": lambda value: held.__setitem__("value", stores(value)),
                "h.read": lambda: held["value"],
            },
        )

        result = _verdict((await check_world_conformance(flag)).results, "round_trip", "door_open")

        assert result.outcome == outcome, result.detail

    async def test_a_world_holding_a_number_does_not_already_hold_the_boolean_it_equals_in_python(self) -> None:
        """Choosing a value the world does not hold uses the same equality: ``0`` is not ``False``.

        Under ``==`` a ``const: false`` dimension whose world starts at ``0`` read as already holding every
        value its schema admits, and the round trip was recorded as unprovable rather than run.
        """
        held: dict[str, Any] = {"value": 0}
        flag = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="door_open",
                    schema={"const": False},
                    matters="a cold-chain alarm scenario presumes the door is shut before the alarm fires",
                    seed="h.seed",
                    read="h.read",
                ),
            ),
            bindings={"h.seed": lambda value: held.__setitem__("value", value), "h.read": lambda: held["value"]},
        )

        result = _verdict((await check_world_conformance(flag)).results, "round_trip", "door_open")

        assert result.outcome == "passed", result.detail
        assert held["value"] is False

    async def test_a_negative_only_numeric_schema_is_not_called_empty(self) -> None:
        """A schema with only a ``maximum`` admits every value below it, and is not empty.

        ``maximum: -1`` admits every integer below it. Reporting otherwise would state the kit's
        own assumption as a fact about the host's registration.
        """
        held: dict[str, int] = {"value": 0}
        negative = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="temperature_offset",
                    schema={"type": "integer", "maximum": -1},
                    matters="cold-chain scenarios presume the cabinet is running below its setpoint",
                    seed="h.seed",
                    read="h.read",
                ),
            ),
            bindings={"h.seed": lambda value: held.__setitem__("value", value), "h.read": lambda: held["value"]},
        )

        result = _verdict((await check_world_conformance(negative)).results, "round_trip", "temperature_offset")

        assert result.outcome == "passed"
        assert held["value"] <= -1

    async def test_a_range_narrower_than_one_still_yields_two_values(self) -> None:
        """A step of 1 over a range of 0.5 produced one candidate and aborted the run.

        A narrow float range is a real declaration, so the generator divides the range instead of
        assuming the host meant integers.
        """
        held: dict[str, float] = {"value": 0.0}
        narrow = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="skew_radians",
                    schema={"type": "number", "minimum": 0, "maximum": 0.5},
                    matters="OCR recovery scenarios presume the page is skewed within the deskewer's range",
                    seed="h.seed",
                    read="h.read",
                    perceived_by=("page",),
                ),
            ),
            bindings={
                "h.seed": lambda value: held.__setitem__("value", value),
                "h.read": lambda: held["value"],
                "h.view": lambda **_: {"page": f"skew={held['value']}"},
            },
            subject_view="h.view",
        )

        results = (await check_world_conformance(narrow)).results

        assert _verdict(results, "perception_ab", "skew_radians").outcome == "passed"
        assert 0 <= held["value"] <= 0.5

    async def test_a_dimension_with_one_legal_value_is_recorded_rather_than_raised(self) -> None:
        """A schema admitting one value is the HOST's declaration making a proof unreachable.

        Which is what ``unavailable`` means. Raising instead cost every other dimension in the
        registry its verdict and pointed whoever read the traceback at the synthesizer for a
        registration decision.
        """
        pinned = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="shelf_stock",
                    schema={"const": 0},
                    matters="restocking scenarios presume a depleted shelf and prove nothing against a full one",
                    read="h.read",
                    perturb="h.force",
                    perceived_by=("aisle",),
                ),
            ),
            bindings={"h.force": lambda value: None, "h.read": lambda: 0, "h.view": lambda **_: {}},
            subject_view="h.view",
        )

        result = _verdict((await check_world_conformance(pinned)).results, "perception_ab", "shelf_stock")

        assert result.outcome == "unavailable"
        assert result.qualification == "schema_admits_too_few_values"
        assert not result.proved


class TestTheEngineNeverInterpretsADimensionName:
    """Host-agnosticism again, on the kit rather than the registry: the verdicts follow shape, not vocabulary."""

    async def test_renaming_every_dimension_leaves_every_verdict_where_it_was(self) -> None:
        """A branch on a name would show up here and nowhere else.

        Every other test uses names that happen to read sensibly, so a kit that special-cased one
        would pass all of them.
        """
        registry, _state = toyhost_world()
        original = await check_world_conformance(registry)
        renamed_registry, _renamed_state = toyhost_world()
        to_q = {declared.name: f"q{index}" for index, declared in enumerate(renamed_registry.declarations)}
        from_q = {q: name for name, q in to_q.items()}
        holds = renamed_registry.bindings["toy.holds"]
        renamed = WorldRegistry(
            [
                WorldDimension(
                    carrier="rig",
                    name=f"q{index}",
                    schema=declared.schema,
                    matters=declared.matters,
                    seed=declared.seed,
                    read=declared.read,
                    perturb=declared.perturb,
                    evidence=declared.evidence,
                    perceived_by=declared.perceived_by,
                    when=declared.when,
                )
                for index, declared in enumerate(renamed_registry.declarations)
            ],
            # The coherence handle is host code, which may read its own names; the engine composes the world it
            # is handed under whatever names were registered, so the host's handle translates them back.
            bindings={
                **renamed_registry.bindings,
                "q.holds": lambda world: holds({from_q[q]: v for q, v in world.items()}),
            },
            subject_view=renamed_registry.subject_view,
            perturb_ambient=renamed_registry.perturb_ambient,
            base_world={to_q[name]: value for name, value in renamed_registry.base_world.items()},
            coherence="q.holds",
        )

        assert [(result.check, result.outcome, result.qualification) for result in original.results] == [
            (result.check, result.outcome, result.qualification)
            for result in (await check_world_conformance(renamed)).results
        ]


class TestTheVocabularyIsClosedWhereItClaimsToBe:
    """Outcomes and qualifications are engine-owned, so a stray value would be a silent widening."""

    async def test_every_verdict_uses_a_declared_outcome_and_qualification(self) -> None:
        """Asserted over a real run rather than by reading the Literal, which proves nothing."""
        results = await _report(optional_capabilities=False)

        assert {result.outcome for result in results} <= set(get_args(Outcome))
        assert {result.qualification for result in results} - {None} <= set(get_args(Qualification))
        assert {result.check for result in results} <= set(get_args(CheckName))


class TestVocabularyCompleteness:
    """Does every path a scenario reads resolve to something the host declared?

    The one check that needs no run, and therefore the one an authoring gate can reuse before a
    subject is scored down for a state nobody set. Its fixtures are the three classes a path
    falls into: one the registry declares, one it declares but no run controls, and one that is a
    typo of a real dimension.
    """

    async def test_paths_over_declared_dimensions_resolve(self) -> None:
        """A plain dimension, a synthetic length, a generator's iterable, and a witnessed one."""
        registry, _state = toyhost_world()

        report = await check_world_conformance(registry, expressions=list(TOY_RESOLVABLE_EXPRESSIONS))

        assert _verdict(report.results, "vocabulary_completeness", None).proved

    async def test_a_judge_only_path_resolves_because_the_vocabulary_exists(self) -> None:
        """A goal check may legitimately read what the subject never saw.

        Refusing it here would answer the authoring gate's question — *should a precondition
        presume this* — with the vocabulary check's verdict, and the two are different questions
        with different remedies.
        """
        registry, _state = toyhost_world()

        report = await check_world_conformance(registry, expressions=[TOY_JUDGE_ONLY_EXPRESSION])

        assert _verdict(report.results, "vocabulary_completeness", None).proved

    async def test_an_unregistered_path_fails_and_is_named(self) -> None:
        """The founding incident's postcondition half, caught without spending a run.

        Named rather than counted: an author told *"a path does not resolve"* has to find it, and
        the whole reason the language is engine-owned is that the engine can say which one.
        """
        registry, _state = toyhost_world()

        report = await check_world_conformance(registry, expressions=[TOY_UNRESOLVABLE_EXPRESSION])
        result = _verdict(report.results, "vocabulary_completeness", None)

        assert result.outcome == "failed"
        assert "documnet_language" in result.detail

    async def test_one_bad_path_among_good_ones_still_fails_and_names_only_itself(self) -> None:
        """Both directions on one fixture: the sound corpus passes and the same corpus plus one
        typo fails, so a check that always failed could not pass this pair."""
        registry, _state = toyhost_world()

        report = await check_world_conformance(
            registry,
            expressions=[*TOY_RESOLVABLE_EXPRESSIONS, TOY_UNRESOLVABLE_EXPRESSION],
        )
        result = _verdict(report.results, "vocabulary_completeness", None)

        assert result.outcome == "failed"
        assert "documnet_language" in result.detail
        assert "document_language'" not in result.detail

    async def test_an_expression_that_cannot_be_parsed_fails_rather_than_being_passed_over(self) -> None:
        """A scenario nothing can read must not sit in a corpus reported conformant."""
        registry, _state = toyhost_world()

        report = await check_world_conformance(registry, expressions=["state.document_language =="])
        result = _verdict(report.results, "vocabulary_completeness", None)

        assert result.outcome == "failed"
        assert "does not parse" in result.detail

    async def test_no_expressions_is_recorded_rather_than_passed(self) -> None:
        """ "Every path resolved" over no paths is vacuously true and would render as a proof."""
        registry, _state = toyhost_world()

        report = await check_world_conformance(registry)
        result = _verdict(report.results, "vocabulary_completeness", None)

        assert result.outcome == "unavailable"
        assert result.qualification == "nothing_to_resolve"
        assert not result.proved

    async def test_expressions_reading_no_world_state_are_the_same_answer(self) -> None:
        """A corpus of ledger predicates exercises no name, so it proves no more than an empty
        one — and the two reaching different verdicts would be the same vacuous pass by another
        door."""
        registry, _state = toyhost_world()

        report = await check_world_conformance(registry, expressions=['called_before("a.b", "c.d")'])
        result = _verdict(report.results, "vocabulary_completeness", None)

        assert result.outcome == "unavailable"
        assert result.qualification == "nothing_to_resolve"

    def test_a_path_addressing_beneath_a_dimension_resolves_to_that_dimension(self) -> None:
        """The registry owns granularity, so resolution takes the longest declared prefix rather
        than demanding the whole path be a name."""
        registry, _state = toyhost_world()

        assert registry.resolve_path("operator_corrections.length") == "operator_corrections"
        assert registry.resolve_path("operator_corrections") == "operator_corrections"
        assert registry.resolve_path("documnet_language") is None


class TestAVerdictRefusesItsOwnIncoherence:
    """Every other record in this contract refuses incoherence where it is written.

    Only this module constructs results and every path it takes is right, so nothing is wrong
    today — which is exactly when a rule is cheap to make structural, and exactly when it
    otherwise stays prose until somebody adds a sixth check. Asserted in the direction that
    matters: the existing suite already proves the guard does not refuse a valid verdict, and
    nothing proved it refuses an invalid one.
    """

    def test_an_unavailable_with_no_reason_is_refused(self) -> None:
        """The reasons a proof was not reached ARE the disclosure surface, so a gap with none is
        a row a coverage reader cannot act on."""
        with pytest.raises(WorldConformanceError, match="no qualification"):
            ConformanceResult(check="round_trip", outcome="unavailable", detail="something stopped it")

    def test_a_failure_carrying_a_gap_reason_is_refused(self) -> None:
        """A failure names its defect in its detail; a gap reason beside it is a second answer."""
        with pytest.raises(WorldConformanceError, match="qualified it"):
            ConformanceResult(
                check="round_trip",
                outcome="failed",
                detail="the value did not read back",
                qualification="arming_only",
            )

    def test_a_pass_may_only_be_narrowed_by_the_one_reason_that_narrows_a_pass(self) -> None:
        """``plumbing_only`` qualifies a pass; every other reason replaces one."""
        with pytest.raises(WorldConformanceError, match="plumbing_only"):
            ConformanceResult(
                check="round_trip",
                outcome="passed",
                detail="it carried",
                qualification="no_perturbation_binding",
            )

        assert (
            ConformanceResult(
                check="round_trip",
                outcome="passed",
                detail="the plumbing carried the value",
                qualification="plumbing_only",
            ).outcome
            == "passed"
        )


class TestTheVocabularyFailureReadsAsASentence:
    """Both halves of the failure are reported, and either alone still reads."""

    async def test_unresolved_and_unreadable_are_reported_together(self) -> None:
        """A reader fixing a corpus wants the whole list, not one pass per typo."""
        registry, _state = toyhost_world()

        report = await check_world_conformance(
            registry,
            expressions=[TOY_UNRESOLVABLE_EXPRESSION, "state.document_language =="],
        )
        result = _verdict(report.results, "vocabulary_completeness", None)

        assert result.outcome == "failed"
        assert "documnet_language" in result.detail
        assert "does not parse" in result.detail
        assert not result.detail.startswith("and ")


def _shelf_world(*, summary: str) -> tuple[WorldRegistry, dict[str, Any]]:
    """A one-dimension world whose value reaches the subject on two surfaces: in full, and summarised.

    The value is an object, whose generated values always populate every property — so the
    generated pair is two non-empty values, and only the schema's empty boundary can move a summary
    that says whether anything is there.

    Args:
        summary: How the summary surface renders — ``"wired"`` reads the seeded value (``none`` or
            ``some``), ``"stuck"`` shows a fresh component's default whatever was seeded.

    Returns:
        The registry and the state it seeds into.
    """
    state: dict[str, Any] = {"featured": {"title": "a"}}

    def seed(value: Any) -> None:
        state["featured"] = dict(value)

    def view(*, surfaces: tuple[str, ...]) -> dict[str, str]:
        featured = state["featured"]
        bodies = {
            "shelf": repr(featured),
            "badge": {"wired": "some" if featured else "none", "stuck": "none"}[summary],
        }
        return {surface: bodies[surface] for surface in surfaces}

    registry = WorldRegistry(
        (
            WorldDimension(
                carrier="shop",
                name="featured_item",
                schema={"type": "object", "properties": {"title": {"type": "string"}}},
                matters="a promotion scenario presumes what is featured, or that nothing is",
                seed="shop.seed",
                read="shop.read",
                perceived_by=("shelf", "badge"),
            ),
        ),
        bindings={"shop.seed": seed, "shop.read": lambda: dict(state["featured"]), "shop.view": view},
        subject_view="shop.view",
    )
    return registry, state


class TestEverySurfaceAnswersToTheDimension:
    """A dimension reaching the subject several ways is held to each, not to the whole view."""

    async def test_a_summary_stuck_on_its_default_fails_beside_a_full_rendering_that_moves(self) -> None:
        registry, _state = _shelf_world(summary="stuck")
        result = _verdict((await check_world_conformance(registry)).results, "perception_ab", "featured_item")

        assert result.outcome == "failed"
        assert "['badge']" in result.detail
        assert "'shelf'" not in result.detail.split("surface(s)")[1].split("rendered")[0]

    async def test_a_summary_that_moves_only_at_the_empty_boundary_passes(self) -> None:
        """The generated pair is two non-empty objects; only the empty boundary moves the badge."""
        registry, _state = _shelf_world(summary="wired")
        result = _verdict((await check_world_conformance(registry)).results, "perception_ab", "featured_item")

        assert result.outcome == "passed"
        assert "{}" in result.detail  # the empty boundary was among the values tried

    async def test_ambient_isolation_watches_every_named_surface(self) -> None:
        registry, state = _shelf_world(summary="wired")
        ambient = WorldRegistry(
            registry.declarations,
            bindings={**registry.bindings, "shop.jostle": lambda: state.update(jostled=True)},
            subject_view="shop.view",
            perturb_ambient="shop.jostle",
        )
        seen: list[tuple[str, ...]] = []
        original = ambient.bindings["shop.view"]

        def watching(*, surfaces: tuple[str, ...]) -> dict[str, str]:
            seen.append(surfaces)
            return original(surfaces=surfaces)

        watched = WorldRegistry(
            ambient.declarations,
            bindings={**ambient.bindings, "shop.view": watching},
            subject_view="shop.view",
            perturb_ambient="shop.jostle",
        )
        assert _verdict((await check_world_conformance(watched)).results, "ambient_isolation", None).outcome == "passed"
        assert ("shelf", "badge") in seen


class TestPerceptionStillness:
    """The other half of A/B: a surface a dimension does not name must not move when it does.

    A/B proves the named surfaces move. A leak that ADDS hidden state to a surface — the answer key
    printed beside the work — leaves every declared distinction intact, so A/B stays green; only this
    check sees it. Each fault is paired with the sound toy world it was derived from.
    """

    async def test_the_sound_toy_world_holds_every_unnamed_surface_still(self) -> None:
        report = await check_world_conformance(toyhost_world()[0])
        stillness = {r.dimension: r for r in report.results if r.check == "perception_stillness"}

        assert set(stillness) == {declared.name for declared in toyhost_world()[0].declarations}
        assert report.failures == ()
        # supervisor_signoff only a person moves: its judge-only claim is recorded as unchecked, never passed.
        assert stillness["supervisor_signoff"].qualification == "not_instantiable_unattended"
        assert all(r.proved for name, r in stillness.items() if name != "supervisor_signoff"), stillness

    async def test_the_judge_only_claim_is_checked_against_every_surface(self) -> None:
        result = _verdict(await _report(), "perception_stillness", "vendor_template")

        assert result.proved
        assert "perceived by no surface" in result.detail
        assert "'document_header'" in result.detail and "'operator_context'" in result.detail

    @pytest.mark.parametrize(
        ("fault", "dimension", "surface"),
        [
            (ToyWorldFaults(operator_context_shows_vendor_template=True), "vendor_template", "operator_context"),
            (ToyWorldFaults(document_header_shows_payment_hold=True), "payment_hold", "document_header"),
        ],
        ids=["judge_only_leaks", "unnamed_surface_moves"],
    )
    async def test_a_leak_turns_exactly_this_check_red(
        self, fault: ToyWorldFaults, dimension: str, surface: str
    ) -> None:
        """One defect, one finding: the leak fails stillness for the leaking dimension and nothing else.

        The sound half is asserted on the same dimension, so an implementation that always fails cannot pass.
        """
        sound = _verdict(await _report(), "perception_stillness", dimension)
        report = await check_world_conformance(toyhost_world(faults=fault)[0])

        assert sound.outcome == "passed"
        assert [(r.check, r.dimension) for r in report.failures] == [("perception_stillness", dimension)]
        (failed,) = report.failures
        assert f"['{surface}']" in failed.detail
        assert "shows this dimension without declaring it" in failed.detail
        # A/B over the leaking dimension is untouched: the leak adds, it erases nothing.
        if dimension == "payment_hold":
            assert _verdict(report.results, "perception_ab", dimension).outcome == "passed"

    async def test_a_seed_that_never_lands_is_not_reported_as_a_still_surface(self) -> None:
        """Round trip owns a dead seeder; stillness records that it never reached its question."""
        results = await _report(faults=ToyWorldFaults(seeding_vendor_template_does_nothing=True))
        result = _verdict(results, "perception_stillness", "vendor_template")

        assert (result.outcome, result.qualification) == ("unavailable", "seeding_did_not_take")
        assert _verdict(results, "round_trip", "vendor_template").outcome == "failed"

    async def test_a_dimension_nothing_can_move_is_unproved_rather_than_passed(self) -> None:
        results = await _report(optional_capabilities=False)

        assert _verdict(results, "perception_stillness", "ingest_backlog").qualification == "no_perturbation_binding"
        assert _verdict(results, "perception_stillness", "payment_hold").qualification == "arming_only"

    async def test_a_world_with_no_surfaces_has_nothing_to_watch(self) -> None:
        """The courier's judge-only dimension: no subject view at all, so recorded, never passed."""
        report = await check_world_conformance(courier_world())
        result = _verdict(report.results, "perception_stillness", "road_closures")

        assert report.failures == ()
        assert (result.outcome, result.qualification) == ("unavailable", "nothing_to_observe")

    async def test_a_dimension_perceived_by_every_surface_has_nothing_to_watch(self) -> None:
        registry, _state = _shelf_world(summary="wired")
        result = _verdict((await check_world_conformance(registry)).results, "perception_stillness", "featured_item")

        assert (result.outcome, result.qualification) == ("unavailable", "nothing_to_observe")
        assert "every surface this registry names" in result.detail


def _mirror_world(*, leak: bool = False, third: bool = True) -> WorldRegistry:
    """Three dimensions on three surfaces, where seeding ``a`` also writes ``b`` — a shared write path.

    Args:
        leak: Whether the ``c`` surface also shows ``a``.
        third: Whether ``c`` is declared at all; without it, ``b_view`` is ``a``'s only other surface.

    Returns:
        The registry.
    """
    state = {"a": "x", "b": "x", "c": "x"}

    def seed_a(value: str) -> None:
        state["a"] = state["b"] = value

    def view(*, surfaces: tuple[str, ...]) -> dict[str, str]:
        bodies = {"a_view": state["a"], "b_view": state["b"], "c_view": state["c"] + (state["a"] if leak else "")}
        return {surface: bodies[surface] for surface in surfaces}

    def dimension(name: str) -> WorldDimension:
        return WorldDimension(
            carrier="rig",
            name=name,
            schema={"enum": ["x", "y", "z"]},
            matters=f"a scenario presumes {name}",
            seed=f"{name}.seed",
            read=f"{name}.read",
            perceived_by=(f"{name}_view",),
        )

    return WorldRegistry(
        (dimension("a"), dimension("b"), *((dimension("c"),) if third else ())),
        bindings={
            "a.seed": seed_a,
            "b.seed": lambda value: state.__setitem__("b", value),
            "c.seed": lambda value: state.__setitem__("c", value),
            "a.read": lambda: state["a"],
            "b.read": lambda: state["b"],
            "c.read": lambda: state["c"],
            "view": view,
        },
        subject_view="view",
    )


class TestMovementASiblingAccountsForIsNotALeak:
    """A surface perceiving a sibling that moved with the dimension moves for a declared reason."""

    async def test_the_sibling_s_surface_is_excused_and_named_and_independence_owns_the_defect(self) -> None:
        results = (await check_world_conformance(_mirror_world(leak=False))).results
        result = _verdict(results, "perception_stillness", "a")

        assert result.outcome == "passed", result.detail
        assert "['b_view'] were not judged" in result.detail
        assert "['c_view']" in result.detail
        assert _verdict(results, "independence", "b").outcome == "failed"

    async def test_the_remaining_surfaces_are_still_judged(self) -> None:
        result = _verdict(
            (await check_world_conformance(_mirror_world(leak=True))).results, "perception_stillness", "a"
        )

        assert result.outcome == "failed"
        assert "['c_view']" in result.detail

    async def test_with_every_other_surface_excused_there_is_nothing_left_to_judge(self) -> None:
        result = _verdict(
            (await check_world_conformance(_mirror_world(third=False))).results, "perception_stillness", "a"
        )

        assert (result.outcome, result.qualification) == ("unavailable", "nothing_to_observe")
        assert "also perceives a sibling that moved" in result.detail


class TestPerceivingSurfacesAreDeclaredAsATuple:
    @pytest.mark.parametrize("surfaces", ["shelf", ("shelf", "shelf"), ("shelf", "")])
    def test_a_bare_string_a_duplicate_or_a_blank_is_refused(self, surfaces: object) -> None:
        from threetears.evals.contracts.host.world import WorldRegistrationError

        with pytest.raises(WorldRegistrationError, match="perceived_by|perceiving surface"):
            WorldDimension(
                carrier="shop",
                name="shelf_stock",
                schema={"type": "string"},
                matters="m",
                seed="s",
                read="r",
                perceived_by=surfaces,  # type: ignore[arg-type]
            )


def _deck_world(*, base: bool, coherent: bool) -> tuple[WorldRegistry, dict[str, Any]]:
    """Two dimensions coupled the way a work deck couples them: a queue over an idle deck starts its head.

    Seeding either one re-applies both and lets the deck settle, as a host whose seed writes production
    state does: when nothing is on the deck and something is queued, the head goes into work. So a queue seeded
    over an idle deck reads back one short, and a deck idled under a queue reads back busy — a coupling
    the host's own production has, not a defect in either seeder.

    Args:
        base: Whether the host names the base world a check composes over (something on the deck).
        coherent: Whether the host binds the coherence handle saying it does not hold a queue over an idle deck.

    Returns:
        The registry and the state it moves.
    """
    state: dict[str, Any] = {"deck": "", "queue": [], "seeded": {"deck": "", "queue": []}}

    def settle() -> None:
        state["deck"], state["queue"] = state["seeded"]["deck"], list(state["seeded"]["queue"])
        if not state["deck"] and state["queue"]:
            state["deck"] = state["queue"].pop(0)

    def seed(key: str) -> Any:
        def write(value: Any) -> None:
            state["seeded"][key] = value
            settle()

        return write

    def holds(world: dict[str, Any]) -> list[str]:
        return ["a queue over an idle deck starts its head"] if world.get("queue") and not world.get("deck") else []

    # The queue is declared first, so the first check meets it over the idle deck a fresh world starts with.
    registry = WorldRegistry(
        (
            WorldDimension(
                carrier="deck",
                name="queue",
                schema={"type": "array", "items": {"type": "string"}},
                matters="a scenario presumes what is lined up next",
                seed="queue.seed",
                read="queue.read",
            ),
            WorldDimension(
                carrier="deck",
                name="deck",
                schema={"enum": ["", "busy", "a", "b"]},
                matters="a scenario presumes what is in work, or that nothing is",
                seed="deck.seed",
                read="deck.read",
            ),
        ),
        bindings={
            "deck.seed": seed("deck"),
            "deck.read": lambda: state["deck"],
            "queue.seed": seed("queue"),
            "queue.read": lambda: list(state["queue"]),
            "deck.holds": holds,
        },
        base_world={"deck": "busy", "queue": []} if base else None,
        coherence="deck.holds" if coherent else None,
    )
    return registry, state


class TestAHostThatDeclaresItsCoupling:
    """A coupled host names a base world and a coherence handle, and the kit composes only worlds it holds.

    Each verdict is asserted with and without the declaration on one fixture, so the declaration is shown
    to be what moves it — rather than a kit that passes coupled hosts whatever they declare.
    """

    async def test_over_an_idle_deck_a_queue_cannot_round_trip(self) -> None:
        registry, _state = _deck_world(base=False, coherent=False)

        assert _verdict((await check_world_conformance(registry)).results, "round_trip", "queue").outcome == "failed"

    async def test_over_the_base_world_it_does(self) -> None:
        registry, _state = _deck_world(base=True, coherent=False)

        assert _verdict((await check_world_conformance(registry)).results, "round_trip", "queue").outcome == "passed"

    async def test_without_coherence_idling_the_deck_reads_as_a_clobber(self) -> None:
        """Independence moves the deck off what it holds — to idle, first — and the queue's head goes into work."""
        registry, _state = _deck_world(base=True, coherent=False)

        result = _verdict((await check_world_conformance(registry)).results, "independence", "queue")

        assert result.outcome == "failed", result.detail

    async def test_with_coherence_the_deck_moves_to_a_value_the_host_holds(self) -> None:
        registry, _state = _deck_world(base=True, coherent=True)

        result = _verdict((await check_world_conformance(registry)).results, "independence", "queue")

        assert result.outcome == "passed", result.detail
        assert "survived setting deck" in result.detail

    async def test_a_sibling_pinned_beside_the_first_value_is_composed_beside_another(self) -> None:
        """An idle deck pins the queue empty, so the queue is moved beside a deck the host holds a queue under."""
        registry, _state = _deck_world(base=True, coherent=True)

        result = _verdict((await check_world_conformance(registry)).results, "independence", "deck")

        assert result.outcome == "passed", result.detail
        assert "deck survived setting queue" in result.detail
        assert "queue beside deck='a'" in result.detail


class TestTheToyHostDeclaresItsCoupling:
    """The reference host's coupled world, proved through the base world and coherence handle it declares.

    A born-digital invoice was never scanned, so the page reader measures it clean whatever scan quality was
    seeded. Each declaration is shown to be what moves its verdict by dropping it from the same registry.
    """

    @staticmethod
    def _without(*, base_world: bool = True, coherence: bool = True) -> WorldRegistry:
        registry, _state = toyhost_world()
        return WorldRegistry(
            registry.declarations,
            bindings=registry.bindings,
            subject_view=registry.subject_view,
            perturb_ambient=registry.perturb_ambient,
            base_world=registry.base_world if base_world else None,
            coherence=registry.coherence if coherence else None,
        )

    async def test_through_both_declarations_nothing_fails(self) -> None:
        registry, _state = toyhost_world()

        report = await check_world_conformance(registry)

        assert report.failures == ()
        assert {result.outcome for result in report.for_dimension("scan_quality")} == {"passed"}
        assert {result.outcome for result in report.for_dimension("vendor_template")} == {"passed"}

    async def test_without_coherence_the_coupling_reads_as_a_clobber(self) -> None:
        result = _verdict(
            (await check_world_conformance(self._without(coherence=False))).results, "independence", "scan_quality"
        )

        assert result.outcome == "failed"
        assert "setting vendor_template left it 'clean'" in result.detail

    async def test_without_the_base_world_scan_quality_cannot_move_and_fails(self) -> None:
        """A fresh world starts on an e-invoice, where the host holds no scan quality but clean."""
        result = _verdict(
            (await check_world_conformance(self._without(base_world=False))).results, "round_trip", "scan_quality"
        )

        assert (result.outcome, result.qualification) == ("failed", None)
        assert "born digital" in result.detail

    @pytest.mark.parametrize("template", ["peppol-einvoice", "acme-2019", "globex-2021"])
    @pytest.mark.parametrize("scan", ["clean", "skewed", "faint"])
    async def test_the_coherence_handle_and_the_page_reader_give_one_answer(self, template: str, scan: str) -> None:
        """The handle states before seeding exactly what the world does after it."""
        registry, _state = toyhost_world()
        await registry.call("toy.seed_vendor_template", template)
        await registry.call("toy.seed_scan_quality", scan)

        held = not await registry.call("toy.holds", {"vendor_template": template, "scan_quality": scan})

        assert held == (await registry.call("toy.read_scan_quality") == scan)


class TestACouplingNeverDecidesWhetherACheckRuns:
    """The coherence handle narrows which values a check draws; it cannot turn a check into a gap.

    A handle answering "not held" for every value a check could use would otherwise be a waiver the host
    authored, and the kit's one rule is that obligations are derived from shape and never waived. So a
    coupling that starves a check fails it, naming the coupling, and ``unavailable`` keeps meaning what the
    declared shape forces.
    """

    async def test_a_dimension_with_no_value_the_host_holds_over_its_base_world_fails(self) -> None:
        """Over an idle deck with no base world, the host holds no queue at all — a contradiction, named."""
        registry, _state = _deck_world(base=False, coherent=True)

        result = _verdict((await check_world_conformance(registry)).results, "round_trip", "queue")

        assert (result.outcome, result.qualification) == ("failed", None)
        assert "a queue over an idle deck starts its head" in result.detail
        assert "name a base world in which the dimension can move" in result.detail

    async def test_a_handle_refusing_every_world_switches_no_check_off(self) -> None:
        """The waiver the bound exists to refuse: every check that draws a value fails, none goes unavailable.

        The two that stay ``unavailable`` are forced by shape, not by the handle: a dimension only a person
        brings into being, and a vocabulary check handed no expressions.
        """
        registry, _state = toyhost_world()
        refusing = WorldRegistry(
            registry.declarations,
            bindings={**registry.bindings, "toy.refuses": lambda world: ["this host holds nothing"]},
            subject_view=registry.subject_view,
            perturb_ambient=registry.perturb_ambient,
            base_world=registry.base_world,
            coherence="toy.refuses",
        )

        report = await check_world_conformance(refusing)

        assert {result.qualification for result in report.unavailable} == {
            "not_instantiable_unattended",
            "nothing_to_resolve",
        }
        assert all(result.dimension == "supervisor_signoff" for result in report.unavailable if result.dimension)
        drawn = [result for result in report.results if result.qualification is None]
        assert {result.check for result in drawn} == {
            "round_trip",
            "perception_ab",
            "perception_stillness",
            "independence",
            "ambient_isolation",
        }
        assert all(result.outcome == "failed" for result in drawn), [(r.check, r.dimension, r.outcome) for r in drawn]
        assert all("this host holds nothing" in result.detail for result in drawn)

    async def test_a_perceived_dimension_s_a_b_fails_rather_than_going_unavailable(self) -> None:
        """Perception A/B draws two values; a handle holding fewer over the base world fails it too."""
        registry, _state = toyhost_world()
        refusing = WorldRegistry(
            registry.declarations,
            bindings={
                **registry.bindings,
                "toy.refuses_skew": lambda world: ["skew"] if world.get("scan_quality") != "clean" else [],
            },
            subject_view=registry.subject_view,
            perturb_ambient=registry.perturb_ambient,
            base_world=registry.base_world,
            coherence="toy.refuses_skew",
        )

        result = _verdict((await check_world_conformance(refusing)).results, "perception_ab", "scan_quality")

        assert (result.outcome, result.qualification) == ("failed", None)
        assert "needs 2 distinct value(s)" in result.detail

    async def test_a_pair_the_host_never_holds_moving_together_fails(self) -> None:
        """A queue held only under the one deck value the base world sets can never move beside the deck."""
        registry, _state = _deck_world(base=True, coherent=True)
        pinned = WorldRegistry(
            registry.declarations,
            bindings={
                **registry.bindings,
                "deck.pins": lambda world: (
                    ["a queue runs only on a busy deck"] if world.get("queue") and world.get("deck") != "busy" else []
                ),
            },
            base_world=registry.base_world,
            coherence="deck.pins",
        )

        result = _verdict((await check_world_conformance(pinned)).results, "independence", "deck")

        assert (result.outcome, result.qualification) == ("failed", None)
        assert "can never move together" in result.detail
        assert "a queue runs only on a busy deck" in result.detail


class TestAnAmbientRigThatSaysWhatItMoved:
    """A rig may name what it moved; the verdict carries the names, and a rig that moved nothing proves nothing."""

    @staticmethod
    def _registry(moved: Any) -> WorldRegistry:
        registry, _state = _shelf_world(summary="wired")
        return WorldRegistry(
            registry.declarations,
            bindings={**registry.bindings, "shop.jostle": lambda: moved},
            subject_view="shop.view",
            perturb_ambient="shop.jostle",
        )

    async def test_the_verdict_names_what_moved(self) -> None:
        result = _verdict(
            (await check_world_conformance(self._registry(["till", "lights"]))).results, "ambient_isolation", None
        )

        assert result.outcome == "passed"
        assert result.detail.endswith("The rig moved: till, lights.")

    async def test_a_rig_that_moved_nothing_is_unavailable(self) -> None:
        result = _verdict((await check_world_conformance(self._registry([]))).results, "ambient_isolation", None)

        assert (result.outcome, result.qualification) == ("unavailable", "no_perturbation_binding")

    async def test_a_rig_that_does_not_report_is_taken_at_its_word(self) -> None:
        result = _verdict((await check_world_conformance(self._registry(None))).results, "ambient_isolation", None)

        assert result.outcome == "passed"
        assert "The rig moved" not in result.detail


def _flag_world(stores: Any, *, perceived: bool = True) -> tuple[WorldRegistry, dict[str, Any]]:
    """One boolean dimension whose seeder stores ``stores(value)`` — the identity, or ``int`` for a lossy one.

    Args:
        stores: What the seeder writes for the value it is handed.
        perceived: Whether a surface renders it.

    Returns:
        The registry and its state.
    """
    state: dict[str, Any] = {"value": True}
    registry = WorldRegistry(
        (
            WorldDimension(
                carrier="rig",
                name="door_open",
                schema={"type": "boolean"},
                matters="a cold-chain alarm scenario presumes whether the door is open",
                seed="h.seed",
                read="h.read",
                perceived_by=("panel",) if perceived else (),
            ),
        ),
        bindings={
            "h.seed": lambda value: state.__setitem__("value", stores(value)),
            "h.read": lambda: state["value"],
            "h.view": lambda **_: {"panel": repr(state["value"])},
        },
        subject_view="h.view",
    )
    return registry, state


class TestEveryComparisonIsJSONEquality:
    """Each place the kit compares what a world holds uses JSON equality, where ``0`` is not ``False``.

    One test per comparison, each on a world built so that Python's ``==`` and JSON disagree exactly
    there — so replacing any one of them with ``==`` turns its own test red.
    """

    async def test_put_does_not_take_a_number_for_the_boolean_it_seeded(self) -> None:
        """``_put``: a world that stored ``int(False)`` has not been put at ``False``."""
        registry, _state = _flag_world(int)

        result = _verdict((await check_world_conformance(registry)).results, "perception_ab", "door_open")

        assert (result.outcome, result.qualification) == ("unavailable", "seeding_did_not_take")

    async def test_independence_sees_a_sibling_turn_a_boolean_into_its_number(self) -> None:
        """The clobber check: a sibling whose seeder rewrites ``False`` as ``0`` has clobbered it."""
        state: dict[str, Any] = {"door": True, "temp": 5}

        def seed_temp(value: int) -> None:
            state["temp"] = value
            state["door"] = int(state["door"])

        registry = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="door_open",
                    schema={"type": "boolean"},
                    matters="a cold-chain alarm scenario presumes whether the door is open",
                    seed="door.seed",
                    read="door.read",
                ),
                WorldDimension(
                    carrier="rig",
                    name="setpoint",
                    schema={"type": "integer"},
                    matters="a cold-chain scenario presumes the cabinet's setpoint",
                    seed="temp.seed",
                    read="temp.read",
                ),
            ),
            bindings={
                "door.seed": lambda value: state.__setitem__("door", value),
                "door.read": lambda: state["door"],
                "temp.seed": seed_temp,
                "temp.read": lambda: state["temp"],
            },
        )

        result = _verdict((await check_world_conformance(registry)).results, "independence", "door_open")

        assert result.outcome == "failed", result.detail

    async def test_independence_does_not_count_a_sibling_holding_the_number_as_moved(self) -> None:
        """The moved check: a sibling that stored ``0`` for ``False`` was not moved to ``False``."""
        state: dict[str, Any] = {"label": "z", "door": 0}
        registry = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="label",
                    schema={"type": "string"},
                    matters="a scenario presumes the shelf label",
                    seed="label.seed",
                    read="label.read",
                ),
                WorldDimension(
                    carrier="rig",
                    name="door_open",
                    schema={"type": "boolean"},
                    matters="a cold-chain alarm scenario presumes whether the door is open",
                    seed="door.seed",
                    read="door.read",
                ),
            ),
            bindings={
                "label.seed": lambda value: state.__setitem__("label", value),
                "label.read": lambda: state["label"],
                "door.seed": lambda value: state.__setitem__("door", int(value)),
                "door.read": lambda: state["door"],
            },
        )

        result = _verdict((await check_world_conformance(registry)).results, "independence", "label")

        assert "no sibling of label could be moved off the value it already held" in result.detail, result.detail
        assert "(door_open)" in result.detail

    @staticmethod
    def _counter_world() -> tuple[WorldRegistry, list[Any]]:
        """A count that is a boolean or a number, whose seeder ignores a zero — so ``0`` is never held.

        The generated pair is ``(False, -1)`` and the integer branch's edge is ``0``, which Python's ``==``
        takes for ``False``.

        Returns:
            The registry, and every value its seeder was handed.
        """
        state: dict[str, Any] = {"value": True}
        handed: list[Any] = []

        def seed(value: Any) -> None:
            handed.append(value)
            if not (isinstance(value, int) and not isinstance(value, bool) and value == 0):
                state["value"] = value

        registry = WorldRegistry(
            (
                WorldDimension(
                    carrier="rig",
                    name="pallet_count",
                    schema={"anyOf": [{"type": "boolean"}, {"type": "integer", "minimum": -1}]},
                    matters="a stocktake scenario presumes either a flag or a count on the pallet",
                    seed="h.seed",
                    read="h.read",
                    perceived_by=("panel",),
                ),
            ),
            bindings={
                "h.seed": seed,
                "h.read": lambda: state["value"],
                "h.view": lambda **_: {"panel": repr(state["value"])},
            },
            subject_view="h.view",
        )
        return registry, handed

    async def test_the_boundary_dedupe_does_not_drop_zero_as_false(self) -> None:
        """``_boundary_values_beside``: ``0`` is an edge of its own, not the ``False`` already in the pair."""
        registry, handed = self._counter_world()

        await check_world_conformance(registry)

        assert any(json_equal(value, 0) for value in handed), handed

    async def test_perception_skips_a_boundary_it_could_not_hold_rather_than_calling_it_generated(self) -> None:
        """Perception A/B: ``0`` not landing is a boundary passed over, not the generated ``False`` failing."""
        registry, _handed = self._counter_world()

        result = _verdict((await check_world_conformance(registry)).results, "perception_ab", "pallet_count")

        assert result.outcome == "passed", result.detail
        assert "boundary value(s) [0] were not held" in result.detail


#: A list whose elements take one of two shapes, each perceived on a surface of its own — the shape of a
#: work queue holding tasks and the notes filed between them. Keyed by ``id``, as a queue under ids is.
_TRAY_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "anyOf": [
            {
                "type": "object",
                "properties": {"id": {"type": "string"}, "task": {"type": "string"}},
                "required": ["id", "task"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"id": {"type": "string"}, "note": {"type": "string"}},
                "required": ["id", "note"],
                "additionalProperties": False,
            },
        ]
    },
}


def _tray_world() -> WorldRegistry:
    """A one-dimension world whose list renders each element shape on its own surface.

    Its coherence handle refuses a tray holding two elements under one id, as a keyed host does, so a
    value that covered both shapes by repeating a key would not be one the kit may try.

    Returns:
        The registry.
    """
    state: dict[str, Any] = {"tray": []}

    def view(*, surfaces: tuple[str, ...]) -> dict[str, str]:
        tray = state["tray"]
        bodies = {
            "tasks": ", ".join(item["task"] for item in tray if "task" in item) or "(no tasks)",
            "notes": ", ".join(item["note"] for item in tray if "note" in item) or "(no notes)",
        }
        return {surface: bodies[surface] for surface in surfaces}

    def coherence(world: dict[str, Any]) -> list[str]:
        ids = [item["id"] for item in world.get("tray", [])]
        return ["two elements share an id"] if len(ids) != len(set(ids)) else []

    return WorldRegistry(
        (
            WorldDimension(
                carrier="desk",
                name="tray",
                schema=_TRAY_SCHEMA,
                matters="a triage scenario presumes the tasks waiting and the notes filed between them",
                seed="desk.seed",
                read="desk.read",
                perceived_by=("tasks", "notes"),
            ),
        ),
        bindings={
            "desk.seed": lambda value: state.update(tray=[dict(item) for item in value]),
            "desk.read": lambda: [dict(item) for item in state["tray"]],
            "desk.view": view,
            "desk.coherence": coherence,
        },
        subject_view="desk.view",
        coherence="desk.coherence",
    )


class TestAListOfShapesIsPerceivedInEveryShape:
    """A list whose schema admits several element shapes is seeded with every one of them.

    The values perception A/B tries are an empty list and one more, and that one used to hold a single
    element — always the first shape — so a surface reading a later shape rendered its empty state on
    every value and a correct renderer could not be declared.
    """

    async def test_both_surfaces_move_so_both_may_be_declared(self) -> None:
        result = _verdict((await check_world_conformance(_tray_world())).results, "perception_ab", "tray")

        assert result.outcome == "passed", result.detail


# =============================================================================
# One defect, one finding — over every fault the toy world can suffer
# =============================================================================

#: Each fault, and the one verdict it must turn red. The kit's honesty claim is this table: a fault that also
#: turns another check red sends the host to fix correct code.
_FAULT_RED_SETS: dict[str, list[tuple[str, str | None]]] = {
    "seeding_vendor_template_does_nothing": [("round_trip", "vendor_template")],
    "document_header_drops_language": [("perception_ab", "document_language")],
    "operator_context_reads_the_processing_shift": [("ambient_isolation", None)],
    "operator_context_shows_vendor_template": [("perception_stillness", "vendor_template")],
    "document_header_shows_payment_hold": [("perception_stillness", "payment_hold")],
    "seeding_scan_quality_resets_language": [("independence", "document_language")],
    "arming_payment_hold_names_no_event": [("round_trip", "payment_hold")],
}


def test_the_red_set_table_names_every_fault() -> None:
    """A fault added to the toy world without a row here would go unchecked while the table claims every one."""
    assert set(_FAULT_RED_SETS) == {field.name for field in dataclasses.fields(ToyWorldFaults)}


@pytest.mark.parametrize("fault", sorted(_FAULT_RED_SETS))
async def test_each_fault_turns_exactly_its_owning_verdict_red(fault: str) -> None:
    """The sound world turns nothing red, so the red set below is the fault's and nothing else's."""
    assert (await check_world_conformance(toyhost_world()[0])).failures == ()

    report = await check_world_conformance(toyhost_world(faults=ToyWorldFaults(**{fault: True}))[0])

    assert [(result.check, result.dimension) for result in report.failures] == _FAULT_RED_SETS[fault]


async def test_a_base_value_that_never_lands_leaves_every_check_composing_over_it_unable_to_start() -> None:
    """The dead vendor_template seeder: the base world's acme-2019 never lands, so the coherence handle answers
    against a world nobody named. Every check composing over the base world records that, naming the base
    dimension, instead of failing on another dimension's account."""
    results = await _report(faults=ToyWorldFaults(seeding_vendor_template_does_nothing=True))

    for check, dimension in [
        ("round_trip", "scan_quality"),
        ("perception_ab", "scan_quality"),
        ("perception_stillness", "scan_quality"),
        ("independence", "scan_quality"),
        ("independence", "document_language"),
        ("ambient_isolation", None),
    ]:
        result = _verdict(results, check, dimension)  # type: ignore[arg-type]
        assert (result.outcome, result.qualification) == ("unavailable", "seeding_did_not_take"), (check, dimension)
        assert "vendor_template did not read back its base value" in result.detail
