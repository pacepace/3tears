"""The check-controls gate, graded over a call ledger and a name-keyed end state.

Each goal check is graded against the do-nothing control (the template's seed, named through the
host's world registry, and an empty ledger) and its named control (the seed with the stated
dimensions replaced, and the stated calls recorded) — through ``grade_goal_checks``, the function a
kind grades a live cell with.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from threetears.evals.contracts import (
    ControlEndState,
    EvalTemplate,
    Firings,
    GoalCheckControl,
    GoalCheckControls,
    RecordedCall,
    ValidationFailedError,
    WorldSeed,
)
from threetears.evals.run.check_controls import (
    check_discriminations,
    control_end_state,
    do_nothing_end_state,
    refuse_non_discriminating_checks,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.kind import PAYMENT_HOLD
from packages.evals.tests.fixtures.toyhost.run import EVERY_FIELD_EMITTED, toyhost_template

#: The toy host's clock-driven dimension: its trigger is the passage of turns, which the toy template's
#: seed arms.
_CLOCK = "operator_corrections"
#: A human-triggered dimension the seed does not arm, which the seeded hold's event can also move.
_SIGNOFF = "supervisor_signoff"

#: A hold check over world state: the document's language stays as the seed put it.
#:
#: Over a dimension set at t=0. A triggered one that waits on an event (the payment hold) is armed by the
#: seed rather than set by it, so the do-nothing control does not hold its seeded value; one that waits on
#: the clock (the operator's corrections) arrives with no candidate action at all, so a check reading it
#: grades the clock rather than the candidate.
_LANGUAGE_UNTOUCHED = 'state.document_language == "de"'


def _with_hold_check() -> EvalTemplate:
    template = toyhost_template()
    controls = template.goal_check_controls
    assert controls is not None
    return template.model_copy(
        update={
            "goal_state_checks": [*template.goal_state_checks, _LANGUAGE_UNTOUCHED],
            "goal_check_controls": GoalCheckControls(
                checks=[
                    *controls.checks,
                    GoalCheckControl(check=_LANGUAGE_UNTOUCHED, intent="hold", control="language-rewritten"),
                ],
                end_states={
                    **controls.end_states,
                    "language-rewritten": ControlEndState(
                        describes="The extractor rewrote the document's language to the one its labels are in.",
                        world={"page_reader": {"document_language": "fr"}},
                    ),
                },
            ),
        }
    )


def test_the_do_nothing_control_is_the_seed_named_by_dimension_with_no_calls_and_the_clock_run() -> None:
    """``operator_corrections`` is turn-triggered: turns pass with no candidate action, so it fires and arrives."""
    profile = toyhost_profile()
    template = toyhost_template()
    assert template.world_seed.namespaces["console"] == {_CLOCK: ["reprice"]}

    idle = do_nothing_end_state(template, world=profile.world)

    assert idle.end_state == {
        "document_language": "de",
        "scan_quality": "faint",
        "vendor_template": "acme-2019",
        _CLOCK: ["reprice"],
    }
    assert idle.ledger.calls == []
    assert idle.fired == Firings(dimensions=frozenset({_CLOCK}), armed=frozenset({_CLOCK}))


def test_an_event_triggered_dimension_the_seed_arms_neither_fires_nor_arrives_when_nothing_is_done() -> None:
    profile = toyhost_profile()
    template = toyhost_template()
    held = template.model_copy(
        update={"world_seed": WorldSeed(namespaces={"console": {PAYMENT_HOLD: "held"}, "page_reader": {}})}
    )

    idle = do_nothing_end_state(held, world=profile.world)

    assert PAYMENT_HOLD not in idle.end_state, "its condition is an event, and nothing brought it about"
    # The clock still runs: an unarmed clock-driven dimension may fire on the world's own clock, and is
    # not the seed's armed event.
    assert idle.fired == Firings(dimensions=frozenset({_CLOCK}))


def _single_check(
    template: EvalTemplate, check: str, intent: str, control: ControlEndState, *, seed: WorldSeed | None = None
) -> EvalTemplate:
    return template.model_copy(
        update={
            "goal_state_checks": [check],
            "goal_check_controls": GoalCheckControls(
                checks=[GoalCheckControl(check=check, intent=intent, control="it")],  # type: ignore[arg-type]
                end_states={"it": control},
            ),
            **({"world_seed": seed} if seed is not None else {}),
        }
    )


class TestAClockDrivenFiringGradesNothing:
    """A check that a clock firing satisfies passes for a candidate that did nothing, and is refused — the gale."""

    @pytest.mark.parametrize(
        "check", [f'fired("{_CLOCK}")', f'fired_armed("{_CLOCK}")', f'contains(state.{_CLOCK}, "reprice")']
    )
    def test_an_act_check_the_seed_armed_clock_satisfies_is_refused(self, check: str) -> None:
        template = _single_check(
            toyhost_template(),
            check,
            "act",
            ControlEndState(describes="The operator's review applied the correction.", fired=[_CLOCK]),
        )

        with pytest.raises(ValidationFailedError, match="the same verdict on both"):
            refuse_non_discriminating_checks(template, profile=toyhost_profile())

    def test_an_act_check_on_an_unarmed_clock_dimension_is_refused_too(self) -> None:
        """The world's own clock may fire it: the seed arming nothing on it does not make fired() discriminate."""
        unarmed = WorldSeed(namespaces={"page_reader": {"document_language": "de"}})
        template = _single_check(
            toyhost_template(),
            f'fired("{_CLOCK}")',
            "act",
            ControlEndState(describes="The operator's review applied the correction.", fired=[_CLOCK]),
            seed=unarmed,
        )

        with pytest.raises(ValidationFailedError, match="the same verdict on both"):
            refuse_non_discriminating_checks(template, profile=toyhost_profile())

    def test_a_check_that_also_reads_what_the_candidate_did_is_admitted(self) -> None:
        """The positive control: the clock fires either way, and the call is what the candidate did."""
        template = toyhost_template()
        controls = template.goal_check_controls
        assert controls is not None
        check = f'{EVERY_FIELD_EMITTED} and fired_armed("{_CLOCK}")'
        admitted = _single_check(template, check, "act", controls.end_states["all-fields-emitted"])

        refuse_non_discriminating_checks(admitted, profile=toyhost_profile())
        (verdict,) = check_discriminations(admitted, profile=toyhost_profile())
        assert (verdict.did_nothing.passed, verdict.controlled.passed) == (False, True)

    def test_a_hold_check_that_the_clock_breaks_is_refused(self) -> None:
        template = _single_check(
            toyhost_template(),
            f'not fired("{_CLOCK}")',
            "hold",
            ControlEndState(describes="The operator's review applied the correction.", fired=[_CLOCK]),
        )

        # The clock fires in the do-nothing control and in the control alike, so it fails both.
        with pytest.raises(ValidationFailedError, match="the same verdict on both"):
            refuse_non_discriminating_checks(template, profile=toyhost_profile())


class TestSeedArmedFirings:
    def test_an_event_dimension_s_armed_firing_discriminates_and_a_world_firing_does_not_satisfy_it(self) -> None:
        held = WorldSeed(namespaces={"console": {PAYMENT_HOLD: "held"}})
        armed_fired = ControlEndState(
            describes="Posting the extraction applied the seeded hold.", fired_armed=[PAYMENT_HOLD]
        )
        worlds_own = ControlEndState(describes="The nightly audit placed a hold of its own.", fired=[PAYMENT_HOLD])
        check = f'fired_armed("{PAYMENT_HOLD}")'

        refuse_non_discriminating_checks(
            _single_check(toyhost_template(), check, "act", armed_fired, seed=held), profile=toyhost_profile()
        )
        with pytest.raises(ValidationFailedError, match="the same verdict on both"):
            refuse_non_discriminating_checks(
                _single_check(toyhost_template(), check, "act", worlds_own, seed=held), profile=toyhost_profile()
            )

    def test_a_control_stating_an_armed_firing_the_seed_never_armed_is_refused(self) -> None:
        unarmed = WorldSeed(namespaces={"page_reader": {"document_language": "de"}})
        template = _single_check(
            toyhost_template(),
            f'fired_armed("{PAYMENT_HOLD}")',
            "act",
            ControlEndState(describes="Posting the extraction applied the seeded hold.", fired_armed=[PAYMENT_HOLD]),
            seed=unarmed,
        )

        with pytest.raises(
            ValidationFailedError, match=r"fired_armed names 'payment_hold', and this template's seed arms no event"
        ):
            refuse_non_discriminating_checks(template, profile=toyhost_profile())

    def test_the_seeds_event_firing_on_another_dimension_it_moves_is_a_state_a_run_can_leave(self) -> None:
        """The session records the seed's event as armed on any dimension it moves, so the gate admits it."""
        held = WorldSeed(namespaces={"console": {PAYMENT_HOLD: "held"}})
        template = _single_check(
            toyhost_template(),
            f'fired_armed("{_SIGNOFF}")',
            "act",
            ControlEndState(
                describes="The seeded hold also brought the supervisor's sign-off.", fired_armed=[_SIGNOFF]
            ),
            seed=held,
        )

        refuse_non_discriminating_checks(template, profile=toyhost_profile())
        (verdict,) = check_discriminations(template, profile=toyhost_profile())
        assert (verdict.did_nothing.passed, verdict.controlled.passed) == (False, True)

    def test_an_armed_firing_naming_no_triggered_dimension_is_refused(self) -> None:
        template = _single_check(
            toyhost_template(),
            'fired("document_language")',
            "act",
            ControlEndState(describes="nothing real", fired_armed=["document_language"]),
        )

        with pytest.raises(ValidationFailedError, match=r"fired\('document_language'\) names a dimension set at t=0"):
            refuse_non_discriminating_checks(template, profile=toyhost_profile())


def test_a_named_control_lays_its_dimensions_over_the_seed_and_records_its_calls() -> None:
    profile = toyhost_profile()
    template = _with_hold_check()
    controls = template.goal_check_controls
    assert controls is not None

    stated = control_end_state(template, controls.end_states["language-rewritten"], world=profile.world)
    emitted = control_end_state(template, controls.end_states["all-fields-emitted"], world=profile.world)

    assert stated.end_state["document_language"] == "fr"
    assert stated.end_state["scan_quality"] == "faint"
    assert len(emitted.ledger.calls) == 4


def test_the_toy_template_s_checks_each_discriminate_in_both_directions() -> None:
    profile = toyhost_profile()

    verdicts = {
        d.check: (d.did_nothing.passed, d.controlled.passed)
        for d in check_discriminations(_with_hold_check(), profile=profile)
    }

    assert verdicts == {EVERY_FIELD_EMITTED: (False, True), _LANGUAGE_UNTOUCHED: (True, False)}
    refuse_non_discriminating_checks(_with_hold_check(), profile=profile)


def test_a_check_whose_control_states_nothing_it_reads_is_refused() -> None:
    """The ledger check against a control that made no calls gives one verdict on both."""
    template = toyhost_template()
    controls = template.goal_check_controls
    assert controls is not None
    idle_control = controls.model_copy(
        update={"end_states": {"all-fields-emitted": ControlEndState(describes="The extractor emitted nothing.")}}
    )

    with pytest.raises(ValidationFailedError, match="the same verdict on both"):
        refuse_non_discriminating_checks(
            template.model_copy(update={"goal_check_controls": idle_control}), profile=toyhost_profile()
        )


class TestAHostThatDeclaresNoWorld:
    """A goal check reads only declared dimensions, so world state stated for a worldless host names nothing."""

    def test_a_control_stating_world_state_is_refused(self) -> None:
        template = toyhost_template().model_copy(update={"world_seed": WorldSeed()})
        controls = template.goal_check_controls
        assert controls is not None
        stated = controls.model_copy(
            update={
                "end_states": {
                    "all-fields-emitted": controls.end_states["all-fields-emitted"].model_copy(
                        update={"world": {"console": {"operator_corrections": ["void"]}}}
                    )
                }
            }
        )
        worldless = replace(toyhost_profile(), world=None)

        with pytest.raises(
            ValidationFailedError, match=r"console\.operator_corrections\), and this host declares no world"
        ):
            refuse_non_discriminating_checks(
                template.model_copy(update={"goal_check_controls": stated}), profile=worldless
            )

    def test_a_seed_is_refused_rather_than_read_as_a_layout(self) -> None:
        worldless = replace(toyhost_profile(), world=None)

        with pytest.raises(ValidationFailedError, match="declares no world"):
            refuse_non_discriminating_checks(toyhost_template(), profile=worldless)

    def test_a_ledger_check_is_graded_with_no_world_at_all(self) -> None:
        template = toyhost_template().model_copy(update={"world_seed": WorldSeed()})
        worldless = replace(toyhost_profile(), world=None)

        refuse_non_discriminating_checks(template, profile=worldless)
        (verdict,) = check_discriminations(template, profile=worldless)
        assert (verdict.did_nothing.passed, verdict.controlled.passed) == (False, True)


def test_a_recorded_call_is_the_control_s_call_shape() -> None:
    """One call shape for a run's ledger and a control's: a control is read exactly as a run is."""
    call = RecordedCall(tool="extractor", action="emit_field", params={"field": "vendor_name"})

    assert ControlEndState(describes="one field emitted", calls=[call]).calls == [call]
