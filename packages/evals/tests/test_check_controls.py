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
from packages.evals.tests.fixtures.toyhost.run import EVERY_FIELD_EMITTED, toyhost_template

#: A hold check over world state: the operator's corrections stay as the seed put them.
_CORRECTIONS_UNTOUCHED = "state.operator_corrections.length == 1"


def _with_hold_check() -> EvalTemplate:
    template = toyhost_template()
    controls = template.goal_check_controls
    assert controls is not None
    return template.model_copy(
        update={
            "goal_state_checks": [*template.goal_state_checks, _CORRECTIONS_UNTOUCHED],
            "goal_check_controls": GoalCheckControls(
                checks=[
                    *controls.checks,
                    GoalCheckControl(check=_CORRECTIONS_UNTOUCHED, intent="hold", control="corrections-added"),
                ],
                end_states={
                    **controls.end_states,
                    "corrections-added": ControlEndState(
                        describes="The extractor appended a correction the operator never made.",
                        world={"console": {"operator_corrections": ["reprice", "void"]}},
                    ),
                },
            ),
        }
    )


def test_the_do_nothing_control_is_the_seed_named_by_dimension_with_no_calls() -> None:
    profile = toyhost_profile()

    idle = do_nothing_end_state(toyhost_template(), world=profile.world)

    assert idle.end_state == {
        "document_language": "de",
        "scan_quality": "faint",
        "vendor_template": "acme-2019",
        "operator_corrections": ["reprice"],
    }
    assert idle.ledger.calls == []


def test_a_named_control_lays_its_dimensions_over_the_seed_and_records_its_calls() -> None:
    profile = toyhost_profile()
    template = _with_hold_check()
    controls = template.goal_check_controls
    assert controls is not None

    stated = control_end_state(template, controls.end_states["corrections-added"], world=profile.world)
    emitted = control_end_state(template, controls.end_states["all-fields-emitted"], world=profile.world)

    assert stated.end_state["operator_corrections"] == ["reprice", "void"]
    assert stated.end_state["document_language"] == "de"
    assert len(emitted.ledger.calls) == 4


def test_the_toy_template_s_checks_each_discriminate_in_both_directions() -> None:
    profile = toyhost_profile()

    verdicts = {
        d.check: (d.did_nothing.passed, d.controlled.passed)
        for d in check_discriminations(_with_hold_check(), profile=profile)
    }

    assert verdicts == {EVERY_FIELD_EMITTED: (False, True), _CORRECTIONS_UNTOUCHED: (True, False)}
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
