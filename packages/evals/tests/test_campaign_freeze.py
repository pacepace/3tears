"""A campaign's declaration is frozen at its first run, and a criterion set after its readings is disclosed, not demoted.

Before this, ``campaign_update`` replaced a design (and added or reworded its questions) whatever runs the campaign
held, and every reader took the questions with no regard to when they were asked. Now:

- **the freeze**: once any aspect holds a run, the declaration (``question``, ``aspects``, ``floors``) is refused in
  any form, and so is re-pointing or clearing the control; ``name`` and ``description`` stay editable, and the
  control may be addressed once, while none is named (the path ``quick.compare`` takes);
- **the disclosure** that needs no trust in that path: each criterion (the aspect, its control, margin, bars and
  floors) is read against the aspect's first stored result, so one set once its readings were visible carries its
  time (``reading_scope.late_criteria``). It reads as one set before them (owner, 2026-10-21 and 2026-10-23:
  operators are trusted to act in good faith, and the disclosure lets a reader judge). When the runs began never
  counts.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import NO_QUESTION_EXPLORATORY, LateCriterion, late_criteria
from threetears.evals.kernel.declaration import GuardrailMargin
from threetears.evals.analysis.campaigns import set_campaign_control, update_campaign
from threetears.evals.analysis.report.campaign_page import late_criteria_lines
from threetears.evals.kernel.errors import ConflictError
from packages.evals.tests.fixtures.toyhost.campaign import (
    TOYHOST_QUESTION,
    toyhost_campaign_of_one_aspect,
    toyhost_aspect_design,
)
from packages.evals.tests.ops_support import TOYHOST_SCOPE, OpsFixture, ops_fixture

#: A moment after every corpus run was created.
_LATER = "2026-03-15T09:30:00+00:00"


def _update(fixture: OpsFixture, updates: dict[str, Any], **control: Any) -> Any:
    host = fixture.host.eval_host
    return update_campaign(
        host.storage, fixture.campaign.id, TOYHOST_SCOPE, updates, updated_by="op", profile=host.profile, **control
    )


def _control(fixture: OpsFixture, run_id: str | None) -> Any:
    host = fixture.host.eval_host
    return set_campaign_control(
        host.storage, fixture.campaign.id, TOYHOST_SCOPE, run_id, set_by="op", profile=host.profile
    )


def _aspects(fixture: OpsFixture, *, runs: bool = False) -> list[dict[str, Any]]:
    """The toy campaign's one aspect, redeclared as an update sends it: its design and a margin, holding no runs."""
    (aspect,) = fixture.campaign.aspects
    assert aspect.design is not None
    declared: dict[str, Any] = {
        "label": aspect.label,
        "design": aspect.design.model_dump(mode="json"),
    }
    if runs:
        declared["run_ids"] = list(aspect.run_ids)
    return [declared]


def _without_runs(fixture: OpsFixture) -> None:
    """Store the toy campaign holding no runs."""
    emptied = [aspect.model_copy(update={"run_ids": []}) for aspect in fixture.campaign.aspects]
    fixture.host.eval_host.storage.save_campaign(fixture.campaign.model_copy(update={"aspects": emptied}))


def _bundle(fixture: OpsFixture) -> Any:
    host = fixture.host.eval_host
    campaign = host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    assert campaign is not None
    return assemble_context_bundle(campaign, storage=host.storage, profile=host.profile)


# --- the freeze --------------------------------------------------------------------------------------------------


def test_a_question_reworded_after_the_first_run_is_refused_with_the_remedy() -> None:
    """The repro: before, this update succeeded and the bundle read the new question as confirmatory."""
    fixture = ops_fixture()
    assert fixture.campaign.run_ids, "the corpus campaign holds its runs"

    with pytest.raises(ConflictError, match=r"holds 2 run\(s\), so what it declared is frozen") as refused:
        _update(fixture, {"question": "does the wider chunk cut the error rate?"})

    assert "create a new campaign with the amended declaration (campaign_create" in refused.value.remedy
    stored = fixture.host.eval_host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    assert stored is not None and stored == fixture.campaign


@pytest.mark.parametrize(
    "updates",
    [
        pytest.param({"floors": [{"aspect": "extract_invoice_fields", "clause": {"kind": "cost_lower"}}]}, id="floors"),
        pytest.param({"aspects": "redeclared"}, id="aspects"),
    ],
)
def test_floors_and_aspects_are_frozen_too(updates: dict[str, Any]) -> None:
    fixture = ops_fixture()
    if updates.get("aspects") == "redeclared":
        updates = {"aspects": _aspects(fixture)}

    with pytest.raises(ConflictError, match="what it declared is frozen"):
        _update(fixture, updates)


def test_withdrawing_a_design_and_declaring_one_on_an_exploratory_campaign_with_runs_are_refused() -> None:
    fixture = ops_fixture()
    (aspect,) = fixture.campaign.aspects
    with pytest.raises(ConflictError, match="what it declared is frozen"):
        _update(fixture, {"aspects": [{"label": aspect.label}]}, exploratory=True)

    exploratory = fixture.campaign.model_copy(
        update={"aspects": [aspect.model_copy(update={"design": None})], "arms": [], "floors": [], "question": ""}
    )
    fixture.host.eval_host.storage.save_campaign(exploratory)
    with pytest.raises(ConflictError, match="what it declared is frozen"):
        _update(fixture, {"aspects": _aspects(fixture)})


def test_name_and_description_stay_editable() -> None:
    fixture = ops_fixture()

    updated = _update(fixture, {"name": "renamed", "description": "a better account of it"})

    assert (updated.name, updated.description) == ("renamed", "a better account of it")
    assert (updated.aspects, updated.question, updated.declared_at) == (
        fixture.campaign.aspects,
        fixture.campaign.question,
        fixture.campaign.declared_at,
    )


def test_the_control_is_addressed_once_and_never_re_pointed_or_cleared() -> None:
    fixture = ops_fixture()
    first, second = fixture.campaign.run_ids
    assert fixture.campaign.control is None

    designated = _control(fixture, first)
    assert designated.control is not None
    assert designated.declared_at == fixture.campaign.declared_at, "naming the control does not re-date the question"

    with pytest.raises(ConflictError, match="re-pointing it"):
        _control(fixture, second)
    with pytest.raises(ConflictError, match="clearing it"):
        _control(fixture, None)
    with pytest.raises(ConflictError, match="clearing it"):
        _update(fixture, {}, clear_control=True)


def test_a_campaign_with_no_run_keeps_its_declaration_editable() -> None:
    fixture = ops_fixture()
    _without_runs(fixture)

    updated = _update(fixture, {"question": "does the wider chunk cut the error rate?", "aspects": _aspects(fixture)})

    assert updated.question == "does the wider chunk cut the error rate?"
    assert updated.declared_by == "op" and updated.declared_at is not None
    assert updated.declared_at != fixture.campaign.declared_at, "a declaration rewritten is dated when it was"


def test_an_update_declaring_aspects_holding_runs_is_refused() -> None:
    fixture = ops_fixture()
    _without_runs(fixture)

    with pytest.raises(Exception, match="runs join an aspect through campaign_runs_add"):
        _update(fixture, {"aspects": _aspects(fixture, runs=True)})


# --- the disclosure: when each criterion was set, read against the first result, and never a demotion ----------------


#: Between the corpus runs' creation (09:30:00) and their first stored result (09:30:01): after the runs began, before
#: anything could be seen.
_BEFORE_ANY_RESULT = "2026-03-14T09:30:00.500000+00:00"


def _asked_late(fixture: OpsFixture, at: str | None = _LATER) -> None:
    """The toy campaign's aspect re-dated to ``at``, as an older build's write or a campaign declared over runs whose
    results were already stored keeps it: no stored field marks it, and its own time says it came after the evidence."""
    (aspect,) = fixture.campaign.aspects
    late = fixture.campaign.model_copy(
        update={"declared_at": at, "aspects": [aspect.model_copy(update={"asked_at": at})]}
    )
    fixture.host.eval_host.storage.save_campaign(late)


def _late(at: str | None) -> list[LateCriterion]:
    """The toy aspect's criteria, each set at ``at``: the aspect itself and its one bar, a ceiling on cost."""
    return [
        LateCriterion(criterion="aspect", set_at=at),
        LateCriterion(criterion="bar", name="cost_usd", set_at=at),
    ]


def test_criteria_set_after_the_first_result_read_as_set_before_it_and_carry_their_time() -> None:
    """Owner rulings (2026-10-21, 2026-10-23): disclosed, not enforced. The question confirms its readings as one
    asked in time does, and the bundle names each criterion set after its readings were visible, with its time."""
    fixture = ops_fixture()
    _asked_late(fixture)

    bundle = _bundle(fixture)

    assert bundle.question_declared is True and bundle.reading_scope.questions_declared is True
    assert bundle.reading_scope.late_criteria == _late(_LATER)
    assert bundle.reading_scope.disclosure is None, "nothing calls its findings exploratory"
    assert all(family.aspect == bundle.aspect for family in bundle.multiple_comparisons.families), (
        "its question's family, never the exploratory one"
    )


def test_criteria_set_after_the_first_result_read_every_reading_as_they_would_in_time() -> None:
    """The same runs, the criteria set before their results and after them: every reading, family and scope is the
    same, and the one difference is the disclosure."""
    timely = _bundle(ops_fixture())
    late_fixture = ops_fixture()
    _asked_late(late_fixture)
    late = _bundle(late_fixture)

    assert timely.reading_scope.late_criteria == []
    assert late.reading_scope.late_criteria != []
    assert late.multiple_comparisons == timely.multiple_comparisons
    assert late.verdict_order == timely.verdict_order
    assert late.bar_adjudications == timely.bar_adjudications
    assert late.reading_scope.model_copy(update={"late_criteria": []}) == timely.reading_scope


def test_criteria_set_after_the_runs_began_and_before_any_result_are_not_late() -> None:
    """Data age is no signal: an aspect asked once its runs were launched, before a result was stored, was set before
    anything could be seen."""
    fixture = ops_fixture()
    _asked_late(fixture, at=_BEFORE_ANY_RESULT)

    bundle = _bundle(fixture)

    assert bundle.design_frozen_at is not None and bundle.design_frozen_at < _BEFORE_ANY_RESULT
    assert bundle.reading_scope.late_criteria == []


def test_criteria_with_no_recorded_time_are_named_with_none_for_their_time() -> None:
    """They cannot be shown to come before the readings, and nothing guesses: each says its time is not recorded."""
    fixture = ops_fixture()
    _asked_late(fixture, at=None)

    bundle = _bundle(fixture)

    assert bundle.question_declared is True
    assert bundle.reading_scope.late_criteria == _late(None)


def test_criteria_declared_before_the_first_result_carry_no_disclosure() -> None:
    """The positive control: the corpus question was asked at the instant its runs were created."""
    bundle = _bundle(ops_fixture())

    assert bundle.question_declared is True and bundle.reading_scope.late_criteria == []
    assert bundle.reading_scope.disclosure is None


def test_a_late_question_beside_an_early_one_dates_the_aspect_and_its_bar_never_the_question() -> None:
    """A campaign asks one question, its live questions' words joined, asked when the last of them was: an aspect
    asked after its first result is named with its bar, and no line names the question."""
    fixture = ops_fixture()
    late = "does the wider chunk cut the error rate?"
    (aspect,) = fixture.campaign.aspects
    stored = toyhost_campaign_of_one_aspect(
        design=toyhost_aspect_design(),
        question=f"{TOYHOST_QUESTION}\n\n{late}",
        asked_at=_LATER,
        id=fixture.campaign.id,
        scope_id=fixture.campaign.scope_id,
        name=fixture.campaign.name,
        subject_id=fixture.campaign.subject_id,
        behavior=fixture.campaign.behavior,
        template_id=aspect.template_id,
        run_ids=list(aspect.run_ids),
        created_by=fixture.campaign.created_by,
    )
    fixture.host.eval_host.storage.save_campaign(stored)

    bundle = _bundle(fixture)

    assert bundle.question_declared is True
    assert bundle.reading_scope.late_criteria == _late(_LATER)


def test_a_design_with_no_question_says_so_as_before() -> None:
    fixture = ops_fixture()
    (aspect,) = fixture.campaign.aspects
    assert aspect.design is not None
    unasked = aspect.model_copy(update={"design": aspect.design.model_copy(update={"merit_axes": None})})
    asked_nothing = fixture.campaign.model_copy(update={"question": "", "floors": [], "aspects": [unasked]})
    fixture.host.eval_host.storage.save_campaign(asked_nothing)

    bundle = _bundle(fixture)

    assert bundle.reading_scope.disclosure == NO_QUESTION_EXPLORATORY
    assert bundle.reading_scope.late_criteria == [], "no question asked, and its bar was set before its readings"


def test_one_result_whose_time_cannot_be_ordered_hides_no_late_criterion_and_is_counted() -> None:
    """The readable results order the readings' first visibility; the one stamped with no offset is counted, and the
    page says whether a criterion came after it is not known."""
    fixture = ops_fixture()
    _asked_late(fixture)
    storage = fixture.host.eval_host.storage
    (aspect,) = fixture.campaign.aspects
    first, *_ = storage.query_eval_results_by_runs(list(aspect.run_ids), TOYHOST_SCOPE)
    storage.save_eval_result(first.model_copy(update={"scored_at": "2026-03-14T09:30:05"}))

    bundle = _bundle(fixture)

    assert bundle.reading_scope.late_criteria == _late(_LATER)
    assert bundle.reading_scope.untimed_results == 1
    assert bundle.first_result_at is not None and bundle.first_result_at.endswith("+00:00")
    assert late_criteria_lines([bundle])[-1] == (
        "1 result carries no time that can be ordered, so whether a criterion was set after it was visible is not known."
    )


def test_a_guardrail_s_margin_is_timed_where_a_question_asks_the_aspect_and_never_where_none_does() -> None:
    """A guardrail bears on recommending a candidate, which needs an asked aspect: under a question its margin is timed
    like the aspect's other criteria; with none asked it is not, as the aspect and its control are not."""
    fixture = ops_fixture()
    _asked_late(fixture)
    bundle = _bundle(fixture)
    declared = bundle.declared
    assert declared is not None and bundle.first_result_at is not None
    guarded = declared.model_copy(
        update={
            "design": declared.design.model_copy(
                update={"guardrail_margins": [GuardrailMargin(dimension="tone", margin=0.5)]}
            )
        }
    )
    unasked = guarded.model_copy(update={"design": guarded.design.model_copy(update={"merit_axes": None})})

    assert [(late.criterion, late.name) for late in late_criteria(guarded, bundle.first_result_at)] == [
        ("aspect", None),
        ("bar", "cost_usd"),
        ("guardrail", "tone"),
    ]
    assert [(late.criterion, late.name) for late in late_criteria(unasked, bundle.first_result_at)] == [
        ("bar", "cost_usd"),
    ]
