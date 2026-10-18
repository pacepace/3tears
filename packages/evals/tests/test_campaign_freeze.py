"""A campaign's declaration is frozen at its first run, and a question declared after it confirms nothing.

Before this, ``campaign_update`` replaced a design (and added or reworded its questions) whatever runs the campaign
held, and every reader took the questions with no regard to when they were asked. Now:

- **the freeze**: once any aspect holds a run, the declaration (``question``, ``aspects``, ``floors``) is refused in
  any form, and so is re-pointing or clearing the control; ``name`` and ``description`` stay editable, and the
  control may be addressed once, while none is named (the path ``quick.compare`` takes);
- **the backstop** that needs no trust in that path: the campaign's ``declared_at`` (when its question was asked in
  full) is read against the first member run's ``created_at``, so a question declared after the evidence reads as
  post hoc.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import NO_QUESTION_EXPLORATORY, POST_HOC_EXPLORATORY
from threetears.evals.analysis.campaigns import set_campaign_control, update_campaign
from threetears.evals.kernel.declaration import Question, preregistered_questions
from threetears.evals.kernel.errors import ConflictError
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign_of_one_aspect, toyhost_design
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_INSTANT
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
        "margin": 5,
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


# --- the backstop: when a question was asked, read against the first run -------------------------------------------


def test_a_question_written_past_the_freeze_after_the_first_run_reads_as_post_hoc() -> None:
    """An older build's write: no stored field marks it, and its own time says it came after the evidence."""
    fixture = ops_fixture()
    (aspect,) = fixture.campaign.aspects
    late = fixture.campaign.model_copy(
        update={
            "question": "is the wider chunk better?",
            "declared_at": _LATER,
            "aspects": [aspect.model_copy(update={"asked_at": _LATER})],
        }
    )
    fixture.host.eval_host.storage.save_campaign(late)

    bundle = _bundle(fixture)

    assert bundle.question_declared is False and bundle.reading_scope.questions_declared is False
    assert bundle.reading_scope.question_post_hoc is True
    assert bundle.reading_scope.disclosure == POST_HOC_EXPLORATORY
    assert all(family.aspect is None for family in bundle.multiple_comparisons.families), (
        "no confirmatory family is built for a question asked after the evidence"
    )
    assert bundle.verdict_order.question is None


def test_a_question_declared_before_the_first_run_stays_confirmatory() -> None:
    """The positive control: the corpus question was asked at the instant its runs were created."""
    bundle = _bundle(ops_fixture())

    assert bundle.question_declared is True and bundle.reading_scope.question_post_hoc is False
    assert bundle.reading_scope.disclosure is None


def test_a_late_question_beside_an_early_one_makes_the_campaign_s_one_question_post_hoc() -> None:
    """A campaign asks one question, its live questions' words joined, asked when the last of them was: a question
    added after the first run makes the whole question post hoc (until aspects, the early one still confirmed)."""
    fixture = ops_fixture()
    design = toyhost_design()
    late = Question(id="q-post", text="does the wider chunk cut the error rate?", asked_at=_LATER)
    both = design.model_copy(update={"questions": [*design.questions, late]})
    (aspect,) = fixture.campaign.aspects
    stored = toyhost_campaign_of_one_aspect(
        declared_design=both,
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

    assert bundle.question_declared is False and bundle.reading_scope.question_post_hoc is True


def test_preregistration_with_no_run_reads_every_live_question_and_refuses_an_unordered_stamp() -> None:
    design = toyhost_design()

    assert preregistered_questions(design, None) == design.live_questions()
    assert preregistered_questions(design, TOYHOST_INSTANT) == design.live_questions()
    with pytest.raises(ValueError, match="no offset"):
        preregistered_questions(design, "2026-03-14T09:30:00")


def test_a_design_with_no_question_says_so_as_before() -> None:
    fixture = ops_fixture()
    (aspect,) = fixture.campaign.aspects
    assert aspect.design is not None
    unasked = aspect.model_copy(update={"design": aspect.design.model_copy(update={"merit_axes": None})})
    asked_nothing = fixture.campaign.model_copy(update={"question": "", "floors": [], "aspects": [unasked]})
    fixture.host.eval_host.storage.save_campaign(asked_nothing)

    bundle = _bundle(fixture)

    assert bundle.reading_scope.disclosure == NO_QUESTION_EXPLORATORY and not bundle.reading_scope.question_post_hoc
