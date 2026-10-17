"""A campaign's design is frozen at its first run, and a question declared after it confirms nothing.

Before this, ``campaign_update`` replaced a design (and added or reworded its questions) whatever runs the campaign
held, and every reader took ``live_questions()`` with no regard to when a question was asked: a question added after
the results read as confirmatory, and a campaign with no question carried the exploratory label nowhere a reader
would look first. Now:

- **the freeze**: once a run has joined, ``declared_design`` is refused in any form, and so is re-pointing or
  clearing the control; ``name`` and ``description`` stay editable, and the control may be addressed once, while
  the design names none (the path ``quick.compare`` takes);
- **the backstop** that needs no stored field: :func:`preregistered_questions` reads each question's own
  ``asked_at`` against the first member run's ``created_at``, and every reader that scopes a reading by the
  questions reads it, so a design changed some other way (an older build) still reads as post hoc.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import NO_QUESTION_EXPLORATORY, post_hoc_disclosure
from threetears.evals.analysis.campaigns import set_campaign_control, update_campaign
from threetears.evals.kernel.declaration import Question, preregistered_questions
from threetears.evals.kernel.errors import ConflictError
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_design
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


def _with_question(asked_at: str) -> dict[str, Any]:
    design = toyhost_design()
    added = Question(id="q-post", text="does the wider chunk cut the error rate?", asked_at=asked_at)
    return design.model_copy(update={"questions": [*design.questions, added]}).model_dump(mode="json")


def _bundle(fixture: OpsFixture) -> Any:
    host = fixture.host.eval_host
    campaign = host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    assert campaign is not None
    return assemble_context_bundle(campaign, storage=host.storage, profile=host.profile)


# --- the freeze --------------------------------------------------------------------------------------------------


def test_a_question_added_after_the_first_run_is_refused_with_the_remedy() -> None:
    """The repro: before, this update succeeded and the bundle read the new question as confirmatory."""
    fixture = ops_fixture()
    assert fixture.campaign.run_ids, "the corpus campaign holds its runs"

    with pytest.raises(ConflictError, match=r"holds 2 run\(s\), so its design is frozen") as refused:
        _update(fixture, {"declared_design": _with_question(_LATER)})

    assert "create a new campaign with the amended design (campaign_create)" in refused.value.remedy
    stored = fixture.host.eval_host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    assert stored is not None and stored.declared_design == fixture.campaign.declared_design


def test_withdrawing_a_design_and_declaring_one_on_an_exploratory_campaign_with_runs_are_refused() -> None:
    fixture = ops_fixture()
    with pytest.raises(ConflictError, match="design is frozen"):
        _update(fixture, {"declared_design": None}, exploratory=True)

    exploratory = fixture.campaign.model_copy(update={"declared_design": None})
    fixture.host.eval_host.storage.save_campaign(exploratory)
    with pytest.raises(ConflictError, match="design is frozen"):
        _update(fixture, {"declared_design": toyhost_design().model_dump(mode="json")})


def test_name_and_description_stay_editable() -> None:
    fixture = ops_fixture()

    updated = _update(fixture, {"name": "renamed", "description": "a better account of it"})

    assert (updated.name, updated.description) == ("renamed", "a better account of it")
    assert updated.declared_design == fixture.campaign.declared_design


def test_the_control_is_addressed_once_and_never_re_pointed_or_cleared() -> None:
    fixture = ops_fixture()
    first, second = fixture.campaign.run_ids
    assert fixture.campaign.declared_design is not None and fixture.campaign.declared_design.control is None

    designated = _control(fixture, first)
    assert designated.declared_design is not None and designated.declared_design.control is not None

    with pytest.raises(ConflictError, match="re-pointing it"):
        _control(fixture, second)
    with pytest.raises(ConflictError, match="clearing it"):
        _control(fixture, None)
    with pytest.raises(ConflictError, match="clearing it"):
        _update(fixture, {}, clear_control=True)


def test_a_campaign_with_no_run_keeps_its_design_editable() -> None:
    fixture = ops_fixture()
    fixture.host.eval_host.storage.save_campaign(fixture.campaign.model_copy(update={"run_ids": []}))

    updated = _update(fixture, {"declared_design": _with_question(_LATER)})

    assert updated.declared_design is not None
    assert [question.id for question in updated.declared_design.live_questions()][-1] == "q-post"


# --- the backstop: when a question was asked, read against the first run -------------------------------------------


def test_a_question_written_past_the_freeze_after_the_first_run_reads_as_post_hoc() -> None:
    """An older build's write: no stored field marks it, and its own asked_at says it came after the evidence."""
    fixture = ops_fixture()
    design = fixture.campaign.declared_design
    assert design is not None
    only_late = design.model_copy(
        update={"questions": [Question(id="q-post", text="is the wider chunk better?", asked_at=_LATER)]}
    )
    fixture.host.eval_host.storage.save_campaign(fixture.campaign.model_copy(update={"declared_design": only_late}))

    bundle = _bundle(fixture)

    assert bundle.question_declared is False and bundle.reading_scope.questions_declared is False
    assert bundle.post_hoc_questions == ["q-post"] == bundle.reading_scope.post_hoc_questions
    assert bundle.reading_scope.disclosure == post_hoc_disclosure(["q-post"])
    assert all(family.question_id is None for family in bundle.multiple_comparisons.families), (
        "no confirmatory family is built for a question asked after the evidence"
    )
    assert bundle.verdict_order.questions == []


def test_a_question_declared_before_the_first_run_stays_confirmatory() -> None:
    """The positive control: the corpus question was asked at the instant its runs were created."""
    bundle = _bundle(ops_fixture())

    assert bundle.question_declared is True and bundle.post_hoc_questions == []
    assert bundle.reading_scope.disclosure is None


def test_a_late_question_beside_an_early_one_is_named_and_the_early_one_still_confirms() -> None:
    fixture = ops_fixture()
    design = toyhost_design()
    late = Question(id="q-post", text="does the wider chunk cut the error rate?", asked_at=_LATER)
    both = design.model_copy(update={"questions": [*design.questions, late]})
    fixture.host.eval_host.storage.save_campaign(fixture.campaign.model_copy(update={"declared_design": both}))

    bundle = _bundle(fixture)

    assert bundle.question_declared is True and bundle.post_hoc_questions == ["q-post"]


def test_preregistration_with_no_run_reads_every_live_question_and_refuses_an_unordered_stamp() -> None:
    design = toyhost_design()

    assert preregistered_questions(design, None) == design.live_questions()
    assert preregistered_questions(design, TOYHOST_INSTANT) == design.live_questions()
    with pytest.raises(ValueError, match="no offset"):
        preregistered_questions(design, "2026-03-14T09:30:00")


def test_a_design_with_no_question_says_so_as_before() -> None:
    fixture = ops_fixture()
    design = fixture.campaign.declared_design
    assert design is not None
    asked_nothing = design.model_copy(update={"questions": []})
    fixture.host.eval_host.storage.save_campaign(fixture.campaign.model_copy(update={"declared_design": asked_nothing}))

    bundle = _bundle(fixture)

    assert bundle.reading_scope.disclosure == NO_QUESTION_EXPLORATORY and bundle.post_hoc_questions == []
