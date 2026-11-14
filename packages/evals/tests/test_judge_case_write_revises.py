"""An operator's judge-case write into a judge template that holds cases writes the template's next revision.

Runs of two revisions of one template never pool (``analysis.revisions``), and a definition reset has always written a
judge template's new battery as a new revision. An operator's own ``judge_case_author`` or ``judge_cases_freeze`` used
to add to the revision in place, so a judge's runs before the write and after it shared a revision id while their
cases differed, and pooled. Held here:

- the first cases of a template are written in place: no run can have measured a template with no case;
- a new case written into a template that holds cases is a case of its next revision, which carries every case of the
  one it supersedes (each copy naming its source), retired ones still retired; the superseded revision is archived,
  unedited and readable by id, and the name resolves to the new one;
- a run of each revision in one campaign is refused at creation and at assembly, each alone is not;
- a write that adds nothing writes no revision;
- a write into a superseded revision is refused, naming the newest, and writes nothing;
- a freeze from judged runs revises the same way (``test_judge_kind.py``, which holds the judged runs): the case set
  it mints lists the new revision's cases, and a case set left listing the superseded revision's cases is named.

**Mutations these catch:** the operator doors passing no ``revise``; a revision written before the freeze was checked
(the stray-result refusal leaving one behind); carrying only live cases; a carried copy not found by the identity a
freeze derives (the same case stored twice in the new revision); the superseded revision edited or left unarchived;
a write into a superseded revision archiving the current one.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.analysis import assemble_context_bundle, create_campaign
from threetears.evals.kernel import JUDGE_KIND, EvalStorage, ValidationFailedError
from threetears.evals.ops import JudgeCaseAuthor, judge_case_author
from threetears.evals.run import JudgeCaseFreezeReport
from threetears.evals.schema import EvalTemplate
from packages.evals.tests.factories import make_campaign, make_eval_run
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_INSTANT
from packages.evals.tests.fixtures.toyhost.judge import FAITHFULNESS_DIM, toyhost_judged_template
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, OpsFixture, ops_fixture

_AUTHOR = "cy@example.com"
_JUDGE = "Revised judge"


def _judge_template() -> EvalTemplate:
    return EvalTemplate(
        id="revised-judge",
        scope_id=TOYHOST_SCOPE,
        name=_JUDGE,
        intent="Score how faithfully an extraction reproduces its invoice.",
        candidate_kind=JUDGE_KIND,
        created_at=TOYHOST_INSTANT,
        updated_at=TOYHOST_INSTANT,
    )


def _vouches_for_itself(*, scope_id: str, caller: str, person: str) -> str | None:
    """A host's check that attests each caller authoring as itself, and for no one else."""
    return person if person == caller else None


def _fixture() -> OpsFixture:
    fixture = off_loop(ops_fixture)
    storage = fixture.host.eval_host.storage
    off_loop(storage.save_template, toyhost_judged_template())
    off_loop(storage.save_template, _judge_template())
    return fixture


def _storage(fixture: OpsFixture) -> EvalStorage:
    return fixture.host.eval_host.storage


def _author(fixture: OpsFixture, template_id: str, invoice: str, expected: int = 1) -> JudgeCaseFreezeReport:
    """One hard negative on the toy criterion, written through the operator's door into ``template_id``."""

    def write() -> JudgeCaseFreezeReport:
        return judge_case_author(
            replace(fixture.host.eval_host, attests_person=_vouches_for_itself),
            JudgeCaseAuthor(
                template_id=template_id,
                criterion_template_id=toyhost_judged_template().id,
                criterion=FAITHFULNESS_DIM,
                output=json.dumps({"invoice_number": invoice}),
                case_material="Invoice doc-42",
                expected=expected,
                note="because",
            ),
            TOYHOST_SCOPE,
            collected_by=_AUTHOR,
        )

    return off_loop(write)


def _templates(fixture: OpsFixture) -> list[EvalTemplate]:
    """Every record under the judge template's name, archived or not, oldest revision first."""
    records = off_loop(lambda: _storage(fixture).query_templates(TOYHOST_SCOPE, archived=None))
    return sorted((t for t in records if t.name == _JUDGE), key=lambda t: t.revision)


def _cases(fixture: OpsFixture, template_id: str) -> list[Any]:
    return off_loop(lambda: _storage(fixture).query_test_cases(TOYHOST_SCOPE, template_id=template_id))


def test_the_first_case_of_a_template_is_written_in_place() -> None:
    fixture = _fixture()

    report = _author(fixture, _judge_template().id, "INV-1")

    assert (report.template_id, report.revision, report.superseded_template_id) == (_judge_template().id, None, None)
    assert report.carried == {} and report.stranded_case_sets == []
    assert [t.id for t in _templates(fixture)] == [_judge_template().id]


def test_a_new_case_into_a_template_that_holds_cases_is_a_case_of_its_next_revision() -> None:
    fixture = _fixture()
    first = _author(fixture, _judge_template().id, "INV-1")
    (first_case,) = first.cases

    second = _author(fixture, first.template_id, "INV-2")

    old, new = _templates(fixture)
    assert (old.id, old.revision, old.archived) == (_judge_template().id, 1, True)
    assert (new.id, new.revision, new.archived) == (second.template_id, 2, False)
    assert new.id != old.id
    assert (second.revision, second.superseded_template_id) == (2, old.id)
    # The new revision is the template unchanged but for what makes it a revision.
    unchanged = {"id", "revision", "archived", "created_at", "updated_at"}
    assert new.model_dump(exclude=unchanged) == _judge_template().model_dump(exclude=unchanged)
    # The superseded revision keeps exactly the case its runs measured, unedited.
    (kept,) = _cases(fixture, old.id)
    assert (kept.id, kept.carried_from) == (first_case.test_case_id, None)
    # The new one holds that case's copy, naming its source, beside the case just written.
    held = {case.id: case for case in _cases(fixture, new.id)}
    assert second.carried == {first_case.test_case_id: next(i for i, c in held.items() if c.carried_from)}
    copy = held[second.carried[first_case.test_case_id]]
    assert (copy.carried_from, copy.content_hash, copy.host_payload) == (kept.id, kept.content_hash, kept.host_payload)
    (written,) = second.cases
    assert written.created and set(held) == {copy.id, written.test_case_id}
    # The name resolves to the new revision, so a launch by name runs both cases.
    resolved = off_loop(lambda: _storage(fixture).load_template_by_name(_JUDGE, TOYHOST_SCOPE))
    assert resolved is not None and resolved.id == new.id


def test_runs_before_and_after_the_write_never_pool() -> None:
    """The property the revision exists for: one battery's runs and the other's are refused as one campaign's."""
    fixture = _fixture()
    first = _author(fixture, _judge_template().id, "INV-1")
    second = _author(fixture, first.template_id, "INV-2")
    storage = _storage(fixture)
    before = make_eval_run(
        id="run-before",
        scope_id=TOYHOST_SCOPE,
        template_id=first.template_id,
        candidate_kind=JUDGE_KIND,
        test_case_ids=[case.test_case_id for case in first.cases],
    )
    after = make_eval_run(
        id="run-after",
        scope_id=TOYHOST_SCOPE,
        template_id=second.template_id,
        candidate_kind=JUDGE_KIND,
        test_case_ids=[*second.carried.values(), *(case.test_case_id for case in second.cases)],
    )
    off_loop(storage.save_eval_run, before)
    off_loop(storage.save_eval_run, after)
    profile = toyhost_profile()

    def create(name: str, run_ids: list[str]) -> Any:
        return create_campaign(
            storage,
            {"name": name, "subject_id": "judge", "behavior": "b", "aspects": [{"label": "b", "run_ids": run_ids}]},
            scope_id=TOYHOST_SCOPE,
            created_by="t",
            profile=profile,
            exploratory=True,
        )

    with pytest.raises(ValidationFailedError) as refused:
        off_loop(create, "spanning", [before.id, after.id])
    said = str(refused.value)
    assert repr(_JUDGE) in said and "revision 1" in said and "revision 2" in said, said
    # A campaign record written before the rule, or by a host's own store, is refused where its bundle is assembled.
    spanning = make_campaign(
        scope_id=TOYHOST_SCOPE, name="stored spanning", template_id="", run_ids=[before.id, after.id]
    )
    with pytest.raises(ValidationFailedError, match="more than one revision"):
        off_loop(lambda: assemble_context_bundle(spanning, storage=storage, profile=profile))
    assert off_loop(create, "before only", [before.id]).id
    assert off_loop(create, "after only", [after.id]).id


def test_a_write_that_adds_nothing_writes_no_revision() -> None:
    fixture = _fixture()
    first = _author(fixture, _judge_template().id, "INV-1")

    again = _author(fixture, first.template_id, "INV-1")

    assert (again.template_id, again.revision) == (first.template_id, None)
    assert [case.test_case_id for case in again.cases] == [case.test_case_id for case in first.cases]
    assert not again.cases[0].created
    assert [t.revision for t in _templates(fixture)] == [1]


def test_a_case_written_before_is_the_carried_copy_in_the_new_revision_and_a_retired_case_stays_retired() -> None:
    fixture = _fixture()
    storage = _storage(fixture)
    first = _author(fixture, _judge_template().id, "INV-1")
    second = _author(fixture, first.template_id, "INV-2")
    retired = off_loop(storage.load_test_case, second.cases[0].test_case_id, TOYHOST_SCOPE)
    assert retired is not None
    off_loop(storage.save_test_case, retired.model_copy(update={"archived": True, "archived_reason": "superseded"}))

    third = _author(fixture, second.template_id, "INV-3")

    assert [t.revision for t in _templates(fixture)] == [1, 2, 3]
    held = {case.id: case for case in _cases(fixture, third.template_id)}
    assert len(held) == 3
    assert held[third.carried[retired.id]].archived, "a retired case is carried retired, never brought back"
    # The first case, carried twice, is found by what it is: authoring it again names its copy and stores nothing.
    again = _author(fixture, third.template_id, "INV-1")
    assert (again.revision, again.cases[0].created) == (None, False)
    assert again.cases[0].test_case_id in held
    assert len(_cases(fixture, third.template_id)) == 3


def test_a_write_into_a_superseded_revision_is_refused_naming_the_newest_and_writes_nothing() -> None:
    fixture = _fixture()
    first = _author(fixture, _judge_template().id, "INV-1")
    second = _author(fixture, first.template_id, "INV-2")

    with pytest.raises(ValidationFailedError) as refused:
        _author(fixture, first.template_id, "INV-3")

    assert second.template_id in str(refused.value) and second.template_id in (refused.value.remedy or "")
    old, new = _templates(fixture)
    assert (old.archived, new.archived) == (True, False), "the current revision is not archived by the refused write"
    assert len(_cases(fixture, old.id)) == 1 and len(_cases(fixture, new.id)) == 2
