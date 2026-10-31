"""A report written from a version of the analyst skill under test: the version both sessions start from, and records.

``write_report(skill=...)`` takes an :class:`~threetears.evals.analysis.analyst.skill.AnalystSkill` built from any
files, so a campaign can run a candidate skill beside the shipped one. The shipped skill is the default, and the
generation records which skill wrote the report either way, so two reports written from different versions are told
apart by their digests. A version is checked before anything is frozen, sent or spent, its withheld notes included.
"""

from __future__ import annotations

from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans.analyst_cli import seeded_host
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.report_writer_support import (
    FakeToolTurnClient,
    FakeTurnModel,
    call,
    kairo_id,
    reply,
    review,
    submitted,
    written,
)
from threetears.evals.actions.report_writing import write_report
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.ops.report_writer.tally import ReportBudget

#: A rule only the candidate's analyst text holds, and one only its editor's does.
_ANALYST_RULE = "Candidate rule: lead with the arm a reader would ship."
_EDITOR_RULE = "Candidate rule: score First screen 0 when the lead names no arm."


def _candidate() -> AnalystSkill:
    """The shipped skill with one line added to the analyst's role file and one to the editor's text."""
    files = dict(load_skill().files)
    files["SKILL.md"] = f"{files['SKILL.md']}\n\n{_ANALYST_RULE}\n"
    files["editor.md"] = f"{files['editor.md']}\n\n{_EDITOR_RULE}\n"
    return AnalystSkill(files=files)


def _passing_model() -> FakeTurnModel:
    """An analyst that submits the Kairo document, and an editor whose first review clears it."""
    return FakeTurnModel(
        analyst=FakeToolTurnClient(reply(call(1, "options_dominance", kairo_id())), reply(submitted())),
        editor_1=FakeToolTurnClient(reply(call(9, "options_dominance", kairo_id())), reply(review())),
    )


def _system(client: FakeToolTurnClient) -> str:
    """The system text a session's every turn was sent with: one text for the whole session."""
    (system,) = {request["system"] for request in client.requests}
    return system


async def test_a_supplied_skill_is_the_text_both_sessions_start_from_and_the_generation_records_it() -> None:
    candidate = _candidate()
    model = _passing_model()

    result = await written(model, withhold=["kairo-round2.md"], skill=candidate)

    assert result.ending == "stored", result.outcome.reason
    analyst_system, editor_system = _system(model.sessions["analyst"]), _system(model.sessions["editor-1"])
    assert _ANALYST_RULE in analyst_system and _EDITOR_RULE not in analyst_system
    assert _EDITOR_RULE in editor_system and _ANALYST_RULE not in editor_system
    assert result.generation["skill"] == {
        "analyst": candidate.analyst(withhold=["kairo-round2.md"]).digest,
        "editor": candidate.editor().digest,
        "package": candidate.digest,
    }


async def test_two_versions_of_the_skill_record_different_digests_and_the_default_is_the_shipped_one() -> None:
    shipped_model, candidate_model = _passing_model(), _passing_model()

    shipped = await written(shipped_model)
    candidate = await written(candidate_model, skill=_candidate())

    assert shipped.generation["skill"] == {
        "analyst": load_skill().analyst().digest,
        "editor": load_skill().editor().digest,
        "package": load_skill().digest,
    }
    differs: dict[str, Any] = {  # Any: each role's digest
        role: (shipped.generation["skill"][role], candidate.generation["skill"][role])
        for role in ("analyst", "editor", "package")
    }
    assert all(left != right for left, right in differs.values()), differs
    assert _ANALYST_RULE not in _system(shipped_model.sessions["analyst"])


@pytest.mark.parametrize("skill", [None, "candidate"])
async def test_a_withheld_note_the_skill_lacks_is_refused_before_anything_is_frozen_sent_or_spent(
    skill: str | None,
) -> None:
    model = _passing_model()

    with pytest.raises(ValueError, match="no exemplar is named no-such-note.md") as refused:
        # A campaign that does not exist: had the freeze run first, it would have refused on that instead.
        await write_report(
            off_loop(seeded_host),
            CALLER,
            model=model,
            budget=ReportBudget(usd=20.0),
            campaign_id="no-such-campaign",
            withhold=["no-such-note.md"],
            skill=_candidate() if skill == "candidate" else None,
        )

    assert "campaign" not in str(refused.value)
    assert model.asked == [], "no session was opened"


def test_a_version_missing_a_file_the_skill_needs_is_refused_when_it_is_built() -> None:
    files = dict(load_skill().files)
    del files["editor.md"]

    with pytest.raises(ValueError, match="the analyst skill is missing editor.md"):
        AnalystSkill(files=files)
