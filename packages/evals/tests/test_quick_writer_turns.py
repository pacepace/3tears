"""The recorded writer models (``quick/writer_turns.py``): one recording per case, routed by the condition that made it.

Each cell's sessions go to ``<recordings>/<arm>-<skill12>-<model>-<effort>/<case>.jsonl``, by the snapshot the
session reads. A file that exists replays with no key, through ``RecordedTurns`` and never ``Model``; a missing one
records when ``ANTHROPIC_API_KEY`` is set and is refused, naming the file and the variable, when it is not; a replay
that leaves recorded lines unasked is disclosed by session. Nothing here reaches a model: the replay is the
hand-scripted ``hand-pass`` recording, run through ``write_report`` with its tools live on the fixture snapshot, and
the recording case prices a turn from the shipped rates without sending one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Final

import pytest

from packages.evals.tests.report_writer_support import HAND_RECORDINGS, MODEL, kairo_id, written
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.ops.report_writer.writer_contract import EffortWord
from threetears.evals.quick.writer_turns import KEY_VARIABLE, TurnsLeft, WriterTurns, writer_turns
from threetears.evals.schema.turns import TurnMessage
from threetears.evals.testing import RecordingRefused

#: The hand-scripted recording replayed: Kairo's report, cleared by the first review.
HAND_PASS: Final = HAND_RECORDINGS / "hand-pass"

#: A Claude model the shipped rates price, so a recording cell bounds a turn with no catalogue read.
PRICED: Final = "claude-opus-5-5"

SHIPPED = load_skill()


def _variant() -> AnalystSkill:
    """The shipped skill with one line added to its editor: another skill by its digest."""
    files = dict(SHIPPED.files)
    files["editor.md"] += "\nOne more line.\n"
    return AnalystSkill(files=files)


def _turns(tmp_path: Path, *, skill: AnalystSkill = SHIPPED, arm: str = "shipped") -> WriterTurns:
    return writer_turns(tmp_path / "turns", arm=arm, skill=skill, cases={kairo_id(): "kairo"})


def _laid(turns: WriterTurns, *, model: str = MODEL, effort: EffortWord | None = None) -> Path:
    """Copy the hand-scripted recording to where ``turns`` routes Kairo's sessions; the path."""
    path = turns.path(model, effort, "kairo")
    path.parent.mkdir(parents=True)
    shutil.copyfile(HAND_PASS / "turns.jsonl", path)
    return path


@pytest.fixture
def no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_VARIABLE, raising=False)


def test_a_case_is_routed_by_arm_skill_model_effort_and_case(tmp_path: Path) -> None:
    turns = _turns(tmp_path)
    skill12 = SHIPPED.digest[:12]

    assert turns.path("claude-sonnet-5-5", "medium", "kairo") == (
        tmp_path / "turns" / f"shipped-{skill12}-claude-sonnet-5-5-medium" / "kairo.jsonl"
    )
    assert turns.path("claude-sonnet-5-5", None, "r2-a").name == "r2-a.jsonl"
    assert turns.path("claude-sonnet-5-5", None, "r2-a").parent.name == f"shipped-{skill12}-claude-sonnet-5-5-default"
    other_arm = _turns(tmp_path, arm="lead-first").path("claude-sonnet-5-5", "medium", "kairo")
    assert other_arm.parent.name == f"lead-first-{skill12}-claude-sonnet-5-5-medium"


def test_two_skills_never_share_a_recording(tmp_path: Path, no_key: None) -> None:
    variant = _variant()
    assert variant.digest[:12] != SHIPPED.digest[:12]
    shipped, varied = _turns(tmp_path), _turns(tmp_path, skill=variant)
    assert shipped.path(MODEL, "medium", "kairo") != varied.path(MODEL, "medium", "kairo")

    # The control's recording exists; the variant, on the same arm label, model, effort and case, never replays it.
    _laid(shipped, effort="medium")
    with pytest.raises(RecordingRefused, match=KEY_VARIABLE) as refused:
        varied(MODEL)("medium").turn_session("analyst", snapshot=kairo_id())
    assert refused.value.path == varied.path(MODEL, "medium", "kairo")


async def test_an_existing_recording_replays_with_no_key(tmp_path: Path, no_key: None) -> None:
    turns = _turns(tmp_path)
    _laid(turns)
    expected = json.loads((HAND_PASS / "replay.json").read_text())

    model = turns(MODEL)(None)
    assert await model.turn_pricing_refusal() is None
    result = await written(model, budget_usd=expected["budget_usd"])

    assert result.ending == expected["ending"] and result.stored is not None
    assert result.outcome.analyst_rounds == expected["analyst_rounds"]
    assert result.outcome.editor_rounds == expected["editor_rounds"]
    assert turns.unconsumed() == []


def test_a_missing_recording_with_no_key_is_refused_naming_the_file_and_the_key(tmp_path: Path, no_key: None) -> None:
    turns = _turns(tmp_path)
    path = turns.path(MODEL, "high", "kairo")
    model = turns(MODEL)("high")

    with pytest.raises(RecordingRefused) as refused:
        model.turn_session("analyst", snapshot=kairo_id())

    assert refused.value.path == path
    assert str(path) in str(refused.value) and KEY_VARIABLE in str(refused.value)
    # An apparatus fault, so the runner excludes the cell rather than scoring it.
    assert not path.exists()


async def test_a_replay_that_leaves_lines_unasked_is_disclosed_by_session(tmp_path: Path, no_key: None) -> None:
    turns = _turns(tmp_path)
    path = _laid(turns)
    line = json.loads(path.read_text().splitlines()[1])  # the analyst's first reply
    leftover = {**line, "session": "editor-9", "index": 0}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(leftover) + "\n")

    result = await written(turns(MODEL)(None), budget_usd=5.0)

    assert result.stored is not None, "the control: the recorded sessions still run as recorded"
    assert turns.unconsumed() == [TurnsLeft("kairo", "editor-9", 1, path)]
    said = str(turns.unconsumed()[0])
    assert "'editor-9'" in said and "1 recorded lines unasked" in said and str(path) in said


def test_a_second_cell_on_a_recorded_case_is_refused(tmp_path: Path, no_key: None) -> None:
    turns = _turns(tmp_path)
    _laid(turns)
    first = turns(MODEL)(None)
    first.turn_session("analyst", snapshot=kairo_id())
    first.turn_session("editor-1", snapshot=kairo_id())  # the same cell's next session reads the same recording

    with pytest.raises(RecordingRefused, match="k_runs=1"):
        turns(MODEL)(None).turn_session("analyst", snapshot=kairo_id())


def test_a_snapshot_the_map_does_not_name_is_refused(tmp_path: Path, no_key: None) -> None:
    turns = _turns(tmp_path)

    with pytest.raises(RecordingRefused, match="is no case of arm 'shipped'"):
        turns(MODEL)(None).turn_session("analyst", snapshot="snapshot-of-nothing")


async def test_a_missing_recording_with_a_key_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_VARIABLE, "unused-no-turn-is-sent")
    turns = _turns(tmp_path)
    path = turns.path(PRICED, None, "kairo")
    model = turns(PRICED)(None)
    try:
        assert await model.turn_pricing_refusal() is None, "the shipped rates bound a turn on the priced model"
        session = model.turn_session("analyst", snapshot=kairo_id())
        ceiling = session.turn_price_ceiling(
            system="Write the report.", messages=[TurnMessage("user", "Begin.")], tools=[], max_tokens=1000
        )
    finally:
        await turns.aclose()

    assert ceiling is not None
    (line,) = [json.loads(text) for text in path.read_text().splitlines()]
    assert (line["session"], line["kind"], line["ceiling"]) == ("analyst", "ceiling", ceiling.usd)


@pytest.mark.parametrize(
    ("arm", "cases", "refused"),
    [
        ("a/b", {"s1": "kairo"}, "arm label 'a/b'"),
        ("shipped", {"s1": "../kairo"}, "case name '../kairo'"),
        ("shipped", {"s1": "kairo", "s2": "kairo"}, "both case 'kairo'"),
    ],
)
def test_a_name_that_would_not_route_to_one_file_is_refused(
    tmp_path: Path, arm: str, cases: dict[str, str], refused: str
) -> None:
    with pytest.raises(ValueError, match=refused):
        writer_turns(tmp_path, arm=arm, skill=SHIPPED, cases=cases)
