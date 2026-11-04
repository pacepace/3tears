"""The recorded writer models (``harness/writer_turns.py``): one recording per case, routed by the condition that made it.

Each cell's sessions go to ``<recordings>/<arm>-<skill12>-<model>-<effort>-r<revisions>/<case>-<snapshot12>.jsonl``, by
the snapshot the session reads, and an arm whose editor runs on a model of its own keeps the editor's in a file beside
it (``-e<editor>`` on the directory). A file that exists replays with no key, through ``RecordedTurns`` and never
``Model``; a missing one records on the arm's provider when its key (``ANTHROPIC_API_KEY`` by default) is set and is
refused, naming the file and the variable, when it is not; a replay
that leaves recorded lines unasked is disclosed by session. Nothing here reaches a model: the replay is the
hand-scripted ``hand-pass`` recording, run through ``write_report`` with its tools live on the fixture snapshot, and
the recording case prices a turn from the shipped rates without sending one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import ClassVar, Final, get_args

import pytest

from packages.evals.tests.fixtures.writer_cases.writer_smoke import WRITER
from packages.evals.tests.report_writer_support import HAND_RECORDINGS, MODEL, kairo_id, written
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals import providers
from threetears.evals.harness.writer_turns import TurnsLeft, WriterTurns, writer_turns
from threetears.evals.providers import Provider
from threetears.evals.providers.wire import KEY_VARIABLES as SHIPPED_KEY_VARIABLES
from threetears.evals.ops.report_writer.loop import MOST_REVISIONS
from threetears.evals.ops.report_writer.writer_contract import EffortWord, WriterArm
from threetears.evals.schema.turns import TurnMessage
from threetears.evals.testing import RecordingRefused

#: The hand-scripted recording replayed: Kairo's report, cleared by the first review.
HAND_PASS: Final = HAND_RECORDINGS / "hand-pass"

#: A Claude model the shipped rates price, so a recording cell bounds a turn with no catalogue read: the live smoke's.
PRICED: Final = WRITER

#: The variable a recording run reads its key from, as a developer sets it.
KEY_VARIABLE: Final = "ANTHROPIC_API_KEY"

#: A second snapshot id, for a case that keeps its name over moved evidence.
MOVED: Final = "report-snapshot:0123456789abcdef0123456789abcdef"

SHIPPED = load_skill()


def _variant() -> AnalystSkill:
    """The shipped skill with one line added to its editor: another skill by its digest."""
    files = dict(SHIPPED.files)
    files["editor.md"] += "\nOne more line.\n"
    return AnalystSkill(files=files)


def _turns(
    tmp_path: Path,
    *,
    skill: AnalystSkill = SHIPPED,
    arm: str = "shipped",
    snapshot: str | None = None,
    revisions: int = MOST_REVISIONS,
) -> WriterTurns:
    cases = {snapshot or kairo_id(): "kairo"}
    return writer_turns(tmp_path / "turns", arm=arm, skill=skill, cases=cases, revisions=revisions)


def _laid(turns: WriterTurns, *, model: str = MODEL, effort: EffortWord | None = None) -> Path:
    """Copy the hand-scripted recording to where ``turns`` routes Kairo's sessions; the path."""
    path = turns.path(model, effort, kairo_id())
    path.parent.mkdir(parents=True)
    shutil.copyfile(HAND_PASS / "turns.jsonl", path)
    return path


@pytest.fixture
def no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_VARIABLE, raising=False)


def test_a_case_is_routed_by_arm_skill_model_effort_revisions_and_snapshot(tmp_path: Path) -> None:
    turns = _turns(tmp_path)
    skill12, snapshot12 = SHIPPED.digest[:12], kairo_id().rpartition(":")[2][:12]
    revisions = f"r{MOST_REVISIONS}"

    assert turns.path(MODEL, "medium", kairo_id()) == (
        tmp_path / "turns" / f"shipped-{skill12}-{MODEL}-medium-{revisions}" / f"kairo-{snapshot12}.jsonl"
    )
    assert turns.path(MODEL, None, kairo_id()).parent.name == f"shipped-{skill12}-{MODEL}-default-{revisions}"
    other_arm = _turns(tmp_path, arm="lead-first").path(MODEL, "medium", kairo_id())
    assert other_arm.parent.name == f"lead-first-{skill12}-{MODEL}-medium-{revisions}"
    fewer = _turns(tmp_path, revisions=0).path(MODEL, "medium", kairo_id())
    assert fewer.parent.name == f"shipped-{skill12}-{MODEL}-medium-r0"


def test_an_arm_with_other_revisions_never_replays_the_recording(tmp_path: Path, no_key: None) -> None:
    _laid(_turns(tmp_path))
    fewer = _turns(tmp_path, revisions=0)

    with pytest.raises(RecordingRefused, match=KEY_VARIABLE) as refused:
        fewer(WriterArm(MODEL, SHIPPED, revisions=0))(None).turn_session("analyst", snapshot=kairo_id())
    assert refused.value.path == fewer.path(MODEL, None, kairo_id())


def test_a_case_over_moved_evidence_never_replays_the_old_recording(tmp_path: Path, no_key: None) -> None:
    _laid(_turns(tmp_path))
    moved = _turns(tmp_path, snapshot=MOVED)
    assert moved.path(MODEL, None, MOVED).name == "kairo-0123456789ab.jsonl"

    # The same case name over another snapshot is another recording, so the old one is never found.
    with pytest.raises(RecordingRefused, match=f"case 'kairo' over snapshot {MOVED}") as refused:
        moved(WriterArm(MODEL, SHIPPED))(None).turn_session("analyst", snapshot=MOVED)
    assert refused.value.path == moved.path(MODEL, None, MOVED)


def test_the_factory_refuses_a_skill_other_than_the_arm_s(tmp_path: Path) -> None:
    variant = _variant()
    turns = _turns(tmp_path)

    with pytest.raises(RecordingRefused, match=f"launched on skill {variant.digest[:12]}") as refused:
        turns(WriterArm(MODEL, variant))
    assert SHIPPED.digest[:12] in str(refused.value)


def test_two_skills_never_share_a_recording(tmp_path: Path, no_key: None) -> None:
    variant = _variant()
    assert variant.digest[:12] != SHIPPED.digest[:12]
    shipped, varied = _turns(tmp_path), _turns(tmp_path, skill=variant)
    assert shipped.path(MODEL, "medium", kairo_id()) != varied.path(MODEL, "medium", kairo_id())

    # The control's recording exists; the variant, on the same arm label, model, effort and case, never replays it.
    _laid(shipped, effort="medium")
    with pytest.raises(RecordingRefused, match=KEY_VARIABLE) as refused:
        varied(WriterArm(MODEL, variant))("medium").turn_session("analyst", snapshot=kairo_id())
    assert refused.value.path == varied.path(MODEL, "medium", kairo_id())


async def test_an_existing_recording_replays_with_no_key(tmp_path: Path, no_key: None) -> None:
    turns = _turns(tmp_path)
    _laid(turns)
    expected = json.loads((HAND_PASS / "replay.json").read_text())

    model = turns(WriterArm(MODEL, SHIPPED))(None)
    assert await model.turn_pricing_refusal() is None
    result = await written(model, budget_usd=expected["budget_usd"])

    assert result.ending == expected["ending"] and result.stored is not None
    assert result.outcome.analyst_rounds == expected["analyst_rounds"]
    assert result.outcome.editor_rounds == expected["editor_rounds"]
    assert turns.unconsumed() == []


def test_a_missing_recording_with_no_key_is_refused_naming_the_file_and_the_key(tmp_path: Path, no_key: None) -> None:
    turns = _turns(tmp_path)
    path = turns.path(MODEL, "high", kairo_id())
    model = turns(WriterArm(MODEL, SHIPPED))("high")

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

    result = await written(turns(WriterArm(MODEL, SHIPPED))(None), budget_usd=5.0)

    assert result.stored is not None, "the control: the recorded sessions still run as recorded"
    assert turns.unconsumed() == [TurnsLeft("kairo", "editor-9", 1, path)]
    said = str(turns.unconsumed()[0])
    assert "'editor-9'" in said and "1 recorded lines unasked" in said and str(path) in said


def test_a_second_cell_on_a_recorded_case_is_refused(tmp_path: Path, no_key: None) -> None:
    turns = _turns(tmp_path)
    _laid(turns)
    first = turns(WriterArm(MODEL, SHIPPED))(None)
    first.turn_session("analyst", snapshot=kairo_id())
    first.turn_session("editor-1", snapshot=kairo_id())  # the same cell's next session reads the same recording

    with pytest.raises(RecordingRefused, match="k_runs=1"):
        turns(WriterArm(MODEL, SHIPPED))(None).turn_session("analyst", snapshot=kairo_id())


def test_a_snapshot_the_map_does_not_name_is_refused(tmp_path: Path, no_key: None) -> None:
    turns = _turns(tmp_path)

    with pytest.raises(RecordingRefused, match="is no case of arm 'shipped'"):
        turns(WriterArm(MODEL, SHIPPED))(None).turn_session("analyst", snapshot="snapshot-of-nothing")


async def test_a_missing_recording_with_a_key_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_VARIABLE, "unused-no-turn-is-sent")
    turns = _turns(tmp_path)
    path = turns.path(PRICED, None, kairo_id())
    model = turns(WriterArm(PRICED, SHIPPED))(None)
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
        ("shipped", {"report-snapshot:": "kairo"}, "snapshot id ''"),
    ],
)
def test_a_name_that_would_not_route_to_one_file_is_refused(
    tmp_path: Path, arm: str, cases: dict[str, str], refused: str
) -> None:
    with pytest.raises(ValueError, match=refused):
        writer_turns(tmp_path, arm=arm, skill=SHIPPED, cases=cases)


def test_revisions_outside_the_lever_s_range_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="revisions are 0 to"):
        writer_turns(tmp_path, arm="shipped", skill=SHIPPED, cases={"s1": "kairo"}, revisions=MOST_REVISIONS + 1)


def test_each_provider_s_key_variable_is_the_shipped_client_s_and_another_provider_is_refused(tmp_path: Path) -> None:
    read = {
        provider: writer_turns(tmp_path, arm="a", skill=SHIPPED, cases={}, provider=provider).key_variable
        for provider in get_args(Provider)
    }
    assert read == dict(SHIPPED_KEY_VARIABLES)
    with pytest.raises(ValueError, match="not 'carrier-pigeon'"):
        writer_turns(tmp_path, arm="a", skill=SHIPPED, cases={}, provider="carrier-pigeon")


# parity-exempt: stands in for the client's constructor only, to keep the route a recording model is built on
class FakeRecordingModel:
    """What a recording run builds, kept by its route; it sends nothing."""

    built: ClassVar[list[tuple[str, str, Path]]] = []

    def __init__(self, model: str, *, provider: str, effort: EffortWord | None, turn_recording: Path) -> None:
        self.model_name = model
        self.built.append((model, provider, turn_recording))

    async def aclose(self) -> None:
        return None


async def test_a_writer_on_another_provider_records_on_it_and_is_refused_without_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_VARIABLE, "the-anthropic-key-is-not-this-arm-s")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cases = {kairo_id(): "kairo"}
    turns = writer_turns(tmp_path / "turns", arm="routed", skill=SHIPPED, cases=cases, provider="openrouter")
    assert turns.key_variable == "OPENROUTER_API_KEY"

    with pytest.raises(RecordingRefused, match="OPENROUTER_API_KEY") as refused:
        turns(WriterArm(MODEL, SHIPPED))(None).turn_session("analyst", snapshot=kairo_id())
    assert refused.value.path == turns.path(MODEL, None, kairo_id())

    # With its own key set, the model built to record the case is built on the arm's provider.
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused-no-turn-is-sent")
    monkeypatch.setattr(providers, "Model", FakeRecordingModel)
    monkeypatch.setattr(FakeRecordingModel, "built", [])
    again = writer_turns(tmp_path / "again", arm="routed", skill=SHIPPED, cases=cases, provider="openrouter")
    again(WriterArm(MODEL, SHIPPED))(None)
    await again.aclose()
    assert FakeRecordingModel.built == [(MODEL, "openrouter", again.path(MODEL, None, kairo_id()))]


def test_an_arm_s_editor_routes_to_a_file_of_its_own_and_another_editor_is_refused(tmp_path: Path) -> None:
    turns = writer_turns(
        tmp_path / "turns", arm="edited", skill=SHIPPED, cases={kairo_id(): "kairo"}, editor_model="ed"
    )
    analyst, editor = turns.paths(MODEL, None, kairo_id())

    assert analyst.parent == editor.parent and analyst.parent.name.endswith(f"-r{MOST_REVISIONS}-eed")
    assert editor.name == analyst.name.removesuffix(".jsonl") + "-editor.jsonl"
    # The arm with no editor of its own keeps one file, and is asked for no editor's.
    plain = _turns(tmp_path)
    assert plain.paths(MODEL, None, kairo_id()) == [plain.path(MODEL, None, kairo_id())]
    with pytest.raises(ValueError, match="runs its editor on the candidate model"):
        plain.path(MODEL, None, kairo_id(), role="editor")
    # An arm launched on another editor, or on none, never replays this arm's recordings.
    for launched in (WriterArm(MODEL, SHIPPED, editor_model="other"), WriterArm(MODEL, SHIPPED)):
        with pytest.raises(RecordingRefused, match="was launched on editor"):
            turns(launched)
