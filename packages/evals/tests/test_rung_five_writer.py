"""Rung 5, ``examples/5-writer/``: the report writer evaluated as the thing under test, replayed with no key.

The rung's turn recordings are an owner-approved paid step, not yet made, so nothing here replays a real writer. What
is held: the eleven case files read back and split 7 held-out to 4 dev; ``writer.py`` stays short and reaches the
engine through public roots; with nothing recorded the door refuses, first for the judge's recording, then, with that
in place, for the writer's, naming the file and ``ANTHROPIC_API_KEY``; and with a hand-scripted recording for two
cases (laid in the test's own copy, never under ``examples/``) ``python -m threetears.evals writer.py`` replays with no
key and prints the promotion read, ``undecided`` on two cases, with its reason. Every socket connect fails.
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
from collections import Counter
from pathlib import Path
from typing import Any, Final

import pytest

from packages.evals.tests.example_loader import EXAMPLES
from packages.evals.tests.report_writer_support import reply
from packages.evals.tests.rung_support import program_lines, rung_copy
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations
from packages.evals.tests.writer_eval_support import ScoredReply
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.harness import AnalystSkill, WriterEval, read_writer_cases
from threetears.evals.harness.writer_turns import writer_turns
from threetears.evals.ops.report_writer.writer_bank import writer_group_side
from threetears.evals.quick.files import load_evals, main
from threetears.evals.schema.turns import TurnText
from threetears.evals.testing import RecordedCompletions

#: The rung's directory.
RUNG: Final = EXAMPLES / "5-writer"

#: The effort both arms write at; their model is read off the eval the file builds, never spelled here.
EFFORT: Final = "medium"

#: The scoring judge's model.
JUDGE: Final = "google/gemini-3.8-flash"

#: How many analyst replies a silent report asks for before the loop gives up on it.
SILENT_TURNS: Final = 4


@pytest.fixture
def writer_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of ``examples/5-writer/`` as the working directory, with no key, and every socket connect refused."""
    copy = rung_copy("5-writer", tmp_path, monkeypatch)

    def connect(_self: socket.socket, address: object) -> None:
        raise AssertionError(f"a replay-only run connected to {address}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    return copy


def test_the_eleven_case_files_read_back_and_split_seven_held_out_to_four_dev() -> None:
    cases = read_writer_cases(RUNG / "cases")
    assert len(cases) == 11
    sides = Counter(writer_group_side(case.group or case.name, 0.35) for case in cases)
    assert sides == {"held_out": 7, "dev": 4}
    # The 0.3 default the README says it moves from leaves fewer than the 5 held-out cases a rate needs.
    assert Counter(writer_group_side(case.group or case.name, 0.3) for case in cases)["held_out"] < 5
    # Every ladder rung's campaign is a case, in a stratum of its own, beside the misfits.
    ladder = sorted(path.name for path in EXAMPLES.iterdir() if path.is_dir() and path.name != RUNG.name)
    assert {case.name for case in cases} >= set(ladder)
    assert Counter(case.stratum for case in cases)["misfit"] == 6
    # Every withheld note is one the frozen control skill holds.
    control = AnalystSkill.at(RUNG / "skills" / "control")
    for case in cases:
        control.analyst(withhold=case.withhold)


def test_the_control_is_a_frozen_copy_of_a_shipped_skill_and_the_variant_differs_in_one_file() -> None:
    control = AnalystSkill.at(RUNG / "skills" / "control")
    readme = (RUNG / "skills" / "README.md").read_text(encoding="utf-8")
    assert control.digest in readme, "the skills README names the digest the control was frozen at"
    variant = control.with_file("SKILL.md", (RUNG / "skills" / "lead-first.md").read_text(encoding="utf-8"))
    differs = {name for name in control.files if control.files[name] != variant.files[name]}
    assert differs == {"SKILL.md"} and variant.digest != control.digest
    # The variant shares the control's notes, so every case runs on both.
    assert variant.analyst().exemplars == control.analyst().exemplars
    # The shipped skill may move on; the copy is what was recorded. Today they are the same.
    assert set(control.files) == set(load_skill().files)


def test_writer_py_is_short_and_reaches_the_engine_only_through_public_roots() -> None:
    path = RUNG / "writer.py"
    text = path.read_text(encoding="utf-8")
    assert sum(1 for line in text.splitlines() if line.strip()) <= 60
    assert program_lines(path) <= 60
    assert public_root_violations(SOURCE_ROOT, [("5-writer/writer.py", path)], consumer_root=REPO_ROOT) == []


def test_the_recordings_readme_names_the_one_command_that_records_and_its_keys() -> None:
    readme = (RUNG / "recordings" / "README.md").read_text(encoding="utf-8")
    for needed in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "python -m threetears.evals examples/5-writer/writer.py"):
        assert needed in readme
    writer = (RUNG / "writer.py").read_text(encoding="utf-8")
    assert 'effort="medium"' in writer and "max_cost_usd=15.0" in writer


def test_with_nothing_recorded_the_door_refuses_naming_the_judge_s_file_and_its_key(
    writer_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["writer.py"]) == 2
    refused = capsys.readouterr().err
    assert "recordings/judge.jsonl" in refused and "OPENROUTER_API_KEY" in refused
    assert not (writer_dir / ".evals").exists(), "the file is refused as it is loaded, before a store is opened"


def test_with_the_writer_s_turns_unrecorded_the_door_refuses_naming_the_file_and_the_key(
    writer_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _judge_recording(writer_dir)

    assert main(["writer.py"]) == 2
    refused = capsys.readouterr().err
    assert "arm 'shipped' has no turn recording for 7 of 7 case(s)" in refused
    named = re.search(r"\((\S+\.jsonl) first\)", refused)
    assert named is not None and Path(named[1]).is_relative_to(writer_dir / "recordings" / "turns")
    assert "ANTHROPIC_API_KEY is not set" in refused


def test_a_two_case_scripted_recording_replays_with_no_key_and_prints_the_promotion_read(
    writer_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    kept = _two_held_out_cases(writer_dir)
    _judge_recording(writer_dir)
    (writer_eval,) = load_evals(writer_dir / "writer.py")
    assert isinstance(writer_eval, WriterEval)
    _silent_recordings(writer_eval, kept)

    assert main(["writer.py"]) == 0
    printed = capsys.readouterr().out
    lines = printed.splitlines()
    assert lines[0] == f"writer: lead-first against shipped, 2 held-out case(s) x 1, judged by {JUDGE}"
    (promotion,) = [line for line in lines if line.startswith("promotion of lead-first: ")]
    # Undecided, and why, in the read's own words: undecided never promotes.
    assert re.fullmatch(r"promotion of lead-first: Undecided: .{20,}", promotion), promotion
    assert "shipped: writer turns replayed from" in printed and "lead-first: writer turns replayed from" in printed
    assert "| misfit | 2 |" in printed
    assert "report: .evals/reports/writer/report.html" in lines
    assert (writer_dir / ".evals" / "reports" / "writer" / "report.html").is_file()


def _two_held_out_cases(directory: Path) -> dict[str, str]:
    """Keep the two cases of one held-out group in the copy's cases, and drop the rest: their snapshot ids, by name."""
    cases = read_writer_cases(directory / "cases")
    groups = Counter(case.group for case in cases if writer_group_side(case.group or case.name, 0.35) == "held_out")
    group = next(name for name, n in sorted(groups.items(), key=str) if n == 2)
    kept: dict[str, str] = {}
    for path in sorted((directory / "cases").glob("*.json.gz")):
        case = next(case for case in cases if path.name == f"{case.name.replace(' ', '-')}.json.gz")
        if case.group == group:
            kept[case.snapshot.id] = case.name
        else:
            path.unlink()
    assert len(kept) == 2
    return kept


def _judge_recording(directory: Path) -> None:
    """One scripted call in the judge's recording, so the judge builds with no key: nothing in a silent replay asks
    it anything."""

    class ScriptedJudge:
        model_name = JUDGE

        async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
            return ScoredReply(content=json.dumps({"score": 4}), temperature=0.0)

    async def record() -> None:
        recorder = RecordedCompletions.record(ScriptedJudge(), directory / "recordings" / "judge.jsonl", settings=None)
        await recorder.generate(system="scripted", user="scripted")

    asyncio.run(record())


def _silent_recordings(writer_eval: WriterEval, cases: dict[str, str]) -> None:
    """Every arm's analyst session for every kept case: replies holding no document, as a recording keeps them."""
    for arm, writer in writer_eval.arms.items():
        routed = writer_turns(
            writer_eval.recordings, arm=arm, skill=writer.analyst_skill, cases=cases, revisions=writer.revisions
        )
        for snapshot in cases:
            assert writer.effort == EFFORT
            path = routed.path(writer.model_id, EFFORT, snapshot)
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = _silent_lines(writer.model_id)
            path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def _silent_lines(model: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for index in range(SILENT_TURNS):
        stamp = {"recorder_version": 1, "recorded_at": "2026-11-03T00:00:00+00:00", "session": "analyst"}
        lines.append({**stamp, "index": index, "kind": "ceiling", "ceiling": 0.05})
        said = reply(TurnText("I have no report yet."), model=model).to_json()
        lines.append({**stamp, "index": index, "kind": "reply", "reply": said})
    return lines
