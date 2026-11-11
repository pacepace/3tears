"""Rung 5, ``examples/5-writer/``: the report writer evaluated as the thing under test, replayed with no key.

The rung's recordings (14 reports on ``claude-sonnet-5-5`` recorded 2026-11-03, and the flash judge and its pairwise
pass recorded again 2026-11-08 over the pages as the package now prints them) ship beside it. What is held: the eleven case files read back and split 7 held-out to 4 dev;
``writer.py`` stays short and reaches the engine through public roots; ``python -m threetears.evals writer.py``
replays the shipped recordings with no key, every turn and judge call served from them and none left unasked, and
prints the judge's preference and the promotion read. In a copy with its recordings stripped, the door refuses,
first for the judge's recording, then, with that in place, for the writer's, naming the file and
``ANTHROPIC_API_KEY``; and with a hand-scripted recording for two cases laid in that copy (never under
``examples/``) it replays and prints the promotion read, ``undecided`` on two cases, with its reason. Every socket
connect fails.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import socket
import sqlite3
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

#: How each of the 14 replayed reports ends, by arm and case, under the clearing rule of the owner's ruling of
#: 2026-11-09 (R1: a round-1 should_fix no longer blocks), which moved five of them from stored_with_findings to stored:
#: shipped's r1 reading only bars and r2 with its gates read in each task, lead-first's 1-triage, r2 with its gates read
#: in each task and r2 with one gate. A rule change that moves any ending turns this red.
ENDINGS: Final = {
    ("shipped", "1-triage"): "stored_with_findings",
    ("shipped", "kairo round 2"): "stored_with_findings",
    ("shipped", "r1 reading only bars"): "stored",
    ("shipped", "r1 without its gates"): "stored_with_findings",
    ("shipped", "r2 with its gates read in each task"): "stored",
    ("shipped", "r2 with one gate"): "stored",
    ("shipped", "r2 without its gates"): "stored_with_findings",
    ("lead-first", "1-triage"): "stored",
    ("lead-first", "kairo round 2"): "stored_with_findings",
    ("lead-first", "r1 reading only bars"): "stored",
    ("lead-first", "r1 without its gates"): "stored",
    ("lead-first", "r2 with its gates read in each task"): "stored",
    ("lead-first", "r2 with one gate"): "stored",
    ("lead-first", "r2 without its gates"): "stored",
}

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


@pytest.fixture
def unrecorded_dir(writer_dir: Path) -> Path:
    """The copy with its recordings stripped, the README that says how to record them left in place."""
    recordings = writer_dir / "recordings"
    (recordings / "judge.jsonl").unlink()
    shutil.rmtree(recordings / "turns")
    assert [path.name for path in recordings.iterdir()] == ["README.md"]
    return writer_dir


def test_the_eleven_case_files_read_back_and_split_seven_held_out_to_four_dev() -> None:
    cases = read_writer_cases(RUNG / "cases")
    assert len(cases) == 11
    sides = Counter(writer_group_side(case.group or case.name, 0.35) for case in cases)
    assert sides == {"held_out": 7, "dev": 4}
    # 0.3, below WriterEval's 0.35 default, would leave fewer than the 5 held-out cases a rate needs.
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
    assert 'effort="medium"' in writer and "max_cost_usd=39.0" in writer
    assert "budget_usd=2.5" in writer and "turns=45" in writer and "max_tokens=2048" in writer
    assert "owner-approved $39 cap" in readme and "awaiting" not in readme
    # The recording's date and the head it was made at; and when the judge's calls were recorded again, naming the
    # commit whose pages the judge read, which is the one a replay of its keys must stand on.
    assert "2026-11-03" in readme and "`28efc1fd1`" in readme
    assert "2026-11-10, over the pages built at `7601153ed`" in readme
    # What it cost: the judge's dollars are what judge.jsonl recorded, and the total is the writer's and the judge's.
    judged = sum(
        json.loads(line)["result"]["cost_usd"]
        for line in (RUNG / "recordings" / "judge.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    (writer_usd,) = re.findall(r"\| Writer \| .*: \$(\d+\.\d\d) \|", readme)
    (judge_usd,) = re.findall(r"\| Judge \| .*, \$(\d+\.\d\d) \|", readme)
    (total_usd,) = re.findall(r"\| Total \| \$(\d+\.\d\d) \|", readme)
    assert judge_usd == f"{judged:.2f}"
    assert total_usd == f"{float(writer_usd) + float(judge_usd):.2f}"
    assert "`b9164e5ec`" not in readme, "the merge under the recording, whose pages the judge never read"
    assert "Not recorded" not in readme


def test_the_shipped_recordings_replay_with_no_key_and_print_the_preference_and_the_promotion_read(
    writer_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["writer.py"]) == 0
    printed = capsys.readouterr().out
    lines = printed.splitlines()
    assert lines[0] == f"writer: lead-first against shipped, 7 held-out case(s) x 1, judged by {JUDGE}"
    # The pairwise preference, the primary reading: lead-first's share beside shipped's and the interval on it, all on
    # the share scale, which the evidence cannot tell from even at 7 cases (the glossary's words, by owner ruling), a
    # row with columns; its two orders agreed on 6 of 7 cases, so the pass gives the held-out cases that would settle
    # it at the share it read, and its spend in the money form.
    # Re-pinned 2026-11-10, the judge recorded again over the pages with one form per measure and the new layout:
    # from 70 · 30 · 33
    # to 100, a lean of -2 points and 33 cases to settle.
    assert "| lead-first | every pair | cannot tell apart | 61 | 39 | 28 to 96 |" in lines
    assert "| lead-first | 86% (6 of 7) | 7 | 0 | 120 | $0.13 of $1.95 |" in lines
    assert "not separated" not in printed
    (promotion,) = [line for line in lines if line.startswith("promotion of lead-first: ")]
    assert promotion == (
        "promotion of lead-first: Undecided: on the primary reading the evidence cannot tell lead-first and shipped "
        "apart, and only an improved primary reading promotes."
    )
    # Every arm replayed whole: nothing recorded now, and no recorded turn left unasked.
    assert "shipped: writer turns replayed from" in printed and "lead-first: writer turns replayed from" in printed
    assert "0 replayed" not in printed and "unasked" not in printed
    # Every judge call served from its recording: a refused one would exclude its result.
    assert printed.count("7 result(s): 7 scored, 0 excluded") == 2
    # The page's first screen opens on its question, then the answer: the promotion reading and the judge's
    # preference, in its words; and the eval declared its criteria as it launched, so the replay never reads them as
    # set after its results.
    page = (writer_dir / ".evals" / "reports" / "writer" / "report.md").read_text(encoding="utf-8")
    first_screen = page.split("\n## ", 1)[0].splitlines()
    assert first_screen[1] == '"Does lead-first write reports better than shipped?"'
    # What the page compares and on what is the line under the question, then the answer.
    assert first_screen[3] == "shipped vs lead-first · report-writing"
    assert first_screen[5] == (
        "**Undecided, so shipped stays: across 7 held-out cases the judge gave lead-first 61% of its preference over "
        "shipped (50% is even), too few cases to tell from an even split; if the share stayed near 61%, roughly 100 "
        "cases would settle it.**"
    )
    assert "set after the results" not in page and "Criteria set late" not in page
    # No surface of the page prints the retired words (owner ruling).
    html = (writer_dir / ".evals" / "reports" / "writer" / "report.html").read_text(encoding="utf-8")
    assert "not separated" not in page and "not separated" not in html
    assert _endings(writer_dir) == ENDINGS


def _endings(directory: Path) -> dict[tuple[str, str], str]:
    """Each replayed report's ending, by arm (named by its skill's digest) and case (named by its snapshot)."""
    control = AnalystSkill.at(directory / "skills" / "control")
    lead_first = control.with_file("SKILL.md", (directory / "skills" / "lead-first.md").read_text(encoding="utf-8"))
    arms = {control.digest: "shipped", lead_first.digest: "lead-first"}
    cases = {case.snapshot.id: case.name for case in read_writer_cases(directory / "cases")}
    with sqlite3.connect(f"file:{directory / '.evals' / 'evals.sqlite'}?mode=ro", uri=True) as db:
        bodies = [
            json.loads(body)
            for (body,) in db.execute(
                "select body from documents where doc_type in (?, ?)", ("eval_result", "eval_test_case")
            )
        ]
    snapshot_of = {
        body["id"]: body["host_payload"]["writer_case"]["snapshot"]["id"]
        for body in bodies
        if body["doc_type"] == "eval_test_case"
    }
    return {
        (arms[body["kind_payload"]["skill"]], cases[snapshot_of[body["test_case_id"]]]): body["kind_payload"]["ending"]
        for body in bodies
        if body["doc_type"] == "eval_result"
    }


def test_with_nothing_recorded_the_door_refuses_naming_the_judge_s_file_and_its_key(
    unrecorded_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["writer.py"]) == 2
    refused = capsys.readouterr().err
    assert "recordings/judge.jsonl" in refused and "OPENROUTER_API_KEY" in refused
    assert not (unrecorded_dir / ".evals").exists(), "the file is refused as it is loaded, before a store is opened"


def test_with_the_writer_s_turns_unrecorded_the_door_refuses_naming_the_file_and_the_key(
    unrecorded_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _judge_recording(unrecorded_dir)

    assert main(["writer.py"]) == 2
    refused = capsys.readouterr().err
    assert "arm 'shipped' has no turn recording for 7 of 7 case(s)" in refused
    named = re.search(r"\((\S+\.jsonl) first\)", refused)
    assert named is not None and Path(named[1]).is_relative_to(unrecorded_dir / "recordings" / "turns")
    assert "ANTHROPIC_API_KEY is not set" in refused


def test_a_two_case_scripted_recording_replays_with_no_key_and_prints_the_promotion_read(
    unrecorded_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    kept = _two_held_out_cases(unrecorded_dir)
    _judge_recording(unrecorded_dir)
    (writer_eval,) = load_evals(unrecorded_dir / "writer.py")
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
    assert (unrecorded_dir / ".evals" / "reports" / "writer" / "report.html").is_file()


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
