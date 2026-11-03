"""``WriterEval``: the report writer measured over portable case files, replayed from turn recordings, offline.

Every case is a package fixture snapshot written as a writer case file; every arm's writer is a turn recording laid
where the arm's router reads it, whose analyst replies hold no document, so each cell ends with nothing stored and
nothing reaches the judge. The judge is a :class:`~threetears.evals.providers.Model` whose transport fails any request,
and every socket connect fails too: a replay-only run makes no network call.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from packages.evals.tests.analyst_support import fixture_snapshots
from packages.evals.tests.report_writer_support import reply
from packages.evals.tests.writer_case_support import SEED
from threetears.evals import Eval
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.kernel import EvalStorage
from threetears.evals.harness import Writer, WriterCaseFile, WriterEval, WriterResult, read_writer_cases
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.writer_bank import writer_case_bank
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND, WRITER_KIND_MEASURES
from threetears.evals.providers import Model
from threetears.evals.harness.writer_cases import writer_case_freeze
from threetears.evals.harness.writer_turns import KEY_VARIABLE, writer_turns
from threetears.evals.schema.turns import TurnText
from threetears.evals.storage import SqliteDocumentStore

#: The writer every arm names.
WRITER: Final = "claude-haiku-5-5"

#: The scoring judge: OpenRouter's, of the google family.
JUDGE: Final = "google/gemini-3.8-flash"

#: The cases, by fixture name: their stratum. At a 0.35 share r1 and r2 are held-out groups and r3 a dev one.
CASES: Final = {"r1": "reference", "r2": "reference", "r2 without its gates": "misfit", "r3": "reference"}

#: The cases on the held-out side: three fixtures, and a quick eval's campaign frozen in its own store, in r1's group.
HELD_OUT: Final = ("r1", "r2", "r2 without its gates", "triage")

#: The quick eval's tickets: each runner misses some, so the frozen page shows examples of stored results.
TICKETS: Final = [
    {"text": "refund please", "label": "billing"},
    {"text": "app crashes", "label": "bug"},
    {"text": "refund page crashes", "label": "billing"},
    {"text": "charged though the button is broken", "label": "billing"},
]

#: How many analyst replies a silent report asks for before the loop gives up on it.
SILENT_TURNS: Final = 4

SCOPE: Final = "writer"


def _variant() -> AnalystSkill:
    """The shipped skill with one blank line added to ``SKILL.md``: another digest, the same instructions."""
    files = dict(load_skill().files)
    files["SKILL.md"] = f"{files['SKILL.md']}\n"
    return AnalystSkill(files=files)


def _refuse(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"the judge was sent {request.url}, and a replay sends nothing")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """No key in the environment, and every socket connect refused and counted."""
    monkeypatch.delenv(KEY_VARIABLE, raising=False)
    attempts: list[object] = []

    def connect(_self: socket.socket, address: object) -> None:
        attempts.append(address)
        raise AssertionError(f"a replay-only run connected to {address}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    return attempts


def _case_files(directory: Path) -> None:
    snapshots = fixture_snapshots("frozen-elsewhere")
    for name, stratum in CASES.items():
        group, withhold = SEED[name]
        case = WriterCaseFile(
            name=name, group=group, stratum=stratum, withhold=list(withhold), snapshot=snapshots[name]
        )
        case.write(directory / f"{name.replace(' ', '-')}.json.gz")


async def _careful(case: Mapping[str, Any]) -> str:
    return "billing" if "refund" in case["text"] or "charged" in case["text"] else "bug"


async def _hasty(case: Mapping[str, Any]) -> str:
    return "bug" if "crash" in case["text"] or "broken" in case["text"] else "billing"


async def _quick_case_file(directory: Path) -> None:
    """A two-arm quick eval run in memory, its campaign frozen through its own host into a case file."""
    ran = await Eval(TICKETS, {"hasty": _hasty, "careful": _careful}, ["label"], k=1, name="triage").run(
        store=":memory:", scope="triage-source"
    )
    case = await writer_case_freeze(
        OpsHost.reading(ran.host),
        ran.scope_id,
        campaign_id=ran.campaign_id,
        name="triage",
        group="r1",
        stratum="support-triage",
        created_by="eval",
    )
    assert case.snapshot.example_result_ids, "the control: the frozen page names the stored result of an example"
    case.write(directory / "triage.json.gz")


def _silent_lines(turns: int) -> list[dict[str, Any]]:
    """An analyst session whose every reply holds no document, as a recording keeps it."""
    lines: list[dict[str, Any]] = []
    for index in range(turns):
        stamp = {"recorder_version": 1, "recorded_at": "2026-11-03T00:00:00+00:00", "session": "analyst"}
        lines.append({**stamp, "index": index, "kind": "ceiling", "ceiling": 0.05})
        said = reply(TurnText("I have no report yet."), model=WRITER).to_json()
        lines.append({**stamp, "index": index, "kind": "reply", "reply": said})
    return lines


def _recordings(directory: Path, cases: Path, arms: dict[str, Writer], *, extra: str | None = None) -> None:
    """Lay a silent recording for every held-out case of every arm; ``extra`` gets one line no replay asks."""
    held_out = {case.snapshot.id: case.name for case in read_writer_cases(cases) if case.name in HELD_OUT}
    assert sorted(held_out.values()) == sorted(HELD_OUT)
    for arm, writer in arms.items():
        routed = writer_turns(
            directory, arm=arm, skill=writer.analyst_skill, cases=held_out, revisions=writer.revisions
        )
        for snapshot, name in held_out.items():
            path = routed.path(WRITER, None, snapshot)
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = _silent_lines(SILENT_TURNS + (1 if arm == extra and name == "r1" else 0))
            path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def _eval(tmp_path: Path, arms: dict[str, Writer], **overrides: Any) -> WriterEval:
    arguments: dict[str, Any] = {
        "arms": arms,
        "judge": Model(JUDGE, api_key="unused", transport=httpx.MockTransport(_refuse)),
        "recordings": tmp_path / "turns",
        "budget_usd": 0.75,
        "max_cost_usd": 6.0,
        "held_out_fraction": 0.35,
        "name": SCOPE,
        "store": tmp_path / "evals.sqlite",
    }
    return WriterEval(tmp_path / "cases", **{**arguments, **overrides})


def _arms() -> dict[str, Writer]:
    return {"shipped": Writer(WRITER, skill=load_skill()), "variant": Writer(WRITER, skill=_variant())}


def _holds_the_cases_template_and_campaign(tmp_path: Path, result: WriterResult) -> None:
    """What the store holds after the run: the template, every case with its stratum and side, the campaign."""
    storage = EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite"))
    template = storage.load_template_by_name("report writer", SCOPE)
    assert template is not None and template.candidate_kind == WRITER_KIND
    bank = writer_case_bank(storage.query_test_cases(SCOPE, template_id=template.id))
    sides = {case.origin.group: case.origin.side for _, case in bank.cases if case.origin is not None}
    assert sides == {"r1": "held_out", "r2": "held_out", "r3": "dev"}
    assert sorted(stored.stratum or "" for stored, _ in bank.cases) == [
        "misfit",
        "reference",
        "reference",
        "reference",
        "support-triage",
    ]
    # The campaign holds both arms, the skill axis and the declared margin.
    campaign = storage.load_campaign(result.campaign_id, SCOPE)
    assert campaign is not None
    (aspect,) = campaign.aspects
    assert aspect.margin == 5.0
    assert aspect.design is not None
    assert [axis.axis_id for axis in aspect.design.axes] == ["report_writer.skill"]


async def test_a_two_arm_replay_mints_launches_files_reads_the_promotion_and_writes_the_page(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = _arms()
    _case_files(tmp_path / "cases")
    await _quick_case_file(tmp_path / "cases")
    _recordings(tmp_path / "turns", tmp_path / "cases", arms, extra="variant")

    result = await _eval(tmp_path, arms).run()

    assert isinstance(result, WriterResult)
    await asyncio.to_thread(_holds_the_cases_template_and_campaign, tmp_path, result)
    assert result.strata == {"misfit": 1, "reference": 2, "support-triage": 1}
    # Both arms ran the held-out cases alone, each under its share of the ceiling.
    for summary in result.arms.values():
        assert summary.status == "completed" and summary.n_cases == len(HELD_OUT)
        assert summary.max_cost_usd == pytest.approx(3.0)
    # The promotion read answered on the held-out cases.
    assert result.decisions == {"variant": "undecided"}
    assert result.exit_code() == 0 and result.why_failed() is None
    # The page is written.
    written = result.write(tmp_path / "report")
    assert written["html"].is_file()
    # Nothing was sent anywhere.
    assert offline == []

    text = result.render()
    assert text.startswith("writer: variant against shipped, 4 held-out case(s) x 1, judged by google/gemini-3.8-flash")
    assert "| reading | shipped | variant |" in text
    readers = {measure.reader_name for measure in WRITER_KIND_MEASURES}
    assert any(f"| {name} |" in text for name in readers)
    assert "promotion of variant: Undecided" in text
    # The replay that left a line unasked is printed, by arm, case and session.
    (drift,) = [line for line in text.splitlines() if "unasked" in line]
    assert drift.startswith("variant: case 'r1': session 'analyst' left 2 recorded lines unasked")
    assert "shipped: writer turns replayed from" in text
    # A case naming an example result this store does not hold is said, not left silent.
    assert "1 case(s) name example results stored only where they were frozen" in text


async def test_a_missing_recording_with_no_key_is_refused_before_anything_runs(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = _arms()
    _case_files(tmp_path / "cases")

    with pytest.raises(ValueError, match=f"no turn recording for 3 of 3 case.*{KEY_VARIABLE}"):
        await _eval(tmp_path, arms).run()
    assert offline == []


async def test_a_case_withholding_a_note_an_arm_lacks_is_refused_before_anything_is_minted(
    tmp_path: Path, offline: list[object]
) -> None:
    files = dict(load_skill().files)
    del files["exemplars/r1-model-selection.md"]
    files["exemplars/INDEX.md"] = "\n".join(
        line for line in files["exemplars/INDEX.md"].splitlines() if "`r1-model-selection.md`" not in line
    )
    arms = {
        "shipped": Writer(WRITER, skill=load_skill()),
        "without r1": Writer(WRITER, skill=AnalystSkill(files=files)),
    }
    _case_files(tmp_path / "cases")

    with pytest.raises(ValueError, match="case 'r1' on arm 'without r1'"):
        await _eval(tmp_path, arms).run()
    assert not (tmp_path / "evals.sqlite").exists()


def test_two_arms_alike_and_one_arm_are_refused() -> None:
    with pytest.raises(ValueError, match="one model, skill and effort"):
        WriterEval(
            "cases",
            arms={"a": Writer(WRITER), "b": Writer(WRITER)},
            judge=Model(JUDGE, api_key="unused"),
            recordings="turns",
            budget_usd=1.0,
            max_cost_usd=4.0,
        )
    with pytest.raises(ValueError, match="at least one candidate"):
        WriterEval(
            "cases",
            arms={"a": Writer(WRITER)},
            judge=Model(JUDGE, api_key="unused"),
            recordings="turns",
            budget_usd=1.0,
            max_cost_usd=4.0,
        )
