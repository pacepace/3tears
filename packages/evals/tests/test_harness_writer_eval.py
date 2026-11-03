"""``WriterEval``: the report writer measured over portable case files, replayed from turn recordings, offline.

Every case is a package fixture snapshot written as a writer case file; every arm's writer is a turn recording laid
where the arm's router reads it. Most recordings' analyst replies hold no document, so each cell ends with nothing
stored and nothing reaches the judge, a :class:`~threetears.evals.providers.Model` whose transport fails any request.
One recording stores Kairo's report, whose page a scripted judge scores, and whose two arms' pages the eval's own
pairwise pass shows that judge side by side. Every socket connect fails too: a replay-only run makes no network call,
whether or not its pages reach the judge.

Mutations that turn this file red: the pass not run, run in one order, or run without its part of the cap; the
promotion read made without the pass's job; the pairwise readings missing from the summary or the written page; a cap
that covers the reports but not the pass admitted; ``question=`` not filed; ``k`` above 1 admitted; a pass refused
before any call left unsaid or exiting 0.
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from packages.evals.tests.analyst_support import fixture_snapshots
from packages.evals.tests.report_writer_support import reply
from packages.evals.tests.writer_case_support import SEED
from packages.evals.tests.writer_eval_support import (
    JUDGE,
    JUDGED_SCORE,
    FakeScoringJudge,
    kairo_case,
    kairo_report,
    lay_recording,
)
from threetears.evals import Eval
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.kernel import EvalStorage
from threetears.evals.analysis import gate_verdicts
from threetears.evals.harness import Writer, WriterCaseFile, WriterEval, WriterResult, read_writer_cases
from threetears.evals.harness.analyse import Filed
from threetears.evals.ops import OpsHost, job_poll
from threetears.evals.ops.report_writer.loop import MOST_REVISIONS
from threetears.evals.ops.report_writer.preference import preference_rubric
from threetears.evals.ops.report_writer.writer_pairwise import WriterPairwiseReport
from threetears.evals.ops.report_writer.writer_bank import writer_case_bank
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND, WRITER_KIND_MEASURES, writer_scoring_rubric
from threetears.evals.providers import Model, ProviderError
from threetears.evals.quick.files import RunResult, Runnable, runnable
from threetears.evals.harness.writer_cases import writer_case_freeze
from threetears.evals.harness.writer_turns import writer_turns
from threetears.evals.schema.models import DECLARED_NAME_CHARS, EvalTemplate
from threetears.evals.schema.turns import TurnText
from threetears.evals.storage import SqliteDocumentStore

#: The writer every arm names.
WRITER: Final = "report-writer-candidate"

#: The cases, by fixture name: their stratum. At a 0.35 share r1 and r2 are held-out groups and r3 a dev one.
CASES: Final = {"r1": "reference", "r2": "reference", "r2 without its gates": "misfit", "r3": "reference"}

#: The cases on the held-out side: three fixtures, and a quick eval's campaign frozen in its own store, in r1's group.
HELD_OUT: Final = ("r1", "r2", "r2 without its gates", "triage")

#: The held-out side's fixtures alone, where no quick eval's campaign is frozen beside them.
HELD_OUT_FIXTURES: Final = tuple(name for name in HELD_OUT if name in CASES)

#: The cases on the dev side: r3's group.
DEV: Final = ("r3",)

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

#: The variable a recording run reads the default writer provider's key from.
KEY_VARIABLE: Final = "ANTHROPIC_API_KEY"


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


def _recordings(
    directory: Path,
    cases: Path,
    arms: dict[str, Writer],
    *,
    extra: str | None = None,
    names: tuple[str, ...] = HELD_OUT,
) -> None:
    """Lay a silent recording for every case of ``names`` (the held-out side's) of every arm; ``extra`` gets one line
    no replay asks."""
    held_out = {case.snapshot.id: case.name for case in read_writer_cases(cases) if case.name in names}
    assert sorted(held_out.values()) == sorted(names)
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
        "max_cost_usd": 8.0,
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

    writer_eval = _eval(tmp_path, arms)
    # The file door's protocols, held by the type checker and at run time: the eval is a runnable, its result a
    # result the door prints, writes and exits on, and a filed campaign the harness door's writer reads.
    door: Runnable = writer_eval
    assert runnable(door)
    result = await writer_eval.run()
    ran: RunResult = result
    assert isinstance(ran, Filed)

    assert isinstance(result, WriterResult)
    await asyncio.to_thread(_holds_the_cases_template_and_campaign, tmp_path, result)
    assert result.strata == {"misfit": 1, "reference": 2, "support-triage": 1}
    # Both arms ran the held-out cases alone, each under its part of what the pairwise reserve (a fifth) leaves.
    for summary in result.arms.values():
        assert summary.status == "completed" and summary.n_cases == len(HELD_OUT)
        assert summary.max_cost_usd == pytest.approx(8.0 * 0.8 / 2)
    # The pass ran, under the reserve: no page was stored, so every pair is a tie and no call was made.
    read = result.pairwise["variant"]
    assert read.calls_made == 0 and read.share == 0.5
    assert read.cap_usd == pytest.approx(8.0 * 0.2)
    assert result.pairwise_unread == {}
    # The promotion read answered on the held-out cases, the pass as its primary reading.
    assert result.decisions == {"variant": "undecided"}
    assert read.read_on == read.job_id
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
    # The whole result, as the file door prints it with --verbose: the quiet summary, then each arm's and the page.
    assert result.render(full=True).startswith(f"{text}\n\n")


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
    with pytest.raises(ValueError, match=re.escape("alike in every lever (model, skill, effort, revisions, editor's")):
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


async def test_reports_whose_budgets_could_pass_an_arms_share_are_refused_before_any_arm_launches(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = _arms()
    _case_files(tmp_path / "cases")
    _recordings(tmp_path / "turns", tmp_path / "cases", arms, names=HELD_OUT_FIXTURES)
    budget_usd, max_cost_usd = 1.25, 8.0
    share = max_cost_usd * 0.8 / len(arms)
    reports = len(HELD_OUT_FIXTURES) * budget_usd
    # The control: the cap would cover the reports alone, split with no reserve; with the pass's part taken, it does not.
    assert reports <= max_cost_usd / len(arms) and reports > share

    expected = (
        f"{len(HELD_OUT_FIXTURES)} report(s) at ${budget_usd:.2f} each could pass an arm's ${share:.2f} part of "
        f"max_cost_usd ${max_cost_usd:.2f}, after the pairwise pass's $1.60 (pairwise_share 0.2); raise max_cost_usd "
        "or lower budget_usd"
    )
    with pytest.raises(ValueError) as refused:
        await _eval(tmp_path, arms, budget_usd=budget_usd, max_cost_usd=max_cost_usd).run()
    assert str(refused.value) == expected
    storage = EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite"))
    assert await asyncio.to_thread(storage.query_eval_runs, SCOPE) == [], "no arm was launched"
    assert offline == []


async def test_a_held_out_set_listing_a_case_no_file_holds_is_refused(tmp_path: Path, offline: list[object]) -> None:
    arms = _arms()
    _case_files(tmp_path / "cases")
    # A first run mints every case into its side's set; no recording is laid, so it stops there.
    with pytest.raises(ValueError, match="no turn recording"):
        await _eval(tmp_path, arms).run()
    gone = await asyncio.to_thread(_the_misfit_case, tmp_path)
    (tmp_path / "cases" / "r2-without-its-gates.json.gz").unlink()

    with pytest.raises(ValueError) as refused:
        await _eval(tmp_path, arms).run()
    assert str(refused.value) == (
        f"the held-out set of scope {SCOPE!r} lists 1 case(s) no file in {tmp_path / 'cases'} holds ({gone}); retire "
        "them (writer_case_archive), or run in a fresh scope (scope=)"
    )
    assert offline == []


def _the_misfit_case(tmp_path: Path) -> str:
    """The stored id of the one case minted with the misfit stratum."""
    storage = EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite"))
    template = storage.load_template_by_name("report writer", SCOPE)
    assert template is not None
    bank = writer_case_bank(storage.query_test_cases(SCOPE, template_id=template.id))
    (misfit,) = [stored.id for stored, _ in bank.cases if stored.stratum == "misfit"]
    return misfit


async def test_a_scope_holding_a_report_writer_template_of_another_kind_is_refused(
    tmp_path: Path, offline: list[object]
) -> None:
    _case_files(tmp_path / "cases")
    storage = EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite"))
    other = EvalTemplate(
        id="not-a-writer", scope_id=SCOPE, name="report writer", intent="Score a callable.", candidate_kind="callable"
    )
    await asyncio.to_thread(storage.save_template, other)

    with pytest.raises(ValueError) as refused:
        await _eval(tmp_path, _arms()).run()
    assert str(refused.value) == (
        f"scope {SCOPE!r} holds a template named 'report writer' of kind 'callable'; run the writer eval in another "
        "scope (scope=)"
    )
    assert offline == []


async def test_a_dev_side_run_launches_the_dev_cases_and_reads_no_promotion(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = _arms()
    _case_files(tmp_path / "cases")
    _recordings(tmp_path / "turns", tmp_path / "cases", arms, names=DEV)

    result = await _eval(tmp_path, arms, side="dev").run()

    assert result.side == "dev"
    for summary in result.arms.values():
        assert summary.status == "completed" and summary.n_cases == len(DEV)
    assert result.decisions == {} and result.gates == {}
    # The pass is still run and printed on the dev side, with no promotion read on it.
    assert list(result.pairwise) == ["variant"] and result.pairwise["variant"].read_on is None
    assert result.exit_code() == 0 and result.why_failed() is None
    text = result.render()
    assert text.startswith(f"writer: variant against shipped, {len(DEV)} dev case(s) x 1")
    assert "promotion: read on the held-out side only" in text
    assert "promotion of" not in text
    assert offline == []


async def test_a_failed_promotion_gate_exits_4_naming_it_and_an_unfinished_arm_exits_1(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = _arms()
    _case_files(tmp_path / "cases")
    _recordings(tmp_path / "turns", tmp_path / "cases", arms, names=HELD_OUT_FIXTURES)
    result = await _eval(tmp_path, arms).run()
    assert result.exit_code() == 0 and result.why_failed() is None, "the control: the replay read no regression"

    # The same read, its first verdict shown worse than the control: the gate the promotion read gates with fails.
    read = result.gates["variant"].gated[0]
    regressed = read.model_copy(update={"kind": "contrast", "outcome": "regressed"})
    gate = gate_verdicts([regressed])
    assert gate.failed and gate.why() is not None
    failing = replace(result, gates={"variant": gate})
    assert failing.exit_code() == 4
    assert failing.why_failed() == f"variant's promotion gate {gate.why()}"

    # An arm whose run did not complete exits 1, whatever its gates read.
    unfinished = {**result.arms, "variant": result.arms["variant"].model_copy(update={"status": "failed"})}
    assert replace(failing, arms=unfinished).exit_code() == 1
    assert offline == []


@pytest.mark.parametrize(
    ("build", "refusal"),
    [
        (lambda: Writer(" "), "names its model by id, and this one is blank"),
        (lambda: Writer(WRITER, effort="loud"), "effort 'loud' is none of none, minimal, low"),
        (lambda: Writer(WRITER, revisions=MOST_REVISIONS + 1), f"revisions are 0 to {MOST_REVISIONS}, not 3"),
        (lambda: Writer(WRITER, revisions=-1), f"revisions are 0 to {MOST_REVISIONS}, not -1"),
        (lambda: Writer("vendor/model"), "model 'vendor/model' names its recordings' directory, so it holds no '/'"),
        (
            lambda: Writer(WRITER, editor_model="vendor/editor"),
            "editor's model 'vendor/editor' names its recordings' directory",
        ),
        (lambda: _declared(held_out_fraction=0.0), "strictly between 0 and 1, not 0.0"),
        (lambda: _declared(held_out_fraction=1.0), "strictly between 0 and 1, not 1.0"),
        (lambda: _declared(side="both"), "side is 'held_out' or 'dev', not 'both'"),
        (lambda: _declared(name="w" * (DECLARED_NAME_CHARS + 1)), f"at most {DECLARED_NAME_CHARS} characters"),
        (lambda: _declared(budget_usd=0.0), "budget_usd is above 0, not 0.0"),
        (lambda: _declared(max_cost_usd=-1.0), "max_cost_usd is above 0, not -1.0"),
        (lambda: _declared(turns=0), "turns is above 0, not 0"),
        (lambda: _declared(request_s=0.0), "request_s is above 0, not 0.0"),
        (
            lambda: _declared(k=2),
            "k is 1 for a writer eval, not 2: each arm replays one recording per case, so a second repeat would "
            "replay the first's turns and repeat its report rather than measure another; run more cases instead",
        ),
        (lambda: _declared(k=0), "k is 1 for a writer eval, not 0"),
        (lambda: _declared(question=" "), "a writer eval's question is filed on its campaign, and this one is blank"),
        (lambda: _declared(pairwise_share=0.0), "pairwise_share is a share strictly between 0 and 1, not 0.0"),
        (lambda: _declared(pairwise_share=1.0), "pairwise_share is a share strictly between 0 and 1, not 1.0"),
    ],
)
def test_a_writer_or_writer_eval_declared_out_of_range_is_refused(build: Callable[[], object], refusal: str) -> None:
    with pytest.raises(ValueError, match=re.escape(refusal)):
        build()


def _declared(**overrides: Any) -> WriterEval:
    """A writer eval declared with two distinct arms and ``overrides``; nothing is read until it runs."""
    arguments: dict[str, Any] = {
        "arms": _arms(),
        "judge": Model(JUDGE, api_key="unused"),
        "recordings": "turns",
        "budget_usd": 1.0,
        "max_cost_usd": 4.0,
    }
    return WriterEval("cases", **{**arguments, **overrides})


def _kairo_arms(tmp_path: Path) -> dict[str, Writer]:
    """Two arms over Kairo's case, each laid Kairo's report, so both pages are stored and judged."""
    arms = _arms()
    kairo = kairo_case(tmp_path / "cases")
    for arm, writer in arms.items():
        routed = writer_turns(
            tmp_path / "turns", arm=arm, skill=writer.analyst_skill, cases={kairo.snapshot.id: "kairo"}
        )
        lay_recording(routed.path(WRITER, None, kairo.snapshot.id), kairo_report(WRITER, WRITER))
    return arms


#: The question a test files its campaign asking.
QUESTION: Final = "Does a blank line in SKILL.md change the reports?"


async def test_a_replay_whose_pages_reach_the_judge_runs_the_pass_in_both_orders_and_reads_the_promotion_on_it(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = _kairo_arms(tmp_path)
    judge = FakeScoringJudge()

    result = await _eval(tmp_path, arms, judge=judge, question=QUESTION).run()

    # Every arm's page was stored and judged on all seven dimensions, then the pair once in each order, and nothing
    # was sent anywhere.
    dims = [dim.name for dim in writer_scoring_rubric()]
    preferred = preference_rubric().name
    assert sorted(judge.calls) == sorted([*(dims * len(arms)), preferred, preferred])
    for summary in result.arms.values():
        assert summary.status == "completed" and summary.n_cases == 1
        assert summary.errors == []
        assert {dimension.name for dimension in summary.judged} == set(dims)
    judged = f"(judged) | {JUDGED_SCORE:.2f} | {JUDGED_SCORE:.2f} |"
    assert any(line.endswith(judged) for line in result.render().splitlines())
    assert offline == []

    # The pass: the candidate's page read second, then first, each order scored, under the pass's part of the cap.
    read = result.pairwise["variant"]
    status = await job_poll(OpsHost.reading(result.host), read.job_id, SCOPE)
    report = WriterPairwiseReport.model_validate(status.result)
    assert (report.candidate, report.control, report.campaign_id) == ("variant", "shipped", result.campaign_id)
    (pair,) = report.pairs
    assert pair.missing is None
    assert [(answer.candidate, answer.answer) for answer in pair.answers] == [("B", "scored"), ("A", "scored")]
    assert read.calls_made == 2 and read.cap_usd == pytest.approx(8.0 * 0.2)
    assert read.primary == report.whole.verdict and read.share == report.whole.candidate_share
    # The judge scored 4 both times, so it preferred Report B in both orders: the orders disagree on the candidate.
    assert (read.both_orders, read.order_consistent, read.order_consistency) == (1, 0, 0.0)
    # The promotion read was given the pass's completed job, and voided its primary on that disagreement.
    assert read.read_on == read.job_id and read.job_id.startswith("action:")
    assert read.void is not None and result.decisions == {"variant": "undecided"}

    # The summary and the written page carry the pass's readings as one table, a column per candidate.
    text = result.render()
    table = result.pairwise_text()
    assert table in text
    rows = {line.split(" | ")[0].removeprefix("| "): line for line in table.splitlines() if line.startswith("| ")}
    assert rows["preference"] == "| preference | variant |"
    # One case is too few for the test to run, and the orders' disagreement voids it besides.
    assert read.primary.outcome == "untested"
    assert rows["reads"] == "| reads | untested (void) |"
    assert rows["share"] == "| share | 0.5 |"
    assert rows["order consistency"] == "| order consistency | 0 (0 of 1) |"
    # Report B was preferred in both orders: a lean away from the page read first.
    assert rows["first-slot lean"] == "| first-slot lean | -0.25 |"
    assert rows["can't tell"] == "| can't tell | 0 |"
    assert rows["held-out cases to settle"] == "| held-out cases to settle | - |"
    assert rows["spent"] == "| spent | $0 of $1.6 |"
    for label in ("difference", "reads, both pages finished", "share, both pages finished"):
        assert label in rows
    assert sum(1 for line in table.splitlines() if line and not line.startswith("|")) == 1, "one sentence over it"
    written = result.write(tmp_path / "report")
    assert table in written["pairwise"].read_text(encoding="utf-8")

    # The campaign is filed asking the eval's question.
    campaign = await asyncio.to_thread(
        EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite")).load_campaign, result.campaign_id, SCOPE
    )
    assert campaign is not None and campaign.question == QUESTION
    assert result.exit_code() == 0


# parity-with: threetears.evals.harness.writer_host.WriterJudge
class FakeDearPairJudge(FakeScoringJudge):
    """The scripted judge, pricing each side-by-side call above any part of the cap the pass could be given."""

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """$100 for a preference call, nothing for an absolute one."""
        return 100.0 if f'"{preference_rubric().name}"' in system else 0.0


# parity-with: threetears.evals.harness.writer_host.WriterJudge
class FakeBrokeJudge(FakeScoringJudge):
    """The scripted judge, whose account is refused on a side-by-side call: out of credit."""

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        """The scripted reply, or the provider's 402 for a preference call."""
        if f'"{preference_rubric().name}"' in system:
            raise ProviderError("openrouter", 402, "insufficient credits")
        return await super().generate(system=system, user=user, response_format=response_format)


async def test_a_pass_stopped_by_an_account_refusal_is_read_said_and_exits_1(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = _kairo_arms(tmp_path)

    result = await _eval(tmp_path, arms, judge=FakeBrokeJudge()).run()

    report = result.pairwise["variant"]
    assert report.stopped is not None and report.primary.outcome == "untested"
    assert result.pairwise_unread == {"variant": f"stopped before its last pair: {report.stopped}"}
    assert report.read_on == report.job_id
    assert result.exit_code() == 1
    text = result.render()
    assert "| reads | untested |" in text
    assert f"pairwise pass of variant: stopped before its last pair: {report.stopped}" in text
    assert offline == []


async def test_three_arms_split_the_pass_s_reserve_between_the_two_candidates(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = {**_arms(), "fewer": Writer(WRITER, skill=load_skill(), revisions=0)}
    _case_files(tmp_path / "cases")
    _recordings(tmp_path / "turns", tmp_path / "cases", arms, names=HELD_OUT_FIXTURES)

    result = await _eval(tmp_path, arms, budget_usd=0.5).run()

    for summary in result.arms.values():
        assert summary.status == "completed" and summary.max_cost_usd == pytest.approx(8.0 * 0.8 / 3)
    assert list(result.pairwise) == ["variant", "fewer"]
    for read in result.pairwise.values():
        assert read.primary.control == "shipped" and read.cap_usd == pytest.approx(8.0 * 0.2 / 2)
        assert read.read_on == read.job_id
    assert list(result.decisions) == ["variant", "fewer"]
    assert "| preference | variant | fewer |" in result.render()
    assert offline == []


async def test_a_pass_its_part_cannot_cover_is_refused_with_nothing_spent_said_and_exits_1(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = _kairo_arms(tmp_path)
    judge = FakeDearPairJudge()

    result = await _eval(tmp_path, arms, judge=judge).run()

    assert all(summary.status == "completed" for summary in result.arms.values())
    assert preference_rubric().name not in judge.calls, "no preference call was made"
    assert result.pairwise == {}
    (why,) = result.pairwise_unread.values()
    assert why.startswith("refused before any call: ") and "above the out-of-run cap $1.60" in why
    # The read had no pass, so it never promotes; the run exits 1, the pass not completed.
    assert result.decisions == {"variant": "undecided"}
    assert result.promotion_why["variant"].startswith("Undecided: no pairwise pass was read")
    assert result.exit_code() == 1
    assert f"pairwise pass of variant: {why}" in result.render()
    assert "| reads | - |" in result.render()
    # The campaign is filed asking the generic question.
    campaign = await asyncio.to_thread(
        EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite")).load_campaign, result.campaign_id, SCOPE
    )
    assert campaign is not None and campaign.question == "Does variant write reports better than shipped?"
    assert offline == []


async def test_two_arms_differing_only_in_revisions_each_replay_their_own_recordings(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = {"shipped": Writer(WRITER, skill=load_skill()), "fewer": Writer(WRITER, skill=load_skill(), revisions=0)}
    _case_files(tmp_path / "cases")
    _recordings(tmp_path / "turns", tmp_path / "cases", arms, names=HELD_OUT_FIXTURES)

    result = await _eval(tmp_path, arms).run()

    for summary in result.arms.values():
        assert summary.status == "completed" and summary.n_cases == len(HELD_OUT_FIXTURES)
    campaign = await asyncio.to_thread(
        EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite")).load_campaign, result.campaign_id, SCOPE
    )
    assert campaign is not None
    (aspect,) = campaign.aspects
    assert aspect.design is not None and [axis.axis_id for axis in aspect.design.axes] == ["report_writer.revisions"]
    # Each arm replayed its own directory, by the revisions segment its recordings are keyed by, to the end.
    text = result.render()
    skill12 = load_skill().digest[:12]
    for arm, writer in arms.items():
        assert (
            f"{arm}: writer turns replayed from {tmp_path / 'turns'}/{arm}-{skill12}-{WRITER}-default-r{writer.revisions}"
            in text
        )
    assert "unasked" not in text
    assert offline == []


def _renamed_note() -> AnalystSkill:
    """The shipped skill with r1's note renamed, file and index row: a frozen copy holding a note the shipped lacks."""
    files = dict(load_skill().files)
    files["exemplars/r1-renamed.md"] = files.pop("exemplars/r1-model-selection.md")
    files["exemplars/INDEX.md"] = files["exemplars/INDEX.md"].replace("`r1-model-selection.md`", "`r1-renamed.md`")
    return AnalystSkill(files=files)


async def test_cases_are_minted_against_the_control_arm_s_skill_not_the_shipped_one(
    tmp_path: Path, offline: list[object]
) -> None:
    control = _renamed_note()
    variant = AnalystSkill(files={**control.files, "SKILL.md": f"{control.files['SKILL.md']}\n"})
    arms = {"frozen": Writer(WRITER, skill=control), "variant": Writer(WRITER, skill=variant)}
    snapshots = fixture_snapshots("frozen-elsewhere")
    for name in ("r1", "r2"):
        group, withhold = SEED[name]
        notes = ["r1-renamed.md"] if name == "r1" else list(withhold)
        case = WriterCaseFile(name=name, group=group, stratum="reference", withhold=notes, snapshot=snapshots[name])
        case.write(tmp_path / "cases" / f"{name}.json.gz")
    with pytest.raises(ValueError, match="no exemplar is named r1-renamed.md"):
        load_skill().analyst(withhold=["r1-renamed.md"])  # the control: the shipped skill lacks the note
    _recordings(tmp_path / "turns", tmp_path / "cases", arms, names=("r1", "r2"))

    result = await _eval(tmp_path, arms).run()

    for summary in result.arms.values():
        assert summary.status == "completed" and summary.n_cases == 2
    origins = await asyncio.to_thread(_origins, tmp_path)
    assert origins == {"r1": control.digest, "r2": control.digest}
    assert offline == []


def _origins(tmp_path: Path) -> dict[str, str]:
    """Each stored case's group: the skill its withheld notes were checked against."""
    storage = EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite"))
    template = storage.load_template_by_name("report writer", SCOPE)
    assert template is not None
    bank = writer_case_bank(storage.query_test_cases(SCOPE, template_id=template.id))
    return {case.origin.group: case.origin.skill for _, case in bank.cases if case.origin is not None}
