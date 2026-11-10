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
before any call left unsaid or exiting 0; a candidate counted as running the control's editor while its
``editor.md``, its editor's model or its effort differs, or the action claiming one editor over arms whose skills differ.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import socket
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, get_args

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
from threetears.evals.actions import Caller, eval_catalogue, read_only_tools
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
from threetears.evals.ops.report_writer.writer_contract import (
    WRITER_KIND,
    WRITER_KIND_MEASURES,
    EffortWord,
    uncompared_readings,
    writer_scoring_rubric,
)
from threetears.evals.kernel.capped_completion import PROMPT_FRAMING_TOKENS
from threetears.evals.kernel.report_snapshot import report_snapshot_id
from threetears.evals.kernel.numbers import spell_dollars
from threetears.evals.providers import Catalogue, Model, ProviderError
from threetears.evals.providers.model import DEFAULT_MAX_TOKENS
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS
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
            directory,
            arm=arm,
            skill=writer.analyst_skill,
            cases=held_out,
            revisions=writer.revisions,
            editor_model=writer.editor_model,
        )
        effort = next((word for word in get_args(EffortWord) if word == writer.effort), None)
        for snapshot, name in held_out.items():
            # An arm whose editor runs on a model of its own is routed an editor's file too, never asked: no analyst
            # reply holds a document for it to review.
            for path in routed.paths(WRITER, effort, snapshot):
                path.parent.mkdir(parents=True, exist_ok=True)
                lines = _silent_lines(SILENT_TURNS + (1 if arm == extra and name == "r1" else 0))
                path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def _rates(rows: list[dict[str, Any]]) -> Catalogue:
    """A catalogue read from ``rows`` over a scripted fetch, never the network."""
    payload = {"data": rows}
    return Catalogue(http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))))


def _eval(tmp_path: Path, arms: dict[str, Writer], **overrides: Any) -> WriterEval:
    arguments: dict[str, Any] = {
        "arms": arms,
        # A live judge the run never asks, whose catalogue prices nothing: it cannot price a call before launch.
        "judge": Model(JUDGE, api_key="unused", transport=httpx.MockTransport(_refuse), catalogue=_rates([])),
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
    # The campaign holds both arms and the skill axis; it asks a superiority question, so it files no margin.
    campaign = storage.load_campaign(result.campaign_id, SCOPE)
    assert campaign is not None
    (aspect,) = campaign.aspects
    assert aspect.margin is None, "a writer eval files no margin unless its author writes one"
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

    # The smallest cap whose arm's part holds the reports: they are the first batch, every case launching at once.
    smallest = spell_dollars(math.ceil(reports * len(arms) / 0.8 * 100 - 1e-9) / 100)
    expected = (
        f"the {len(HELD_OUT_FIXTURES)} report(s) at {spell_dollars(budget_usd)} each could pass an arm's {spell_dollars(share)} part of "
        f"max_cost_usd {spell_dollars(max_cost_usd)}, after the pairwise pass's $1.60 (pairwise_share 0.2); a max_cost_usd of "
        f"{smallest} or more admits it, or lower budget_usd"
    )
    with pytest.raises(ValueError) as refused:
        await _eval(tmp_path, arms, budget_usd=budget_usd, max_cost_usd=max_cost_usd).run()
    assert str(refused.value) == expected
    storage = EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite"))
    assert await asyncio.to_thread(storage.query_eval_runs, SCOPE) == [], "no arm was launched"
    assert offline == []


#: The judge's rates, per token, as OpenRouter's catalogue lists the rung-5 judge's: prompt, then output.
PROMPT_RATE: Final = 0.825e-6
OUTPUT_RATE: Final = 4.125e-6

#: The judge's output cap, as rung 5 builds it.
JUDGE_CAP: Final = 2048

#: The judge's catalogue row.
JUDGE_ROW: Final = {
    "id": JUDGE,
    "pricing": {"prompt": f"{PROMPT_RATE:.10f}", "completion": f"{OUTPUT_RATE:.10f}"},
    "supported_parameters": ["max_tokens", "temperature", "reasoning", "response_format", "structured_outputs"],
}

#: One judge call with no page in it: the output cap and the prompt every call is allowed, at the judge's rates.
FLOOR: Final = JUDGE_CAP * OUTPUT_RATE + PROMPT_FRAMING_TOKENS * PROMPT_RATE


def _priced_judge() -> Model:
    """A live judge priced from its catalogue row at rung 5's output cap, which the run never asks."""
    return Model(
        JUDGE,
        api_key="unused",
        transport=httpx.MockTransport(_refuse),
        catalogue=_rates([JUDGE_ROW]),
        max_tokens=JUDGE_CAP,
    )


async def test_the_judge_s_floor_is_its_own_output_cap_and_the_prompt_allowance_at_its_rates() -> None:
    judge = _priced_judge()
    await judge.refresh_prices()
    assert judge.price_ceiling(system="", user="") == pytest.approx(FLOOR)
    # The cap the judge is built with prices each call; built with none, it is priced at the client's default cap.
    uncapped = Model(JUDGE, api_key="unused", transport=httpx.MockTransport(_refuse), catalogue=_rates([JUDGE_ROW]))
    await uncapped.refresh_prices()
    assert uncapped.price_ceiling(system="", user="") == pytest.approx(
        DEFAULT_MAX_TOKENS * OUTPUT_RATE + PROMPT_FRAMING_TOKENS * PROMPT_RATE
    )


@pytest.mark.parametrize("what", ["pass", "arm"])
async def test_a_judge_whose_calls_at_their_floor_could_pass_a_part_is_refused_before_any_arm_launches(
    tmp_path: Path, offline: list[object], what: str
) -> None:
    arms = _arms()
    _case_files(tmp_path / "cases")
    _recordings(tmp_path / "turns", tmp_path / "cases", arms, names=HELD_OUT_FIXTURES)
    cases, dims, max_cost_usd = len(HELD_OUT_FIXTURES), len(writer_scoring_rubric()), 8.0
    each = f"${FLOOR:.4f} each before any page is read ({JUDGE}'s output cap and {PROMPT_FRAMING_TOKENS} prompt tokens)"
    pass_calls = cases * 2 * JUDGE_CALL_ATTEMPTS
    judge_calls = cases * dims * JUDGE_CALL_ATTEMPTS

    def admits(share: float, budget_usd: float) -> str:
        """The smallest cap whose parts hold every case's reports, judging and pairs at the floor."""
        arm = cases * budget_usd + judge_calls * FLOOR
        smallest = max(arm * len(arms) / (1 - share), pass_calls * FLOOR / share)
        return f"a max_cost_usd of {spell_dollars(math.ceil(smallest * 100 - 1e-9) / 100)} or more admits it"

    if what == "pass":
        # A hundredth of $8 for the pass: under its calls at their floor, though the arms' parts hold theirs.
        share, budget_usd = 0.01, 0.75
        pass_part, arm_part = max_cost_usd * share, max_cost_usd * (1 - share) / len(arms)
        assert pass_calls * FLOOR > pass_part and cases * budget_usd + judge_calls * FLOOR <= arm_part
        expected = (
            f"a pairwise pass's {pass_calls} judge call(s) ({cases} pair(s) in both orders, {JUDGE_CALL_ATTEMPTS} "
            f"attempts each) are priced at {each}, {spell_dollars(pass_calls * FLOOR)} in all, above the pass's "
            f"{spell_dollars(pass_part)} part of max_cost_usd {spell_dollars(max_cost_usd)} (pairwise_share {share}); "
            f"{admits(share, budget_usd)}, or raise pairwise_share or lower the judge's max_tokens"
        )
    else:
        # The reports alone fit the arm's part, and do not once its judging is priced beside them.
        share, budget_usd = 0.2, 0.95
        pass_part, arm_part = max_cost_usd * share, max_cost_usd * (1 - share) / len(arms)
        assert pass_calls * FLOOR <= pass_part and cases * budget_usd <= arm_part
        total = cases * budget_usd + judge_calls * FLOOR
        assert total > arm_part
        expected = (
            f"an arm's {cases} report(s) at {spell_dollars(budget_usd)} and its {judge_calls} judge call(s) ({cases} case(s) x "
            f"{dims} dimension(s) x {JUDGE_CALL_ATTEMPTS} attempts), priced at {each}, come to {spell_dollars(total)}, above "
            f"the arm's {spell_dollars(arm_part)} part of max_cost_usd {spell_dollars(max_cost_usd)}; {admits(share, budget_usd)}, or "
            "lower budget_usd or the judge's max_tokens"
        )
    with pytest.raises(ValueError) as refused:
        await _eval(
            tmp_path,
            arms,
            judge=_priced_judge(),
            budget_usd=budget_usd,
            max_cost_usd=max_cost_usd,
            pairwise_share=share,
        ).run()
    assert str(refused.value) == expected
    if what == "pass":
        # Every amount the refusal weighs is money's one form, the smallest cap it names included: the pass's part of
        # $8 at a hundredth reads to two figures, never cut to the cent.
        assert "the pass's $0.080 part of max_cost_usd $8.00" in str(refused.value)
    storage = EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite"))
    assert await asyncio.to_thread(storage.query_eval_runs, SCOPE) == [], "no arm was launched"
    assert offline == []


async def test_a_judge_whose_calls_at_their_floor_fit_every_part_is_admitted(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = _arms()
    _case_files(tmp_path / "cases")
    _recordings(tmp_path / "turns", tmp_path / "cases", arms, names=HELD_OUT_FIXTURES)
    cases, dims = len(HELD_OUT_FIXTURES), len(writer_scoring_rubric())
    assert cases * 0.75 + cases * dims * JUDGE_CALL_ATTEMPTS * FLOOR <= 8.0 * 0.8 / 2
    assert cases * 2 * JUDGE_CALL_ATTEMPTS * FLOOR <= 8.0 * 0.2

    result = await _eval(tmp_path, arms, judge=_priced_judge()).run()

    assert all(summary.status == "completed" for summary in result.arms.values())
    assert result.pairwise_unread == {}
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
    # A judged score over one case reads a tenth: a hundredth is noise below about 30 cases.
    judged = f"(judged) | {JUDGED_SCORE:.1f} | {JUDGED_SCORE:.1f} |"
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

    # The summary and the written page carry the pass's readings as two tables of rows with columns: the preference,
    # a row per candidate and the pairs it reads, and the judge's checks, a row per candidate.
    text = result.render()
    table = result.pairwise_text()
    assert table in text
    lines = [line for line in table.splitlines() if line.startswith("| ")]
    assert lines[0] == ("| Candidate | Pairs | Reads | Candidate's share (%) | Control's share (%) | Interval (%) |")
    # One case is too few for the test to run, and the orders' disagreement voids it besides.
    assert read.primary.outcome == "untested"
    every, finished, checks = [line for line in lines if line.startswith("| variant | ")]
    assert every.startswith(f"| variant | every pair | untested: {read.primary.reason_detail} (void) | 50 | 50 | ")
    assert read.primary.reason_detail == "1 case with a share, and the test needs two"
    assert finished.startswith("| variant | both pages finished | ")
    assert (
        "| Candidate | Order consistency | First-slot lean (pts) | Can't tell | Held-out cases to settle | Spent |"
        in lines
    )
    # Report B was preferred in both orders: a lean away from the page read first.
    assert checks == "| variant | 0% (0 of 1) | -25 | 0 | — | $0 of $1.60 |"
    assert sum(1 for line in table.splitlines() if line and not line.startswith("|")) == 1, "one sentence over it"
    written = result.write(tmp_path / "report")
    assert table in written["pairwise"].read_text(encoding="utf-8")

    # The campaign is filed asking the eval's question.
    campaign = await asyncio.to_thread(
        EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite")).load_campaign, result.campaign_id, SCOPE
    )
    assert campaign is not None and campaign.question == QUESTION
    assert result.exit_code() == 0


def _scored(request: httpx.Request) -> httpx.Response:
    """OpenRouter's reply to a judge call, as the scripted judge scores it: the dimension its system prompt names, 4."""
    body = json.loads(request.content)
    system = next(message["content"] for message in body["messages"] if message["role"] == "system")
    match = re.search(r'single key "([^"]+)"', system)
    assert match is not None, "the judge's system prompt names no dimension"
    content = json.dumps({"reasoning": f"read {match[1]}", "criteria_scores": {match[1]: JUDGED_SCORE}})
    choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
    usage = {"prompt_tokens": 200, "completion_tokens": 30}
    return httpx.Response(200, json={"model": JUDGE, "choices": [choice], "usage": usage})


async def test_a_judge_recorded_over_stored_pages_replays_in_a_fresh_store_with_no_key(
    tmp_path: Path, offline: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    arms = _kairo_arms(tmp_path)
    recording = tmp_path / "recordings" / "judge.jsonl"
    monkeypatch.delenv("RECORD", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    recorder = Model(
        JUDGE,
        api_key="scripted",
        transport=httpx.MockTransport(_scored),
        catalogue=_rates([JUDGE_ROW]),
        max_tokens=JUDGE_CAP,
        recording=recording,
    )

    recorded = await _eval(tmp_path, arms, judge=recorder).run()
    await recorder.aclose()

    # Both pages were judged on every dimension and side by side in both orders, and every call was recorded.
    asked = len(arms) * len(writer_scoring_rubric()) + 2
    assert len(recording.read_text(encoding="utf-8").splitlines()) == asked
    assert recorded.pairwise["variant"].calls_made == 2 and recorded.pairwise_unread == {}

    # A fresh store, no key: the judge replays what it recorded, and nothing it is asked is missing from it.
    replayer = Model(JUDGE, max_tokens=JUDGE_CAP, recording=recording)
    replayed = await _eval(tmp_path, arms, judge=replayer, store=tmp_path / "fresh.sqlite").run()

    assert len(recording.read_text(encoding="utf-8").splitlines()) == asked, "the replay recorded nothing"
    for summary in replayed.arms.values():
        assert summary.status == "completed" and summary.errors == []
        assert {dimension.name: dimension.mean for dimension in summary.judged} == {
            dim.name: JUDGED_SCORE for dim in writer_scoring_rubric()
        }
    read = replayed.pairwise["variant"]
    status = await job_poll(OpsHost.reading(replayed.host), read.job_id, SCOPE)
    (pair,) = WriterPairwiseReport.model_validate(status.result).pairs
    assert [(answer.candidate, answer.answer) for answer in pair.answers] == [("B", "scored"), ("A", "scored")]
    assert read.calls_made == 2 and read.stopped is None and replayed.pairwise_unread == {}
    # The promotion read took the replayed pass as its primary reading.
    assert read.read_on == read.job_id and replayed.decisions == {"variant": "undecided"}
    assert replayed.exit_code() == 0
    assert offline == []


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
    assert any(line.startswith(f"| variant | every pair | untested: {report.stopped} | ") for line in text.splitlines())
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
    rows = [line for line in result.render().splitlines() if line.startswith(("| variant | ", "| fewer | "))]
    assert [row.split(" | ")[0] for row in rows] == [
        "| variant",
        "| variant",
        "| fewer",
        "| fewer",
        "| variant",
        "| fewer",
    ]
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
    # The unread pass's row in each table: the preference table's six columns and the checks table's six, all dashes.
    tables = result.pairwise_text().split("\n\n")
    preference = next(table for table in tables if table.startswith("| Candidate | Pairs |"))
    checks_table = next(table for table in tables if table.startswith("| Candidate | Order consistency |"))
    assert "| variant | — | — | — | — | — |" in preference.splitlines()
    assert "| variant | — | — | — | — | — |" in checks_table.splitlines()
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


def _editor_variant() -> AnalystSkill:
    """The shipped skill with one sentence added to ``editor.md``: another editor (its text is read stripped, so a
    blank line would not differ)."""
    files = dict(load_skill().files)
    files["editor.md"] = f"{files['editor.md'].rstrip()}\n\nReview each report once more before you answer.\n"
    return AnalystSkill(files=files)


@pytest.mark.parametrize(
    ("variant", "one_editor"),
    [
        pytest.param(lambda: Writer(WRITER, skill=_variant()), True, id="only SKILL.md differs"),
        pytest.param(lambda: Writer(WRITER, skill=_editor_variant()), False, id="editor.md differs"),
        pytest.param(
            lambda: Writer(WRITER, skill=load_skill(), editor_model="report-writer-editor"),
            False,
            id="the editor's model differs",
        ),
        pytest.param(lambda: Writer(WRITER, skill=load_skill(), effort="low"), False, id="the effort differs"),
    ],
)
async def test_a_candidate_runs_the_control_s_editor_only_when_its_editor_md_model_and_effort_all_match(
    tmp_path: Path, offline: list[object], variant: Callable[[], Writer], one_editor: bool
) -> None:
    arms = {"shipped": Writer(WRITER, skill=load_skill()), "variant": variant()}
    _case_files(tmp_path / "cases")
    _recordings(tmp_path / "turns", tmp_path / "cases", arms, names=HELD_OUT_FIXTURES)

    result = await _eval(tmp_path, arms).run()

    assert result.decisions.keys() == {"variant"}
    # What the promotion read was told, as it records it: read once, at the end, every case launched at once.
    assert result.one_editor == {"variant": one_editor}
    if not one_editor:
        assert not any(heading in result.promotion_why["variant"] for heading in _uncompared_headings())
    assert offline == []


def _uncompared_headings() -> set[str]:
    """Each editor-decided reading's heading, as a promotion's sentence would name it."""
    uncompared = uncompared_readings(one_editor=False)
    return {measure.reader_name for measure in WRITER_KIND_MEASURES if measure.name in uncompared}


async def test_a_candidate_held_at_its_pass_s_stop_is_read_as_running_another_editor(
    tmp_path: Path, offline: list[object]
) -> None:
    """Under a rule to stop, two cases launched one at a time: the pass is refused after the first, so the candidate is
    held there and its promotion read at its stop. It differs from the control in its effort alone, a skill the read
    would count as one editor on its own, so only the eval telling it says two."""
    arms = {"shipped": Writer(WRITER, skill=load_skill()), "variant": Writer(WRITER, skill=load_skill(), effort="low")}
    kairo = kairo_case(tmp_path / "cases")
    # The same report written over another campaign's snapshot: its id is the campaign's and the input's, so another.
    snapshot = kairo.snapshot
    assert snapshot.campaign_id is not None
    campaign = f"{snapshot.campaign_id}-again"
    again = kairo.model_copy(
        update={
            "name": "kairo-again",
            "snapshot": snapshot.model_copy(
                update={
                    "campaign_id": campaign,
                    "id": report_snapshot_id(campaign, snapshot.report_input_digest),
                }
            ),
        }
    )
    again.write(tmp_path / "cases" / "kairo-again.json.gz")
    for arm, writer in arms.items():
        cases = {case.snapshot.id: case.name for case in (kairo, again)}
        routed = writer_turns(tmp_path / "turns", arm=arm, skill=writer.analyst_skill, cases=cases)
        effort = next((word for word in get_args(EffortWord) if word == writer.effort), None)
        for snapshot in cases:
            lay_recording(routed.path(WRITER, effort, snapshot), kairo_report(WRITER, WRITER))

    result = await _eval(tmp_path, arms, judge=FakeDearPairJudge(), stop_when="separated", batch_size=1).run()

    (why,) = result.pairwise_unread.values()
    assert why.startswith("refused before any call: ")
    assert result.launch_line, "the run launched in batches"
    assert result.one_editor == {"variant": False}
    assert not any(heading in result.promotion_why["variant"] for heading in _uncompared_headings())
    assert offline == []


async def test_the_action_unsaid_reads_arms_whose_skills_differ_as_two_editors(
    tmp_path: Path, offline: list[object]
) -> None:
    """The writer eval knows the two arms' editor.md match, though their skills differ; the action, told nothing,
    reads the arms' recorded skill levels and counts two editors, the direction that keeps an editor's reading out
    of the sentence."""
    arms = _arms()
    _case_files(tmp_path / "cases")
    _recordings(tmp_path / "turns", tmp_path / "cases", arms, names=HELD_OUT_FIXTURES)
    result = await _eval(tmp_path, arms).run()
    assert result.one_editor == {"variant": True}

    (read_only,) = eval_catalogue().mount_all(read_only_tools())
    read = await read_only.call(
        {
            "action": "writer_promotion",
            "campaign_id": result.campaign_id,
            "candidate": "variant",
            "pairwise_job": result.pairwise["variant"].job_id,
        },
        host=OpsHost.reading(result.host),
        caller=Caller(scope_id=SCOPE, identity="test"),
    )
    assert not read.is_error, read.text
    assert read.structured is not None
    assert read.structured["one_editor"] is False
    assert read.structured["decision"] == result.decisions["variant"]
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
