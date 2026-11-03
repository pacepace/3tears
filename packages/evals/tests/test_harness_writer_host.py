"""The quick path's writer host: the report writer launched over a SQLite store, as the thing under test.

Every writer is scripted to write nothing (each reply holds no document), so no page reaches the judge and nothing is
sent anywhere; the judge is a :class:`~threetears.evals.providers.Model` whose transport fails any request it is
handed. Cases are the package's fixture snapshots, restored into the scope and minted onto the held-out side, as a
host mints its own. One campaign's pages are stored and judged, by a scripted judge that sends nothing, so the
pairwise pass reads them on the host and checks its judge's family as the host's launches do.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from packages.evals.tests.analyst_support import fixture_snapshots
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.report_writer_support import FakeToolTurnClient, reply
from packages.evals.tests.writer_case_support import SEED
from packages.evals.tests.writer_eval_support import FakeScoringJudge, kairo_case, kairo_report, lay_recording
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.harness import Writer, WriterEval
from threetears.evals.ops import OpsHost, await_job
from threetears.evals.ops.analysis import CampaignDefinition, campaign_create
from threetears.evals.ops.analyst import report_snapshot_restore
from threetears.evals.ops.report_writer.writer_pairwise import WriterPairwiseReport, writer_pairwise
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.ops.report_writer.writer_bank import (
    WriterCaseMint,
    WriterSplit,
    WriterTemplateCreate,
    writer_case_mint,
    writer_template_create,
)
from threetears.evals.ops.report_writer.writer_contract import (
    WriterArm,
    WriterModelFactory,
    ENDING_MEASURE,
    WRITER_KIND,
    WRITER_KIND_MEASURES,
    EffortWord,
)
from threetears.evals.ops.report_writer.writer_promotion import writer_promotion
from threetears.evals.providers import Model
from threetears.evals.quick.judged import BorrowedJudgeClient
from threetears.evals.harness.writer_host import writer_ops_host
from threetears.evals.harness.writer_turns import writer_turns
from threetears.evals.run import quote_launch, start_run
from threetears.evals.schema.models import CaseSetRef, EvalRun
from threetears.evals.schema.turns import ToolTurnClient, TurnText
from threetears.evals.storage import SqliteDocumentStore

#: The writer every arm names, on the factory's route (anthropic): a placeholder, since nothing is sent to it.
WRITER = "report-writer-candidate"

#: The scoring judge: OpenRouter's, of the google family.
JUDGE = "google/gemini-3.8-flash"

#: Five fixture cases, none of them a consumer's campaign.
CASES = ("r1", "r2", "r3", "r4", "r2 without its gates")

#: A report capped in dollars and in turns, so a cell's deadline is bounded.
BUDGET = ReportBudget(usd=0.75, turns=40)

#: How many turns an analyst that never writes a document is given before its report ends with nothing stored.
SILENT_TURNS = 4


# parity-with: threetears.evals.ops.report_writer.loop.TurnModel
@dataclass
class FakeSilentWriter:
    """A writer whose every analyst reply holds no document: each cell ends with nothing stored, and nothing judged."""

    model_name: str = WRITER
    skill: str = ""
    sessions: list[str] = field(default_factory=list)

    async def turn_pricing_refusal(self) -> str | None:
        return None

    def turn_session(self, name: str, *, snapshot: str) -> ToolTurnClient:
        self.sessions.append(name)
        return FakeToolTurnClient(*(reply(TurnText("I have no report yet.")) for _ in range(4)))


def _refuse(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"the judge was sent {request.url}, and these tests send nothing")


def _judge(model_id: str = JUDGE, **route: Any) -> Model:
    """A judge on ``route`` whose transport fails any request, so nothing is ever sent."""
    return Model(model_id, api_key="unused", transport=httpx.MockTransport(_refuse), **route)


def _host(
    tmp_path: Path,
    *,
    judge: Model | None = None,
    budget: ReportBudget = BUDGET,
    request_s: float = 120.0,
    skills: dict[str, AnalystSkill] | None = None,
    built: list[FakeSilentWriter] | None = None,
    models: WriterModelFactory | None = None,
) -> OpsHost:
    def scripted(arm: WriterArm) -> Callable[[EffortWord | None], FakeSilentWriter]:
        def at(_effort: EffortWord | None) -> FakeSilentWriter:
            writer = FakeSilentWriter(model_name=arm.session_model, skill=arm.skill.digest)
            if built is not None:
                built.append(writer)
            return writer

        return at

    return writer_ops_host(
        SqliteDocumentStore(tmp_path / "evals.sqlite"),
        judge=judge or _judge(),
        models=models or scripted,
        budget=budget,
        request_s=lambda _max_tokens: request_s,
        max_cost_usd=5.0,
        skills=skills or {},
    )


async def _mint(host: OpsHost, names: tuple[str, ...] = CASES) -> tuple[str, CaseSetRef]:
    """The writer template, created on the host, and ``names`` minted onto its held-out side; the set's latest version."""
    template = await asyncio.to_thread(
        writer_template_create,
        host,
        WriterTemplateCreate(template_name="report writer", intent="Write the report."),
        CALLER.scope_id,
    )
    snapshots = fixture_snapshots(CALLER.scope_id)
    ref: CaseSetRef | None = None
    for name in names:
        group, withhold = SEED[name]
        restored = await report_snapshot_restore(host, CALLER.scope_id, snapshots[name])
        minted = await writer_case_mint(
            host,
            WriterCaseMint(
                template_id=template.id,
                snapshot_id=restored.snapshot_id,
                withhold=list(withhold),
                group=group,
                split=WriterSplit(held_out_fraction=0.99),
            ),
            CALLER.scope_id,
            created_by=CALLER.identity,
        )
        assert minted.side == "held_out", name
        ref = minted.case_set
    assert ref is not None
    return template.id, ref


def _arm(template_id: str, case_set: CaseSetRef, skill: AnalystSkill, *, judge: str = JUDGE) -> dict[str, Any]:
    return {
        "max_cost_usd": 2.5,
        "template_id": template_id,
        "scope_id": CALLER.scope_id,
        "subject_id": "the report writer",
        "models": [WRITER],
        "k_runs": 1,
        "case_set": case_set,
        "judge_model": judge,
        "overlays": {"skill": skill.digest},
    }


async def _launched(host: OpsHost, arguments: dict[str, Any]) -> EvalRun:
    (run,) = await start_run(host.launch, **arguments)
    async with asyncio.timeout(60.0):
        while host.launch.job_manager.is_active(run.id):
            await asyncio.sleep(0.2)
    stored = await asyncio.to_thread(host.eval_host.storage.load_eval_run, run.id, CALLER.scope_id)
    assert stored is not None
    return stored


def _variant() -> AnalystSkill:
    """The shipped skill with one blank line added to ``SKILL.md``: another digest, the same instructions."""
    files = dict(load_skill().files)
    files["SKILL.md"] = f"{files['SKILL.md']}\n"
    return AnalystSkill(files=files)


async def test_two_arms_launch_over_five_held_out_cases_and_the_promotion_read_answers(tmp_path: Path) -> None:
    shipped, variant = load_skill(), _variant()
    built: list[FakeSilentWriter] = []
    host = _host(tmp_path, skills={variant.digest: variant}, built=built)
    template_id, case_set = await _mint(host)

    runs = {
        label: await _launched(host, _arm(template_id, case_set, skill))
        for label, skill in (("shipped", shipped), ("variant", variant))
    }

    for run in runs.values():
        results = await asyncio.to_thread(host.eval_host.storage.query_eval_results_by_run, run.id, CALLER.scope_id)
        assert len(results) == len(CASES)
        assert {result.termination for result in results} == {"completed"}
        assert all(ENDING_MEASURE in result.host_measures for result in results)
        assert run.judge_model == JUDGE
        assert run.host_payload[WRITER_KIND]["judge_family"] == "writer anthropic, judge google"
    # Every cell built its own writer from the factory, on the model and the skill its arm named.
    assert len(built) == 2 * len(CASES) and {writer.model_name for writer in built} == {WRITER}
    by_skill = {digest: sum(writer.skill == digest for writer in built) for digest in (shipped.digest, variant.digest)}
    assert by_skill == {shipped.digest: len(CASES), variant.digest: len(CASES)}

    line = await asyncio.to_thread(
        campaign_create,
        host.eval_host,
        CampaignDefinition(
            name="writer host",
            subject_id="the report writer",
            behavior="report-writing",
            question="Does the variant write as the shipped skill does?",
            arms=[
                {"label": "shipped", "role": "control", "from_run_ids": [runs["shipped"].id]},
                {"label": "variant", "role": "candidate", "from_run_ids": [runs["variant"].id]},
            ],
            aspects=[
                {
                    "template_id": template_id,
                    "run_ids": [run.id for run in runs.values()],
                    "margin": 5,
                    "design": {
                        "axes": [
                            {
                                "axis_id": "report_writer.skill",
                                "values": [
                                    {"content": shipped.digest, "display": "shipped"},
                                    {"content": variant.digest, "display": "variant"},
                                ],
                            }
                        ],
                        "held_fixed": {"stimulus": "controlled", "apparatus": "commissioned"},
                    },
                }
            ],
        ),
        CALLER.scope_id,
        created_by=CALLER.identity,
    )
    read = await asyncio.to_thread(writer_promotion, host.eval_host, line.id, CALLER.scope_id, candidate="variant")
    assert read.decision == "undecided", read.why
    assert read.cases == len(CASES)
    assert read.candidate == "variant" and read.control == "shipped"

    # The runs are in the file: a host built afresh over it reads them back.
    again = _host(tmp_path)
    assert await asyncio.to_thread(again.eval_host.storage.load_eval_run, runs["shipped"].id, CALLER.scope_id)


def test_the_host_declares_the_writer_and_writes_two_cells_at_once_under_enforced_ceilings(tmp_path: Path) -> None:
    host = _host(tmp_path)
    profile = host.eval_host.profile

    assert [contract.kind for contract in profile.kinds] == [WRITER_KIND]
    assert set(profile.measures.names) >= {measure.name for measure in WRITER_KIND_MEASURES}
    assert set(host.launch.kinds) == {WRITER_KIND}
    settings = host.launch.settings()
    assert settings.max_concurrent_cells == 2
    assert settings.enforcement_enabled and settings.max_cost_usd == 5.0
    assert host.generation is None


def test_the_judge_is_lent_for_the_judge_role_on_its_own_model_and_nothing_else(tmp_path: Path) -> None:
    clients = _host(tmp_path).eval_host.clients
    assert clients is not None

    lent = clients("judge", None, temperature=0.0)
    assert isinstance(lent, BorrowedJudgeClient) and lent.model_name == JUDGE
    assert isinstance(clients("judge", JUDGE), BorrowedJudgeClient)
    with pytest.raises(ValueError, match="'analysis' role"):
        clients("analysis", None)
    with pytest.raises(ValueError, match="asked for 'openai/gpt-9'"):
        clients("judge", "openai/gpt-9")


@pytest.mark.parametrize(
    ("judge", "family"),
    [
        (lambda: _judge("deepseek-chat", base_url="https://api.deepseek.com/v1"), "unchecked"),
        (lambda: _judge("gpt-9-mini", provider="openai"), "writer anthropic, judge openai"),
    ],
    ids=["an endpoint of the caller's own places no family", "a provider's own model is of its family"],
)
async def test_the_judge_s_family_is_read_from_its_route(
    tmp_path: Path, judge: Callable[[], Model], family: str
) -> None:
    model = judge()
    host = _host(tmp_path, judge=model)
    template_id, case_set = await _mint(host, ("r1",))

    run = await _launched(host, _arm(template_id, case_set, load_skill(), judge=model.model_name))

    assert run.host_payload[WRITER_KIND]["judge_family"].startswith(family)


async def test_a_judge_routed_to_the_writer_s_own_vendor_is_refused_before_anything_runs(tmp_path: Path) -> None:
    model = _judge("anthropic/report-judge")
    host = _host(tmp_path, judge=model)
    template_id, case_set = await _mint(host, ("r1",))

    with pytest.raises(ValidationFailedError, match=rf"of the writer '{re.escape(WRITER)}''s family \(anthropic\)"):
        await quote_launch(host.launch, **_arm(template_id, case_set, load_skill(), judge=model.model_name))


async def test_an_arm_whose_cells_could_outlast_its_job_is_refused_naming_what_to_lower(tmp_path: Path) -> None:
    # No turn cap: a report's deadline is every turn the loop can send, past what one job is given.
    host = _host(tmp_path, budget=ReportBudget(usd=0.75), request_s=600.0)
    template_id, case_set = await _mint(host)

    with pytest.raises(ValidationFailedError, match="past its job's") as refused:
        await quote_launch(host.launch, **_arm(template_id, case_set, load_skill()))
    assert "turn cap" in (refused.value.remedy or "")
    # The same cases under a turn cap fit.
    capped = _host(tmp_path, request_s=120.0)
    await quote_launch(capped.launch, **_arm(template_id, case_set, load_skill()))


def _silent_recording(path: Path, *, turns: int) -> None:
    """A turn recording of an analyst session whose every reply holds no document, as a recording run writes one."""
    path.parent.mkdir(parents=True)
    silent = reply(TurnText("I have no report yet."), model=WRITER).to_json()
    lines = [
        {"recorder_version": 1, "recorded_at": "2026-11-03T00:00:00+00:00", "session": "analyst", "index": index}
        | entry
        for index in range(turns)
        for entry in ({"kind": "ceiling", "ceiling": 0.05}, {"kind": "reply", "reply": silent})
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


async def test_an_arm_launches_through_its_recorded_models_and_replays_each_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    shipped = load_skill()
    snapshot = fixture_snapshots(CALLER.scope_id)["r1"].id
    turns = writer_turns(tmp_path / "turns", arm="shipped", skill=shipped, cases={snapshot: "r1"})
    _silent_recording(turns.path(WRITER, None, snapshot), turns=SILENT_TURNS)
    host = _host(tmp_path, models=turns)
    template_id, case_set = await _mint(host, ("r1",))

    run = await _launched(host, _arm(template_id, case_set, shipped))

    (result,) = await asyncio.to_thread(host.eval_host.storage.query_eval_results_by_run, run.id, CALLER.scope_id)
    # The recorded analyst never submits, so the report ends refused, as recorded, never failed.
    assert result.termination == "completed" and result.host_measures[ENDING_MEASURE] == "refused"
    # The launcher handed the factory the arm's skill, and the cell replayed its case's recording to the end.
    assert turns.unconsumed() == []


#: The pairwise pass's judge: a bare id, so its family is its provider's.
PAIR_JUDGE = "scoring-judge"


async def _judged_campaign(tmp_path: Path) -> str:
    """A two-arm writer eval over Kairo's case, each page judged by :data:`PAIR_JUDGE` on OpenRouter: its campaign."""
    arms = {"shipped": Writer(WRITER, skill=load_skill()), "variant": Writer(WRITER, skill=_variant())}
    kairo = kairo_case(tmp_path / "cases")
    for arm, writer in arms.items():
        routed = writer_turns(
            tmp_path / "turns", arm=arm, skill=writer.analyst_skill, cases={kairo.snapshot.id: "kairo"}
        )
        lay_recording(routed.path(WRITER, None, kairo.snapshot.id), kairo_report(WRITER, WRITER))
    result = await WriterEval(
        tmp_path / "cases",
        arms=arms,
        judge=FakeScoringJudge(PAIR_JUDGE),
        recordings=tmp_path / "turns",
        budget_usd=0.75,
        max_cost_usd=5.0,
        name=CALLER.scope_id,
        store=tmp_path / "evals.sqlite",
    ).run()
    assert all(summary.status == "completed" for summary in result.arms.values())
    return result.campaign_id


async def _pairwise(tmp_path: Path, campaign_id: str, *, writer_provider: str) -> WriterPairwiseReport:
    """The pairwise pass over ``campaign_id`` on a writer host over the same store, its writers on ``writer_provider``."""
    host = writer_ops_host(
        SqliteDocumentStore(tmp_path / "evals.sqlite"),
        judge=FakeScoringJudge(PAIR_JUDGE),
        models=lambda arm: lambda effort: FakeSilentWriter(model_name=arm.session_model, skill=arm.skill.digest),
        budget=BUDGET,
        request_s=lambda _max_tokens: 120.0,
        max_cost_usd=5.0,
        writer_provider=writer_provider,
    )
    try:
        started = await writer_pairwise(host, campaign_id, CALLER.scope_id, candidate="variant")
        (status,) = await await_job(host, started, CALLER.scope_id)
    finally:
        await host.launch.job_manager.shutdown()
    assert status.state == "completed" and status.result is not None, status.detail
    return WriterPairwiseReport.model_validate(status.result)


async def test_the_pairwise_pass_on_the_writer_host_checks_the_judge_s_family_as_its_launches_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    campaign_id = await _judged_campaign(tmp_path)

    report = await _pairwise(tmp_path, campaign_id, writer_provider="anthropic")

    # The host places both: the check is made, never recorded as not made.
    assert report.judge_family == ["writer anthropic, judge openrouter"]
    # Writers the host places in the judge's own family: the pass is refused before any call.
    with pytest.raises(ValidationFailedError, match=r"is of the writer .* family \(openrouter\)"):
        await _pairwise(tmp_path, campaign_id, writer_provider="openrouter")
