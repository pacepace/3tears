"""The report writer campaign, end to end on live models: #1009 chunk 06's mechanism smoke.

Two arms over three writer cases, launched through the writer's launcher as a host launches them: the shipped skill
(the control) and an inert variant of it (one blank line added to ``SKILL.md``, so its digest differs and nothing it
says does). Each cell writes a report on the arm's model, the editor reviews it, and a scoring judge of another model
family scores the stored page on the editor rubric's seven items. The two runs then become a campaign, and the
promotion read answers on it. This proves the pipeline runs; it decides nothing, and the skill ledger is never
written::

    uv run python -m packages.evals.tests.fixtures.report_plans.writer_smoke --dry
    uv run python -m packages.evals.tests.fixtures.report_plans.writer_smoke --cap-usd 5 --out <dir>

``--dry`` drives the same path on scripted models that write nothing, so every refusal before a model is called shows
up for free. The live run reads ``ANTHROPIC_API_KEY`` (the writer) and ``OPENROUTER_API_KEY`` (the judge) from the
environment.

**How ``--cap-usd`` holds.** The host's ceilings are on, and each arm is launched naming ``max_cost_usd``, an equal
share of the cap, which the engine enforces on the run's whole spend, its judge's included. The engine's quote is
read first and printed: where it prices an arm above that share, the smoke stops before anything is sent. The toy
host's rate card prices no Claude model, so for the writer the quote answers unpriced, and the engine admits an
unpriced arm only because the launch names its ``max_cost_usd``. Inside that, each report is also capped by its own
budget (:data:`BUDGET`), and the smoke refuses a cap the reports' budgets could pass.

The three cases are minted onto the held-out side of this process's own in-memory bank (``held_out_fraction`` 0.99,
which puts r1, r3 and r4 there), because the promotion read reads held-out cases only. Nothing outside this process
holds that bank.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from packages.evals.tests.analyst_support import fixture_snapshots
from packages.evals.tests.fixtures.toyhost.host import with_kinds
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.report_writer_support import FakeToolTurnClient, reply
from packages.evals.tests.writer_case_support import SEED, WRITER_TEMPLATE, writer_host
from threetears.evals.actions.report_writing import launchable_report_writer_kind
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.ops import OpsHost
from threetears.evals.ops.analysis import CampaignDefinition, campaign_create
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.ops.report_writer.writer_bank import WriterCaseMint, WriterSplit, writer_case_mint
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND, EffortWord, WriterArm, writer_scoring_rubric
from threetears.evals.ops.report_writer.writer_promotion import writer_promotion
from threetears.evals.ops.analyst import report_snapshot_restore
from threetears.evals.providers import Model
from threetears.evals.providers.failures import describe_failure
from threetears.evals.run import quote_launch, start_run
from threetears.evals.schema.models import CaseSetRef, EvalRun
from threetears.evals.schema.turns import ToolTurnClient, TurnText

#: The writer every cell runs on: the ceiling of the build-and-test models, and the one a Claude path needs.
WRITER: Final = "claude-haiku-5-5"

#: The scoring judge: another family than the writer, as the launcher requires, and cheap.
JUDGE: Final = "google/gemini-3.8-flash"

#: Each model's family as this host places it.
FAMILIES: Final = {WRITER: "anthropic", JUDGE: "google"}

#: The cases: three fixtures in three groups, none of them r5 (its input is being re-recorded on its own branch).
CASES: Final = ("r1", "r3", "r4")

#: Each report's cap: dollars, and a turn cap that bounds its wall clock. Six reports at this cap stay under $5.
BUDGET: Final = ReportBudget(usd=0.75, turns=40)

#: The host's ceiling for one request.
REQUEST_S: Final = 120.0

#: The writer's thinking effort when an arm names none.
EFFORT: Final = "low"

#: The writer template as a host creates it, its rubric seeded from the shipped skill.
SCORED = WRITER_TEMPLATE.model_copy(update={"rubric": writer_scoring_rubric()})


def inert_variant() -> AnalystSkill:
    """The shipped skill with one blank line added to ``SKILL.md``: another digest, the same instructions."""
    files = dict(load_skill().files)
    files["SKILL.md"] = f"{files['SKILL.md']}\n"
    return AnalystSkill(files=files)


# parity-with: threetears.evals.ops.report_writer.loop.TurnModel
@dataclass
class _FakeDryModel:
    """The dry mode's model: an analyst whose every reply holds no document, so no page reaches the judge."""

    model_name: str = WRITER
    sessions: dict[str, FakeToolTurnClient] = field(default_factory=dict)

    async def turn_pricing_refusal(self) -> str | None:
        return None

    def turn_session(self, name: str, *, snapshot: str) -> ToolTurnClient:
        client = FakeToolTurnClient(*(reply(TurnText("I have no report yet.")) for _ in range(4)))
        self.sessions[name] = client
        return client


def _keys() -> dict[str, str]:
    """The two keys from the environment; refused by name when either is missing."""
    keys = {name: os.environ.get(name, "") for name in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")}
    if missing := [name for name, value in keys.items() if not value]:
        raise SystemExit(f"writer_smoke.py: a live run needs {', '.join(missing)} in the environment; --dry needs none")
    return keys


def _host(*, dry: bool, out: Path | None, skills: dict[str, AnalystSkill]) -> OpsHost:
    """The toy host with the writer registered and, live, its judge and writer on real models."""
    fixture = writer_host(settings=TOYHOST_LAUNCH_SETTINGS.model_copy(update={"enforcement_enabled": True}))
    fixture.host.eval_host.storage.save_template(SCORED)
    # The package's own describer for its own Model, so a failed turn says what the provider said.
    launch = replace(
        fixture.host.launch,
        eval_host=replace(fixture.host.launch.eval_host, failure_describer=describe_failure),
    )
    built: list[int] = []
    if dry:

        def models(arm: WriterArm) -> Callable[[EffortWord | None], Any]:
            return lambda _effort: _FakeDryModel(model_name=arm.session_model)

    else:
        keys = _keys()
        assert out is not None

        def judge_clients(role: str, model: str | None, *, temperature: float | None = None) -> Any:
            if role != "judge":
                raise SystemExit(f"writer_smoke.py: the smoke builds no {role!r} client")
            return Model(model or JUDGE, provider="openrouter", api_key=keys["OPENROUTER_API_KEY"])

        launch = replace(launch, eval_host=replace(launch.eval_host, clients=judge_clients))

        def models(arm: WriterArm) -> Callable[[EffortWord | None], Any]:
            def at(effort: EffortWord | None) -> Model:
                built.append(1)
                return Model(
                    arm.session_model,
                    provider="anthropic",
                    api_key=keys["ANTHROPIC_API_KEY"],
                    effort=effort or EFFORT,
                    turn_recording=out / f"turns-{len(built):02d}.jsonl",
                )

            return at

    holder: list[OpsHost] = []
    entry = launchable_report_writer_kind(
        lambda: holder[0],
        identity="report-writer-smoke",
        models=models,
        budget=BUDGET,
        request_s=lambda _max_tokens: REQUEST_S,
        model_family=FAMILIES.get,
        skills=skills,
    )
    host = OpsHost(launch=with_kinds(launch, {WRITER_KIND: entry}))
    holder.append(host)
    return host


async def _seed(host: OpsHost) -> CaseSetRef:
    """Mint :data:`CASES` onto the held-out side of this bank; the held-out set at its latest version."""
    split = WriterSplit(held_out_fraction=0.99)
    snapshots = fixture_snapshots(CALLER.scope_id)
    sides: set[str] = set()
    ref: CaseSetRef | None = None
    for name in CASES:
        group, withhold = SEED[name]
        restored = await report_snapshot_restore(host, CALLER.scope_id, snapshots[name])
        minted = await writer_case_mint(
            host,
            WriterCaseMint(
                template_id=SCORED.id,
                snapshot_id=restored.snapshot_id,
                withhold=list(withhold),
                group=group,
                split=split,
            ),
            CALLER.scope_id,
            created_by=CALLER.identity,
        )
        sides.add(minted.side)
        ref = minted.case_set
    if sides != {"held_out"} or ref is None:
        raise SystemExit(f"writer_smoke.py: the cases landed on {sorted(sides)}, and the promotion reads held-out only")
    # The last mint's version of the held-out set lists all three cases.
    return ref


def _launched(case_set: CaseSetRef, skill: AnalystSkill, *, max_cost_usd: float) -> dict[str, Any]:
    return {
        "max_cost_usd": max_cost_usd,
        "template_id": SCORED.id,
        "scope_id": CALLER.scope_id,
        "subject_id": "the report writer",
        "models": [WRITER],
        "k_runs": 1,
        "case_set": case_set,
        "judge_model": JUDGE,
        "overlays": {"skill": skill.digest},
    }


async def _run(host: OpsHost, arguments: dict[str, Any], *, deadline_s: float) -> EvalRun:
    (run,) = await start_run(host.launch, **arguments)
    async with asyncio.timeout(deadline_s):
        while host.launch.job_manager.is_active(run.id):
            await asyncio.sleep(1.0)
    stored = await asyncio.to_thread(host.eval_host.storage.load_eval_run, run.id, CALLER.scope_id)
    assert stored is not None
    return stored


def _summary(host: OpsHost, run: EvalRun) -> list[dict[str, Any]]:
    results = host.eval_host.storage.query_eval_results_by_run(run.id, CALLER.scope_id)
    return [
        {
            "case": result.test_case_id,
            "termination": result.termination,
            "runner_error": result.runner_error,
            "host_measures": dict(sorted(result.host_measures.items())),
            "rubric_scores": {score.dim: score.score for score in result.rubric_scores},
        }
        for result in sorted(results, key=lambda r: r.test_case_id)
    ]


async def smoke(*, dry: bool, cap_usd: float, out: Path | None) -> dict[str, Any]:
    """Run the smoke; what each arm landed and what the promotion read answered."""
    shipped, variant = load_skill(), inert_variant()
    host = await asyncio.to_thread(lambda: _host(dry=dry, out=out, skills={variant.digest: variant}))
    case_set = await _seed(host)
    arms = {"shipped": shipped, "inert variant": variant}
    share = cap_usd / len(arms)
    quotes = {
        label: await quote_launch(host.launch, **_launched(case_set, skill, max_cost_usd=share))
        for label, skill in arms.items()
    }
    over = [
        f"{label} quoted ${verdict.price.predicted_usd:.2f}"
        for label, quote in quotes.items()
        for verdict in quote.arms
        if verdict.price.predicted_usd is not None and verdict.price.predicted_usd > share
    ]
    reports = len(CASES)
    if over or reports * BUDGET.usd > share:
        raise SystemExit(
            f"writer_smoke.py: an arm could pass its ${share:.2f} share of the ${cap_usd} cap "
            f"({'; '.join(over) or f'{reports} reports at ${BUDGET.usd} each'}); nothing sent"
        )
    runs: dict[str, EvalRun] = {}
    for label, skill in arms.items():
        launched = _launched(case_set, skill, max_cost_usd=share)
        runs[label] = await _run(host, launched, deadline_s=60.0 if dry else 3 * 3600.0)
    line = await asyncio.to_thread(
        campaign_create,
        host.eval_host,
        CampaignDefinition(
            name="writer smoke",
            subject_id="the report writer",
            behavior="report-writing",
            question="Does the inert variant write as the shipped skill does?",
            arms=[
                {"label": "shipped", "role": "control", "from_run_ids": [runs["shipped"].id]},
                {"label": "inert variant", "role": "candidate", "from_run_ids": [runs["inert variant"].id]},
            ],
            aspects=[
                {
                    "template_id": SCORED.id,
                    "run_ids": [run.id for run in runs.values()],
                    "margin": 5,
                    "design": {
                        "axes": [
                            {
                                "axis_id": "report_writer.skill",
                                "values": [
                                    {"content": shipped.digest, "display": "shipped"},
                                    {"content": variant.digest, "display": "inert variant"},
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
    promotion: Any
    try:
        read = await asyncio.to_thread(
            writer_promotion, host.eval_host, line.id, CALLER.scope_id, candidate="inert variant"
        )
        promotion = read.model_dump(mode="json")
    except (NotFoundError, ValidationFailedError) as refused:  # the read's refusal is the smoke's finding; exit 1
        promotion = {"refused": type(refused).__name__, "message": str(refused)}
    return {
        "dry": dry,
        "quotes": {label: repr(quote) for label, quote in quotes.items()},
        "arms": {
            label: {"run": run.id, "results": await asyncio.to_thread(_summary, host, run)}
            for label, run in runs.items()
        },
        "campaign": line.id,
        "promotion": promotion,
    }


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="writer_smoke.py", description=__doc__.splitlines()[0])
    parser.add_argument("--dry", action="store_true", help="scripted models, nothing sent")
    parser.add_argument("--cap-usd", type=float, default=5.0, help="the owner's cap for the whole smoke")
    parser.add_argument("--out", type=Path, help="where the live run's turn recordings and result go")
    options = parser.parse_args(list(argv))
    if not options.dry:
        if options.out is None:
            parser.error("a live run needs --out")
        options.out.mkdir(parents=True, exist_ok=False)
    found = asyncio.run(smoke(dry=options.dry, cap_usd=options.cap_usd, out=options.out))
    text = json.dumps(found, indent=2, default=str)
    if options.out is not None:
        (options.out / "smoke.json").write_text(text, encoding="utf-8")
    print(text)
    return 1 if "refused" in found["promotion"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
