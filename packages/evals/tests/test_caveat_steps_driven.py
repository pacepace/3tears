"""A next step that ships a sequence of calls is followed, as written, through the real actions on a fixture campaign.

A step is a promise: followed as written, it does what it says. Each sequence a first-screen step can ship is taken
here by one generic follower (:func:`_follow`), which knows nothing of what the step is for: it makes each call in
order through :meth:`MountedTool.call` as every transport makes it, supplies exactly what the step leaves to its taker
(``needs``, ``as_launched``) and fills what it reads off an earlier call (``after``) from that call's result. Nothing
is completed by a model: the extractor is the toy host's scripted one, and the variation writer a scripted answer.

- **More cases, on arms that differ only in their model** (a campaign too small to read, gated): the gate's step
  launches every arm in one call on new cases written by the template's llm axis and joins the runs; read again, the
  arms share every case, the gate is lifted, and the launch kept every case it generated and reused none.
- **The same step on an axis of stated values** (enum): the launch reuses the cases the arms already ran, so the
  campaign is still too small. The step states it cannot know which (``axis_may_not_generate``): the reading holds no
  template's axes.
- **A re-run that holds an apparatus confound's dimension alike**, on arms launched at every non-default setting the
  reading holds: the launch fills the repeats, the latency declaration and the schedule from the runs, names the rest,
  holds the reviewer pool at one value on every call, joins the new runs and takes the old ones out; read again, the
  confound is gone and the arms are the same arms.
- **A confirmation in a new campaign**: created with its criteria, the launch's runs joined as soon as they start and
  the control named from them before any result is scored; the new campaign reads no criterion and no control as
  named late, on cases the original campaign never ran, while the original report's caveat stays raised.

What no sequence here drives makes no call (``lift_not_shown``), held in ``test_caveat_list.py``.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.caveats import Plan, Setup, caveat_step, evidence_of, next_step
from threetears.evals.analysis.report_input import RemedyStep, ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.gen import generate_variations
from threetears.evals.kernel.declaration import SweptAxis
from threetears.evals.kernel.host import CompletionRole
from threetears.evals.ops import JobsStarted, OpsHost
from threetears.evals.run import ArmPlan, ArmPrice, ArmQuote, KindWiring, LaunchableKind, LaunchRequest, launch_run
from threetears.evals.schema import DEFAULT_LAUNCH_K_RUNS, SweepableValue
from threetears.evals.schema.models import EvalRun, EvalTemplate, VariationAxis
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.kernel import EvalStorage
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_aspect_design
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.host import with_registry
from packages.evals.tests.fixtures.toyhost.kind import (
    DOCUMENT_PARAM,
    TOY_DOCUMENTS,
    TOY_EXTRACTOR_KIND,
    ExtractionRequest,
    ExtractionResult,
    ScriptedExtractionClient,
    ToyDocument,
    ToyExtractorKind,
)
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.ops_support import CALLER, ops_fixture
from packages.evals.tests.off_loop import off_loop

#: The control's model and the candidate's: the two arms differ only in their model.
CONTROL_MODEL, CANDIDATE_MODEL = RUN_MODELS

#: The model the launches name to write their cases.
WRITER = "writer-a"

#: Every invoice a generated case can name: more than any launch here asks for, so a writer never runs out.
_DOCUMENTS: tuple[ToyDocument, ...] = tuple(
    replace(TOY_DOCUMENTS[index % len(TOY_DOCUMENTS)], document_id=f"doc-g{index:02d}") for index in range(60)
)

#: The axis a model writes, and one of stated values over the first three invoices.
_LLM_AXIS = VariationAxis(name=DOCUMENT_PARAM, generator="llm", description="an invoice id")
_ENUM_AXIS = VariationAxis(
    name=DOCUMENT_PARAM, generator="enum", values=[document.document_id for document in _DOCUMENTS[:3]]
)


@dataclass(frozen=True)
class _Completion:
    content: str


# parity-with: threetears.evals.schema.completion.BoundCompletionClient
@dataclass
class _FakeWriter:
    """A variation writer that answers with invoices no earlier answer named, so every value it writes is new."""

    model_name: str
    fresh: itertools.count[int]

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        return 0.001

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        values = ", ".join(f'"{_DOCUMENTS[next(self.fresh)].document_id}"' for _ in range(8))
        return _Completion("{" + f'"values": [{values}]' + "}")

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> _FakeWriter:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


# parity-with: threetears.evals.kernel.host.eval_host.CompletionClients
@dataclass
class _FakeClients:
    """The host's client factory: one count of invoices across every writer it builds."""

    fresh: itertools.count[int] = field(default_factory=itertools.count)

    def __call__(self, role: CompletionRole, model: str | None, *, temperature: float | None = None) -> _FakeWriter:
        assert role == "variation" and model is not None, "only the variation writer is asked for"
        return _FakeWriter(model_name=model, fresh=self.fresh)


class _HeldClient(ScriptedExtractionClient):
    """The scripted extractor, holding every extraction until ``open`` is set: no result is scored before then."""

    def __init__(self, held: asyncio.Event) -> None:
        super().__init__()
        self.held = held

    async def extract(self, request: ExtractionRequest) -> ExtractionResult:
        await self.held.wait()
        return await super().extract(request)


def _template(axis: VariationAxis) -> EvalTemplate:
    return toyhost_template().model_copy(update={"variation_axes": [axis]})


def _priced_at_nothing(_quote: ArmQuote) -> ArmPrice:
    return ArmPrice(predicted_usd=0.0, basis="the toy extractor's scripted rate card")


def _generating_host(template: EvalTemplate, client: ScriptedExtractionClient) -> OpsHost:
    """The toy host as an operations host whose launcher generates its cases as an adopter's does: once per launch,
    through the launch's admitted generation, every arm of the launch running the cases it generated."""
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(template)
    host, _client = toyhost_launch_host(storage=storage, clients=_FakeClients())
    world = host.eval_host.profile.world
    assert world is not None

    async def launch(request: LaunchRequest) -> EvalRun:
        async def generate() -> Any:
            return await generate_variations(
                request.template,
                request.n_variations,
                storage=storage,
                scope_id=request.scope_id,
                generation=request.generation,
                blocking_executor=None,
            )

        generation = await request.launch_group.resolve_once(
            ("cases", request.template.id, str(request.n_variations)), generate
        )
        return await launch_run(
            generating,
            request,
            KindWiring(
                kind_factory=lambda cell: ToyExtractorKind(
                    client=client, world=world, documents=_DOCUMENTS, goal_checks=tuple(cell.template.goal_state_checks)
                ),
                subject=TOYHOST_SUBJECT,
                test_cases=generation.cases,
                variation_counts=generation.counts,
            ),
        )

    def plan_arm(request: LaunchRequest) -> ArmPlan:
        return ArmPlan(
            case_count=request.n_variations,
            candidate_model=request.candidate_model or CONTROL_MODEL,
            judge=None,
            simulator_model=None,
        )

    generating = replace(
        host,
        eval_host=with_registry(
            host.eval_host,
            {
                TOY_EXTRACTOR_KIND: LaunchableKind(
                    launch=launch, resolve_subject=lambda _subject, _scope: None, plan_arm=plan_arm
                )
            },
        ),
        launch_pricer=_priced_at_nothing,
    )
    return OpsHost(launch=generating)


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


async def _call(tool: MountedTool, host: OpsHost, action: str, **arguments: Any) -> dict[str, Any]:
    outcome = await tool.call({"action": action, **arguments}, host=host, caller=CALLER)
    assert not outcome.is_error, outcome.text
    assert isinstance(outcome.structured, dict)
    return outcome.structured


async def _started_runs(tool: MountedTool, host: OpsHost, started: Mapping[str, Any]) -> list[str]:
    """The runs a launch's job names on job_poll once they start, polled as a taker polls it."""
    (job,) = JobsStarted.model_validate(started).jobs
    async with asyncio.timeout(10):
        while not (status := await _call(tool, host, "job_poll", job_id=job.job_id))["run_ids"]:
            await asyncio.sleep(0.005)
    return list(status["run_ids"])


async def _ended(tool: MountedTool, host: OpsHost, job_ids: list[str]) -> None:
    async with asyncio.timeout(20):
        for job_id in job_ids:
            while not (await _call(tool, host, "job_poll", job_id=job_id))["done"]:
                await asyncio.sleep(0.005)


@dataclass
class _Taken:
    """What following a step returned: the runs its launches started, the campaign it created, its jobs."""

    runs: list[str] = field(default_factory=list)
    campaign: str | None = None
    jobs: list[str] = field(default_factory=list)


async def _follow(
    tool: MountedTool,
    host: OpsHost,
    step: RemedyStep,
    supplied: Mapping[str, Any],
    *,
    control_model: str | None = None,
    before_results: Callable[[], None] = lambda: None,
) -> _Taken:
    """Take ``step`` as written: each call in order, every argument the step leaves to its taker from ``supplied``
    (None leaves it out, as a launch that never set it did), every one it reads off an earlier call from that call's
    result. A launch's runs are read off job_poll once they start; its jobs are awaited to their end once every call
    is made, and ``before_results`` runs just before, when nothing is left to name."""
    taken = _Taken()
    for part in (step, *step.then):
        left = [*part.needs, *part.as_launched]
        assert set(left) <= set(supplied), f"the taker is not told {sorted(set(left) - set(supplied))}"
        given = {name: supplied[name] for name in left if supplied[name] is not None}
        for call in part.calls:
            after: dict[str, Any] = {}
            for name, source in part.after.items():
                if source == "launched_runs":
                    after[name] = list(taken.runs)
                elif source == "created_campaign":
                    after[name] = taken.campaign
                else:
                    ran = [
                        run
                        for run in taken.runs
                        if (await _call(tool, host, "run_get", run_id=run))["candidate_model"] == control_model
                    ]
                    (after[name],) = ran
            result = await _call(tool, host, str(part.action), **{**call, **given, **after})
            if part.action == "run_launch":
                taken.runs += await _started_runs(tool, host, result)
                taken.jobs += [job["job_id"] for job in result["jobs"]]
            elif part.action == "campaign_create":
                taken.campaign = str(result["id"])
    before_results()
    await _ended(tool, host, taken.jobs)
    return taken


def _bundle(host: OpsHost, campaign_id: str) -> CampaignBundle:
    return off_loop(assemble_campaign_bundle, host.launch.eval_host, campaign_id, TOYHOST_SCOPE)


def _cases(host: OpsHost, run_ids: list[str]) -> set[str]:
    storage = host.launch.eval_host.storage
    runs = [off_loop(storage.load_eval_run, run_id, TOYHOST_SCOPE) for run_id in run_ids]
    return {case for run in runs if run is not None for case in run.test_case_ids}


def _design() -> dict[str, Any]:
    """An aspect that sweeps the candidate model, asking whether the candidate is better than the control."""
    axis = SweptAxis(axis_id="model", values=[SweepableValue.of(model, display=model) for model in RUN_MODELS])
    return toyhost_aspect_design(axes=[axis], bars=[]).model_dump(mode="json")


async def _campaign(
    tool: MountedTool, host: OpsHost, *, template: EvalTemplate, launches: list[dict[str, Any]]
) -> tuple[str, dict[str, list[str]]]:
    """A campaign of two arms over the launches' runs, its control the control model's: each launch's runs by model."""
    by_model: dict[str, list[str]] = {}
    for launched in launches:
        started = await _call(
            tool, host, "run_launch", template_id=template.id, subject_id=TOYHOST_SUBJECT.subject_id, **launched
        )
        runs = await _started_runs(tool, host, started)
        await _ended(tool, host, [job["job_id"] for job in started["jobs"]])
        for run in runs:
            model = (await _call(tool, host, "run_get", run_id=run))["candidate_model"]
            by_model.setdefault(model, []).append(run)
    created = await _call(
        tool,
        host,
        "campaign_create",
        name="steps followed",
        subject_id=TOYHOST_SUBJECT.subject_id,
        behavior="extraction",
        question="is the candidate extractor better than the control?",
        aspects=[
            {"label": "extract", "run_ids": [run for runs in by_model.values() for run in runs], "design": _design()}
        ],
        campaign_arms=[{"label": "control", "role": "control", "from_run_ids": by_model[CONTROL_MODEL]}],
    )
    return str(created["id"]), by_model


def _read(host: OpsHost, campaign_id: str) -> tuple[CampaignBundle, ReportInput]:
    bundle = _bundle(host, campaign_id)
    return bundle, report_input(bundle)


def _setup(bundle: CampaignBundle, built: ReportInput) -> Setup:
    """What a launch can fill from the reading, as the first screen reads it off the campaign's bundle."""
    evidence = evidence_of(bundle)
    return Setup(
        subject_id=evidence.subject_id,
        templates=evidence.templates,
        arms=evidence.arms,
        control="control",
        aspects=tuple(aspect.id for aspect in built.aspects),
    )


# --- more cases, on arms that differ only in their model -------------------------------------------------------------


async def test_a_gate_s_step_followed_on_an_llm_axis_lifts_the_gate_on_cases_every_arm_shares(
    evals: MountedTool,
) -> None:
    template = _template(_LLM_AXIS)
    host = off_loop(_generating_host, template, ScriptedExtractionClient())
    first = {"models": list(RUN_MODELS), "n_variations": 3, "variation_model": WRITER, "k_runs": 2}
    campaign_id, before = await _campaign(evals, host, template=template, launches=[first])
    _bundle_before, built = _read(host, campaign_id)
    assert built.first_screen is not None and built.first_screen.gate is not None
    assert (built.first_screen.gate.rule, built.first_screen.gate.fewest_cases) == ("too_small", 3)
    (step,) = built.first_screen.steps
    assert step.planned_cases is not None and step.planned_cases > 0
    assert len(step.calls) == 1 and sorted(step.call_arms[0]) == sorted(arm.id for arm in built.arms), (
        "arms that differ only in their model share one call, so its new cases are every one of theirs"
    )
    assert step.limits == ["axis_may_not_generate"] and step.unpaired_arms == []

    taken = await _follow(
        evals,
        host,
        step,
        {
            "campaign_id": campaign_id,
            "variation_model": WRITER,
            "judge_model": None,
            "simulator_model": None,
            "overlays": None,
            "apparatus_settings": None,
            "cell_timeout_s": None,
            "measure_latency": None,
            "arm_schedule": None,
        },
    )

    _bundle_after, after = _read(host, campaign_id)
    assert after.first_screen is not None and after.first_screen.gate is None, "the campaign is no longer too small"
    added = _cases(host, taken.runs)
    assert len(added) == step.planned_cases
    assert not added & _cases(host, [run for runs in before.values() for run in runs]), "every case is new"
    storage = host.launch.eval_host.storage
    for run_id in taken.runs:
        run = off_loop(storage.load_eval_run, run_id, TOYHOST_SCOPE)
        assert run is not None and run.variation_counts is not None
        assert (run.variation_counts.kept, run.variation_counts.reused) == (step.planned_cases, 0)
    per_model = {
        model: _cases(host, [*runs, *(run for run in taken.runs if _model_of(host, run) == model)])
        for model, runs in before.items()
    }
    assert per_model[CONTROL_MODEL] == per_model[CANDIDATE_MODEL], "the arms share every case, old and new"


def _model_of(host: OpsHost, run_id: str) -> str:
    run = off_loop(host.launch.eval_host.storage.load_eval_run, run_id, TOYHOST_SCOPE)
    assert run is not None
    return run.candidate_model


async def test_the_same_step_on_an_axis_of_stated_values_reuses_the_cases_and_says_it_may(evals: MountedTool) -> None:
    template = _template(_ENUM_AXIS)
    host = off_loop(_generating_host, template, ScriptedExtractionClient())
    first = {"models": list(RUN_MODELS), "n_variations": 3, "k_runs": 2}
    campaign_id, before = await _campaign(evals, host, template=template, launches=[first])
    _bundle_before, built = _read(host, campaign_id)
    assert built.first_screen is not None and built.first_screen.gate is not None
    (step,) = built.first_screen.steps
    assert "axis_may_not_generate" in step.limits, "the reading holds no template's axes, so the step cannot promise"

    taken = await _follow(
        evals,
        host,
        step,
        {
            "campaign_id": campaign_id,
            "variation_model": None,
            "judge_model": None,
            "simulator_model": None,
            "overlays": None,
            "apparatus_settings": None,
            "cell_timeout_s": None,
            "measure_latency": None,
            "arm_schedule": None,
        },
    )

    _bundle_after, after = _read(host, campaign_id)
    assert _cases(host, taken.runs) <= _cases(host, [run for runs in before.values() for run in runs])
    assert after.first_screen is not None and after.first_screen.gate is not None, "no new case, still too small"
    assert after.first_screen.gate.rule == "too_small"


# --- a re-run that holds an apparatus confound's dimension alike -----------------------------------------------------


async def test_a_balanced_re_run_followed_with_the_dimension_held_alike_removes_the_confound_from_the_same_arms(
    evals: MountedTool,
) -> None:
    fixture = off_loop(ops_fixture)
    host = fixture.host
    template = toyhost_template()
    settings = {"k_runs": 2, "measure_latency": True, "arm_schedule": "sequential", "cell_timeout_s": 30.0}
    assert settings["k_runs"] != DEFAULT_LAUNCH_K_RUNS
    launches = [
        {"models": [CONTROL_MODEL], "apparatus_settings": {"reviewer_pool": "pool-a"}, **settings},
        {"models": [CANDIDATE_MODEL], "apparatus_settings": {"reviewer_pool": "pool-b"}, **settings},
    ]
    campaign_id, before = await _campaign(evals, host, template=template, launches=launches)
    bundle, built = _read(host, campaign_id)
    caveat = next(entry for entry in built.caveat_list or [] if entry.id == "apparatus:reviewer_pool")
    # No contrast names an apparatus dimension, so this caveat lands method-level and the first screen shows no step
    # for it (filed); the step is the one the first screen offers wherever it does show it.
    step = caveat_step(caveat, setup=_setup(bundle, built))
    assert step.remedy == "rerun_balanced" and step.alike == ["reviewer_pool"]
    for call in step.calls:
        assert {name: call[name] for name in ("k_runs", "measure_latency", "arm_schedule")} == {
            "k_runs": 2,
            "measure_latency": True,
            "arm_schedule": "sequential",
        }, "the repeats, the latency declaration and the schedule are filled from what the runs recorded"
    assert step.as_launched == [
        "judge_model",
        "simulator_model",
        "overlays",
        "apparatus_settings",
        "cell_timeout_s",
        "case_set_name",
        "case_set_version",
    ]
    keys = {arm.id: arm for arm in built.arms}

    await _follow(
        evals,
        host,
        step,
        {
            "campaign_id": campaign_id,
            # Held alike: one value on every call, never each arm's own.
            "apparatus_settings": {"reviewer_pool": "pool-a"},
            # As launched.
            "cell_timeout_s": settings["cell_timeout_s"],
            "judge_model": None,
            "simulator_model": None,
            "overlays": None,
            "case_set_name": None,
            "case_set_version": None,
        },
    )

    _bundle_after, after = _read(host, campaign_id)
    assert not [entry.id for entry in after.caveat_list or [] if entry.kind in ("apparatus", "confound")]
    assert {arm.id: arm for arm in after.arms}.keys() == keys.keys(), "the same arms, read again"
    stored = off_loop(host.launch.eval_host.storage.load_campaign, campaign_id, TOYHOST_SCOPE)
    assert stored is not None
    assert not set(stored.run_ids) & {run for runs in before.values() for run in runs}, "the replaced runs left"


# --- a confirmation in a new campaign --------------------------------------------------------------------------------


async def test_a_confirmation_followed_in_order_reads_nothing_late_in_its_campaign_on_unseen_cases(
    evals: MountedTool,
) -> None:
    held = asyncio.Event()
    held.set()
    template = _template(_LLM_AXIS)
    host = off_loop(_generating_host, template, _HeldClient(held))
    first = {"models": list(RUN_MODELS), "n_variations": 3, "variation_model": WRITER, "k_runs": 2}
    campaign_id, before = await _campaign(evals, host, template=template, launches=[first])
    bundle, built = _read(host, campaign_id)
    flagged = sorted(entry.id for entry in built.caveat_list or [] if entry.kind == "late")
    assert flagged, "the fixture campaign was declared after its results"
    setup = _setup(bundle, built)
    candidate = next(arm.id for arm in built.arms if arm.id != "control")
    step = next_step(
        "confirm_on_fresh_cases",
        closes=flagged[0],
        arms=[candidate, "control"],
        readings=["extract/passed"],
        plan=Plan(4),
        setup=setup,
    )
    assert step.limits == ["this_report_stays_flagged", "control_may_read_late", "axis_may_not_generate"]
    held.clear()

    taken = await _follow(
        evals,
        host,
        step,
        {
            "name": "the confirmation",
            "behavior": "extraction",
            "question": "is the candidate extractor better than the control?",
            "aspects": [{"label": "extract", "design": _design()}],
            "floors": [],
            "variation_model": WRITER,
            "judge_model": None,
            "simulator_model": None,
            "overlays": None,
            "apparatus_settings": None,
            "cell_timeout_s": None,
            "measure_latency": None,
            "arm_schedule": None,
        },
        control_model=CONTROL_MODEL,
        before_results=held.set,
    )

    assert taken.campaign is not None
    _bundle_new, confirmed = _read(host, taken.campaign)
    assert not [entry.id for entry in confirmed.caveat_list or [] if entry.kind == "late"], (
        "declared before its results, its control named from its runs before any was scored"
    )
    assert not _cases(host, taken.runs) & _cases(host, [run for runs in before.values() for run in runs])
    _bundle_again, again = _read(host, campaign_id)
    assert sorted(entry.id for entry in again.caveat_list or [] if entry.kind == "late") == flagged, (
        "this report's caveat stays raised: the confirmation is read in the campaign it created"
    )
