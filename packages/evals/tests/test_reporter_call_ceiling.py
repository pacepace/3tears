"""A reporter arm whose largest generator call is reserved above its cap on its own is refused before anything is spent.

A reporter's generator call is known whole before launch: each case's frozen bundle, the preset's text and the
admitted model. Until the reporter priced it, an arm whose generator call's ceiling was above its cap passed the
estimate and the launch, and was refused only when its run reached the call, after the cases before it were
paid for. The host composes :func:`~threetears.evals.ops.price_largest_reporter_call` into its reporter entry's
``price_largest_call``, which builds each case's first generator request as the cell builds it and prices it on
the host's ``analysis``-role client. The estimate and the launch then refuse that arm alike, naming the cap one
call needs, with nothing called; a cap at or above the ceiling is admitted, and the call the run makes is the
one that was priced.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from typing import Any

import pytest

from threetears.evals.analysis import REPORTER_KIND, ReporterKind
from threetears.evals.kernel import ANALYSIS_GEN_PROMPT, ValidationFailedError
from threetears.evals.ops import (
    REPORTER_UNHONOURED_LAUNCH_ARGUMENTS,
    LaunchArguments,
    OpsHost,
    ReporterCaseFreeze,
    ReporterModelRole,
    admit_reporter_launch,
    launch_estimate,
    plan_reporter_arm,
    price_largest_reporter_call,
    reporter_case_freeze,
)
from threetears.evals.run import (
    ArmPrice,
    ArmQuote,
    KindWiring,
    LaunchableKind,
    LaunchHost,
    LaunchRequest,
    build_judge_service,
    launch_run,
    quote_launch,
    start_run,
)
from threetears.evals.schema.models import EvalTemplate, JudgedArtifact, RubricDim
from threetears.evals.schema.subject import SubjectSnapshot
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_INSTANT
from packages.evals.tests.ops_support import TOYHOST_SCOPE, MemoWriter, OpsFixture, ops_fixture
from packages.evals.tests.toyhost_memo import MODEL, PROMPT
from packages.evals.tests.off_loop import off_loop

TEMPLATE_ID = "toyhost-reporter"
SUBJECT = "the analysis writer"

#: What the host's generator-role client prices one generator request at.
_GENERATOR_CEILING_USD = 0.25

#: What the host's pricer predicted for the whole arm.
_PREDICTED_USD = 0.01


class _Prompts:
    """The host's prompt registry: the system default only."""

    def resolve(self, preset: str | None) -> str | None:
        return PROMPT if preset is None else None


def _resolve_model(role: ReporterModelRole, named: str | None) -> str:
    """The host's model resolution: the fixtured writer for both roles, every name as given."""
    return named if named is not None else MODEL


class _ReporterCampaign:
    """The toy host launching the analysis reporter over one frozen case, its generator priced per request."""

    def __init__(self) -> None:
        self.fixture: OpsFixture = ops_fixture(ceiling_usd=_GENERATOR_CEILING_USD)
        self.launched: list[str] = []
        self.eval_host = self.fixture.host.launch.eval_host
        self.eval_host.storage.save_template(
            EvalTemplate(
                id=TEMPLATE_ID,
                scope_id=TOYHOST_SCOPE,
                name="Memo reporter",
                intent="Judge the analysis memo against the evidence its writer read.",
                candidate_kind=REPORTER_KIND,
                rubric=[RubricDim(name="reporter.groundedness", description="Claims trace.", scale="ordinal")],
                created_at=TOYHOST_INSTANT,
                updated_at=TOYHOST_INSTANT,
            )
        )
        self.case_id = reporter_case_freeze(
            self.eval_host,
            ReporterCaseFreeze(template_id=TEMPLATE_ID, campaign_id=self.fixture.campaign.id),
            TOYHOST_SCOPE,
        ).test_case_id

        def priced(_quote: ArmQuote) -> ArmPrice:
            return ArmPrice(predicted_usd=_PREDICTED_USD, basis="from past reporter runs")

        self.corpus_run_ids = self.run_ids()
        holder: list[LaunchHost] = []
        launch = self.fixture.host.launch
        self.host: LaunchHost = dataclasses.replace(
            launch,
            kinds={**launch.kinds, REPORTER_KIND: self._entry(lambda: holder[0])},
            launch_pricer=priced,
        )
        holder.append(self.host)

    def _entry(self, launch_host: Any) -> LaunchableKind:
        eval_host = self.eval_host

        async def launch(request: LaunchRequest) -> Any:
            self.launched.append(request.candidate_model or "")
            admitted = off_loop(
                admit_reporter_launch,
                request,
                storage=eval_host.storage,
                prompts=_Prompts(),
                resolve_model=_resolve_model,
            )
            judge = off_loop(
                build_judge_service,
                eval_host,
                request.template,
                admitted.judge_model,
                request.judge_config_ids,
                judged_artifact=JudgedArtifact.DOCUMENT,
            )
            client = eval_host.completion_clients("a reporter run")("analysis", admitted.candidate_model)
            teardown = contextlib.AsyncExitStack()
            teardown.push_async_callback(client.aclose)
            kind = ReporterKind(
                prompt=admitted.prompt,
                prompt_id=ANALYSIS_GEN_PROMPT,
                prompt_version=None,
                client=client,
                model=admitted.candidate_model,
                host=eval_host,
            )
            return await launch_run(
                launch_host(),
                request,
                KindWiring(
                    kind_factory=lambda _cell: kind,
                    subject=SubjectSnapshot(subject_id=SUBJECT, subject_label=SUBJECT, state={}),
                    test_cases=admitted.cases,
                    judge=judge,
                    teardown=teardown,
                ),
            )

        return LaunchableKind(
            launch=launch,
            resolve_subject=lambda _subject, _scope: None,
            unhonoured_launch_arguments=REPORTER_UNHONOURED_LAUNCH_ARGUMENTS,
            plan_arm=lambda request: plan_reporter_arm(
                eval_host, request, prompts=_Prompts(), resolve_model=_resolve_model
            ),
            price_largest_call=lambda request: price_largest_reporter_call(
                eval_host, request, prompts=_Prompts(), resolve_model=_resolve_model
            ),
        )

    def _arguments(self, max_cost_usd: float) -> dict[str, Any]:
        return {
            "template_id": TEMPLATE_ID,
            "subject_id": SUBJECT,
            "models": [MODEL],
            "k_runs": 1,
            "max_cost_usd": max_cost_usd,
        }

    async def estimate(self, max_cost_usd: float) -> Any:
        [arm] = (await quote_launch(self.host, **self._arguments(max_cost_usd), scope_id=TOYHOST_SCOPE)).arms
        return arm

    async def ops_estimate(self, max_cost_usd: float) -> Any:
        arguments = LaunchArguments(**self._arguments(max_cost_usd))
        [arm] = (await launch_estimate(OpsHost(launch=self.host), arguments, TOYHOST_SCOPE)).arms
        return arm

    async def launch(self, max_cost_usd: float) -> list[Any]:
        runs = await start_run(self.host, **self._arguments(max_cost_usd), scope_id=TOYHOST_SCOPE)
        async with asyncio.timeout(10):
            while any(self.host.job_manager.is_active(run.id) for run in runs):
                await asyncio.sleep(0.01)
        return runs

    def generator_calls(self) -> list[dict[str, str]]:
        """Every request sent with the generator's system prompt — the judge's are the rest."""
        return [call for writer in self.fixture.writers for call in writer.calls if PROMPT[:40] in call["system"]]

    def run_ids(self) -> set[str]:
        return {run.id for run in self.eval_host.storage.query_eval_runs(TOYHOST_SCOPE)}

    def priced(self) -> list[dict[str, str]]:
        return [pair for writer in self.fixture.writers for pair in writer.priced]


def _writers(campaign: _ReporterCampaign) -> list[MemoWriter]:
    return campaign.fixture.writers


async def test_a_cap_below_one_generator_calls_ceiling_is_refused_by_the_estimate_and_the_launch_with_nothing_spent() -> (
    None
):
    campaign = off_loop(_ReporterCampaign)

    arm = await campaign.estimate(0.2)
    assert arm.outcome == "refused"
    assert arm.plan.call_ceiling.usd == pytest.approx(_GENERATOR_CEILING_USD)
    assert campaign.case_id in arm.plan.call_ceiling.call
    assert f"the generator call on case '{campaign.case_id}'" in arm.refusal
    assert "reserved at up to $0.2500 before it is sent, more than the whole cap" in arm.refusal
    assert arm.remedy.startswith("name max_cost_usd of at least $0.25, the cap one such call needs to be admitted")
    assert "a cheaper generator model" in arm.remedy, "the remedy names what lowers the ceiling, not only the cap"

    reported = await campaign.ops_estimate(0.2)
    assert (reported.outcome, reported.refusal, reported.remedy) == ("refused", arm.refusal, arm.remedy)
    assert reported.call_ceiling_usd == pytest.approx(_GENERATOR_CEILING_USD)

    with pytest.raises(ValidationFailedError) as refused:
        await campaign.launch(0.2)
    assert refused.value.message == arm.refusal, "the estimate refuses word for word what the launch refuses"
    assert refused.value.remedy == arm.remedy

    assert campaign.launched == [], "no launcher ran"
    assert campaign.priced(), "the arm's generator call was priced"
    assert all(writer.calls == [] for writer in _writers(campaign)), "no call was made"
    assert all(writer.closed == 1 for writer in _writers(campaign)), "every client built to price was released"
    assert off_loop(campaign.run_ids) == campaign.corpus_run_ids, "no run was stored beyond the campaign's own"


async def test_a_cap_at_the_ceiling_is_admitted_and_the_call_the_run_makes_is_the_one_priced() -> None:
    campaign = off_loop(_ReporterCampaign)

    arm = await campaign.estimate(_GENERATOR_CEILING_USD)
    assert (arm.outcome, arm.refusal) == ("admitted", None)
    assert arm.plan.call_ceiling.usd == pytest.approx(_GENERATOR_CEILING_USD)

    runs = await campaign.launch(_GENERATOR_CEILING_USD)
    assert len(runs) == 1 and campaign.launched == [MODEL]
    [sent] = campaign.generator_calls()
    priced = campaign.priced()
    assert priced[0] == sent, "the pricer built the request the cell sent, byte for byte"


def test_a_case_the_cell_makes_no_call_for_prices_nothing() -> None:
    """A case carrying no reporter case is excluded by the cell with no call made, so it holds no ceiling."""
    campaign = _ReporterCampaign()
    writer = MemoWriter(campaign.eval_host, campaign.fixture.campaign)
    writer.ceiling_usd = _GENERATOR_CEILING_USD
    kind = ReporterKind(
        prompt=PROMPT,
        prompt_id=ANALYSIS_GEN_PROMPT,
        prompt_version=None,
        client=writer,
        model=MODEL,
        host=campaign.eval_host,
    )
    [stored] = campaign.eval_host.storage.query_test_cases(TOYHOST_SCOPE, template_id=TEMPLATE_ID)
    assert kind.price_case(stored) == pytest.approx(_GENERATOR_CEILING_USD)
    unreadable = stored.model_copy(update={"host_payload": {}})
    assert kind.price_case(unreadable) is None
    assert len(writer.priced) == 1, "only the readable case's request was priced"
