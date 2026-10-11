"""An arm whose largest call's reservation is above its cap on its own is refused before anything is spent.

A judge campaign on a flash model (6 cases, k=1, ``max_cost_usd=0.5``) was estimated at $0.023 and admitted, then
went ``budget_stopped`` after one result: "a call priced at up to $0.5801 would have taken the run to $0.5801
against a $0.5000 cap". A run reserves a priced call's ceiling before making it (``CellSink.reserve``) and refuses
the call when the ceiling would take it past its cap — so a call whose ceiling is above the whole cap can never be
made, whatever the run is predicted to spend, and the case reaching it is excluded after the cases before it were
paid for. Neither the estimate nor the launch said so.

The judge kind prices every case's call before launch, as its cell reserves it (the client's ``price_ceiling`` for
the case's prompt, times the judge's parse-retry attempts), and the arm carries the largest
(``ArmPlan.call_ceiling``). One rule refuses the arm when that ceiling is above its cap, for the estimate and the
launch alike, naming the cap one call needs and what lowers the ceiling. A cap at or above it is admitted.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any

import pytest

from threetears.evals.kernel import (
    JUDGE_KIND,
    JUDGE_KIND_CONTRACT,
    JudgeCase,
    JUDGE_CASE_KEY,
    JudgeCaseSource,
    ValidationFailedError,
    judge_case_payload,
)
from threetears.evals.ops import LaunchArguments, OpsHost, launch_estimate
from threetears.evals.run import (
    JUDGE_CALL_ATTEMPTS,
    ArmPrice,
    ArmQuote,
    LaunchHost,
    launchable_judge_kind,
    quote_launch,
    start_run,
)
from threetears.evals.schema import EvalTemplate, EvalTestCase, JudgedArtifact, JudgeEvidence
from threetears.evals.schema.models import TRANSCRIPT_DIM_ID
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.judge import ToyJudgeCompletion
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: The judge under test, as the report named it.
_FLASH = "google/gemini-3.8-flash"

#: The template the judge cases are frozen into.
_TEMPLATE_ID = "dj_judge_calibration"

#: The evidence that makes one case's prompt the largest of the six.
_LONG = "LONG-EVIDENCE"

#: What the host's client prices one request of the largest case at, and of every other one.
_LONG_REQUEST_USD = 0.29005
_SHORT_REQUEST_USD = 0.0019

#: What the run reserves for the largest case's call: every attempt the judge's parse retry can make.
_LARGEST_CALL_USD = _LONG_REQUEST_USD * JUDGE_CALL_ATTEMPTS

#: What the host's pricer predicted for the whole arm.
_PREDICTED_USD = 0.023


class _PricedJudge:
    """A judge-role client that prices each request by its prompt, and records every call it is sent."""

    def __init__(self) -> None:
        self.priced: list[str] = []
        self.calls: list[str] = []
        self.model_name = _FLASH

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float:
        self.priced.append(user)
        return _LONG_REQUEST_USD if _LONG in user else _SHORT_REQUEST_USD

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        self.calls.append(user)
        return ToyJudgeCompletion(
            content=json.dumps({"reasoning": "read it", "criteria_scores": {TRANSCRIPT_DIM_ID: 3}}),
            input_tokens=100,
            output_tokens=40,
            cost_usd=0.001,
            price_source="a rate card",
            model=_FLASH,
        )

    async def aclose(self) -> None:
        """Nothing to release."""


def _judge_case(index: int, artifact: str) -> EvalTestCase:
    case = JudgeCase(
        dim=TRANSCRIPT_DIM_ID,
        scale="ordinal",
        intent="Answer the member's question as the bot would.",
        judged_artifact=JudgedArtifact.TRANSCRIPT,
        evidence=JudgeEvidence(case_material="The member asked about the raid schedule.", artifact=artifact),
        source=JudgeCaseSource(run_id="source-run", result_id=f"result-{index}", test_case_id=f"source-case-{index}"),
    )
    return EvalTestCase(
        id=f"judge-case-{index}",
        scope_id=TOYHOST_SCOPE,
        template_id=_TEMPLATE_ID,
        variation_params={},
        host_payload=judge_case_payload(case),
    )


class _JudgeCampaign:
    """The toy host launching the judge kind over six frozen cases, the fourth the largest."""

    def __init__(self, *, host_ceiling_usd: float | None = None) -> None:
        self.client = _PricedJudge()
        settings = (
            TOYHOST_LAUNCH_SETTINGS
            if host_ceiling_usd is None
            else TOYHOST_LAUNCH_SETTINGS.model_copy(update={"max_cost_usd": host_ceiling_usd})
        )
        self.quotes: list[ArmQuote] = []
        profile = toyhost_profile()
        toy, _ = toyhost_launch_host(
            profile=dataclasses.replace(
                profile, kinds=(*profile.kinds, dataclasses.replace(JUDGE_KIND_CONTRACT, seats=None))
            ),
            clients=lambda _role, _model, *, temperature=None: self.client,  # type: ignore[arg-type, return-value]
            settings=lambda: settings,
        )

        def priced(quote: ArmQuote) -> ArmPrice:
            self.quotes.append(quote)
            return ArmPrice(predicted_usd=_PREDICTED_USD, basis="from 6 priced past results")

        holder: list[LaunchHost] = []
        self.host: LaunchHost = dataclasses.replace(
            toy, kinds={JUDGE_KIND: launchable_judge_kind(lambda: holder[0])}, launch_pricer=priced
        )
        holder.append(self.host)
        storage = self.host.eval_host.storage
        storage.save_template(
            EvalTemplate(
                id=_TEMPLATE_ID,
                scope_id=TOYHOST_SCOPE,
                name="judge calibration",
                intent="Score as a person would.",
                candidate_kind=JUDGE_KIND,
            )
        )
        for index in range(6):
            storage.save_test_case(_judge_case(index, f"{_LONG} the whole log" if index == 3 else "a short reply"))

    async def estimate(self, max_cost_usd: float) -> Any:
        [arm] = (
            await quote_launch(
                self.host,
                template_id=_TEMPLATE_ID,
                subject_id="the flash judge",
                models=[_FLASH],
                k_runs=1,
                max_cost_usd=max_cost_usd,
                scope_id=TOYHOST_SCOPE,
            )
        ).arms
        return arm

    async def ops_estimate(self, max_cost_usd: float) -> Any:
        arguments = LaunchArguments(
            template_id=_TEMPLATE_ID,
            subject_id="the flash judge",
            models=[_FLASH],
            k_runs=1,
            max_cost_usd=max_cost_usd,
        )
        [arm] = (await launch_estimate(OpsHost(launch=self.host), arguments, TOYHOST_SCOPE)).arms
        return arm

    async def launch(self, max_cost_usd: float) -> list[Any]:
        runs = await start_run(
            self.host,
            template_id=_TEMPLATE_ID,
            subject_id="the flash judge",
            models=[_FLASH],
            k_runs=1,
            max_cost_usd=max_cost_usd,
            scope_id=TOYHOST_SCOPE,
        )
        async with asyncio.timeout(10):
            while any(self.host.job_manager.is_active(run.id) for run in runs):
                await asyncio.sleep(0.01)
        return runs


async def test_a_cap_below_one_calls_ceiling_is_refused_by_the_estimate_and_the_launch_with_nothing_spent() -> None:
    campaign = _JudgeCampaign()

    arm = await campaign.estimate(0.5)
    assert arm.outcome == "refused"
    assert arm.plan.call_ceiling.usd == pytest.approx(_LARGEST_CALL_USD)
    assert "judge-case-3" in arm.plan.call_ceiling.call, "the largest call is the case with the most evidence"
    assert arm.refusal.startswith(
        f"the arm on '{_FLASH}' (6 stored case(s) x 1 repeat(s) of template '{_TEMPLATE_ID}') cannot make its "
        "largest call under its $0.50 cap: the judge call on case 'judge-case-3'"
    )
    assert "reserved at up to $0.5801 before it is sent, more than the whole cap" in arm.refusal
    assert "Refused before any launcher ran" in arm.refusal
    assert arm.remedy.startswith("name max_cost_usd of at least $0.59, the cap one such call needs to be admitted")
    assert "$0.61 also leaves room for the $0.02 the arm is predicted to spend around it" in arm.remedy
    assert "a cheaper judge model" in arm.remedy, "the remedy names what lowers the ceiling, not only the cap"

    reported = await campaign.ops_estimate(0.5)
    assert (reported.outcome, reported.refusal, reported.remedy) == ("refused", arm.refusal, arm.remedy)
    assert reported.call_ceiling_usd == pytest.approx(_LARGEST_CALL_USD)

    with pytest.raises(ValidationFailedError) as refused:
        await campaign.launch(0.5)
    assert refused.value.message == arm.refusal, "the estimate refuses word for word what the launch refuses"
    assert refused.value.remedy == arm.remedy

    assert campaign.client.calls == [], "no call was made"
    assert campaign.host.eval_host.storage.query_eval_runs(TOYHOST_SCOPE) == []
    assert campaign.host.job_manager.admitted_count == 0


async def test_a_cap_at_or_above_one_calls_ceiling_is_admitted() -> None:
    campaign = _JudgeCampaign()

    arm = await campaign.estimate(0.59)
    assert (arm.outcome, arm.refusal) == ("admitted", None)
    assert arm.plan.call_ceiling.usd == pytest.approx(_LARGEST_CALL_USD)

    runs = await campaign.launch(0.59)
    assert len(runs) == 1
    assert len(campaign.client.calls) == 6, "every case's call was admitted and made"


async def test_a_call_above_the_hosts_own_ceiling_is_named_as_the_hosts_to_raise() -> None:
    campaign = _JudgeCampaign(host_ceiling_usd=0.5)

    arm = await campaign.estimate(0.5)
    assert arm.outcome == "refused"
    assert arm.remedy.startswith(
        "one such call needs a cap of at least $0.59, above the host's $0.50 ceiling, which a launch cannot raise"
    )
    assert "a cheaper judge model" in arm.remedy


async def test_a_case_the_judge_kind_cannot_read_is_refused_by_name_rather_than_dropped() -> None:
    """A stored case this build cannot read used to be left out of the arm without a word: a set nobody chose."""
    campaign = _JudgeCampaign()
    campaign.host.eval_host.storage.save_test_case(
        EvalTestCase(
            id="judge-case-unreadable",
            scope_id=TOYHOST_SCOPE,
            template_id=_TEMPLATE_ID,
            variation_params={},
            host_payload={**_judge_case(9, "a short reply").host_payload, JUDGE_CASE_KEY: {"dim": 7}},
        )
    )

    with pytest.raises(ValidationFailedError) as estimated:
        await campaign.estimate(1.0)
    assert "1 case(s) the judge kind cannot ask" in estimated.value.message
    assert "judge-case-unreadable" in estimated.value.message
    assert "judge_cases_freeze" in estimated.value.remedy

    with pytest.raises(ValidationFailedError) as launched:
        await campaign.launch(1.0)
    assert launched.value.message == estimated.value.message, "the estimate refuses what the launch refuses"
    assert campaign.client.calls == [], "no call was made"
    assert campaign.host.eval_host.storage.query_eval_runs(TOYHOST_SCOPE) == []
