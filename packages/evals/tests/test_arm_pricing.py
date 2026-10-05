"""Every arm of every launch is priced before any launcher runs, by one rule.

The engine used to price only a GENERATING launch's arms (from the kind's plan, through the host's
``launch_pricer``, held to the band's upper end), and left every other arm to the host — so one launch
could price two arms by two rules, and a host's own rule for a stored-case arm (a point estimate from one
result, blind to the judge and simulator pins and the rig) priced a session low. These pin the replacement
on the toy host, whose kind generates nothing and plans every arm over its stored cases:

- **A stored-case arm is quoted from its plan** — its stored case count, ``n_variations=0``,
  ``case_source='stored'`` — through the host's pricer, before any launcher runs.
- **Every refusal fires, and fires before any launcher runs**: predicted above the cap; unpriceable
  (no prediction, no pricer, or no plan) under an inherited cap. An unpriceable arm under a cap the launch
  named runs. The plan holds the launcher at the tail, and a plan the kind cannot make is refused before it.
- **The battery prices every template's arms once**, stored-case templates included, and its launches carry
  those plans rather than pricing again.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.contracts import EvalRun, EvalStorage, ValidationFailedError
from threetears.evals.contracts.models import EvalTemplate
from threetears.evals.run import (
    ArmPlan,
    ArmPrice,
    ArmQuote,
    LaunchHost,
    LaunchPricer,
    LaunchRequest,
    LaunchSettings,
    start_run,
    start_universal_battery,
)
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_COST_CEILING_USD, TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND
from packages.evals.tests.fixtures.toyhost.launch import (
    TOYHOST_LAUNCH_SETTINGS,
    plan_toyhost_arm,
    price_toyhost_arm,
    toyhost_launch_host,
)
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template, toyhost_test_cases

#: How many stored cases the toy kind plays per arm.
STORED_CASES = len(toyhost_test_cases(toyhost_template()))


class _Launched:
    """A toy host whose launcher, pricer and plan are recorded, so a refusal before the launcher is visible."""

    def __init__(
        self,
        *templates: EvalTemplate,
        pricer: LaunchPricer | None = price_toyhost_arm,
        priced: bool = True,
        plan: Any = plan_toyhost_arm,
        settings: LaunchSettings = TOYHOST_LAUNCH_SETTINGS,
    ) -> None:
        self.storage = EvalStorage(InMemoryDocumentStore())
        for template in templates or (toyhost_template(),):
            self.storage.save_template(template)
        toy, _client = toyhost_launch_host(storage=self.storage, settings=lambda: settings)
        (launchable,) = toy.kinds.values()
        self.quotes: list[ArmQuote] = []
        self.launched: list[LaunchRequest] = []

        def quoted(quote: ArmQuote) -> ArmPrice:
            self.quotes.append(quote)
            assert pricer is not None
            return pricer(quote)

        async def launch(request: LaunchRequest) -> EvalRun:
            self.launched.append(request)
            return await launchable.launch(request)

        self.host: LaunchHost = replace(
            toy,
            kinds={TOY_EXTRACTOR_KIND: replace(launchable, launch=launch, plan_arm=plan)},
            launch_pricer=quoted if priced and pricer is not None else None,
        )

    async def launch(self, **arguments: Any) -> list[EvalRun]:
        launched: dict[str, Any] = {"template_id": toyhost_template().id, "models": list(RUN_MODELS), **arguments}
        runs = await start_run(self.host, scope_id=TOYHOST_SCOPE, subject_id=TOYHOST_SUBJECT.subject_id, **launched)
        async with asyncio.timeout(10):
            while any(self.host.job_manager.is_active(run.id) for run in runs):
                await asyncio.sleep(0.01)
        return runs

    def assert_nothing_launched(self) -> None:
        assert self.launched == [], "refused before any launcher ran"
        assert self.storage.query_eval_runs(TOYHOST_SCOPE) == []
        assert self.host.job_manager.admitted_count == 0


def _at(usd: float | None, basis: str = "a rate card") -> LaunchPricer:
    def price(_quote: ArmQuote) -> ArmPrice:
        return ArmPrice(predicted_usd=usd, basis=basis)

    return price


# =============================================================================
# A stored-case arm is quoted, by the same port, before its launcher runs
# =============================================================================


async def test_a_stored_case_arm_is_quoted_from_its_plan_before_its_launcher_runs():
    launched = _Launched()
    (launchable,) = launched.host.kinds.values()
    assert "n_variations" in launchable.unhonoured_launch_arguments and launchable.plan_arm is not None, (
        "a kind that never generates plans its arms too"
    )

    runs = await launched.launch(k_runs=2)

    assert [(q.candidate_model, q.case_count, q.n_variations, q.case_source, q.k_runs) for q in launched.quotes] == [
        (model, STORED_CASES, 0, "stored", 2) for model in RUN_MODELS
    ]
    assert [q.template_id for q in launched.quotes] == [toyhost_template().id] * len(RUN_MODELS)
    assert [q.apparatus_settings for q in launched.quotes] == [{"reviewer_pool": "pool-a"}] * len(RUN_MODELS)
    assert len(launched.launched) == len(RUN_MODELS) and len(runs) == len(RUN_MODELS)
    assert all(request.arm_plan is not None for request in launched.launched), "the tail holds each arm to its plan"


async def test_every_arm_is_priced_before_the_first_launcher_runs():
    """The second arm over its cap refuses the launch before the first arm's launcher built anything."""

    def second_is_dear(quote: ArmQuote) -> ArmPrice:
        return ArmPrice(predicted_usd=9.5 if quote.candidate_model == RUN_MODELS[1] else 0.0, basis="a rate card")

    launched = _Launched(pricer=second_is_dear)

    with pytest.raises(ValidationFailedError, match=f"the arm on '{RUN_MODELS[1]}'"):
        await launched.launch()

    assert len(launched.quotes) == 2
    launched.assert_nothing_launched()


async def test_a_stored_case_arm_predicted_above_its_cap_is_refused_before_any_launcher_runs():
    launched = _Launched(pricer=_at(9.5, "from 3 past results"))

    with pytest.raises(
        ValidationFailedError,
        match=(
            f"the arm on '{RUN_MODELS[0]}' \\({STORED_CASES} stored case\\(s\\) x 1 repeat\\(s\\) of template "
            f"'{toyhost_template().id}'\\) is predicted to cost \\$9\\.50 \\(from 3 past results\\), above its "
            f"\\${TOYHOST_COST_CEILING_USD:.2f} cap.*Refused before any launcher ran"
        ),
    ):
        await launched.launch(k_runs=1)

    launched.assert_nothing_launched()


async def test_a_stored_case_arm_over_a_cap_the_launch_named_is_refused_too():
    """A named cap licenses an UNPRICED arm; one predicted above it is refused all the same."""
    launched = _Launched(pricer=_at(2.0))

    with pytest.raises(ValidationFailedError, match="predicted to cost \\$2\\.00.*above its \\$1\\.50 cap"):
        await launched.launch(max_cost_usd=1.5)

    launched.assert_nothing_launched()


@pytest.mark.parametrize(
    ("unpriceable", "said"),
    [
        ({"pricer": _at(None, "no priced result of this template on that model")}, "no priced result"),
        ({"priced": False}, "host 'toyhost' prices no launch \\(LaunchHost.launch_pricer\\)"),
        ({"plan": None}, f"kind '{TOY_EXTRACTOR_KIND}' plans no arm \\(LaunchableKind.plan_arm\\)"),
    ],
    ids=["no-prediction", "no-pricer", "no-plan"],
)
async def test_an_unpriceable_stored_case_arm_is_refused_under_an_inherited_cap_and_runs_under_a_chosen_one(
    unpriceable, said
):
    launched = _Launched(**unpriceable)

    with pytest.raises(
        ValidationFailedError,
        match=f"cannot be priced: {said}.*unknown, not \\$0.*inherited from max_cost_usd.*Refused before any launcher ran",
    ):
        await launched.launch()
    launched.assert_nothing_launched()

    runs = await launched.launch(max_cost_usd=1.5)
    assert [(run.max_cost_usd, run.max_cost_usd_origin) for run in runs] == [(1.5, "chosen")] * len(RUN_MODELS)


async def test_with_enforcement_off_no_arm_is_priced_and_every_arm_is_still_planned():
    def refuses(_quote: ArmQuote) -> ArmPrice:
        raise AssertionError("no cap is in force, so nothing is priced")

    launched = _Launched(
        pricer=refuses, settings=TOYHOST_LAUNCH_SETTINGS.model_copy(update={"enforcement_enabled": False})
    )

    runs = await launched.launch()

    assert len(runs) == len(RUN_MODELS) and launched.quotes == []
    assert all(request.arm_plan is not None for request in launched.launched)


async def test_a_stored_case_arm_freezing_more_cases_than_its_plan_is_refused_at_the_tail():
    """It was priced at its plan's count, so it may run fewer, never more."""

    def plans_one(request: LaunchRequest) -> ArmPlan:
        return replace(plan_toyhost_arm(request), case_count=1)

    launched = _Launched(plan=plans_one)

    with pytest.raises(
        ValueError, match=f"planned this arm at most 1 case\\(s\\) and its launcher froze {STORED_CASES}"
    ):
        await launched.launch(models=[RUN_MODELS[0]])

    assert launched.quotes[0].case_count == 1, "priced at the plan"
    assert launched.storage.query_eval_runs(TOYHOST_SCOPE) == []


async def test_an_arm_its_kind_cannot_plan_is_refused_before_its_launcher_runs():
    launched = _Launched()

    with pytest.raises(ValidationFailedError, match="has no default candidate model; name one"):
        await launched.launch(models=[])

    assert launched.quotes == []
    launched.assert_nothing_launched()


# =============================================================================
# The battery: every template's arms priced once, before any launches
# =============================================================================


async def _no_preflight(_subject_id: str, _models: Any) -> Any:
    async def check(_template: EvalTemplate, _cassette_mode: str) -> None:
        return None

    return check


def _universal(template_id: str) -> EvalTemplate:
    return toyhost_template().model_copy(update={"id": template_id, "universal": True})


async def test_a_battery_prices_every_arm_of_every_template_once():
    """The battery's pre-flight prices each arm; the launch it then makes carries that plan rather than pricing again."""
    launched = _Launched(_universal("first"), _universal("second"))

    run_ids = await start_universal_battery(
        launched.host,
        TOYHOST_SUBJECT.subject_id,
        scope_id=TOYHOST_SCOPE,
        models=list(RUN_MODELS),
        preflight=_no_preflight,
    )
    async with asyncio.timeout(10):
        while any(launched.host.job_manager.is_active(run_id) for run_id in run_ids):
            await asyncio.sleep(0.01)

    assert sorted((q.template_id, q.candidate_model) for q in launched.quotes) == sorted(
        (template, model) for template in ("first", "second") for model in RUN_MODELS
    ), "every arm quoted exactly once"
    assert len(run_ids) == 4 and all(request.arm_plan is not None for request in launched.launched)


async def test_a_battery_refuses_a_stored_case_template_over_its_cap_before_launching_any():
    def by_template(quote: ArmQuote) -> ArmPrice:
        return ArmPrice(predicted_usd=50.0 if quote.template_id == "dear" else 0.0, basis="a rate card")

    launched = _Launched(_universal("cheap"), _universal("dear"), pricer=by_template)

    with pytest.raises(ValidationFailedError, match="predicted to cost \\$50\\.00.*Nothing was launched"):
        await start_universal_battery(
            launched.host,
            TOYHOST_SUBJECT.subject_id,
            scope_id=TOYHOST_SCOPE,
            models=[RUN_MODELS[0]],
            preflight=_no_preflight,
        )

    launched.assert_nothing_launched()
