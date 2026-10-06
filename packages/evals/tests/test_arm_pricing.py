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
  those plans rather than pricing again; it prepares every template before starting any, and launches each
  template as it priced it.
- **An arm is priced by the judges it will be scored by** (``plan_judge``: a dim's config can override the
  pin) and held to them, and to its simulator, at the tail.
- **A kind's plan refusals come before any pricing refusal**, and an arm of a kind that plans nothing is told
  its launch may be missing more than a price.
- **No arm reaches the tail unpriced under an enforced cap**, however the launch was composed.

Mutations that turn this file red (each applied to a saved copy and restored): the tail's judge, simulator or
unpriced check removed; planning's judge-pin check removed; arms priced before they are planned; the quote
handed ``judge=None``; the battery starting each template's group as soon as it is prepared; the battery
re-dispatching its template at launch.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.contracts import EvalRun, EvalStorage, JudgedArtifact, RubricDim, ValidationFailedError
from threetears.evals.contracts.judge_attribution import judges_sharing_a_candidate_model
from threetears.evals.contracts.models import EvalTemplate
from threetears.evals.run import (
    ArmPlan,
    ArmPrice,
    ArmQuote,
    KindWiring,
    LaunchGroup,
    LaunchHost,
    LaunchPricer,
    LaunchRequest,
    LaunchSettings,
    PlannedJudge,
    build_judge_service,
    launch_as_group,
    launch_run,
    plan_judge,
    price_arms,
    quote_launch,
    require_candidate_model,
    resolve_judge_pin,
    start_run,
    start_universal_battery,
)
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_judge_config
from packages.evals.tests.fixtures.toyhost.contract import TOY_EXTRACTOR_CONTRACT
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


async def test_a_cap_over_the_hosts_tells_the_operator_only_the_host_can_raise_it():
    """An arm over the host's own ceiling is not told "a larger max_cost_usd raises the cap" — a launch cannot."""
    launched = _Launched(pricer=_at(9.0))

    with pytest.raises(ValidationFailedError, match="the cap is the host's ceiling, which a launch cannot raise"):
        await launched.launch()
    with pytest.raises(ValidationFailedError, match=f"up to the host's \\${TOYHOST_COST_CEILING_USD:.2f} ceiling"):
        await launched.launch(max_cost_usd=1.5)

    launched.assert_nothing_launched()


# =============================================================================
# A launch's override may only LOWER the host's ceiling — refused on every launch surface, before anything is paid
# =============================================================================

_ABOVE_THE_HOSTS_USD = TOYHOST_COST_CEILING_USD + 0.5


async def test_a_launch_naming_a_cap_above_the_hosts_is_refused_before_anything_is_priced():
    launched = _Launched()

    with pytest.raises(ValidationFailedError, match=f"max_cost_usd={_ABOVE_THE_HOSTS_USD} is above the host's ceiling"):
        await launched.launch(max_cost_usd=_ABOVE_THE_HOSTS_USD)

    assert launched.quotes == [], "refused on its arguments, before any arm was priced"
    launched.assert_nothing_launched()
    # At the ceiling is a choice, not a raise.
    runs = await launched.launch(max_cost_usd=TOYHOST_COST_CEILING_USD)
    assert {(run.max_cost_usd, run.max_cost_usd_origin) for run in runs} == {(TOYHOST_COST_CEILING_USD, "chosen")}


async def test_a_launch_naming_a_metered_ceiling_above_the_hosts_is_refused():
    launched = _Launched()
    above = (TOYHOST_LAUNCH_SETTINGS.max_metered_calls or 0) + 1

    with pytest.raises(ValidationFailedError, match=f"max_metered_calls={above} is above the host's ceiling"):
        await launched.launch(max_metered_calls=above)

    launched.assert_nothing_launched()


async def test_the_lower_only_rule_holds_with_enforcement_off():
    """The argument's contract does not change with the host's enforcement switch."""
    launched = _Launched(settings=TOYHOST_LAUNCH_SETTINGS.model_copy(update={"enforcement_enabled": False}))

    with pytest.raises(ValidationFailedError, match="a launch may only lower the host's ceiling"):
        await launched.launch(max_cost_usd=_ABOVE_THE_HOSTS_USD)

    launched.assert_nothing_launched()


async def test_a_quote_naming_a_cap_above_the_hosts_is_refused():
    launched = _Launched()

    with pytest.raises(ValidationFailedError, match="a launch may only lower the host's ceiling"):
        await quote_launch(
            launched.host,
            template_id=toyhost_template().id,
            subject_id=TOYHOST_SUBJECT.subject_id,
            models=list(RUN_MODELS),
            max_cost_usd=_ABOVE_THE_HOSTS_USD,
            scope_id=TOYHOST_SCOPE,
        )

    assert launched.quotes == []


async def test_a_battery_naming_a_cap_above_the_hosts_is_refused_before_launching_any():
    launched = _Launched(_universal("first"), _universal("second"))

    with pytest.raises(ValidationFailedError, match="a launch may only lower the host's ceiling.*Nothing was launched"):
        await start_universal_battery(
            launched.host,
            TOYHOST_SUBJECT.subject_id,
            scope_id=TOYHOST_SCOPE,
            models=[RUN_MODELS[0]],
            max_cost_usd=_ABOVE_THE_HOSTS_USD,
            preflight=_no_preflight,
        )

    assert launched.quotes == []
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


# =============================================================================
# The plan states the judges and simulator the arm runs under, and the arm is priced and held by them
# =============================================================================

#: A dim the toy template is scored on, for the judged arms below.
_DIM = "toy.fidelity"


def _judged_template() -> EvalTemplate:
    return toyhost_template().model_copy(
        update={"rubric": [RubricDim(name=_DIM, description="the extraction is faithful", scale="ordinal")]}
    )


class _Judged:
    """The toy host as a judged kind: a launcher that builds its judge from the request, and a plan of its own."""

    def __init__(
        self,
        *,
        plan_judge_model: str | None = None,
        wired_simulator: str | None = None,
        role_default: str = "judge-default",
        alternate: str | None = None,
        launcher_steps: bool = True,
    ) -> None:
        self.storage = EvalStorage(InMemoryDocumentStore())
        self.template = _judged_template()
        self.storage.save_template(self.template)
        settings = TOYHOST_LAUNCH_SETTINGS.model_copy(update={"judge_alternate_model": alternate})
        toy, _client = toyhost_launch_host(
            storage=self.storage, clients=lambda role, model, **_: object(), settings=lambda: settings
        )
        (launchable,) = toy.kinds.values()
        self.quotes: list[ArmQuote] = []
        self.launched: list[LaunchRequest] = []
        eval_host = toy.eval_host

        def plan(request: LaunchRequest) -> ArmPlan:
            model = require_candidate_model(request, None)
            judge = plan_judge(
                eval_host,
                request.template,
                plan_judge_model or resolve_judge_pin(request, role_default, candidate_model=model),
                request.judge_config_ids,
                judged_artifact=JudgedArtifact.DOCUMENT,
            )
            return ArmPlan(
                case_count=STORED_CASES,
                candidate_model=require_candidate_model(request, None),
                judge=judge,
                simulator_model=None,
            )

        def quoted(quote: ArmQuote) -> ArmPrice:
            self.quotes.append(quote)
            return price_toyhost_arm(quote)

        async def launch(request: LaunchRequest) -> EvalRun:
            self.launched.append(request)
            model = require_candidate_model(request, None)
            pin = (
                resolve_judge_pin(request, role_default, candidate_model=model)
                if launcher_steps
                else request.judge_model or role_default
            )
            judge = build_judge_service(
                eval_host,
                request.template,
                pin,
                request.judge_config_ids,
                judged_artifact=JudgedArtifact.DOCUMENT,
            )
            return await launch_run(
                self.host,
                request,
                KindWiring(
                    kind_factory=lambda _cell: None,  # type: ignore[arg-type,return-value]
                    subject=TOYHOST_SUBJECT,
                    test_cases=toyhost_test_cases(request.template),
                    judge=judge,
                    simulator_model=wired_simulator,
                ),
            )

        self.host: LaunchHost = replace(
            toy,
            kinds={
                TOY_EXTRACTOR_KIND: replace(
                    launchable,
                    launch=launch,
                    plan_arm=plan,
                    unhonoured_launch_arguments=frozenset({"cassette_mode", "n_variations", "simulator_model"}),
                )
            },
            launch_pricer=quoted,
        )

    async def launch(self, **arguments: Any) -> list[EvalRun]:
        launched: dict[str, Any] = {"template_id": self.template.id, "models": [RUN_MODELS[0]], **arguments}
        return await start_run(self.host, scope_id=TOYHOST_SCOPE, subject_id=TOYHOST_SUBJECT.subject_id, **launched)


def test_plan_judge_resolves_the_model_each_dim_is_scored_by_as_the_judge_service_does():
    """A dim whose config names its own model is scored by that model, whatever the run-level pin."""
    judged = _Judged()
    config = make_judge_config(scope_id=TOYHOST_SCOPE, rubric_dim_id=_DIM, model="judge-dear")
    judged.storage.save_judge_config(config)
    eval_host = judged.host.eval_host

    planned = plan_judge(eval_host, judged.template, "judge-cheap", judged_artifact=JudgedArtifact.DOCUMENT)

    assert planned == PlannedJudge(
        model="judge-cheap", effective_judges={_DIM: "judge-dear"}, config_ids={_DIM: config.id}
    )
    built = build_judge_service(eval_host, judged.template, "judge-cheap", judged_artifact=JudgedArtifact.DOCUMENT)
    assert built.planned == planned, "one config load and one cascade for the plan and the service"
    with pytest.raises(ValidationFailedError, match="which this template does not score"):
        plan_judge(
            eval_host,
            judged.template,
            "judge-cheap",
            {"toy.unscored": config.id},
            judged_artifact=JudgedArtifact.DOCUMENT,
        )


async def test_an_arm_is_quoted_with_the_judges_its_plan_resolved():
    judged = _Judged()
    config = make_judge_config(scope_id=TOYHOST_SCOPE, rubric_dim_id=_DIM, model="judge-dear")
    judged.storage.save_judge_config(config)

    runs = await judged.launch(judge_model="judge-cheap")

    (quote,) = judged.quotes
    assert quote.judge is not None and dict(quote.judge.effective_judges) == {_DIM: "judge-dear"}
    assert dict(quote.judge.config_ids) == {_DIM: config.id} and quote.judge.model == "judge-cheap"
    assert len(runs) == 1 and runs[0].effective_judges == {_DIM: "judge-dear"}, "it ran as it was priced"
    async with asyncio.timeout(10):
        while judged.host.job_manager.is_active(runs[0].id):
            await asyncio.sleep(0.01)


async def test_an_arm_whose_launcher_wires_other_judges_than_its_plan_is_refused_at_the_tail():
    judged = _Judged(plan_judge_model="judge-planned")

    with pytest.raises(ValueError, match="planned this arm judged by .*judge-planned.* and its launcher wired"):
        await judged.launch()

    assert judged.storage.query_eval_runs(TOYHOST_SCOPE) == []


async def test_an_arm_whose_launcher_resolves_another_simulator_than_its_plan_is_refused_at_the_tail():
    judged = _Judged(wired_simulator="sim-unplanned")

    with pytest.raises(
        ValueError, match="planned this arm's simulator on None and its launcher resolved 'sim-unplanned'"
    ):
        await judged.launch()

    assert judged.storage.query_eval_runs(TOYHOST_SCOPE) == []


async def test_a_plan_contradicting_the_launchs_judge_pin_is_refused_before_any_arm_is_priced():
    judged = _Judged(plan_judge_model="judge-planned")

    with pytest.raises(ValueError, match="pinned to judge 'judge-named' under 'judge-planned'"):
        await judged.launch(judge_model="judge-named")

    assert judged.quotes == [] and judged.launched == []


# =============================================================================
# A kind's request-level refusals come before every pricing refusal
# =============================================================================


async def test_a_kinds_plan_refusal_comes_before_cannot_be_priced_on_a_host_that_prices_nothing():
    """The operator who forgot a model hears that, not "cannot be priced — name max_cost_usd"."""
    launched = _Launched(priced=False)

    with pytest.raises(ValidationFailedError) as refused:
        await launched.launch(models=[])

    assert str(refused.value) == f"kind '{TOY_EXTRACTOR_KIND}' has no default candidate model; name one"
    launched.assert_nothing_launched()


async def test_an_arm_of_a_kind_that_plans_nothing_is_told_its_launch_may_be_missing_more():
    launched = _Launched(plan=None)

    with pytest.raises(
        ValidationFailedError, match="The launch may also be missing something its kind needs: a kind that plans no arm"
    ):
        await launched.launch()

    launched.assert_nothing_launched()


async def test_a_quote_reports_every_outcome_the_rule_can_reach_and_launches_nothing():
    unpriced = _Launched(pricer=_at(None, "nothing to go on"))
    chosen = await quote_launch(
        unpriced.host,
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=list(RUN_MODELS),
        max_cost_usd=1.5,
        scope_id=TOYHOST_SCOPE,
    )
    off = _Launched(settings=TOYHOST_LAUNCH_SETTINGS.model_copy(update={"enforcement_enabled": False}))
    uncapped = await quote_launch(
        off.host,
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=list(RUN_MODELS),
        scope_id=TOYHOST_SCOPE,
    )

    assert {arm.outcome for arm in chosen.arms} == {"unpriced-under-chosen-cap"}
    assert {(arm.cap_usd, arm.cap_origin) for arm in chosen.arms} == {(1.5, "chosen")}
    assert {arm.outcome for arm in uncapped.arms} == {"uncapped"} and len(off.quotes) == len(RUN_MODELS), (
        "a quote prices every arm even where the launch would price none"
    )
    unpriced.assert_nothing_launched()
    off.assert_nothing_launched()


@pytest.mark.parametrize(
    ("bad", "said"),
    [
        ({"predicted_usd": None, "central_usd": 1.0, "basis": "b"}, "upper end, which is the prediction"),
        ({"predicted_usd": 1.0, "central_usd": 2.0, "basis": "b"}, "low <= central <= predicted"),
        ({"predicted_usd": 2.0, "central_usd": 1.0, "low_usd": 1.5, "basis": "b"}, "low <= central <= predicted"),
        ({"predicted_usd": 2.0, "low_usd": -1.0, "basis": "b"}, "0 or more"),
    ],
)
def test_an_arm_price_refuses_a_range_out_of_order(bad, said):
    with pytest.raises(ValueError, match=said):
        ArmPrice(**bad)
    assert ArmPrice(predicted_usd=2.0, central_usd=1.0, low_usd=0.5, basis="b").central_usd == 1.0


# =============================================================================
# An unpriced arm cannot reach the tail under an enforced cap, however the launch was composed
# =============================================================================


async def test_a_launch_composed_through_launch_as_group_must_price_its_arms():
    launched = _Launched()
    host = launched.host
    (launchable,) = host.kinds.values()
    template = toyhost_template()

    def request(group: LaunchGroup) -> LaunchRequest:
        return LaunchRequest(
            template=template,
            kind=TOY_EXTRACTOR_KIND,
            subject_id=TOYHOST_SUBJECT.subject_id,
            candidate_model=RUN_MODELS[0],
            k_runs=1,
            scope_id=TOYHOST_SCOPE,
            n_variations=0,
            variation_model=None,
            judge_model=None,
            judge_config_ids=None,
            simulator_model=None,
            cassette_mode="off",
            cassette_corpus_id=None,
            overlays=TOY_EXTRACTOR_CONTRACT.validate_overlays({}),
            kind_spec=TOY_EXTRACTOR_CONTRACT.validate_spec(template.kind_spec),
            max_cost_usd=None,
            max_metered_calls=None,
            apparatus_settings={"reviewer_pool": "pool-a"},
            generation_budget=None,
            arm_plan=None,
            arm_price=None,
            launch_group=group,
            settings=TOYHOST_LAUNCH_SETTINGS,
        )

    async def form() -> tuple[LaunchGroup, None]:
        return LaunchGroup(candidate_models=[RUN_MODELS[0]]), None

    async def unpriced(group: LaunchGroup, _formed: None) -> list[EvalRun]:
        return [await launchable.launch(request(group))]

    async def priced(group: LaunchGroup, _formed: None) -> list[EvalRun]:
        return [await launchable.launch(arm) for arm in await price_arms(host, launchable, [request(group)])]

    with pytest.raises(ValueError, match="reached the launch tail unpriced under an enforced cap.*price_arms"):
        await launch_as_group(host, 1, settings=TOYHOST_LAUNCH_SETTINGS, form=form, prepare=unpriced, event="t")
    assert launched.storage.query_eval_runs(TOYHOST_SCOPE) == []

    (run,) = await launch_as_group(host, 1, settings=TOYHOST_LAUNCH_SETTINGS, form=form, prepare=priced, event="t")
    assert [q.candidate_model for q in launched.quotes] == [RUN_MODELS[0]]
    async with asyncio.timeout(10):
        while host.job_manager.is_active(run.id):
            await asyncio.sleep(0.01)


# =============================================================================
# The battery: every template prepared before any starts, each launching the template it priced
# =============================================================================


async def test_a_battery_starts_no_template_when_a_later_templates_preparation_refuses():
    """A refusal only a launcher can make, on the second template, used to leave the first template's runs running."""
    launched = _Launched(_universal("first"), _universal("second"))
    (launchable,) = launched.host.kinds.values()
    prepared: list[str] = []

    async def refuses_second(request: LaunchRequest) -> EvalRun:
        prepared.append(request.template.id)
        if request.template.id == "second":
            raise ValidationFailedError("the second template's subject cannot be captured")
        return await launchable.launch(request)

    host = replace(launched.host, kinds={TOY_EXTRACTOR_KIND: replace(launchable, launch=refuses_second)})

    with pytest.raises(ValidationFailedError, match="cannot be captured"):
        await start_universal_battery(
            host, TOYHOST_SUBJECT.subject_id, scope_id=TOYHOST_SCOPE, models=[RUN_MODELS[0]], preflight=_no_preflight
        )

    assert sorted(set(prepared)) == ["first", "second"], "the first template was prepared, and abandoned"
    assert launched.storage.query_eval_runs(TOYHOST_SCOPE) == [] and host.job_manager.active_count == 0
    assert host.job_manager.admitted_count == 0


async def test_a_battery_launches_the_template_it_priced_not_one_edited_since():
    """A template edited between the pre-flight and its launch would otherwise launch under plans made for another."""
    launched = _Launched(_universal("first"))

    async def edits_the_template(_subject_id: str, _models: Any) -> Any:
        async def check(template: EvalTemplate, _cassette_mode: str) -> None:
            launched.storage.save_template(template.model_copy(update={"candidate_kind": "no-launcher-for-this"}))

        return check

    run_ids = await start_universal_battery(
        launched.host,
        TOYHOST_SUBJECT.subject_id,
        scope_id=TOYHOST_SCOPE,
        models=[RUN_MODELS[0]],
        preflight=edits_the_template,
    )
    async with asyncio.timeout(10):
        while any(launched.host.job_manager.is_active(run_id) for run_id in run_ids):
            await asyncio.sleep(0.01)

    assert len(run_ids) == 1 and launched.launched[0].template.candidate_kind == TOY_EXTRACTOR_KIND


# =============================================================================
# A candidate does not judge its own output when an alternate judge stands ready
# =============================================================================


async def _judged_run(judged: _Judged, **arguments: Any) -> EvalRun:
    """Launch, wait for every run, and return the first arm's."""
    runs = await judged.launch(**arguments)
    async with asyncio.timeout(10):
        while any(judged.host.job_manager.is_active(run.id) for run in runs):
            await asyncio.sleep(0.01)
    return runs[0]


async def test_an_inherited_judge_on_a_candidates_model_steps_to_the_alternate():
    judged = _Judged(role_default=RUN_MODELS[0], alternate="judge-alternate")

    run = await _judged_run(judged)

    # Recorded as the alternate it is, not as the role default it stepped off: re-running the same arguments
    # picks the alternate setting, and only while the default is still a candidate.
    assert run.judge_model == "judge-alternate" and run.model_role_provenance == {"judge": "alternate"}
    assert run.effective_judges == {_DIM: "judge-alternate"}
    (quote,) = judged.quotes
    assert quote.judge is not None and quote.judge.model == "judge-alternate", "priced by the judge that scores"
    assert judges_sharing_a_candidate_model(run.effective_judges, [run.candidate_model]) == {}


@pytest.mark.parametrize(
    ("judged", "arguments", "judge", "origin"),
    [
        # Not a candidate: the default stands.
        ({"role_default": "judge-default", "alternate": "judge-alternate"}, {}, "judge-default", "inherited"),
        # A judge the launch named is a choice, recorded and never overridden, even on a candidate's model.
        (
            {"role_default": "judge-default", "alternate": "judge-alternate"},
            {"judge_model": RUN_MODELS[0]},
            RUN_MODELS[0],
            "chosen",
        ),
        # No alternate configured, or one that is itself a candidate: the default stands and the overlap is disclosed.
        ({"role_default": RUN_MODELS[0], "alternate": None}, {}, RUN_MODELS[0], "inherited"),
        # The alternate is a SIBLING arm's model: it is a candidate of the launch, so it may not judge either.
        (
            {"role_default": RUN_MODELS[0], "alternate": RUN_MODELS[1]},
            {"models": list(RUN_MODELS[:2])},
            RUN_MODELS[0],
            "inherited",
        ),
    ],
    ids=["default-not-a-candidate", "named-judge-on-a-candidate", "no-alternate", "alternate-is-a-candidate"],
)
async def test_the_judge_stays_where_no_usable_alternate_or_a_choice_applies(judged, arguments, judge, origin):
    run = await _judged_run(_Judged(**judged), **arguments)

    assert run.judge_model == judge
    assert run.model_role_provenance == {"judge": origin}
    shared = judges_sharing_a_candidate_model(run.effective_judges, [run.candidate_model])
    assert (shared == {_DIM: judge}) == (judge == RUN_MODELS[0]), "an overlap left in place is disclosed"


async def test_a_launcher_that_keeps_a_candidate_judge_where_an_alternate_stood_ready_is_refused_at_the_tail():
    judged = _Judged(
        role_default=RUN_MODELS[0], alternate="judge-alternate", launcher_steps=False, plan_judge_model=RUN_MODELS[0]
    )

    with pytest.raises(
        ValueError, match="kept the inherited judge .* one of the launch's candidates.*resolve_judge_pin"
    ):
        await judged.launch()

    assert judged.storage.query_eval_runs(TOYHOST_SCOPE) == []
