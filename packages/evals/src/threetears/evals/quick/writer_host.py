"""A host that launches the report writer over the quick path's own store: the writer measured as the thing under test.

The quick path's host (:func:`~threetears.evals.quick.one_call.callable_host`) declares the callable kinds only, and
builds a private launching host for each run. The writer is another kind, so it gets a host of its own, built here
over the same store, with nothing of the callable host changed:

- **its profile** declares the writer's kind contract and its measures
  (:data:`~threetears.evals.ops.report_writer.writer_contract.WRITER_KIND_CONTRACT`,
  :data:`~threetears.evals.ops.report_writer.writer_contract.WRITER_KIND_MEASURES`) on the engine's shared core, which
  seats the judge and the spend ceiling a writer run fills;
- **its launch registry** holds one entry,
  :func:`~threetears.evals.actions.report_writing.launchable_report_writer_kind`, and its template checks admit
  everything, since the writer's launch refuses a template it cannot honour itself;
- **its scoring judge** is the judge :class:`~threetears.evals.providers.Model` the caller built, lent for the judge
  role only, as :meth:`~threetears.evals.quick.judged.Judge.clients` lends a judged eval's;
- **its launch settings** run at most two cells of a run at once, with the cost ceilings enforced, so every arm is
  launched naming its share of ``max_cost_usd``.

**A model's family is read from the route it is built on**, never from a table of vendors: a model sent to
``anthropic`` or ``openai`` is of that provider's family, an OpenRouter id is of the vendor its id names before the
``/``, and a model on an endpoint of the caller's own (``base_url``) is of no family the host can tell, so its run
records the check as not made. The judge's route is its ``Model``'s; every writer is built by the model factory on
``writer_provider``.

**Every arm is held to its job's budget before it is priced.** A run's job is given a budget sized as if its cells ran
one after another, and clamped at the engine's ceiling (:func:`~threetears.evals.run.jobs.adaptive_job_timeout_s`); a
writer cell's deadline is a whole report (:func:`~threetears.evals.ops.report_writer.writer_launch.writer_launch_deadline`).
An arm whose cells, two at a time, could outlast that budget would be cut off mid-report after it was paid for, so its
plan refuses it, naming what to lower.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Final

from threetears.evals.actions.report_writing import launchable_report_writer_kind
from threetears.evals.analysis.analyst.skill import AnalystSkill
from threetears.evals.kernel import EvalStorage, default_job_timeout
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host import (
    SHARED_CORE,
    CompletionClients,
    CompletionRole,
    EvalHost,
    HostProfile,
    MeasureRegistry,
    TemplateChecks,
    default_cell_timeout,
)
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.ops.report_writer.writer_contract import (
    WRITER_KIND,
    WRITER_KIND_CONTRACT,
    WRITER_KIND_MEASURES,
    WriterModelFactory,
)
from threetears.evals.ops.report_writer.writer_launch import (
    admit_writer_launch,
    plan_writer_arm,
    writer_launch_deadline,
)
from threetears.evals.providers import Model, Provider, describe_failure
from threetears.evals.quick.judged import BorrowedJudgeClient
from threetears.evals.quick.result import CREATED_BY
from threetears.evals.run import ArmPlan, LaunchHost, LaunchRequest, LaunchSettings
from threetears.evals.run.jobs import adaptive_job_timeout_s
from threetears.evals.schema import DocumentStore
from threetears.evals.schema.completion import RequestCeiling

#: The host id the writer host's profile carries.
_WRITER_HOST_ID: Final = "eval-writer"

#: How many of a run's cells write at once: two reports side by side, never more.
_WRITER_CELLS_AT_ONCE: Final = 2


def _family(model_id: str, provider: Provider, base_url: str | None) -> str | None:
    """A model's family from its route: the provider's, the vendor an OpenRouter id names, or none for an endpoint."""
    if base_url is not None:
        return None
    if provider == "openrouter":
        vendor, slash, _ = model_id.partition("/")
        return vendor if slash and vendor else None
    return provider


def _lent(judge: Model) -> CompletionClients:
    """The host's client factory: ``judge``, lent for the judge role on its own model, and nothing else."""

    def lend(role: CompletionRole, model: str | None, *, temperature: float | None = None) -> BorrowedJudgeClient:
        if role != "judge":
            raise ValueError(f"the writer host calls no model in the {role!r} role; its scoring judge is its only one")
        if model is not None and model != judge.model_name:
            raise ValueError(
                f"a judge config asked for {model!r}, and the writer host's judge calls {judge.model_name!r}; "
                "every page is scored on its one judge"
            )
        return BorrowedJudgeClient(judge, model_name=judge.model_name, temperature=temperature)

    return lend


def _admit(_checked: object) -> None:
    """A template check the writer host leaves to the writer's launch, which refuses what a writer cannot honour."""


def writer_ops_host(
    store: DocumentStore,
    *,
    judge: Model,
    models: WriterModelFactory,
    budget: ReportBudget,
    request_s: RequestCeiling,
    max_cost_usd: float,
    skills: Mapping[str, AnalystSkill] = MappingProxyType({}),
    writer_provider: Provider = "anthropic",
    max_arms: int = 2,
    identity: str = CREATED_BY,
) -> OpsHost:
    """A host that launches report writer arms over ``store``, beside the quick host's runs in it (the module docstring).

    :param store: The store the quick path keeps its runs in; the writer's cases, runs and campaigns join them.
    :ptype store: DocumentStore
    :param judge: The scoring judge, built with its route and its recording; every arm's pages are scored on it, and a
        launch names its ``model_name`` as the judge.
    :ptype judge: Model
    :param models: The model factory: an arm's writer, by the model id its launch names and the skill the arm runs, at
        a reasoning effort.
    :ptype models: WriterModelFactory
    :param budget: Each report's budget; its turn cap bounds how long a cell can run.
    :ptype budget: ReportBudget
    :param request_s: The ceiling for one request on the writer's client, by output cap, in seconds.
    :ptype request_s: RequestCeiling
    :param max_cost_usd: The most one run may spend, judge included; a launch names its share, at most this.
    :ptype max_cost_usd: float
    :param skills: The skill versions a launch may run, by digest; the shipped skill is always one.
    :ptype skills: Mapping[str, AnalystSkill]
    :param writer_provider: The provider the factory builds every writer on, which tells a writer's family.
    :ptype writer_provider: Provider
    :param max_arms: The most arms launched together, and the most runs unfinished at once.
    :ptype max_arms: int
    :param identity: Who the cells' tool calls run as.
    :ptype identity: str
    :return: The host: launches the writer kind, generates no analysis.
    :rtype: OpsHost
    """
    judge_family = _family(judge.model_name, judge.provider, judge.base_url)

    def model_family(model_id: str) -> str | None:
        if model_id == judge.model_name:
            return judge_family
        return _family(model_id, writer_provider, None)

    settings = LaunchSettings(
        max_launch_arms=max_arms,
        max_admitted_runs=max_arms,
        judge_concurrency=1,
        max_concurrent_cells=_WRITER_CELLS_AT_ONCE,
        enforcement_enabled=True,
        max_cost_usd=max_cost_usd,
        max_metered_calls=None,
        max_out_of_run_cost_usd=max_cost_usd,
    )
    eval_host = EvalHost(
        profile=HostProfile(
            host_id=_WRITER_HOST_ID,
            host_sweepables=SHARED_CORE,
            measures=MeasureRegistry(WRITER_KIND_MEASURES),
            kinds=(WRITER_KIND_CONTRACT,),
        ),
        storage=EvalStorage(store),
        failure_describer=describe_failure,
        trace_sink=None,
        blocking_executor=None,
        cell_timeout=default_cell_timeout,
        clients=_lent(judge),
    )
    holder: list[OpsHost] = []
    entry = launchable_report_writer_kind(
        lambda: holder[0],
        identity=identity,
        models=models,
        budget=budget,
        request_s=request_s,
        model_family=model_family,
        skills=skills,
    )

    def plan(request: LaunchRequest) -> ArmPlan:
        storage = holder[0].eval_host.storage
        stored = (
            []
            if request.case_set is not None
            else storage.query_test_cases(request.scope_id, template_id=request.template.id)
        )
        admitted = admit_writer_launch(request, stored, skills=skills, model_family=model_family)
        planned = plan_writer_arm(holder[0].eval_host, request, admitted)
        cell_s = writer_launch_deadline(request, admitted, budget=budget, request_s=request_s)
        cells = planned.case_count * request.k_runs
        width = 1 if request.measure_latency else min(request.settings.max_concurrent_cells, cells)
        needed_s, job_s = math.ceil(cells / max(width, 1)) * cell_s, adaptive_job_timeout_s(cells, cell_s)
        if needed_s > job_s:
            raise ValidationFailedError(
                f"the arm's {cells} writer cell(s), {width} at a time at {cell_s:.0f}s each, could run {needed_s:.0f}s, "
                f"past its job's {job_s:.0f}s budget, which would cut a report off after it was paid for",
                remedy="launch fewer cases, give the report budget a lower turn cap, or lower the request ceiling",
            )
        return planned

    checks = TemplateChecks(
        require_known_tools_allowed=_admit,
        refuse_undeclared_world_seed=_admit,
        refuse_undeliverable_template=_admit,
        refuse_inadmissible_template_case=_admit,
        kinds={WRITER_KIND: replace(entry, plan_arm=plan)},
    )
    host = OpsHost(
        launch=LaunchHost(
            eval_host=replace(eval_host, template_checks=checks),
            settings=lambda: settings,
            job_timeout_factory=default_job_timeout,
        )
    )
    holder.append(host)
    return host


#: ``writer_ops_host`` joins ``__all__`` with its first caller outside this module, the quick path's writer eval; until
#: then the public-API caller gate (``tests/enforcement/test_public_api_has_a_caller.py``) would read it as test-only.
__all__: list[str] = []
