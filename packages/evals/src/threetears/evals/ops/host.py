"""What the operations run over: a launching host, and how it generates an analysis in the background.

Every operation that reads takes the :class:`~threetears.evals.kernel.host.EvalHost`; every one that
starts long work — a launch, an analysis generation — needs the process's job manager too, which the
:class:`~threetears.evals.run.LaunchHost` builds. :class:`OpsHost` is the two together with the
things neither carries: what a background generation is held to (the report's budget and its wall-clock budgets) —
the host's configuration and never the engine's guess. How many cases a launch
would run is its kinds' answer (``LaunchableKind.plan_arm``), which a launch estimate asks as the launch does.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.job_timeout import default_job_timeout
from threetears.evals.ops.report_writer.loop import WriterLimits
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.run.definition_seed import SeedCorpus
from threetears.evals.run.launch import LaunchHost, LaunchSettings
from threetears.evals.schema import EvalRun, WorldPlacement


def _reading_settings() -> LaunchSettings:
    """The settings of a host that launches nothing (:meth:`OpsHost.reading`), built fresh on every read.

    One run at a time, uncapped, no metered tools. Read only by the launch and the out-of-run cap, and a reading
    host has no kind to launch. Built per call rather than held at module level, so no two hosts share an instance.

    :return: The settings.
    :rtype: LaunchSettings
    """
    return LaunchSettings(
        max_launch_arms=1,
        max_admitted_runs=1,
        judge_concurrency=1,
        enforcement_enabled=False,
        max_cost_usd=1.0,
        max_metered_calls=None,
        max_out_of_run_cost_usd=1.0,
    )


#: The default wall-clock ceiling on a generation job's preparation, used when a host supplies none
#: (``AnalysisGeneration.preparation_budget_s``, read on every call): building the ``analyst`` client, holding it to the
#: host's allowed writers and asking whether its turns can be priced — in seconds. The job starts with no budget of its
#: own (the writing enters ``AnalysisGeneration.budget_s`` itself), so without this a client that hangs while it is
#: built or refreshes its prices would leave the attempt ``running`` for as long as its lease kept renewing.
#: Preparation makes no model call, so the bound is tight, unlike a launch's (``LAUNCH_PREPARATION_BUDGET_S``), whose
#: launchers call providers: a hang reads as refused within minutes rather than ``running`` for hours. A host that
#: wants another bound sets it per call.
ANALYSIS_PREPARATION_BUDGET_S = 900.0


@dataclass(frozen=True, kw_only=True)
class AnalysisGeneration:
    """How this host generates an analysis as a background job: the report writer, on the host's model and budget.

    ``analysis_generate`` writes the campaign's report document with the analyst and the report editor on the client
    the host returns for the ``analyst`` role, admitting every turn against :attr:`report_budget`. Nothing here names
    a prompt: the writer's words are the analyst skill the package ships, whose digest is each analysis's provenance.

    :ivar report_budget: The host's budget for one report (``ReportBudget``), read when each generation starts: the
        only admission on its turns, which a caller of ``analysis_generate`` cannot raise. ``None`` for a host that has
        not set one, whose generation start is refused saying where to set it, with nothing frozen or sent.
    :ivar limits: What the writing loop is held to (its rounds' tool calls, bounces, reviews and revisions, every
        turn's output cap), passed to the report writer exactly as set; ``WriterLimits()`` unless the host sets it.
    :ivar budget_s: The job's wall-clock ceiling on the writing (the snapshot's freeze, every turn of every session,
        the store); a report that outruns it is recorded ``timed_out``. Derive it from the same :attr:`limits` object,
        ABOVE the writer's own ceiling (``Ceilings().writer_ceiling_s(limits, request_s=..., turns=...)`` of
        :class:`~threetears.evals.run.Ceilings`, over the host's own ceiling for one request on its client and the turn
        cap of the budget :attr:`report_budget` returns, ``ReportBudget.turns``) plus the host's headroom: a budget below it cancels a turn still writing inside its cap, after it was billed. The
        snapshot's freeze (assembling the campaign's evidence) runs inside this budget, not the preparation's, so a
        freeze that hangs reads ``running`` until this budget ends it ``timed_out``.
    :ivar preparation_budget_s: The job's wall-clock ceiling on its preparation (the client, the allowed-writer check
        and the pricing question), in seconds; a generation whose preparation outruns it ends refused before any
        turn. :data:`ANALYSIS_PREPARATION_BUDGET_S` unless the host sets it, and read on every call, so a host may set
        it per call.
    """

    report_budget: Callable[[], ReportBudget] | None = None
    limits: WriterLimits = field(default_factory=WriterLimits)
    budget_s: float
    preparation_budget_s: float = ANALYSIS_PREPARATION_BUDGET_S

    def __post_init__(self) -> None:
        """Refuse a budget that is not positive, and a preparation budget that is not positive and finite.

        :raises ValueError: ``budget_s`` is not positive, or ``preparation_budget_s`` is not a positive finite number.
        """
        if self.budget_s <= 0:
            raise ValueError(f"AnalysisGeneration.budget_s must be positive; got {self.budget_s}")
        if not (math.isfinite(self.preparation_budget_s) and self.preparation_budget_s > 0):
            raise ValueError(
                f"AnalysisGeneration.preparation_budget_s must be positive and finite; got {self.preparation_budget_s}"
            )


def _placed_nowhere(run: EvalRun) -> dict[str, WorldPlacement]:
    """No placement: a reading host's world is declared for its runs' reads, and it launches none to place."""
    return {}


@dataclass(frozen=True, kw_only=True)
class OpsHost:
    """The host the operations, and the actions over them, work in.

    :ivar launch: The launching host: its :class:`~threetears.evals.kernel.host.EvalHost`, its kinds and
        the job manager every long-running operation is tracked by.
    :ivar generation: How an analysis is generated in the background; ``None`` for a host that does not
        generate analyses here, whose generation start is refused saying so.
    :ivar seed_corpus: The host's seed corpus for a scope (``load_seed_corpus(its directory, scope_id)``), which a case
        rename reads so a rename never changes how definition drift finds the case; ``None`` for a host that
        seeds no template cases.
    """

    launch: LaunchHost
    generation: AnalysisGeneration | None = None
    seed_corpus: Callable[[str], SeedCorpus] | None = None

    @classmethod
    def reading(cls, eval_host: EvalHost) -> OpsHost:
        """The host an eval host's runs are read through, and nothing launched: what serving ``as_host(...)`` needs.

        Its launching host declares no kind, so a launch is refused by name, and it generates no analysis. Every read
        (templates, runs, campaigns, reports, gates) works over ``eval_host``'s storage.

        :param eval_host: The host whose runs are read, as :func:`~threetears.evals.quick.as_host` builds one.
        :ptype eval_host: EvalHost
        :return: The host.
        :rtype: OpsHost
        """
        # The registry is the host's template checks' ``kinds``, so nothing launches here by emptying it.
        checks = eval_host.template_checks
        if checks is not None and checks.kinds:
            eval_host = replace(eval_host, template_checks=replace(checks, kinds={}))
        launch = LaunchHost(
            eval_host=eval_host,
            settings=_reading_settings,
            job_timeout_factory=default_job_timeout,
            # A world's placement is stamped on a launched run, and nothing launches here.
            world_placements=None if eval_host.profile.world is None else _placed_nowhere,
        )
        return cls(launch=launch)

    @property
    def eval_host(self) -> EvalHost:
        """The host every read goes through — the launching host's own."""
        return self.launch.eval_host

    def out_of_run_cap(self) -> float | None:
        """The cap a call made outside any run is held to: the host's out-of-run cap, read once, or ``None`` with enforcement off.

        One reading for every out-of-run operation — an analysis generation and a judge repeat — so the two
        cannot hold their calls to different ceilings.
        """
        settings = self.launch.settings()
        return settings.max_out_of_run_cost_usd if settings.enforcement_enabled else None


__all__ = ["AnalysisGeneration", "OpsHost"]
