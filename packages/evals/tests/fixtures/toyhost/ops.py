"""The toy host as an operations host that writes a campaign's report: what a host adds to launching to get one.

Three things, each on the :class:`~threetears.evals.ops.OpsHost` or its profile:

- **how a report is written** (:class:`~threetears.evals.ops.AnalysisGeneration`): the budget one report is admitted
  against, the writing loop's limits, and the job's wall clock, derived from those limits over the host's own ceiling
  for one request (``Ceilings().writer_ceiling_s``) plus headroom, so the job never cancels a turn still writing
  inside its cap. ``analysis_generate`` writes on the client the host returns for the ``analyst`` role, a turn model;
  the toy host's clients are the caller's (``clients``), as its launch host takes them;
- **its reading of a model's family** (:func:`toyhost_model_family`): a judge is held to another family than the
  writer whose report it scores, which a pairwise pass over writer campaigns checks;
- **the writer's kind contract and measures** on the profile (:func:`toyhost_writer_profile`), so a writer template is
  admitted against the profile and a writer run's measures read. The contract also seats ``reviewer_pool``, the role
  the toy host pins on every kind. Launching writer campaigns takes one more step this host leaves out: the writer's
  launchable kind on the launch registry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Final

from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.host import CompletionClients, HostProfile, MeasureRegistry
from threetears.evals.ops import (
    WRITER_KIND_CONTRACT,
    WRITER_KIND_MEASURES,
    AnalysisGeneration,
    OpsHost,
    ReportBudget,
    WriterLimits,
)
from threetears.evals.run import Ceilings, LaunchSettings
from threetears.evals.schema import DEFAULT_JOB_LEASE_S
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile

#: The most one report may spend, and the turns it may send: the turn cap is what bounds the job's wall clock.
TOYHOST_REPORT_BUDGET: Final = ReportBudget(usd=10.0, turns=60)

#: The toy host's ceiling for one request on its client, in seconds, whatever the request's output cap.
TOYHOST_REQUEST_S: Final = 120.0

#: What the job is given beyond the writer's own ceiling: the snapshot's freeze, the tool calls and the store.
TOYHOST_REPORT_HEADROOM_S: Final = 60.0


def toyhost_model_family(model_id: str) -> str | None:
    """The family of a model, read from the id a launch names it by: the vendor before a ``/``, as a router names it.

    An id with no vendor in it is one this host cannot place, and the check that reads it is recorded as not made.

    :param model_id: The model's id.
    :ptype model_id: str
    :return: The vendor, or ``None``.
    :rtype: str | None
    """
    vendor, slash, _name = model_id.partition("/")
    return vendor if slash and vendor else None


def _writer_seats() -> frozenset[str]:
    """The seats the writer's contract fills: the judge and the spend ceiling."""
    assert WRITER_KIND_CONTRACT.seats is not None, "the writer's contract seats its judge"
    return WRITER_KIND_CONTRACT.seats


def toyhost_writer_profile() -> HostProfile:
    """The toy host's profile with the report writer's kind contract on its kinds and its measures on its registry.

    :return: The profile.
    :rtype: HostProfile
    """
    base = toyhost_profile()
    return replace(
        base,
        kinds=(*base.kinds, replace(WRITER_KIND_CONTRACT, seats=_writer_seats() | {"reviewer_pool"})),
        measures=MeasureRegistry((*TOYHOST_MEASURES, *WRITER_KIND_MEASURES), families=base.measures.families),
    )


def toyhost_generation(limits: WriterLimits | None = None) -> AnalysisGeneration:
    """How the toy host writes a report: its budget, the loop's limits, and the job's ceiling derived from both.

    :param limits: The writing loop's limits; ``None`` is ``WriterLimits()``.
    :ptype limits: WriterLimits | None
    :return: The generation.
    :rtype: AnalysisGeneration
    """
    held = limits if limits is not None else WriterLimits()
    writing_s = Ceilings().writer_ceiling_s(
        held, request_s=lambda _max_tokens: TOYHOST_REQUEST_S, turns=TOYHOST_REPORT_BUDGET.turns
    )
    return AnalysisGeneration(
        report_budget=lambda: TOYHOST_REPORT_BUDGET,
        limits=held,
        budget_s=writing_s + TOYHOST_REPORT_HEADROOM_S,
    )


def toyhost_ops_host(
    *,
    storage: EvalStorage | None = None,
    clients: CompletionClients | None = None,
    settings: Callable[[], LaunchSettings] = lambda: TOYHOST_LAUNCH_SETTINGS,
    generation: AnalysisGeneration | None = None,
    job_lease_s: float = DEFAULT_JOB_LEASE_S,
) -> OpsHost:
    """The toy host as an operations host that writes reports and places models in families.

    :param storage: Where documents live; ``None`` is a fresh in-memory store.
    :ptype storage: EvalStorage | None
    :param clients: The completion-client factory, whose ``analyst`` client writes each report; ``None`` for a host
        that reads and launches only, whose generation start ends refused for want of a client.
    :ptype clients: CompletionClients | None
    :param settings: Reads the launch settings, the out-of-run cap a report is held to among them.
    :ptype settings: Callable[[], LaunchSettings]
    :param generation: How a report is written; ``None`` is :func:`toyhost_generation`'s.
    :ptype generation: AnalysisGeneration | None
    :param job_lease_s: How long the host's job leases last unless renewed.
    :ptype job_lease_s: float
    :return: The host.
    :rtype: OpsHost
    """
    launch, _client = toyhost_launch_host(
        profile=toyhost_writer_profile(), storage=storage, clients=clients, settings=settings, job_lease_s=job_lease_s
    )
    return OpsHost(
        launch=launch,
        generation=generation if generation is not None else toyhost_generation(),
        model_family=toyhost_model_family,
    )


__all__ = [
    "TOYHOST_REPORT_BUDGET",
    "TOYHOST_REPORT_HEADROOM_S",
    "TOYHOST_REQUEST_S",
    "toyhost_generation",
    "toyhost_model_family",
    "toyhost_ops_host",
    "toyhost_writer_profile",
]
