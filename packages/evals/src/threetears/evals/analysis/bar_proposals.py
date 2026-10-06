"""Bar proposals from a baseline campaign — the one caller of the ratchet.

A behavior's bar should start where the incumbent already performs, so that shipping something
worse is a visible regression rather than a matter of opinion. :func:`propose_bars` measures the
incumbent the way every analysis does — it assembles the baseline campaign's bundle and reads its
one cell's measures — and hands each measure with a better end to
:meth:`~threetears.evals.contracts.host.bars.BarRegistry.propose`, which flags a vacuous seed and
registers nothing.

**A baseline measures one configuration under one rig.** A campaign of several cells has several
incumbents, and choosing which one is the standard is the decision a proposal exists to leave to a
person, so it is refused rather than pooled: a mean across arms is the bar of no configuration
anyone ran.

**Nothing here adopts anything.** The proposals are returned; a bar reaches a registry only when
a person writes it into a host's registrations, because the registry has no mutation API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from threetears.evals.analysis.bundle import assemble_context_bundle
from threetears.evals.analysis.numbers import format_number
from threetears.evals.contracts.errors import NotFoundError, ValidationFailedError
from threetears.evals.contracts.host.bars import BarProposal, no_better_end
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.host.eval_host import EvalHost

log = get_logger(__name__)


@dataclass(frozen=True)
class BaselineBarProposals:
    """What a baseline campaign proposes as its behavior's bars, and what it could not propose a bar on.

    Attributes:
        campaign_id: The baseline campaign measured.
        behavior: The behavior every proposal governs — the campaign's.
        variant_key: The incumbent's variant: the baseline's one cell.
        proposals: One per measure with a better end that the host declares, in measure-name order.
            Each carries ``vacuous`` and its ``reason``; nothing is registered.
        not_proposed: ``{reading: why}`` for every reading the cell carries that no bar could be proposed on —
            a measure undeclared by the host, directionless, or with no mean to seed from, and every judged
            dimension, since a registered bar names a declared measure and a judged dimension is not one.
    """

    campaign_id: str
    behavior: str
    variant_key: str
    proposals: tuple[BarProposal, ...]
    not_proposed: dict[str, str] = field(default_factory=dict)

    @property
    def vacuous(self) -> tuple[BarProposal, ...]:
        """The proposals a person must not adopt as written — each nothing could fail, in one of two ways.

        Either every value the measure can take clears it (the baseline sat at the permissive end of the
        declared range), or a bar is already registered and the proposal would not tighten it, so adopting it
        would loosen the standard. Each proposal's ``reason`` says which.
        """
        return tuple(proposal for proposal in self.proposals if proposal.vacuous)


def propose_bars(host: EvalHost, baseline_campaign_id: str, *, scope_id: str) -> BaselineBarProposals:
    """Propose a bar on every measure the baseline campaign's incumbent was measured on.

    Each proposal's threshold is the incumbent's mean on the measure, over the cell's non-faulted
    observations — the same population every bar is later adjudicated over, so a proposed bar and
    the verdict that will read it describe one set of observations.

    Args:
        host: The host whose measures, bars and storage this reads.
        baseline_campaign_id: The campaign that measured the incumbent configuration.
        scope_id: The scope it lives in.

    Returns:
        The proposals, each flagged vacuous where nothing could fail it, and the measures nothing
        could be proposed on, with why.

    Raises:
        NotFoundError: No such campaign in the scope.
        ValidationFailedError: The campaign measured no cell, or more than one — a baseline is one
            configuration under one rig.
    """
    campaign = host.storage.load_campaign(baseline_campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", baseline_campaign_id)
    bundle = assemble_context_bundle(campaign, storage=host.storage, profile=host.profile)
    if len(bundle.cell_measures) != 1:
        raise ValidationFailedError(
            f"campaign {baseline_campaign_id!r} measured {len(bundle.cell_measures)} cells, and a baseline is one "
            "configuration under one rig — which of several is the incumbent is the choice a proposal leaves to a "
            "person, so propose from a campaign of the incumbent alone",
            details={
                "campaign_id": baseline_campaign_id,
                "cells": [[cell.variant_key, cell.apparatus_class_id] for cell in bundle.cell_measures],
            },
        )
    (cell,) = bundle.cell_measures
    measures = host.profile.measures
    proposals: list[BarProposal] = []
    not_proposed: dict[str, str] = {}
    for summary in cell.measures.measures:
        descriptor = measures.get(summary.name)
        if descriptor is None:
            not_proposed[summary.name] = "this host does not declare it, so no registered bar could ever be read on it"
            continue
        if (what := no_better_end(descriptor)) is not None:
            not_proposed[summary.name] = f"it is {what}, with no better end to clear"
            continue
        if summary.mean is None:
            not_proposed[summary.name] = "it carries no mean to seed a threshold from"
            continue
        proposals.append(
            host.profile.bars.propose(
                behavior=campaign.behavior,
                measure=summary.name,
                observed=summary.mean,
                measures=measures,
                rationale=(
                    f"the incumbent's measured baseline: a mean of {format_number(summary.mean)} over {summary.n} "
                    f"observations of {summary.n_independent} cases in campaign {campaign.id}"
                ),
            )
        )
    for judged in cell.judged:
        not_proposed[judged.dimension] = (
            "it is a judged dimension, and a registered bar names a measure the host declares — a judge's score is "
            "not one"
        )
    log.info(
        "eval.propose_bars campaign=%s scope=%s behavior=%s proposed=%d vacuous=%d not_proposed=%d",
        campaign.id,
        scope_id,
        campaign.behavior,
        len(proposals),
        sum(1 for proposal in proposals if proposal.vacuous),
        len(not_proposed),
    )
    return BaselineBarProposals(
        campaign_id=campaign.id,
        behavior=campaign.behavior,
        variant_key=cell.variant_key,
        proposals=tuple(proposals),
        not_proposed=not_proposed,
    )


__all__ = ["BaselineBarProposals", "propose_bars"]
