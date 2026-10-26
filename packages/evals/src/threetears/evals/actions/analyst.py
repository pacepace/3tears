"""The analyst's actions: the tools a campaign analyst and its editor investigate a frozen campaign with.

An analyst writes a campaign report by investigating the campaign through tools (``docs/eval-report-kit/ANALYST.md``).
Each is an :class:`~threetears.evals.actions.catalogue.Action` in the one catalogue, so every host gets them and every
transport mounts them as it mounts the engine's. They take the id of a snapshot (``report_snapshot_freeze``, the one
write) and read only its frozen report input, so the analyst, the editor in its own context and a replay read the same
figures. Every figure comes back as the key that names it, written ``key = reads``
(:func:`~threetears.evals.actions.render.render_cited`).

**A cut of their own.** :func:`analyst_actions` puts every one on the analyst's cut, which the standard tools do not
mount: :func:`analyst_tools` is the tool that does, so the ``evals`` tool an MCP client sees, and its flat parameters,
stay the engine's. The in-process loop presents the actions one tool each, by name, from the toolsets:
:func:`analyst_toolset` for the analyst and :func:`editor_toolset` for the editor, both read off the workflows below,
so no file lists the tools' names.

**The workflows.** ``Investigate: snapshot`` freezes the evidence; ``facts`` reads what it holds; ``analyses``
computes over it; ``blocks`` shows the evidence a claim would carry; ``gaps`` notes a capability the analyst reasoned
around (``capability_gap``, which stores nothing).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated, get_args

from pydantic import Field, model_validator

from threetears.evals.actions.analyst_analyses import ANALYSES_WORKFLOW, analyst_analyses_actions
from threetears.evals.actions.analyst_facts import FACTS_WORKFLOW, analyst_facts_actions
from threetears.evals.actions.analyst_render import BLOCKS_WORKFLOW, analyst_render_actions
from threetears.evals.actions.catalogue import DEFAULT_PREFIX, Action, Caller, PermissionClass, ToolSpec
from threetears.evals.actions.params import AnalysisId, CampaignId, SnapshotId
from threetears.evals.actions.render import render_cited
from threetears.evals.ops.analyst import GapNoted, SnapshotFrozen, capability_gap, report_snapshot_freeze
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema.base import EvalBaseModel

#: The cut the analyst's actions mount on (:attr:`~threetears.evals.actions.catalogue.Action.cut`).
ANALYST_CUT = "analyst"

#: The help index's group for the freeze.
SNAPSHOT_WORKFLOW = "Investigate: snapshot"

#: The help index's group for a noted capability gap.
GAPS_WORKFLOW = "Investigate: gaps"

#: The workflows whose tools the analyst investigates and builds its report with.
_ANALYST_WORKFLOWS = (FACTS_WORKFLOW, ANALYSES_WORKFLOW, BLOCKS_WORKFLOW, GAPS_WORKFLOW)

#: The workflows whose tools the editor tests a report with: no blocks, since it reads the page it is given.
_EDITOR_WORKFLOWS = (FACTS_WORKFLOW, ANALYSES_WORKFLOW, GAPS_WORKFLOW)

#: An example snapshot id, as a freeze returns one.
_EXAMPLE_SNAPSHOT = "report-snapshot:0193a1b2c3d4e5f60718293a4b5c6d7e"


class SnapshotFreezeParams(EvalBaseModel):
    """``report_snapshot_freeze``: the campaign or the stored analysis to freeze, exactly one."""

    campaign_id: Annotated[str | None, get_args(CampaignId)[1]] = None
    analysis_id: Annotated[str | None, get_args(AnalysisId)[1]] = None

    @model_validator(mode="after")
    def _one_source(self) -> SnapshotFreezeParams:
        """Refuse a call naming both or neither.

        :raises ValueError: Both or neither of ``campaign_id`` and ``analysis_id`` are given.
        """
        if (self.campaign_id is None) == (self.analysis_id is None):
            raise ValueError("name a campaign_id or an analysis_id, exactly one")
        return self


class CapabilityGapParams(EvalBaseModel):
    """``capability_gap``: what the analyst needed and no tool gave, why, and what it did instead."""

    snapshot_id: SnapshotId
    need: Annotated[str, Field(min_length=1, description="What you needed and no tool gave, as one capability.")]
    why: Annotated[str, Field(min_length=1, description="What you needed it for: the question it would answer.")]
    workaround: Annotated[str, Field(min_length=1, description="What you did instead, and go on doing.")]
    claim_ids: list[str] = Field(
        default_factory=list, description="The claims of your report the workaround touches; empty for none yet."
    )


async def _report_snapshot_freeze(host: OpsHost, caller: Caller, params: SnapshotFreezeParams) -> SnapshotFrozen:
    return await report_snapshot_freeze(
        host,
        caller.scope_id,
        campaign_id=params.campaign_id,
        analysis_id=params.analysis_id,
        created_by=caller.identity,
    )


async def _capability_gap(host: OpsHost, caller: Caller, params: CapabilityGapParams) -> GapNoted:
    return await capability_gap(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        need=params.need,
        why=params.why,
        workaround=params.workaround,
        claim_ids=params.claim_ids,
    )


def _snapshot_actions() -> tuple[Action, ...]:
    """The freeze and the gap note: the analyst's actions this module builds itself."""
    return (
        Action(
            name="report_snapshot_freeze",
            summary="Freeze a campaign's evidence, or a stored analysis's, as the snapshot every analyst tool reads.",
            workflow=SNAPSHOT_WORKFLOW,
            permission="write",
            params=SnapshotFreezeParams,
            result=SnapshotFrozen,
            handler=_report_snapshot_freeze,
            render=render_cited,
            example={"campaign_id": "0193a1b2-campaign"},
            idempotent=True,
            detail=(
                "A campaign's evidence is assembled as it stands; an analysis's is the bundle frozen with it. The "
                "report input built from it is stored whole, and every analyst tool takes the snapshot_id returned "
                "here and reads that input alone, so every call reads the same figures. Freezing unchanged evidence "
                "names the snapshot already stored (reused true). Makes no model call."
            ),
        ),
        Action(
            name="capability_gap",
            summary="Note a capability you needed and no tool gave, and the workaround you go on with; stores nothing.",
            workflow=GAPS_WORKFLOW,
            permission="read",
            params=CapabilityGapParams,
            result=GapNoted,
            handler=_capability_gap,
            render=render_cited,
            example={
                "snapshot_id": _EXAMPLE_SNAPSHOT,
                "need": "the cases needed to separate two arms' readings",
                "why": "to say what would settle the asked question",
                "workaround": "named the case count from the interval widths, in words",
            },
            detail=(
                "Call it whenever you reason around a missing tool, at the moment you do. It computes nothing and "
                "stores nothing: the call itself, in the transcript, is the record, and the gap id is the same for the "
                "same snapshot, need and why."
            ),
        ),
    )


def analyst_actions() -> tuple[Action, ...]:
    """Every analyst action, on the analyst's cut: the freeze, the facts, the analyses, the blocks and the gap note.

    :return: The actions, in the order help lists them.
    :rtype: tuple[Action, ...]
    """
    (freeze, gap) = _snapshot_actions()
    built = (freeze, *analyst_facts_actions(), *analyst_analyses_actions(), *analyst_render_actions(), gap)
    return tuple(replace(action, cut=ANALYST_CUT) for action in built)


def _named(workflows: tuple[str, ...]) -> tuple[str, ...]:
    """The analyst actions of those workflows, by name, in catalogue order."""
    return tuple(action.name for action in analyst_actions() if action.workflow in workflows)


def analyst_toolset() -> tuple[str, ...]:
    """The tools the analyst works with, by action name: the facts, analyses, blocks and the gap note.

    Not the freeze: the loop running the analyst freezes once and hands the snapshot id to the analyst and the editor.

    :return: The action names, in catalogue order.
    :rtype: tuple[str, ...]
    """
    return _named(_ANALYST_WORKFLOWS)


def editor_toolset() -> tuple[str, ...]:
    """The tools the report editor tests a report with, by action name: the facts, analyses and the gap note.

    No block tools: the editor reads the rendered page it is given, and tests an omission or a counterfactual with
    the facts and analyses.

    :return: The action names, in catalogue order.
    :rtype: tuple[str, ...]
    """
    return _named(_EDITOR_WORKFLOWS)


def analyst_tools(prefix: str = DEFAULT_PREFIX, *, writes: bool = True) -> tuple[ToolSpec]:
    """The one tool carrying the analyst's actions: ``<prefix>_analyst``.

    :param prefix: The host's prefix for its eval tools.
    :ptype prefix: str
    :param writes: Whether it carries the freeze, the analyst's one write; ``False`` for a host whose agent may only
        read, which then reads snapshots frozen another way.
    :ptype writes: bool
    :return: ``<prefix>_analyst``, read actions and, with ``writes``, the freeze.
    :rtype: tuple[ToolSpec]
    """
    classes: frozenset[PermissionClass] = frozenset({"read", "write"} if writes else {"read"})
    return (
        ToolSpec(
            name=f"{prefix}_analyst",
            description=(
                "Investigate a campaign for its report: freeze its evidence as a snapshot, then read every figure from "
                "it by the key that names it."
            ),
            permissions=classes,
            cut=ANALYST_CUT,
        ),
    )


__all__ = [
    "analyst_actions",
    "analyst_toolset",
    "analyst_tools",
    "editor_toolset",
]
