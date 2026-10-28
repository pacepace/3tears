"""The analyst's actions: the tools a campaign analyst and its editor investigate a frozen campaign with.

An analyst writes a campaign report by investigating the campaign through tools (``docs/eval-report-kit/ANALYST.md``).
Each is an :class:`~threetears.evals.actions.catalogue.Action` in the one catalogue, so every host gets them and every
transport mounts them as it mounts the engine's. They take the id of a snapshot (``report_snapshot_freeze``, the one
write) and read only its frozen report input, so the analyst, the editor in its own context and a replay read the same
figures. Every figure comes back as the key that names it, written ``key = reads``
(:func:`~threetears.evals.actions.render.render_cited`).

**A cut of their own.** :func:`analyst_actions` puts every one on the analyst's cut, which the standard tools do not
mount: :func:`analyst_tools` is the tool that does, so the ``evals`` tool an MCP client sees, and its flat parameters,
stay the engine's. The in-process loop mounts the same tool an MCP client does: the analyst's role carries every
workflow below, and the editor's (``role="editor"``) the facts, the analyses and the gap note, read off the workflows,
so no file lists the actions' names.

**The workflows.** ``Investigate: snapshot`` freezes the evidence; ``facts`` reads what it holds; ``analyses``
computes over it; ``blocks`` shows the evidence a claim would carry, and the whole draft as the store and the editor
read it (``report_check``, which stores nothing); ``gaps`` notes a capability the analyst reasoned around
(``capability_gap``, which stores nothing).
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Annotated, Literal, get_args

from pydantic import Field, model_validator

from threetears.evals.actions.analyst_analyses import ANALYSES_WORKFLOW, analyst_analyses_actions
from threetears.evals.actions.analyst_facts import FACTS_WORKFLOW, analyst_facts_actions
from threetears.evals.actions.analyst_keys import analyst_keys_actions
from threetears.evals.actions.analyst_levers import analyst_levers_actions
from threetears.evals.actions.analyst_price import analyst_price_actions
from threetears.evals.actions.analyst_render import analyst_render_actions
from threetears.evals.actions.analyst_write import analyst_write_actions
from threetears.evals.actions.catalogue import DEFAULT_PREFIX, Action, Caller, HelpTopic, PermissionClass, ToolSpec
from threetears.evals.actions.params import AnalysisId, CampaignId, SnapshotId
from threetears.evals.actions.render import render_cited
from threetears.evals.ops.analyst_blocks import figure_key_index, figure_key_pages
from threetears.evals.ops.analyst import GapNoted, SnapshotFrozen, capability_gap, report_snapshot_freeze
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema.base import EvalBaseModel

#: The cut the analyst's actions mount on (:attr:`~threetears.evals.actions.catalogue.Action.cut`).
ANALYST_CUT = "analyst"

#: The help index's group for the freeze.
SNAPSHOT_WORKFLOW = "Investigate: snapshot"

#: The help index's group for a noted capability gap.
GAPS_WORKFLOW = "Investigate: gaps"

#: The workflows whose actions the editor tests a report with: no blocks, since it reads the page it is given.
_EDITOR_WORKFLOWS = frozenset({FACTS_WORKFLOW, ANALYSES_WORKFLOW, GAPS_WORKFLOW})

#: What the figure-key grammar is served as: its index, and each group of its paths a page, ``figure_keys.<group>``.
_FIGURE_KEYS = "figure_keys"

#: The figure-key grammar as a help topic: what every figure an analyst tool returns is keyed by, and what a claim cites.
#: The whole grammar runs past what one result may take of a reader's context, so the topic is its rules, filters and
#: a line per group of paths, and each group's forms, statistics and examples are a page of it.
_FIGURE_KEYS_TOPIC = HelpTopic(
    name=_FIGURE_KEYS,
    summary="The figure-key grammar: how a figure is cited, its filters, and every path by group.",
    text=functools.partial(figure_key_index, _FIGURE_KEYS),
    pages=figure_key_pages,
)

#: The rule every call to the analyst tool is held to, whichever role it is cut for.
_CITE_BY_KEY = (
    "every figure reads key = reads: cite its key in a report, never the number typed; figure_keys_check reads a "
    "draft's keys."
)

#: What a tool carrying the freeze says of the snapshot id, and what one without it says: a tool names only the
#: actions it carries.
_SNAPSHOT_RULES = {
    True: "every call but the freeze names snapshot_id, the snapshot report_snapshot_freeze returned; the same id "
    "throughout.",
    False: "every call names snapshot_id, the snapshot frozen for this work; the same id throughout.",
}

#: Why the editor's tool goes without the blocks and the freeze, and what its caller does instead: what it is told
#: of a call to one, or of a remedy that needs one.
_EDITOR_LEFT_OUT = (
    "the editor tests the report it is given against the snapshot it was written from, so it renders no blocks and "
    "freezes nothing; a snapshot that must be frozen again goes back to the analyst that froze it"
)

#: Each role's tool description.
_DESCRIPTIONS = {
    "analyst": (
        "Investigate a campaign for its report: freeze its evidence as a snapshot, then read every figure from it by "
        "the key that names it."
    ),
    "editor": (
        "Test a campaign report against the snapshot it was written from: read every figure by the key that names it, "
        "and compute what no single figure states."
    ),
}

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
    """Every analyst action, on the analyst's cut: the freeze, the facts, the analyses, the blocks (a draft's check
    among them) and the gap note.

    :return: The actions, in the order help lists them.
    :rtype: tuple[Action, ...]
    """
    (freeze, gap) = _snapshot_actions()
    built = (
        freeze,
        *analyst_facts_actions(),
        *analyst_keys_actions(),
        *analyst_analyses_actions(),
        *analyst_price_actions(),
        *analyst_levers_actions(),
        *analyst_render_actions(),
        *analyst_write_actions(),
        gap,
    )
    return tuple(replace(action, cut=ANALYST_CUT) for action in built)


def analyst_tools(
    prefix: str = DEFAULT_PREFIX, *, writes: bool = True, role: Literal["analyst", "editor"] = "analyst"
) -> tuple[ToolSpec]:
    """The one tool carrying the analyst's actions: ``<prefix>_analyst``, cut for the analyst or for its editor.

    :param prefix: The host's prefix for its eval tools.
    :ptype prefix: str
    :param writes: Whether it carries the freeze, the analyst's one write; ``False`` for a host whose agent may only
        read, which then reads snapshots frozen another way, and for a loop that froze the snapshot itself.
    :ptype writes: bool
    :param role: ``analyst``, every workflow of the cut; or ``editor``, the facts, the analyses and the gap note, with
        no blocks (the editor reads the page it is given) and no freeze, whatever ``writes`` says.
    :ptype role: Literal["analyst", "editor"]
    :return: ``<prefix>_analyst``, read actions and, with ``writes``, the freeze; its help serves the figure-key grammar
        as ``topic='figure_keys'``, an index whose pages are ``topic='figure_keys.<group>'``.
    :rtype: tuple[ToolSpec]
    """
    classes: frozenset[PermissionClass] = frozenset({"read", "write"} if writes else {"read"})
    editor = role == "editor"
    return (
        ToolSpec(
            name=f"{prefix}_analyst",
            description=_DESCRIPTIONS[role],
            permissions=classes,
            cut=ANALYST_CUT,
            workflows=_EDITOR_WORKFLOWS if editor else None,
            topics=(_FIGURE_KEYS_TOPIC,),
            constraints=(_SNAPSHOT_RULES[writes and not editor], _CITE_BY_KEY),
            left_out=_EDITOR_LEFT_OUT if editor else "",
        ),
    )


__all__ = [
    "analyst_actions",
    "analyst_tools",
]
