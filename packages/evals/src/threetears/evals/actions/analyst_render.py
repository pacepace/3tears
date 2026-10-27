"""The analyst's block tools: the evidence block a claim would show, as its values table, and the catalog of block kinds.

The actions of the ``Investigate: blocks`` workflow, each an
:class:`~threetears.evals.actions.catalogue.Action` over its operation in :mod:`threetears.evals.ops` and rendered by
:func:`~threetears.evals.actions.render.render_cited`. :func:`~threetears.evals.actions.analyst.analyst_actions`
gathers them onto the analyst's cut, and the editor's cut of the analyst tool is read off the workflows, so no file
lists their names.

``block_catalog`` comes first: it lists every kind ``block_render`` draws, the sources each reads, and the query
fields that narrow a source's rows. ``block_render`` draws one block the way a planned page draws a chart, by the
page's own code, and returns its values table, never the picture.
"""

from __future__ import annotations

from typing import Annotated, Any, get_args

from pydantic import BeforeValidator, Field, ValidationError, field_validator, model_validator

from threetears.evals.actions.catalogue import Action, Caller
from threetears.evals.actions.params import ARM_FORMS, Limit, Offset, SnapshotId, written_arm
from threetears.evals.actions.render import render_cited
from threetears.evals.kernel.report_plan import ChartQuery, ChartReference, PlannedChart
from threetears.evals.ops.analyst_blocks import BLOCK_ROWS, BlockCatalog, RenderedBlock, block_catalog, block_render
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema.base import EvalBaseModel

#: The help index's group for these actions, which the editor's cut is read off.
_BLOCKS_WORKFLOW = "Investigate: blocks"

#: An example snapshot id, as a freeze returns one.
_EXAMPLE_SNAPSHOT = "report-snapshot:0193a1b2c3d4e5f60718293a4b5c6d7e"

#: Every kind of block, and every reference line, as a page's chart names them.
_KINDS: tuple[str, ...] = get_args(PlannedChart.model_fields["kind"].annotation)
_REFERENCES: tuple[str, ...] = get_args(ChartReference.model_fields["kind"].annotation)

#: A query's fields where a call leaves them out: no filter, no limit, nothing transformed, one panel.
_EMPTY_QUERY: dict[str, Any] = {
    "aspect": "",
    "measures": [],
    "arms": [],
    "strata": [],
    "limit": 0,
    "transform": "none",
    "small_multiples_by": "none",
}


class BlockCatalogParams(EvalBaseModel):
    """``block_catalog``: the snapshot whose blocks to list."""

    snapshot_id: SnapshotId


class BlockRenderParams(EvalBaseModel):
    """``block_render``: the snapshot, the block's kind, its query, its line, what it stresses, and the page."""

    snapshot_id: SnapshotId
    block_kind: str = Field(
        description="The kind of block, as block_catalog lists it.", json_schema_extra={"enum": list(_KINDS)}
    )
    query: dict[str, Any] = Field(
        description=(
            "What the block reads: source (required, a source the kind draws), then any of aspect, measures, arms "
            "(each " + ARM_FORMS + "), strata, limit, transform and small_multiples_by; a field left out is empty: "
            "no filter, no limit."
        )
    )
    reference: str | None = Field(
        default=None,
        description="The reference line to draw, which code keys from the snapshot; omitted draws none.",
        json_schema_extra={"enum": list(_REFERENCES)},
    )
    emphasis: Annotated[str, BeforeValidator(written_arm)] = Field(
        default="",
        description=f"The one arm, stratum or task to stress, by id (an arm also {ARM_FORMS}); empty for none.",
    )
    title: str = Field(default="", description="The block's title, figure keys allowed; empty for none.")
    limit: Limit = BLOCK_ROWS
    offset: Offset = 0

    @field_validator("query")
    @classmethod
    def _a_chart_query(cls, query: dict[str, Any]) -> dict[str, Any]:
        """The query with every field it leaves out empty, each arm it names read to its id by the one rule every
        action's arms share (:func:`~threetears.evals.actions.params.written_arm`), refused where it is no chart query.

        :raises ValueError: The query names no source, a field a chart query does not hold, or a value out of range.
        """
        arms = query.get("arms")
        if isinstance(arms, list):
            query = {**query, "arms": [written_arm(arm) for arm in arms]}
        try:
            filled = ChartQuery.model_validate({**_EMPTY_QUERY, **query})
        except ValidationError as invalid:
            raise ValueError(f"query is not a chart query: {invalid}") from invalid
        return filled.model_dump(mode="json")

    @model_validator(mode="after")
    def _a_block(self) -> BlockRenderParams:
        """Refuse a kind or a line no page draws.

        :raises ValueError: The kind or the reference line is none a page's chart takes.
        """
        if self.block_kind not in _KINDS:
            raise ValueError(f"block_kind is {self.block_kind!r}, and the kinds are {list(_KINDS)}")
        if self.reference is not None and self.reference not in _REFERENCES:
            raise ValueError(f"reference is {self.reference!r}, and the lines are {list(_REFERENCES)}")
        return self


async def _block_catalog(host: OpsHost, caller: Caller, params: BlockCatalogParams) -> BlockCatalog:
    return await block_catalog(host, caller.scope_id, snapshot_id=params.snapshot_id)


async def _block_render(host: OpsHost, caller: Caller, params: BlockRenderParams) -> RenderedBlock:
    return await block_render(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        block_kind=params.block_kind,
        query=ChartQuery.model_validate(params.query),
        reference=params.reference,
        emphasis=params.emphasis,
        title=params.title,
        limit=params.limit,
        offset=params.offset,
    )


def analyst_render_actions() -> tuple[Action, ...]:
    """The ``Investigate: blocks`` actions: the catalog of blocks, then one block drawn.

    :return: The actions, in the order help lists them.
    :rtype: tuple[Action, ...]
    """
    snapshot = {"snapshot_id": _EXAMPLE_SNAPSHOT}
    return (
        Action(
            name="block_catalog",
            summary="List every kind of evidence block, the sources each draws, and the sources this snapshot holds.",
            workflow=_BLOCKS_WORKFLOW,
            permission="read",
            params=BlockCatalogParams,
            result=BlockCatalog,
            handler=_block_catalog,
            render=render_cited,
            example=snapshot,
            detail=(
                "Each kind says what it draws, the sources it reads, whether every row needs an interval, whether it "
                "draws panels, and the reference lines code can key on it. Each source says what a row is, the query "
                "fields that narrow its rows, and whether this snapshot holds any. The figure key forms are the paths "
                "a claim cites a figure by; help topic='figure_keys' gives the grammar's "
                "rules and filters and names a page per group of paths, topic='figure_keys.<group>', with each form's "
                "statistics and an example."
            ),
        ),
        Action(
            name="block_render",
            summary="Draw one evidence block as a page would, and read its values table, level, unit and line.",
            workflow=_BLOCKS_WORKFLOW,
            permission="read",
            params=BlockRenderParams,
            result=RenderedBlock,
            handler=_block_render,
            render=render_cited,
            paged=True,
            example={**snapshot, "block_kind": "forest", "query": {"source": "readings"}},
            detail=(
                "The block is completed and checked by the planner's own rules and drawn by the page's own code, so a "
                "block drawn here is one a report may show. It returns every value the block draws as a table of "
                "text, the interval level, unit and axis code fixed, and the reference line's figure by its key: never "
                "the picture. It echoes the line and the emphasis it drew, which sit outside the query, so a report "
                "document's evidence can carry them beside it. A block the page cannot draw on this snapshot is refused naming every fault at the "
                "parameter it sits at. A page of rows stops at limit, or sooner where cells run long; its last "
                "line names the call that reads the next."
            ),
        ),
    )


__all__ = ["analyst_render_actions"]
