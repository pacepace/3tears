"""The price of a trade, as an action of the ``Investigate: analyses`` workflow.

``price_tradeoff`` is an :class:`~threetears.evals.actions.catalogue.Action` over its operation in
:mod:`threetears.evals.ops.analyst_price`, rendered by :func:`~threetears.evals.actions.render.render_cited`. It
belongs to the analyses workflow, so the analyst's tool and the editor's cut of it, read off the workflows, carry it:
:func:`~threetears.evals.actions.analyst.analyst_actions` gathers :func:`analyst_price_actions` beside the other
analyses.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from threetears.evals.actions.analyst_analyses import ANALYSES_WORKFLOW
from threetears.evals.actions.catalogue import Action, Caller
from threetears.evals.actions.params import ArmFilter, Offset, SizedLimit, SnapshotId
from threetears.evals.actions.render import page_sizing, render_cited
from threetears.evals.ops.analyst_price import PRICE_PAGE, PriceTradeoff, price_tradeoff
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema.base import EvalBaseModel

#: An example snapshot id, as a freeze returns one.
_EXAMPLE_SNAPSHOT = "report-snapshot:0193a1b2c3d4e5f60718293a4b5c6d7e"


class PriceParams(EvalBaseModel):
    """``price_tradeoff``: the snapshot, the gain axis, the option to price the others against, the slice to price
    within, an option to price as a what-if, and the page."""

    snapshot_id: SnapshotId
    gain_key: Annotated[
        str,
        Field(
            min_length=1,
            description="The axis the trade's gain is read on: a measure with a direction, or a judged dimension, by "
            "the key campaign_overview lists.",
        ),
    ]
    arm_filter: ArmFilter = None
    slice_filter: Annotated[
        str | None,
        Field(
            min_length=1,
            description="Price within this one task, stratum or cell only, by its id (a cell as <row>.<column>), on "
            "the slice's own figures; omitted prices overall.",
        ),
    ] = None
    what_if: Annotated[
        str | None,
        Field(
            min_length=1,
            description="Also price this option the gates remove or leave undecided, as if it passed: a labelled "
            "what-if, never a recommendation; omitted for none.",
        ),
    ] = None
    limit: SizedLimit = None
    offset: Offset = 0


async def _price_tradeoff(host: OpsHost, caller: Caller, params: PriceParams) -> PriceTradeoff:
    limit, fits = page_sizing("price_tradeoff", params, PRICE_PAGE)
    return await price_tradeoff(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        gain_key=params.gain_key,
        arm=params.arm_filter,
        slice_filter=params.slice_filter,
        what_if=params.what_if,
        limit=limit,
        offset=params.offset,
        fits=fits,
    )


def analyst_price_actions() -> tuple[Action, ...]:
    """The ``Investigate: analyses`` action that prices a trade.

    :return: ``price_tradeoff``.
    :rtype: tuple[Action, ...]
    """
    return (
        Action(
            name="price_tradeoff",
            summary="Price the trade between the surviving options: cost per point gained on one axis, and where the "
            "best option flips as the price of a point moves.",
            workflow=ANALYSES_WORKFLOW,
            permission="read",
            params=PriceParams,
            result=PriceTradeoff,
            handler=_price_tradeoff,
            render=render_cited,
            paged=True,
            example={"snapshot_id": _EXAMPLE_SNAPSHOT, "gain_key": "facts"},
            detail=(
                "Only options that survive the gates are priced; each other option is listed off the frontier with "
                "why (gated, no cost, no figure on the axis, dominated, or best at no single price). The frontier is "
                "the options best at some price per point, cheapest first; each step's per_point is the price per "
                "point at which the best option moves from the cheaper to the dearer, with its low and high ends "
                "read off the gain's paired interval, and under unpaired off the two options' own half-widths "
                "combined (an end an interval leaves open is unbounded, said in words). Where no option states an "
                "overall figure on the axis, each is pooled over its slices, by traffic share where stated, else "
                "equally (a caveat says so). slice_filter prices within one task, stratum or cell; what_if prices "
                "one option the gates remove or leave undecided, labelled, never a recommendation. An option a gate "
                "has no figure for somewhere, while none it has fails, is listed off the frontier as undecided with "
                "its unknown places, never as removed. Every option is also priced against "
                "the control, or against arm_filter. Pages cover a window of the options, the frontier and its "
                "steps among them, and state the whole frontier's size; without a limit a page holds as many "
                f"options as fit one result, up to {PRICE_PAGE}. Monthly figures appear only at the declared "
                "monthly volume. Cost is read as exact. It prices the trade and chooses nothing; a mix nobody ran "
                "is not priced."
            ),
        ),
    )


__all__ = ["analyst_price_actions"]
