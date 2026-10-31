"""A key check, as an action of the ``Investigate: facts`` workflow.

``figure_keys_check`` is an :class:`~threetears.evals.actions.catalogue.Action` over its operation in
:mod:`threetears.evals.ops.analyst_keys`, rendered by :func:`~threetears.evals.actions.render.render_cited`. It is
sized to one result like every analyst listing, but not ``paged``: a paged action's next call repeats every argument
the caller sent, and this one's argument is the list of keys, which the caller already holds. Its page says where the
unread keys start instead (:mod:`~threetears.evals.analysis.analyst.keys`). It
belongs to the facts workflow, so the analyst's tool and the editor's cut of it, read off the workflows, both carry
it: the analyst checks a draft's keys before it submits, and the editor checks every key of the report it tests.
:func:`~threetears.evals.actions.analyst.analyst_actions` gathers :func:`analyst_keys_actions` beside the facts.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from threetears.evals.actions.analyst_facts import FACTS_WORKFLOW
from threetears.evals.actions.catalogue import Action, Caller
from threetears.evals.actions.params import Offset, SizedLimit, SnapshotId
from threetears.evals.actions.render import page_sizing, render_cited
from threetears.evals.ops.analyst_keys import KEYS_LIMIT, KEYS_MOST, KeysChecked, figure_keys_check
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema.base import EvalBaseModel

#: An example snapshot id, as a freeze returns one.
_EXAMPLE_SNAPSHOT = "report-snapshot:0193a1b2c3d4e5f60718293a4b5c6d7e"

#: One key as a report writes it.
_Key = Annotated[str, Field(min_length=1)]


class KeysCheckParams(EvalBaseModel):
    """``figure_keys_check``: the snapshot, the keys to read on it, and the page."""

    snapshot_id: SnapshotId
    keys: list[_Key] = Field(
        min_length=1,
        max_length=KEYS_MOST,
        description=f"The keys to read, as a report writes them, up to {KEYS_MOST}: a path with any filters, in "
        "braces or not ({cost_per_1k.<arm>|usd}, reading.<arm>.change|pts), or arm:<id>.",
    )
    limit: SizedLimit = None
    offset: Offset = 0


async def _figure_keys_check(host: OpsHost, caller: Caller, params: KeysCheckParams) -> KeysChecked:
    limit, fits = page_sizing("figure_keys_check", params, KEYS_LIMIT)
    return await figure_keys_check(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        keys=params.keys,
        limit=limit,
        offset=params.offset,
        fits=fits,
    )


def analyst_keys_actions() -> tuple[Action, ...]:
    """The ``Investigate: facts`` action that reads a report's keys.

    :return: ``figure_keys_check``.
    :rtype: tuple[Action, ...]
    """
    return (
        Action(
            name="figure_keys_check",
            summary="Read each figure key a report cites, as the page would print it, or why it reads nothing.",
            workflow=FACTS_WORKFLOW,
            permission="read",
            params=KeysCheckParams,
            result=KeysChecked,
            handler=_figure_keys_check,
            render=render_cited,
            example={
                "snapshot_id": _EXAMPLE_SNAPSHOT,
                "keys": ["{n_candidates}", "arm:fast.v2", "reading.fast.v2.change|abs|pts"],
            },
            detail=(
                "Check every key of a draft before submitting it, and every key of a report under review. Each key "
                "reads: what it prints through its filters (key|filters = reads), the family form it reads by and "
                "what that family reads, the arms and aspects it is about, and the filters its figure takes. Or it "
                "reads nothing: unstated (well formed; the snapshot states no such figure) or refused (it names "
                "nothing the snapshot holds, reads two ways, or carries a filter its figure does not suit), with the "
                "resolver's own refusal and what to write instead. Keys are checked once each, in the order given; "
                f"without a limit a page holds as many as fit one result, up to {KEYS_LIMIT}. A page never writes the "
                "keys back: its progress line says which of them it read and, where some are unread, to send the "
                "same keys again with offset=unread_from; the last page says all of them are read."
            ),
        ),
    )


__all__ = ["analyst_keys_actions"]
