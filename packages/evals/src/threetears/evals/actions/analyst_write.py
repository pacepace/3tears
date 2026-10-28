"""A draft check, as an action of the ``Investigate: blocks`` workflow: a report document read before it is submitted.

``report_check`` is an :class:`~threetears.evals.actions.catalogue.Action` over its operation in
:mod:`threetears.evals.ops.analyst`, rendered by :func:`~threetears.evals.actions.render.render_cited`. The analyst
sends its draft ``report-doc/1`` document and reads back what the store and the editor would make of it: every verdict
the invariants find, and the page as text, each block as its values table, written by the function the editor's message
is written by. It is a read: it stores nothing, and the same draft on the same snapshot returns the same result.

**Its pages never send the draft back, and never cut a verdict unsaid.** A result is sized to one result's budget:
the verdicts first, grouped so each remedy is stated once, as many as fit from ``verdict_offset``; then, where every
verdict from it fits, the page's lines from ``offset``. Like ``figure_keys_check``, it is not ``paged``: a paged
action's next call repeats every argument the caller sent, and this one's is the whole draft, which the caller holds.
A result's ``progress`` says where the unread verdicts or lines start instead.

**On the analyst's tool alone.** It belongs to the blocks workflow, which the editor's cut of the tool leaves out: the
editor reads the page it is given, and checks no draft. :func:`~threetears.evals.actions.analyst.analyst_actions`
gathers :func:`analyst_write_actions` beside the blocks.

**A draft that does not read is the call's fault.** A ``doc`` that is not JSON, or not a ``report-doc/1`` document, is
refused as the call, naming every field at fault; a document that reads but names what the snapshot does not hold is
checked, and that is a verdict (validity, ``0``).
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import Field, ValidationError, field_validator

from threetears.evals.actions.analyst_render import BLOCKS_WORKFLOW
from threetears.evals.actions.catalogue import Action, Caller
from threetears.evals.actions.params import Offset, SizedLimit, SnapshotId
from threetears.evals.actions.render import page_fits, render_cited
from threetears.evals.kernel.report_doc import ReportDoc
from threetears.evals.ops.analyst import CHECK_LINES, DraftChecked, report_check
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema.base import EvalBaseModel

#: An example snapshot id, as a freeze returns one.
_EXAMPLE_SNAPSHOT = "report-snapshot:0193a1b2c3d4e5f60718293a4b5c6d7e"

#: A query's fields where a block reads every row of its source: no filter, no limit, nothing transformed, one panel.
_EVERY_ROW: dict[str, Any] = {
    "aspect": "",
    "measures": [],
    "arms": [],
    "strata": [],
    "limit": 0,
    "transform": "none",
    "small_multiples_by": "none",
}

#: The smallest draft that reads as a ``report-doc/1`` document: the example a refusal shows.
_EXAMPLE_DOC: dict[str, Any] = {
    "schema": "report-doc/1",
    "story": "decision",
    "headline": "{arm:fast.v2} holds the pass rate at a lower cost.",
    "lede": "Its reading against the control is {reading.fast.v2.change|pts}.",
    "recommends": ["fast.v2"],
    "confidence": {"form": "low", "caveats": []},
    "decision": [{"slot": "recommendation", "text": "Adopt {arm:fast.v2}."}],
    "claims": [
        {
            "id": "c1",
            "text": "{arm:fast.v2} reads no worse than the control.",
            "weight": "lead",
            "evidence": [
                {
                    "block": "forest",
                    "query": {**_EVERY_ROW, "source": "readings"},
                    "caption": "Each arm's reading against the control.",
                }
            ],
        }
    ],
    "explore": [],
    "omitted": [],
    "next": [],
    "method": [{"term": "Cases", "text": "Every arm ran the same cases."}],
}

#: How a draft that does not read as a document is mended.
_DOC_REMEDY = (
    "send the draft as one report-doc/1 object (or its JSON text whole), every required field present: the skill's "
    "report document table, and the example below, show its shape"
)


def _at(location: tuple[int | str, ...]) -> str:
    """A field's place in the draft as a verdict names one: ``claims[0].evidence``."""
    written = ""
    for part in location:
        written += f"[{part}]" if isinstance(part, int) else f".{part}" if written else str(part)
    return written or "(the document)"


class ReportCheckParams(EvalBaseModel):
    """``report_check``: the snapshot, the draft, and which of its verdicts and its page's lines to read."""

    snapshot_id: SnapshotId
    doc: dict[str, Any] = Field(
        description="The draft report-doc/1 document, as a JSON object (its JSON text is read too): every field, as "
        "you would submit it."
    )
    limit: SizedLimit = None
    offset: Offset = 0
    verdict_offset: Annotated[
        int,
        Field(
            ge=0,
            description="Skip this many verdicts; a result's verdicts_unread_from, where the verdicts run past "
            "one result.",
        ),
    ] = 0

    @field_validator("doc", mode="before")
    @classmethod
    def _read_json(cls, doc: Any) -> Any:
        """A draft sent as JSON text, read to its object.

        :raises ValueError: The text is not JSON.
        """
        if not isinstance(doc, str):
            return doc
        try:
            return json.loads(doc)
        except json.JSONDecodeError as unread:
            raise ValueError(
                f"doc is not JSON ({unread.msg}, line {unread.lineno} column {unread.colno}): {_DOC_REMEDY}"
            ) from unread

    @field_validator("doc")
    @classmethod
    def _a_report_doc(cls, doc: dict[str, Any]) -> dict[str, Any]:
        """Refuse a draft that is no ``report-doc/1`` document, naming every field at fault.

        :raises ValueError: The draft does not validate as a :class:`~threetears.evals.kernel.report_doc.ReportDoc`.
        """
        try:
            ReportDoc.model_validate(doc)
        except ValidationError as invalid:
            faults = "; ".join(f"{_at(error['loc'])}: {error['msg']}" for error in invalid.errors())
            raise ValueError(f"doc is not a report-doc/1 document: {faults}. {_DOC_REMEDY}") from invalid
        return doc


async def _report_check(host: OpsHost, caller: Caller, params: ReportCheckParams) -> DraftChecked:
    return await report_check(
        host,
        caller.scope_id,
        snapshot_id=params.snapshot_id,
        doc=ReportDoc.model_validate(params.doc),
        fits=page_fits("report_check", params),
        limit=params.limit if params.limit is not None else CHECK_LINES,
        offset=params.offset,
        verdict_offset=params.verdict_offset,
    )


def analyst_write_actions() -> tuple[Action, ...]:
    """The ``Investigate: blocks`` action that reads a whole draft: ``report_check``.

    :return: ``report_check``.
    :rtype: tuple[Action, ...]
    """
    return (
        Action(
            name="report_check",
            summary="Check a draft report document: every verdict code finds, and the page as the editor reads it.",
            workflow=BLOCKS_WORKFLOW,
            permission="read",
            params=ReportCheckParams,
            result=DraftChecked,
            handler=_report_check,
            render=render_cited,
            example={"snapshot_id": _EXAMPLE_SNAPSHOT, "doc": _EXAMPLE_DOC},
            detail=(
                "Check the whole draft before you submit it, and again after each revision. The page is drawn strict, "
                "as generation draws it, and every check the store runs is run on the draft and on the page's text "
                "above the appendix. Each verdict names its invariant (0 validity, else 1 to 5: numbers through keys, "
                "blocks that draw, caveat coverage, the confidence form, no id above the appendix), its field path "
                "and its fault; verdicts sharing a remedy are grouped under it, stated once. The store refuses a "
                "draft holding any. A page that does not build is one more verdict, at the field it stopped at, and "
                "there is then no page to read. A draft with no Method item (invariant 6) is no report-doc/1 document, so it is "
                "refused as the call, as is a doc that is not JSON or any other field out of shape, naming every "
                "field at fault. Then the page as a reader and the editor read it: each claim by its id and weight, "
                "each block by what code fixed about it (its level, unit, axis and line) and its values table, then "
                "the appendix. A result fits one result's budget: the verdicts first, as many as fit, then as many "
                "of the page's lines as fit beside them. Its progress line says what is left and how to read it: "
                "send the same draft again with verdict_offset=verdicts_unread_from for the verdicts that did not "
                "fit (the page follows the last of them), or with offset=unread_from for the page's next lines. "
                "Stores nothing, makes no model call, and the same draft always reads the same."
            ),
        ),
    )


__all__ = ["analyst_write_actions"]
