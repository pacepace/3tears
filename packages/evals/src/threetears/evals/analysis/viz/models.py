"""Wire models for compiled charts.

:class:`~threetears.evals.analysis.viz.compiler.CompiledChart` is the compiler's in-process
return value and carries behaviour; this is the shape that crosses the API
boundary. They are kept apart deliberately — one is free to grow methods, the
other is a contract that lands in the generated OpenAPI document — and
:meth:`FindingChart.from_compiled` is the single place the two are mapped.

Being the single site does not by itself keep the two in step, so the mapping is
held total by test rather than by assertion:
``tests/test_viz_compiler.py::TestTheWireShapeCarriesTheCompilation``
compares ``CompiledChart``'s dataclass fields against this model's, and fails on a
field added to either and mapped by neither. Anything deliberately left off the
wire is named in :data:`UNMAPPED_COMPILED_FIELDS` with its reason — an omission
the test then requires, so it cannot be an oversight that merely looks decided.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.prose import ModelProse

if TYPE_CHECKING:
    from threetears.evals.analysis.viz.compiler import CompiledChart

#: ``CompiledChart`` fields deliberately not carried on the wire → why not.
#:
#: Read by the totality test, which requires this to be exactly the set of fields
#: the mapping drops: a field is either served or its absence is a stated decision.
UNMAPPED_COMPILED_FIELDS: dict[str, str] = {
    "unit": (
        "the unit the chart draws in already reaches every reader through the values table's column headers "
        "('Mean (s)') and the spec's axis titles, both built from it during compilation — a third copy on the "
        "wire would be a second thing to keep true, and no consumer reads one"
    ),
}


class FindingChart(EvalBaseModel):
    """One finding's compiled chart, as served.

    ``spec`` deliberately carries no colour: each surface merges its own theme,
    the browser from the live CSS custom properties and the server from the
    generated palette artifact. That is what stops a chart's palette from drifting
    away from the design tokens on either side.

    ``columns`` and ``rows`` are the values as drawn, in drawn order, so a reader
    who cannot see the picture can still check the claim — and so the two
    descriptions of one chart cannot disagree, because they come from one
    compilation. The columns differ per chart shape (a breakdown states a value, a
    null result states an interval), which is why they are DATA rather than a
    layout the reader has to know in advance.

    ``caption`` and ``disclosures`` are two authors and are served apart. The
    caption is the analysis author's one idea about the chart, carried exactly as
    written; the disclosures are what the compiler must add — what it truncated,
    restated or could not place — one line each. Joining them into one string made
    the author's sentence the first of several in a paragraph nobody finished.

    ``error`` carries the reason a stored payload could not be drawn. A chart
    missing from a surface is otherwise indistinguishable from a finding that
    never carried one, and a payload this build's compiler refuses is exactly the
    one a reader most needs told about — so the failure is served, not omitted.
    """

    finding_id: str = Field(description="The finding this chart belongs to.")
    spec: dict[str, Any] = Field(
        default_factory=dict, description="The Vega-Lite spec, without colour. Empty when `error` is set."
    )
    columns: list[dict[str, str]] = Field(
        default_factory=list,
        description="The values-as-drawn table's columns, in display order — each `{key, header}`, the header carrying the unit.",
    )
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="The values-as-drawn rows, in drawn order, keyed by column key; typed loosely on the wire so a chart shape can widen it additively.",
    )
    caption: ModelProse = Field(
        default="",
        description="The analysis author's own line beside the chart, exactly as written — never extended by the compiler; "
        "empty when the author wrote none.",
    )
    disclosures: list[str] = Field(
        default_factory=list,
        description="What this compilation must tell the reader that the author could not — what it truncated, "
        "filtered, restated or could not place, and any key the figure does not draw — one idea per line, in "
        "reading order, rendered after `caption` and apart from it. Empty when there is nothing to disclose.",
    )
    title: str = Field(default="", description="The chart's title.")
    error: str = Field(
        default="", description="Why this finding's stored payload could not be drawn; empty when it was."
    )

    @classmethod
    def from_compiled(cls, finding_id: str, compiled: CompiledChart) -> FindingChart:
        """Build the wire shape from a compilation.

        Every field of :class:`~threetears.evals.analysis.viz.compiler.CompiledChart` is carried
        across except those named in :data:`UNMAPPED_COMPILED_FIELDS`; the module
        docstring names the test that holds that true.

        Args:
            finding_id: The finding the chart belongs to.
            compiled: The compiler's output.

        Returns:
            The serialisable chart.
        """
        return cls(
            finding_id=finding_id,
            spec=compiled.spec,
            columns=[{"key": column["key"], "header": column["header"]} for column in compiled.columns],
            rows=compiled.rows,
            caption=compiled.caption,
            disclosures=list(compiled.disclosures),
            title=compiled.title,
        )

    @classmethod
    def from_failure(cls, finding_id: str, reason: str) -> FindingChart:
        """Build the entry for a finding whose stored payload will not draw.

        Args:
            finding_id: The finding the chart belongs to.
            reason: Why it cannot be drawn, naming the offending field.

        Returns:
            A chart carrying only the reason — no spec, no values.
        """
        return cls(finding_id=finding_id, error=reason)


__all__ = [
    "UNMAPPED_COMPILED_FIELDS",
    "FindingChart",
]
