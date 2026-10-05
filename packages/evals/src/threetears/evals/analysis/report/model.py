"""The ``Report`` — one versioned document every surface renders an analysis from.

**The report is a document, not an assembly.** A stored analysis used to reach a reader through four
reads (the analysis, its arm table, its decision-surface table, its compiled charts) that each surface
put together for itself, and the memo's Markdown dropped every chart's data on the way. A
:class:`Report` is all of it, in reading order, as an ordered list of blocks — ``text`` with a role,
``table``, ``chart`` and ``disclosure`` — each linked to the findings it belongs to or rests on. It has
a published JSON Schema (``schema.json`` beside this module), so a host generates its client types from
the schema rather than mirroring them by hand, and it ships three serializers: JSON (canonical, this
model's own dump), Markdown (the agent-facing form) and HTML that reads without a script.

**Who wrote what is part of the shape.** A ``text`` block holds what the analysis's author wrote, and
nothing else; what code has to add — a chart's restatement, an arm the join could not place, why the
time axis is days — is a ``disclosure``, never appended to the author's words. A chart block carries its
:class:`~threetears.evals.analysis.viz.intent.ChartIntent` (what it draws and must say) or the reason it
cannot be drawn — never a renderer's spec, which is the host's to make.

**Where a caveat goes.** An author's caveat is attached to its finding, always, as written — the author
placed it there, and moving it would be code overruling the author about what qualifies their claim. The
authored :class:`~threetears.evals.contracts.authored.Caveat` carries no magnitude, so placing caveats by
how material they are would mean reading materiality out of prose, which code does not do; the number
that does say whether a difference is worth acting on — a measure's materiality threshold — is code's,
and it already labels the delta it applies to (``immaterial``) where that delta is drawn. What belongs in
an appendix is what no single finding owns: the ``methods`` section holds the analysis-wide disclosures
(the arms the join could not place, the bars no cell could be read against, the time axis's basis, how
the analysis was generated).
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from threetears.evals.analysis.viz.intent import Cell, ChartIntent, ChartType
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.prose import ModelProse

#: The report shape's version. Moves when a field or block kind is added, renamed or removed, or when a
#: field's meaning moves under its name; a host reads it to know what it was handed.
REPORT_VERSION: Literal[1] = 1

#: Where a block sits, in reading order.
ReportSection = Literal["summary", "questions", "decisions", "findings", "arms", "surface", "next", "methods"]

#: Each section's heading, in reading order — the order every serializer lays the sections out in.
SECTION_TITLES: dict[str, str] = {
    "summary": "Summary",
    "questions": "Declared questions",
    "decisions": "Decisions",
    "findings": "Findings",
    "arms": "Arms",
    "surface": "Decision surface",
    "next": "What to run next",
    "methods": "Methods and disclosures",
}

#: What a text block's author wrote it as.
TextRole = Literal[
    "summary",
    "answer",
    "decision",
    "revisit_when",
    "finding_title",
    "finding_body",
    "caveat",
    "carried_forward",
    "next_step",
    "next_step_why",
]

#: Who a disclosure speaks for.
DisclosureSource = Literal["chart", "arms", "surface", "time_axis", "generation"]


class Fact(EvalBaseModel):
    """A labelled fact code states beside an author's words — a confidence, a disposition, a tier."""

    name: str = Field(min_length=1, description="What the fact is, e.g. `Confidence`.")
    value: str = Field(min_length=1, description="The fact, in words.")


class _Block(EvalBaseModel):
    """What every block carries: where it sits, and which findings it belongs to or rests on."""

    section: ReportSection = Field(description="The section the block sits in.")
    finding: int | None = Field(
        default=None,
        ge=0,
        description="The finding this block belongs to, by its position in the document (0 is the first); None when it belongs to none.",
    )
    rests_on: list[int] = Field(
        default_factory=list, description="Positions of the findings this block rests on, as the author linked them."
    )


class TextBlock(_Block):
    """What the analysis's author wrote, exactly as written, with the facts code states beside it."""

    kind: Literal["text"] = "text"
    role: TextRole = Field(description="What the author wrote it as.")
    body: ModelProse = Field(
        description="The author's words, Markdown allowed; empty where the author left a required field blank."
    )
    facts: list[Fact] = Field(default_factory=list, description="Facts code states beside the words, in reading order.")

    @model_validator(mode="after")
    def _a_findings_own_words_name_it(self) -> Self:
        """Refuse a finding's title, body, caveat or carried-forward claim that names no finding.

        Raises:
            ValueError: A block in one of those roles has no ``finding``.
        """
        if self.role in FINDING_ROLES and self.finding is None:
            raise ValueError(f"a {self.role} block belongs to a finding and names none")
        return self


#: The text roles that are part of one finding, and so name it.
FINDING_ROLES: frozenset[str] = frozenset({"finding_title", "finding_body", "caveat", "carried_forward"})


def finding_number(block: TextBlock) -> int:
    """The finding a block belongs to, as a reader counts findings: from one.

    Raises:
        ValueError: The block names no finding.
    """
    if block.finding is None:
        raise ValueError(f"a {block.role} block names no finding to number")
    return block.finding + 1


class TableColumn(EvalBaseModel):
    """One column of a report table."""

    key: str = Field(min_length=1, description="The key in each row.")
    header: str = Field(min_length=1, description="The column's header, carrying its unit where it has one.")


class TableBlock(_Block):
    """A table code laid out — its columns, its rows in their stated order, and how much of it is shown."""

    kind: Literal["table"] = "table"
    name: str = Field(
        min_length=1, description="Which table this is: `evidence`, `arms`, `surface` or `unadjudicated_bars`."
    )
    title: str = Field(min_length=1, description="The table's heading.")
    columns: list[TableColumn] = Field(min_length=1, description="The columns, in display order.")
    rows: list[dict[str, Cell]] = Field(description="The rows shown, in the stated order, keyed by column key.")
    order: str = Field(min_length=1, description="The order the rows are in, in words.")
    total_rows: int = Field(ge=0, description="How many rows the table has; more than are shown when it is truncated.")

    @model_validator(mode="after")
    def _rows_fit_the_table(self) -> Self:
        """Refuse a table showing more rows than it has, or a row stating a value no column shows.

        Raises:
            ValueError: ``total_rows`` is below the rows shown, or a row carries a key that is not a column.
        """
        if self.total_rows < len(self.rows):
            raise ValueError(f"table {self.name!r} shows {len(self.rows)} rows but says it has {self.total_rows}")
        keys = {column.key for column in self.columns}
        if stray := sorted({key for row in self.rows for key in row} - keys):
            raise ValueError(f"table {self.name!r} has rows keyed {', '.join(stray)}, which no column shows")
        return self


class ChartBlock(_Block):
    """A finding's chart: its intent, or why the stored chart cannot be drawn."""

    kind: Literal["chart"] = "chart"
    viz_type: ChartType = Field(description="The chart type the finding carries.")
    intent: ChartIntent | None = Field(
        default=None, description="What the chart draws and must say; None when it cannot be drawn."
    )
    error: str = Field(
        default="",
        description=(
            "Why the stored chart cannot be drawn, naming the offending field; empty when it can. Served rather than "
            "omitted: a chart missing from a report is otherwise indistinguishable from a finding that carried none."
        ),
    )

    @model_validator(mode="after")
    def _drawn_or_says_why_not(self) -> Self:
        """Refuse a chart block that carries both an intent and an error, or neither.

        Raises:
            ValueError: The block is drawable and failed, or neither.
        """
        if (self.intent is None) == (not self.error.strip()):
            raise ValueError("a chart block carries exactly one of an intent and the reason it cannot be drawn")
        if self.intent is not None and self.intent.type != self.viz_type:
            raise ValueError(f"a {self.viz_type} chart block carries a {self.intent.type} intent")
        return self


class DisclosureBlock(_Block):
    """Something code must tell the reader that no author wrote — one idea."""

    kind: Literal["disclosure"] = "disclosure"
    source: DisclosureSource = Field(description="What the disclosure speaks for.")
    text: str = Field(min_length=1, description="The disclosure, one sentence or a few.")


#: A report block, discriminated by ``kind``.
ReportBlock = Annotated[TextBlock | TableBlock | ChartBlock | DisclosureBlock, Field(discriminator="kind")]


class ReportSource(EvalBaseModel):
    """What the report is a report of, and how that analysis was generated."""

    analysis_id: str = Field(min_length=1, description="The analysis the report renders.")
    campaign_id: str = Field(min_length=1, description="The campaign the analysis is of.")
    scope_id: str = Field(min_length=1, description="The scope both live in.")
    subject_id: str = Field(min_length=1, description="The analysed subject.")
    subject_kind: str = Field(description="The subject's kind; empty when the campaign declared none.")
    behavior: str = Field(min_length=1, description="The behaviour under analysis.")
    generated_at: str = Field(min_length=1, description="When the analysis was generated (ISO-8601).")
    generator_model: str = Field(min_length=1, description="The model that wrote it, as the provider reported it.")
    bundle_fingerprint: str = Field(min_length=1, description="The fingerprint of the bundle it was written over.")


class Report(EvalBaseModel):
    """One analysis, as a document every surface renders. See the module docstring for the contract."""

    report_version: Literal[1] = Field(default=REPORT_VERSION, description="This shape's version.")
    headline: ModelProse = Field(description="The author's headline, as written; empty when the author wrote none.")
    finding_count: int = Field(
        ge=0, description="How many findings the document holds — the range every position is in."
    )
    source: ReportSource = Field(description="What this is a report of.")
    blocks: list[ReportBlock] = Field(description="The report, in reading order.")

    @model_validator(mode="after")
    def _positions_point_at_findings(self) -> Self:
        """Refuse a block linked to a finding the report does not hold.

        Raises:
            ValueError: A block's ``finding`` or ``rests_on`` names a position past ``finding_count``.
        """
        for index, block in enumerate(self.blocks):
            named = ([block.finding] if block.finding is not None else []) + list(block.rests_on)
            if stray := sorted({position for position in named if not 0 <= position < self.finding_count}):
                raise ValueError(
                    f"block {index} ({block.kind}) is linked to finding position(s) {stray}, and the report holds "
                    f"{self.finding_count} finding(s)"
                )
        return self

    def to_canonical_json(self) -> str:
        """The report as its canonical JSON — the form the published schema validates.

        Returns:
            The JSON text, indented, keys in model order.
        """
        return self.model_dump_json(indent=2)


__all__ = [
    "REPORT_VERSION",
    "SECTION_TITLES",
    "ChartBlock",
    "DisclosureBlock",
    "DisclosureSource",
    "FINDING_ROLES",
    "Fact",
    "Report",
    "ReportBlock",
    "ReportSection",
    "ReportSource",
    "TableBlock",
    "TableColumn",
    "TextBlock",
    "TextRole",
    "finding_number",
]
