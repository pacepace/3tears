"""The ``Report`` — one versioned document every surface renders an analysis from.

**The report is a document, not an assembly.** A stored analysis used to reach a reader through four
reads (the analysis, its arm table, its decision-surface table, its compiled charts) that each surface
put together for itself, and the memo's Markdown dropped every chart's data on the way. A
:class:`Report` is all of it, in reading order, as an ordered list of blocks — ``text`` with a role,
``table``, ``chart`` and ``disclosure`` — each linked to the findings it belongs to or rests on. It has
a published JSON Schema (``schema.json`` beside this module), so a host generates its client types from
the schema rather than mirroring them by hand. **The schema checks the shape and every cross-field rule JSON
Schema can state** — a code-only report names no analysis or model and holds no headline, finding or text
block; a report with no findings links no block to one; a finding's own words name their finding; a chart
block carries exactly one of an intent and an error, the intent of its own type. Three rules compare a value
with a sibling's, which JSON Schema cannot: a block's finding positions are below ``finding_count`` on a
report that has findings, a table's ``total_rows`` is at least the rows it shows, and a row keys only its
table's columns. Those the model's validators hold, so a document is a report when the model accepts it, and
the schema accepts a superset by exactly those three. It and it ships three serializers: JSON (canonical, this
model's own dump), Markdown (the agent-facing form) and HTML that reads without a script.

**A report is of an analysis, or of the evidence alone — and it says which** (:attr:`Report.basis`). A
campaign with a generated analysis is reported through it (``analysis``). A campaign with none — no
analyst has run, and the package ships no keyless one — is still reported (``code_only``): the arm
table, the decision surface, the contrasts against the control, a chart per measure and every
disclosure the evidence carries, all computed by code, with NO text block at all, and a disclosure
stating plainly that no analysis was generated and what one would add. The shape refuses a code-only
report carrying an author's words, and an analysis report missing its analysis.

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

from typing import Annotated, Literal, Self, get_args

from pydantic import ConfigDict, Field, model_validator

from threetears.evals.analysis.viz.intent import Cell, ChartIntent, ChartType
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.prose import ModelProse

#: The report shape's version. Moves when a field or block kind is added, renamed or removed, or when a
#: field's meaning moves under its name; a host reads it to know what it was handed.
#:
#: 2: ``basis`` added (a report of an analysis, or a code-only report of the evidence alone);
#: ``source.analysis_id`` and ``source.generator_model`` became nullable (None on a code-only report),
#: and ``source.generated_at`` / ``source.bundle_fingerprint`` mean the assembly's on a code-only report;
#: disclosure sources ``runs``, ``measurement``, ``apparatus`` and ``comparisons`` added; the
#: ``comparisons`` and ``questions`` tables added; a chart block with no finding (a chart code chose).
REPORT_VERSION: Literal[2] = 2

#: What a report is of: a generated analysis, or the campaign's evidence alone with no analysis.
ReportBasis = Literal["analysis", "code_only"]

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

#: Who a disclosure speaks for. ``runs``: member runs left out, unfinished or short. ``measurement``: how
#: and when the runs were launched and measured. ``apparatus``: the rig — controls, confounds, cells that
#: did not pool. ``comparisons``: how the contrasts against the control were tested and corrected.
DisclosureSource = Literal[
    "chart", "arms", "surface", "time_axis", "generation", "runs", "measurement", "apparatus", "comparisons"
]


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


#: The text roles that are part of one finding, and so name it.
FINDING_ROLES: frozenset[str] = frozenset({"finding_title", "finding_body", "caveat", "carried_forward"})


#: Only whitespace — what an empty author field, or a chart block's empty error, reads as.
_BLANK = r"^\s*$"

#: Some non-whitespace — what a chart block's error is when it carries one.
_SAID = r"\S"


class TextBlock(_Block):
    """What the analysis's author wrote, exactly as written, with the facts code states beside it."""

    model_config = ConfigDict(
        json_schema_extra={
            # A finding's own words name it (``_a_findings_own_words_name_it``), stated in the schema too.
            "if": {"properties": {"role": {"enum": sorted(FINDING_ROLES)}}, "required": ["role"]},
            "then": {"properties": {"finding": {"type": "integer"}}, "required": ["finding"]},
        }
    )

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
        min_length=1,
        description=(
            "Which table this is: `evidence`, `arms`, `surface`, `unadjudicated_bars`, `comparisons` (the contrasts "
            "against the control, as code tested them) or `questions` (the declared questions, on a code-only report)."
        ),
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
    """A chart: a finding's (its intent, or why the stored chart cannot be drawn), or one code chose.

    A chart code chose — on a code-only report, one per measure the surface can draw — names no finding.
    """

    model_config = ConfigDict(
        json_schema_extra={
            # Exactly one of an intent and the reason it cannot be drawn, and the intent of the block's own type
            # (``_drawn_or_says_why_not``), stated in the schema too.
            "if": {"properties": {"intent": {"type": "null"}}, "required": ["intent"]},
            "then": {"properties": {"error": {"type": "string", "pattern": _SAID}}},
            "else": {"properties": {"error": {"type": "string", "pattern": _BLANK}}},
            "allOf": [
                {
                    "if": {"properties": {"viz_type": {"const": chart_type}}, "required": ["viz_type"]},
                    "then": {
                        "properties": {
                            "intent": {"anyOf": [{"type": "null"}, {"properties": {"type": {"const": chart_type}}}]}
                        }
                    },
                }
                for chart_type in get_args(ChartType)
            ],
        }
    )

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
    """What the report is a report of, and how that analysis was generated — or that none was."""

    analysis_id: str | None = Field(
        default=None, min_length=1, description="The analysis the report renders; None on a code-only report."
    )
    campaign_id: str = Field(min_length=1, description="The campaign the analysis is of.")
    scope_id: str = Field(min_length=1, description="The scope both live in.")
    subject_id: str = Field(min_length=1, description="The analysed subject.")
    subject_kind: str = Field(description="The subject's kind; empty when the campaign declared none.")
    behavior: str = Field(min_length=1, description="The behaviour under analysis.")
    generated_at: str = Field(
        min_length=1,
        description=(
            "When the analysis was generated (ISO-8601); on a code-only report, when the evidence was assembled for it."
        ),
    )
    generator_model: str | None = Field(
        default=None,
        min_length=1,
        description="The model that wrote the analysis, as the provider reported it; None on a code-only report.",
    )
    bundle_fingerprint: str = Field(
        min_length=1,
        description=(
            "The fingerprint of the evidence bundle the analysis was written over; on a code-only report, of the "
            "bundle the report was computed from."
        ),
    )


class Report(EvalBaseModel):
    """One analysis, as a document every surface renders. See the module docstring for the contract."""

    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                # A code-only report names no analysis or model and holds nothing an author wrote
                # (``_the_basis_matches_what_the_report_holds``); an analysis report names both.
                {
                    "if": {"properties": {"basis": {"const": "code_only"}}, "required": ["basis"]},
                    "then": {
                        "properties": {
                            "headline": {"type": "string", "pattern": _BLANK},
                            "finding_count": {"const": 0},
                            "source": {
                                "properties": {"analysis_id": {"type": "null"}, "generator_model": {"type": "null"}}
                            },
                            "blocks": {"items": {"not": {"properties": {"kind": {"const": "text"}}, "required": ["kind"]}}},
                        }
                    },
                    "else": {
                        "properties": {
                            "source": {
                                "properties": {"analysis_id": {"type": "string"}, "generator_model": {"type": "string"}},
                                "required": ["analysis_id", "generator_model"],
                            }
                        }
                    },
                },
                # A report with no findings links no block to one (``_positions_point_at_findings``, at the one
                # finding_count the schema can compare against).
                {
                    "if": {"properties": {"finding_count": {"const": 0}}, "required": ["finding_count"]},
                    "then": {
                        "properties": {
                            "blocks": {"items": {"properties": {"finding": {"type": "null"}, "rests_on": {"maxItems": 0}}}}
                        }
                    },
                },
            ]
        }
    )

    report_version: Literal[2] = Field(default=REPORT_VERSION, description="This shape's version.")
    basis: ReportBasis = Field(
        description=(
            "`analysis` when the report renders a generated analysis; `code_only` when no analysis exists and the "
            "report is the campaign's evidence as code computed it — no headline, no findings, no author's words."
        )
    )
    headline: ModelProse = Field(
        description="The author's headline, as written; empty when the author wrote none, and on a code-only report."
    )
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

    @model_validator(mode="after")
    def _the_basis_matches_what_the_report_holds(self) -> Self:
        """Refuse a report whose basis disagrees with its source or its blocks.

        An analysis report names its analysis and the model that wrote it. A code-only report names
        neither, and holds nothing an author wrote: no headline, no findings, no text block — so a
        reader can never take code's computation for an analyst's judgement, or the reverse.

        Raises:
            ValueError: The basis and what the report holds disagree.
        """
        source = self.source
        if self.basis == "analysis":
            if source.analysis_id is None or source.generator_model is None:
                raise ValueError("an analysis report names its analysis_id and generator_model in its source")
            return self
        if source.analysis_id is not None or source.generator_model is not None:
            raise ValueError(
                "a code-only report renders no analysis, so its source names no analysis_id or generator_model"
            )
        if self.headline.strip():
            raise ValueError("a code-only report has no author, so it carries no headline")
        if self.finding_count:
            raise ValueError(f"a code-only report has no findings, and this one counts {self.finding_count}")
        if authored := [index for index, block in enumerate(self.blocks) if isinstance(block, TextBlock)]:
            raise ValueError(f"a code-only report holds no author's words, and blocks {authored} are text blocks")
        return self

    def to_canonical_json(self) -> str:
        """The report as its canonical JSON — the form the published schema validates.

        Returns:
            The JSON text, indented, keys in model order.
        """
        return self.model_dump_json(indent=2)


def report_title(report: Report) -> str:
    """The report's title, as every serializer prints it: the author's headline, or what a code-only report is.

    Args:
        report: The report.

    Returns:
        The title, one line before escaping.
    """
    if report.basis == "code_only":
        return f"Campaign {report.source.campaign_id}: its evidence, with no analysis"
    return report.headline.strip() or "(blank headline)"


def report_byline(report: Report) -> str:
    """What the report is of and how it was made, as every serializer prints it under the title.

    Args:
        report: The report.

    Returns:
        The line, before escaping.
    """
    source = report.source
    if report.basis == "code_only":
        return (
            f"Code-only report of campaign {source.campaign_id} — {source.behavior}; computed from its evidence on "
            f"{source.generated_at}. No analysis was generated."
        )
    return (
        f"Analysis {source.analysis_id} of campaign {source.campaign_id} — {source.behavior}; generated "
        f"{source.generated_at} by {source.generator_model}."
    )


__all__ = [
    "REPORT_VERSION",
    "SECTION_TITLES",
    "ChartBlock",
    "DisclosureBlock",
    "DisclosureSource",
    "FINDING_ROLES",
    "Fact",
    "Report",
    "ReportBasis",
    "ReportBlock",
    "ReportSection",
    "ReportSource",
    "TableBlock",
    "TableColumn",
    "TextBlock",
    "TextRole",
    "finding_number",
    "report_byline",
    "report_title",
]
