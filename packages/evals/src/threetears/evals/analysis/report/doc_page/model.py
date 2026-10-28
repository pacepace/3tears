"""A report document's page as a document: every part the page draws, in its order, already resolved.

:func:`~threetears.evals.analysis.report.doc_page.build.build` builds one; the writers
(:func:`~threetears.evals.analysis.report.doc_page.html.doc_page_html`,
:func:`~threetears.evals.analysis.report.doc_page.markdown.doc_page_markdown`,
:func:`~threetears.evals.analysis.report.doc_page.text.doc_page_text`) lay it out. Nothing here reads the input or
the document: every figure key is already written out, every block already drawn.

**Who wrote what.** A field the analyst wrote, its figure keys written out, is typed model prose; a line code wrote
(the form words, a caveat's code label and checked line, the slot titles, a block's notes, the masthead) is plain
text. The parts the planned page shares (the masthead, a drawn chart, the arms key, a decision entry) are its models.

**Above the appendix, then the appendix.** The masthead, the first screen (headline, lede, flag, confidence, decision,
options), the claims with their evidence, explore and next steps are what a reader reads first; the appendix is the
record under them: the arms key, every table the input holds, what was left out and why, Method, the editor's
findings left open, and the made line.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from threetears.evals.analysis.report.blocks import TableBlock
from threetears.evals.analysis.report.doc_page.words import CAVEATS_TEMPLATE
from threetears.evals.analysis.report.md_parts import one_line
from threetears.evals.analysis.report.plan_page.model import PageArmsKey, PageChart, PageMasthead, PageSlot
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.prose import ModelProse

__all__ = [
    "DocCaveatLine",
    "DocClaim",
    "DocConfidence",
    "DocEvidence",
    "DocFinding",
    "DocNext",
    "DocOmitted",
    "DocPage",
    "DocTerm",
    "caveats_line",
]


class DocCaveatLine(EvalDocumentModel):
    """One caveat the confidence carries, as the page states it: its label, then the analyst's clause (open) or
    code's checked line (immaterial)."""

    label: str = Field(description="The caveat's name: code's for a caveat code assigns, else the analyst's.")
    text: str = Field(
        description="What would make it material (the analyst's, keys written out), or code's checked line."
    )
    checked: bool = Field(description="Whether code proved it immaterial, so the text is code's checked line.")


class DocConfidence(EvalDocumentModel):
    """How sure the call is: the form in code's words, then each caveat in the document's order."""

    form: Literal["high", "high_if_immaterial", "low"] = Field(description="The document's confidence form.")
    words: str = Field(description="The form, in code's words.")
    caveats: list[DocCaveatLine] = Field(description="Each caveat, in the document's order.")


class DocEvidence(EvalDocumentModel):
    """One evidence block under a claim: the block drawn (or what still draws of it), its caption, and what code fixed
    about it."""

    block: str = Field(description="The block's kind, as the document names it.")
    chart: PageChart | None = Field(
        description="The block as the page draws it; a table of its values where it no longer draws as its kind; None "
        "where nothing of it draws (a page read from a stored document)."
    )
    caption: ModelProse = Field(description="The analyst's line under the block, its keys written out.")
    level: str = Field(description="The interval level it prints; empty where it draws no interval.")
    unit: str = Field(description="The unit its values are in; empty for several or none.")
    domain: str = Field(description="Its value axis, low to high, where code fixes one; empty where it is fitted.")
    reference: str = Field(description="Its reference line and the figure it sits at; empty for none.")
    degraded: str = Field(description="Code's line where the block no longer draws as written; empty where it does.")


class DocClaim(EvalDocumentModel):
    """One claim, in page order: its text, the type size its weight sets, and its evidence."""

    id: str = Field(
        description="The claim's id, as the document and the editor's findings name it; never printed above the "
        "appendix."
    )
    place: int = Field(description="Its place on the page, from 1.")
    text: ModelProse = Field(description="The claim, its keys written out.")
    weight: Literal["lead", "support", "detail"] = Field(description="Its weight, which sets its type size alone.")
    evidence: list[DocEvidence] = Field(description="The one or two blocks that show it.")
    caveats: list[str] = Field(
        description="The labels of the caveats the document lands beside it, in its order, each distinct label once "
        "(a label several share says how many and the arms their records name)."
    )


class DocTerm(EvalDocumentModel):
    """One Method term, its definition, and the labels of the caveats the document lands on it."""

    term: ModelProse
    text: ModelProse
    caveats: list[str] = Field(
        description="The labels of the caveats landed on it, in the document's order, each distinct label once."
    )


class DocNext(EvalDocumentModel):
    """One next step and what it would settle."""

    text: ModelProse
    settles: str = Field(description="What its result would settle (the analyst's), its keys written out.")


class DocOmitted(EvalDocumentModel):
    """One thing the analyst left out of the body, and why."""

    what: ModelProse
    why: ModelProse


class DocFinding(EvalDocumentModel):
    """One editor finding left unresolved, as Method prints it."""

    about: str = Field(description="What it is about, in code's words: a claim by its place, or the page.")
    claim_place: int | None = Field(description="The claim's place on the page; None for the page as a whole.")
    severity: str = Field(description="Its severity, in code's words.")
    finding: ModelProse = Field(description="What the editor found.")


class DocPage(EvalDocumentModel):
    """A campaign page built from its report document: every part, already resolved, in the page's order."""

    title: str = Field(description="The campaign's name, the page's title.")
    masthead: PageMasthead
    degraded: str = Field(
        default="", description="Code's one note where part of a stored document no longer reads; empty where all does."
    )
    headline: ModelProse
    lede: ModelProse
    flag: ModelProse = Field(description="What a reader must know first; empty for none.")
    confidence: DocConfidence
    decision: list[PageSlot] = Field(description="The decision entries, in the document's order, titled by slot word.")
    options: TableBlock = Field(description="Every arm, with its settings, served model and cost, the call marked.")
    recommended_rows: list[int] = Field(description="The options table's rows the document recommends, by position.")
    claims: list[DocClaim] = Field(description="The claims, in the document's order.")
    explore: TableBlock | None = Field(description="What a reader could change, as prose rows; None for none.")
    next: list[DocNext] = Field(description="The next steps, after the claims; may be empty.")
    arms_key: PageArmsKey = Field(description="Each arm's full configuration once, first in the appendix.")
    full_results: list[TableBlock] = Field(description="Every table the input holds a row of.")
    omitted: list[DocOmitted] = Field(description="What was left out of the body, and why.")
    method: list[DocTerm] = Field(description="The method's terms, each with the caveats landed on it.")
    findings: list[DocFinding] = Field(description="The editor's findings left unresolved.")
    made: str = Field(default="", description="How the page was made, in one line; empty for none.")


def caveats_line(labels: Sequence[str]) -> str:
    """The line beside a claim or a Method term naming the caveats landed on it, as every writer prints it.

    The labels are already one per distinct label (the page builder says a shared label once); the Markdown, text and
    HTML writers each wrap this one line, so the three never word it apart.

    :param labels: The landed caveats' labels (:attr:`DocClaim.caveats`, :attr:`DocTerm.caveats`).
    :ptype labels: Sequence[str]
    :return: The line, each label on one line; empty for no label.
    :rtype: str
    """
    if not labels:
        return ""
    return CAVEATS_TEMPLATE.format(labels="; ".join(one_line(label) for label in labels))
