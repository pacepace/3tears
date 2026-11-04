"""The parts a page draws a chart, a table or a decision with: each already resolved, every figure written out.

These were the planned page's parts; the planned page is retired with the report planner, and the report document's
page (:mod:`~threetears.evals.analysis.report.doc_page`) and the analyst's blocks
(:mod:`~threetears.evals.analysis.analyst.blocks`) draw with the ones left: the masthead, a decision slot, a chart
and the arms key. Nothing here reads the input: every figure is already written out, every chart already resolved.

**Who wrote what.** A field a writer wrote, its figure keys written out, is typed model prose; a line code wrote
(slot titles, a chart's notes, the masthead's frame) is plain text.

**Charts and tables are the report's blocks.** A chart is a :class:`~threetears.evals.analysis.report.blocks.ChartBlock`
carrying the kit's data, a table a :class:`~threetears.evals.analysis.report.blocks.TableBlock`, so each prints as its
values table, carries the kit's JSON and is laid out by the writers every page shares. A block's ``section`` is
required of it; a part sets it to :data:`PLANNED_SECTION`, which the page that lays the part out ignores, since that
page's own order is fixed by its model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from threetears.evals.analysis.report.blocks import ChartBlock, ReportSection, TableBlock
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.prose import ModelProse

__all__ = [
    "PLANNED_SECTION",
    "PageArmsKey",
    "PageChart",
    "PageMasthead",
    "PageSlot",
]

#: The ``section`` every block on a planned page carries: required of a block, ignored by the planned page's writers.
PLANNED_SECTION: ReportSection = "memo"


class PageMasthead(EvalDocumentModel):
    """The page's one-line masthead (GENERATOR §2.1), its figure keys written out."""

    kind: str = Field(description="What the campaign is (`Change · control vs candidates`).")
    system: str = Field(description="The system under test.")
    counts: str = Field(description="Arms, aspects or tasks, cases × repeats; empty for none.")
    run_date: str = Field(description="The run date; empty where the input states none.")
    synthetic: bool = Field(description="Whether the data is made up for an example.")


class PageSlot(EvalDocumentModel):
    """One decision slot: its key, its title in code's words, the planner's text, and whether it is the one set
    apart (the recommendation, or the lean)."""

    key: str = Field(description="The slot's key, as the plan names it.")
    title: str = Field(description="Its title, by mode and kind (GENERATOR §2.5).")
    text: ModelProse = Field(description="The planner's text, its keys written out.")
    note: str = Field(
        default="",
        description=(
            "Code's line printed after the planner's text, never part of it: which interval a Watch slot's weakest "
            "margin is read on; empty for none."
        ),
    )
    highlighted: bool = Field(description="Whether it is the recommendation (decide) or the lean (tradeoff).")


class PageChart(EvalDocumentModel):
    """One chart as a page draws it: the planner's title, the resolved chart, and the lines code adds under it."""

    title: ModelProse = Field(description="The planner's title, its keys written out; empty where a headline says it.")
    chart: ChartBlock | TableBlock = Field(description="The chart with every value it draws, or a table.")
    notes: list[str] = Field(description="Code's captions: a comparison not in the evidence, what bold means.")


class PageArmsKey(EvalDocumentModel):
    """The arms key (GENERATOR §2.8): each arm's full configuration once, under code's eyebrow and takeaway."""

    eyebrow: str = Field(description="Its eyebrow, in code's words.")
    takeaway: str = Field(description="Its one line, in code's words.")
    table: TableBlock = Field(description="A row per arm.")
