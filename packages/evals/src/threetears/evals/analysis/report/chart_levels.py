"""Each interval level a page draws, stated once (#1128; owner ruling, 2026-11-14).

A chart drawing intervals prints its level in its head (``95% intervals``). A page of several such charts printed it
on every one, so a first screen of three charts said ``95% intervals`` three times. A page now states each level once
and marks a chart whose level differs from the page's main one, by each chart's ``level_head``
(:func:`~threetears.evals.analysis.report.charts.drawn_head`):

- the page's main level is the one most of its charts draw, the first of them on a tie;
- the first chart at the main level prints it as its head always did; a later one prints nothing, the level the page
  already stated;
- a chart at any other level is marked: its head prints its level and, where its figure is not the main level's,
  ``not`` the main one (``98.3% intervals, not 95%``), so a reader never reads it at the level the page stated. A
  level differing only in its gloss (``95% (adjusted for 3 comparisons)`` or ``95% unpaired`` beside ``95%``) is
  marked by the gloss its head keeps.

The charts are read in the order a reader meets them; one chart, or charts all without a level, are left as they are.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import cast

from threetears.evals.analysis.report.blocks import ChartBlock
from threetears.evals.analysis.report.charts import GateScatterChart, ScatterChart, level_head

__all__ = ["levels_said"]


def _chart_level(block: ChartBlock) -> str | None:
    """The level a chart's head prints, as the chart carries it; None where its head prints none: a kind with no
    level, a scatter drawing no whisker, a chart whose panels carry levels of their own.

    :param block: The chart.
    :ptype block: ChartBlock
    :return: The level, or None.
    :rtype: str | None
    """
    if block.data is None:
        return block.level if block.chart == "cost_quality" and block.level and block.points else None
    level: str | None = getattr(block.data, "level", None)
    panels = getattr(block.data, "panels", None) or []
    if not level or any(getattr(panel, "level", None) not in (None, level) for panel in panels):
        return None
    if isinstance(block.data, (ScatterChart, GateScatterChart)):
        points = [*(block.data.points or []), *(point for panel in panels for point in (panel.points or []))]
        whiskered = any(
            (point.x_lo is not None and point.x_hi is not None) or (point.y_lo is not None and point.y_hi is not None)
            for point in points
        )
        return level if whiskered else None
    return level


def _marked_head(level: str, main: str) -> str:
    """The head of a chart at a level other than its page's main one: its level's own head, then ``not`` the main
    level's figure where the two figures differ (``98.3% intervals, not 95%``); a level differing only in its gloss
    (``95% unpaired``, ``95% (adjusted for 3 comparisons)``) is marked by the gloss its head keeps.

    :param level: The chart's level.
    :ptype level: str
    :param main: The page's main level.
    :ptype main: str
    :return: The head.
    :rtype: str
    """
    figure, main_figure = level.split(" ", 1)[0], main.split(" ", 1)[0]
    return level_head(level) if figure == main_figure else f"{level_head(level)}, not {main_figure}"


def levels_said[T](blocks: Sequence[T]) -> list[T]:
    """A page's blocks with each chart's head set to state each level once (this module's rule), in the order given,
    which is the order a reader meets them; every other block as it is.

    :param blocks: The page's blocks, in reading order.
    :ptype blocks: Sequence[T]
    :return: The blocks, each chart's ``level_head`` set where the rule moves it from the level's own head, which a
        chart whose head the rule leaves as it is keeps unset.
    :rtype: list[T]
    """
    levels = [_chart_level(block) if isinstance(block, ChartBlock) else None for block in blocks]
    counted = Counter(level for level in levels if level is not None)
    if sum(counted.values()) < 2:
        return list(blocks)
    # most_common keeps a tie in the order the levels were first met
    main = counted.most_common(1)[0][0]
    stated: set[str] = set()
    said: list[T] = []
    for block, level in zip(blocks, levels, strict=True):
        if level is None or not isinstance(block, ChartBlock):
            said.append(block)
            continue
        if level != main:
            head: str | None = _marked_head(level, main)
        else:
            head = "" if level in stated else None
        stated.add(level)
        said.append(block if head is None or head == level_head(level) else cast(T, _headed(block, head)))
    return said


def _headed(block: ChartBlock, head: str) -> ChartBlock:
    """The chart with its head set: on its kit data where it carries one, else on the block (a cost/quality chart)."""
    if block.data is not None:
        return block.model_copy(update={"data": block.data.model_copy(update={"level_head": head})})
    return block.model_copy(update={"level_head": head})
