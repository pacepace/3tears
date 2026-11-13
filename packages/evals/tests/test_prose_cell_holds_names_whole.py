"""A prose cell holds a hyphenated name whole, as a paragraph does.

A cell of 40 characters or more wraps at a measure, as a paragraph does (``td.wrap``). A paragraph on a written page
holds a model's name whole (``span.id``), so ``qwen3.8-flash`` never breaks after its hyphen; a prose cell once did
not, so a sensitivity table's sentence could break the name across two lines. A table block's cell and a chart's
values table alike now hold it.

Mutations that turn this file red: a ``wrap`` cell, in a table block or in a chart's values, laid out without
``held_whole``.
"""

from __future__ import annotations

import re

from threetears.evals.analysis.report import ChartBlock
from threetears.evals.analysis.report.blocks import TableBlock, TableCell, TableColumn
from threetears.evals.analysis.report.html_parts import chart_html, table_html

#: A model's name with a hyphen in it, and a sentence about it long enough to be a prose cell.
_NAME = "qwen3.8-flash"
_SENTENCE = f"On the uncorrected readings, {_NAME} (A) stops reading as within the margin below 11 points."


def _wrap_cells(markup: str) -> list[str]:
    return re.findall(r'<td class="wrap"[^>]*>(.*?)</td>', markup, re.S)


def test_a_table_block_s_prose_cell_holds_a_name_whole() -> None:
    markup = table_html(
        TableBlock(
            section="memo",
            name="sensitivity",
            title="Sensitivity",
            columns=[TableColumn(key="figure", header="Figure"), TableColumn(key="reading", header="Reading")],
            rows=[{"figure": TableCell(value="Pass rate"), "reading": TableCell(value=_SENTENCE)}],
            order="as read",
            total_rows=1,
        )
    )
    (cell,) = _wrap_cells(markup)
    assert f'<span class="id">{_NAME}</span>' in cell, cell


def test_a_chart_values_prose_cell_holds_a_name_whole() -> None:
    block = ChartBlock(
        section="memo",
        chart="bars",
        title="t",
        data={  # type: ignore[arg-type]
            "rows": [{"label": "Persona holds character", "value": -7.6, "note": _SENTENCE}],
            "domain": [-9, 6],
            "fmt": "+num:0",
        },
    )
    cells = _wrap_cells(chart_html(block, "c", static=False))
    held = [cell for cell in cells if _NAME in cell]
    assert held and all(f'<span class="id">{_NAME}</span>' in cell for cell in held), cells
