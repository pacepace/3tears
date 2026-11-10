"""A table's cell of figures takes a class no other rule of the page reads.

A cell of a column of figures (``4.5 (3.8 to 5.0)``) is classed so the stylesheet holds its line whole. The page's
stylesheet is the report's own rules and the kit's together, and the kit already styles its figure cards by class
(a surface, a border, padding); a cell sharing a card's class would draw as a card inside its table, every cell of a
column of figures padded and boxed, the table wider than its frame. So every rule of every stylesheet a page can carry
(the live page's, the static page's with its charts' rules, and the kit's own copy) that names a table cell's class
names it on the cell, and no script the kit runs makes an element of that class.
"""

from __future__ import annotations

import re

from packages.evals.tests.report_kit_support import KIT, report_of
from threetears.evals.analysis import report_html
from threetears.evals.analysis.report.model import TableBlock, TableCell, TableColumn

#: One selector's compound naming a class: everything between combinators around it.
_COMPOUND = r"[^\s>+~]*\.{name}(?![\w-])[^\s>+~]*"


def _page(*, static: bool = False) -> str:
    """A page holding a table with a column of figures, a column of intervals and a column of prose."""
    columns = [
        TableColumn(key="reading", header="Reading", term="reading"),
        TableColumn(key="control", header="▸ base", term="control"),
        TableColumn(key="arm", header="fast", term="arm mean"),
        TableColumn(key="interval", header="Interval", term="interval"),
        TableColumn(key="note", header="Note"),
    ]
    rows = [
        {
            "reading": TableCell(value="Score (judged, 1–5)"),
            "control": TableCell(value="4.5 (3.8 to 5.0)"),
            "arm": TableCell(value="420ms"),
            "interval": TableCell(value="-0.9 to +0.1"),
            "note": TableCell(value="what the judge read differs most where the reply ran long"),
        }
    ]
    table = TableBlock(
        section="contrasts", name="comparisons", title="Against base", columns=columns, rows=rows, order="as built",
        total_rows=len(rows),
    )  # fmt: skip
    return report_html(report_of([table]), static=static)


def _cell_classes(page: str) -> set[str]:
    """Every class a table cell of the page carries."""
    return {name for found in re.findall(r'<t[dh]\b[^>]*\bclass="([^"]+)"', page) for name in found.split()}


def _stylesheets() -> str:
    """Every stylesheet a page can carry: the live page's, the static page's (its charts' rules with it), and the kit's
    documentation copy, comments left out."""
    sheets = [
        sheet
        for static in (False, True)
        for sheet in re.findall(r"<style[^>]*>(.*?)</style>", _page(static=static), flags=re.DOTALL)
    ]
    sheets.append((KIT / "kit" / "kit.css").read_text())
    return re.sub(r"/\*.*?\*/", "", "\n".join(sheets), flags=re.DOTALL)


def _selectors(css: str) -> list[str]:
    """Every selector of every rule in ``css``, one per comma that separates selectors (never one inside ``:is(…)``)."""
    selectors: list[str] = []
    for group in re.findall(r"([^{}]+)\{", css):
        depth, part = 0, ""
        for char in group:
            depth += {"(": 1, ")": -1}.get(char, 0)
            if char == "," and depth == 0:
                selectors.append(part.strip())
                part = ""
            else:
                part += char
        selectors.append(part.strip())
    return [selector for selector in selectors if selector]


#: How a compound names a table cell: by its tag, or by a ``:is()`` of the cell tags alone.
_ON_A_CELL = re.compile(r"(?:td|th|:is\((?:td|th)(?:,(?:td|th))*\))(?![\w-])")


def test_a_figure_cell_carries_a_class() -> None:
    assert "figs" in _cell_classes(_page()), "the column of figures is classed, so its line holds whole"


def test_a_compound_on_a_cell_reads_as_one() -> None:
    names = [_ON_A_CELL.match(compound) is not None for compound in ("td.figs", ":is(th,td):first-child.unbroken")]
    assert names == [True, True]
    assert _ON_A_CELL.match(".mult .figs") is None and _ON_A_CELL.match("table.figs") is None
    assert _selectors(".tbl :is(th,td).x,.y{}") == [".tbl :is(th,td).x", ".y"]


def test_every_rule_naming_a_table_cells_class_names_it_on_the_cell() -> None:
    selectors = _selectors(_stylesheets())
    classes = _cell_classes(_page()) | _cell_classes(_page(static=True))
    misread = [
        (name, selector)
        for name in sorted(classes)
        for selector in selectors
        for compound in re.findall(_COMPOUND.format(name=re.escape(name)), selector)
        if _ON_A_CELL.match(compound) is None
    ]
    assert misread == [], f"a rule styles something else by a table cell's class: {misread}"


def test_no_script_of_the_kit_makes_an_element_of_a_figure_cells_class() -> None:
    scripts = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", _page(), flags=re.DOTALL))
    assert scripts, "the page runs the kit's script"
    for script in (scripts, (KIT / "kit" / "kit.js").read_text()):
        assert not re.search(r"""class\s*[:=]\s*["']figs["']""", script)
