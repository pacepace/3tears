"""Nothing a field carries is markup in a report's Markdown, wherever in a line it sits.

A host that renders Markdown with raw HTML turns ``<script>`` or ``<img onerror=...>`` into an element wherever it
sits in a line, not only where a line begins. These tests hold every writer of a report's Markdown to one rule: the
page, rendered by a Markdown parser with raw HTML on, holds no element but the ones the package writes itself
(headings, tables, ``<details>`` and its ``<summary>``, the glossary's anchors), no attribute but the ones those
carry, and shows each field's text as it was written.

The hostile text comes from :mod:`~packages.evals.tests.hostile_page_support` (a campaign's page, an aspect's report)
and, for a report document's page, from a copy of Kairo's page with a marked hostile string in every free-text field,
each marker naming the field so a field the writer printed raw is the one the failure names.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from markdown_it import MarkdownIt

from packages.evals.tests.clock_support import frozen_clock
from packages.evals.tests.hostile_page_support import HOSTILE, hostile_campaign_page, hostile_report
from packages.evals.tests.report_doc_support import converted_doc, snapshot_of
from threetears.evals.analysis.report import Report
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock, TrialBlock
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report.md_parts import (
    chart_lines,
    escaped,
    inert,
    markdown_table,
    paragraphs,
    table_lines,
)
from threetears.evals.analysis.report.model import CampaignReport, Report as ReportModel
from threetears.evals.analysis.report.serialize_md import campaign_report_markdown, report_markdown
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_doc import ReportDoc

#: Every element a page the package writes holds: what Markdown makes of its headings, paragraphs, lists, tables,
#: bold, emphasis and code, and the ``<details>``, ``<summary>`` and ``<a id>`` the package writes itself.
_ALLOWED_TAGS = frozenset(
    {
        "h1",
        "h2",
        "h3",
        "h4",
        "p",
        "ul",
        "ol",
        "li",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "strong",
        "em",
        "code",
        "pre",
        "details",
        "summary",
        "a",
    }
)

#: The attributes the package's own elements carry: a fold's id and whether it is open, an anchor's id.
_ALLOWED_ATTRIBUTES = {"details": {"id", "open"}, "a": {"id"}}

#: Ids the hostile text claims, which a page of Markdown holds none of: its own are ``T<n>`` and ``g-check-<slug>``.
_CLAIMED_IDS = frozenset({"glossary", "aspect-billing"})


class _Page(HTMLParser):
    """What a rendered page holds: each element with its attributes, and its text as a reader reads it."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.other: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_comment(self, data: str) -> None:
        self.other.append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self.other.append(f"<!{decl}>")

    def handle_pi(self, data: str) -> None:
        self.other.append(f"<?{data}>")

    def unknown_decl(self, data: str) -> None:
        self.other.append(f"<![{data}]>")

    @property
    def text(self) -> str:
        return "".join(self.parts)

    @property
    def tags(self) -> Counter[str]:
        return Counter(tag for tag, _ in self.elements)


def _render(markdown: str) -> _Page:
    """``markdown`` as a CommonMark renderer with raw HTML on and GitHub's tables reads it, then parsed."""
    page = _Page()
    page.feed(MarkdownIt("commonmark", {"html": True}).enable("table").render(markdown))
    page.close()
    return page


def _unexpected(page: _Page) -> list[str]:
    """What the page holds beyond what the package writes: an element, an attribute, a comment, a declaration or an id
    the hostile text claims."""
    found = [f"<{tag}>" for tag in page.tags if tag not in _ALLOWED_TAGS]
    for tag, attributes in page.elements:
        if tag not in _ALLOWED_TAGS:
            found += [f"<{tag} {name}>" for name in attributes]
            continue
        allowed = _ALLOWED_ATTRIBUTES.get(tag, set())
        found += [f"<{tag} {name}>" for name in attributes if name not in allowed]
        found += [f"id={value}" for name, value in attributes.items() if name == "id" and value in _CLAIMED_IDS]
    return [*found, *page.other]


def _said(text: str) -> str:
    """``text`` as a reader of a rendered line reads it: its runs of white space one space, since a heading or a cell is
    one line."""
    return " ".join(text.split())


def _holds_nothing_beyond_the_package(markdown: str) -> _Page:
    page = _render(markdown)
    assert _unexpected(page) == []
    return page


# =============================================================================
# The reader: it finds a hostile element where the package left one, and reads the same text where it did not
# =============================================================================


def test_the_reader_finds_every_element_an_unescaped_field_would_open() -> None:
    """The check is not vacuous: the parser, given a field the way the writers used to print it, sees the elements."""
    page = _render("| a |\n|---|\n| <script>alert(1)</script> |\n\n**<img src=x onerror=1>** and <a href=x>y</a>")
    assert {"script", "img", "a"} <= set(page.tags)
    found = _unexpected(page)
    assert {"<script>", "<img>", "<img src>", "<img onerror>", "<a href>"} <= set(found), found
    assert _unexpected(_render("a <!-- c --> b")) == ["<!-- c -->"]


#: Text a field may hold that a renderer could read as markup, an entity or a tag, and the places a writer puts it.
_FIELDS = [
    HOSTILE,
    "<script>alert(1)</script>",
    "text <b>bold</b> text",
    "a < b and c > d",
    "x<y",
    "<!-- comment -->",
    "<https://evil.example/x>",
    '<a href="javascript:alert(1)">x</a>',
    "&lt;script&gt;",
    "&#60;img&#62;",
    "&#x3c;img&#x3e;",
    "&amp;",
    "AT&T; R&D",
    "Q&A &co & more",
    "\\<b>x",
    "\\\\<b>x",
    "C:\\<dir>\\&lt;",
    "trailing <",
    "<",
    "</details></summary>",
    "<details open><summary>x</summary>",
]


def _in_places(field: str) -> dict[str, str]:
    """``field`` in each place a writer prints one: a paragraph, a heading, a bold lead-in, a table cell."""
    return {
        "paragraph": inert(field),
        "paragraph list": "\n".join(paragraphs(f"- {field}\n- {field}")),
        "heading": f"## {escaped(field)}",
        "bold": f"**{escaped(field)}** after",
        "cell": "\n".join(markdown_table([field], [[field]])),
        "summary": f"<details><summary>{escaped(field)}</summary>\n\nbody\n\n</details>",
    }


@pytest.mark.parametrize("field", _FIELDS)
def test_a_field_is_text_wherever_it_sits_and_reads_as_it_was_written(field: str) -> None:
    for place, markdown in _in_places(field).items():
        page = _render(markdown)
        assert _unexpected(page) == [], place
        assert _said(field) in _said(page.text), (place, page.text)


def test_text_already_escaped_reads_as_its_own_source_so_a_field_is_escaped_once() -> None:
    assert _render(inert(escaped("<b>"))).text.strip() == "&lt;b>"


def test_escaped_leaves_what_is_not_markup_as_it_was() -> None:
    for text in (
        "plain words",
        "5 > 3",
        "a & b",
        "R&D",
        "x &y z",
        "50% | 60%",
        "p < 0.001",
        "5<6",
        "p95 > 250 ms",
        "a <= b",
        "a << b",
        'say "hi"',
        "",
    ):
        assert escaped(text) == text


def test_escaped_names_the_entities_it_makes() -> None:
    assert escaped("<b>") == "&lt;b>"
    assert escaped("&lt;") == "&amp;lt;"
    assert escaped("&#60;") == "&amp;#60;"
    assert escaped("\\<") == "&#92;&lt;"
    assert escaped("a <") == "a &lt;"
    assert escaped("p < 0.001 and 5<6") == "p < 0.001 and 5<6"
    assert escaped('a"b<') == 'a"b&lt;'
    assert escaped('a"b<', attribute=True) == "a&quot;b&lt;"


# =============================================================================
# Each path the package writes Markdown by
# =============================================================================


def test_a_table_cell_and_a_header_hold_no_markup() -> None:
    lines = markdown_table([HOSTILE, "plain"], [[HOSTILE, "a | b"], ["x", HOSTILE]])
    page = _holds_nothing_beyond_the_package("\n".join(lines))
    assert page.tags["table"] == 1 and page.tags["td"] == 4 and page.tags["th"] == 2
    assert _said(page.text).count(_said(HOSTILE)) == 3


def test_a_chart_s_title_and_every_value_it_draws_hold_no_markup() -> None:
    report = hostile_report()
    charts = [block for block in report.blocks if isinstance(block, ChartBlock)]
    assert len(charts) > 10
    for chart in charts:
        _holds_nothing_beyond_the_package("\n".join(chart_lines(chart)))


def test_a_table_block_s_rows_and_total_hold_no_markup() -> None:
    report = hostile_report()
    tables = [block for block in report.blocks if isinstance(block, TableBlock)]
    assert tables
    for table in tables:
        page = _holds_nothing_beyond_the_package("\n".join(table_lines(table)))
        assert _said(HOSTILE) in _said(page.text)


def test_a_report_of_hostile_fields_holds_no_element_the_package_did_not_write() -> None:
    """The title, the verdict and what warns, each chart's title and values, each table, each trial's summary and
    record and every other disclosure, one report."""
    report = _with_every_fold_and_world(hostile_report())
    markdown = report_markdown(report)
    page = _holds_nothing_beyond_the_package(markdown)
    assert page.tags["details"] >= 3 and page.tags["table"] >= 3 and page.tags["h1"] == 1
    # The title is a heading that reads as the field was written.
    assert re.search(rf"^# {re.escape(escaped(_said(HOSTILE)))}$", markdown, re.M)
    assert _said(page.text).count(_said(HOSTILE)) >= 10


async def test_a_campaign_page_of_hostile_fields_holds_no_element_the_package_did_not_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The campaign's name, its arms' and aspects' labels, each fold's line and its second folded page, the glossary's
    check names and the aspects it names them on."""
    page = await hostile_campaign_page()
    _name_hostile_checks(monkeypatch)
    markdown = campaign_report_markdown(page)
    read = _holds_nothing_beyond_the_package(markdown)
    assert read.tags["details"] >= 6 and read.tags["h1"] == 1
    # Each aspect's fold: the heading (with its label) and the summary opening its page.
    assert read.tags["h2"] >= 2 and read.tags["summary"] >= 4
    assert any(line.startswith("- <a id=") and "&lt;" in line for line in markdown.splitlines())
    assert _said(read.text).count(_said(HOSTILE)) >= 4


async def test_a_report_document_s_page_and_the_record_after_it_hold_no_element_the_package_did_not_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every free-text field of a written report's page (masthead, headline, lede, flag, confidence, decision, options,
    claims with their captions and notes, explore, next steps, the appendix's tables, omitted, Method, Checks and
    findings) holds a hostile string naming the field; each prints as the text it was."""
    doc = _hostile_doc_page()
    markdown = doc_page_markdown(doc)
    page = _holds_nothing_beyond_the_package(markdown)
    printed = page.text
    missing = [field for field in _PRINTED if _said(_marker(field)) not in _said(printed)]
    assert missing == [], "fields the page did not print, or printed as something other than their text"
    assert page.tags["h1"] == 1 and page.tags["table"] >= 3
    # Then as the campaign page's evidence record: the aspects' folds and the glossary after the document.
    campaign = (await hostile_campaign_page()).model_copy(update={"doc_page": doc, "doc_trials": True})
    _name_hostile_checks(monkeypatch)
    after = _holds_nothing_beyond_the_package(campaign_report_markdown(campaign))
    assert after.tags["details"] >= 6


def test_a_check_s_row_in_the_documents_own_checks_list_holds_no_markup() -> None:
    doc = _hostile_doc_page()
    assert doc.checks, "the document names a goal check, so its Checks list is written"
    markdown = doc_page_markdown(doc)
    row = next(line for line in markdown.splitlines() if line.startswith("- <a id="))
    page = _holds_nothing_beyond_the_package(row)
    assert _said(doc.checks[0].name) in _said(page.text)


# =============================================================================
# One writer of the escape
# =============================================================================

_REPORT = Path(__file__).parents[1] / "src" / "threetears" / "evals" / "analysis" / "report"


@pytest.mark.parametrize("module", ["md_parts.py", "serialize_md.py", "doc_page/markdown.py"])
def test_no_markdown_writer_escapes_markup_by_a_way_of_its_own(module: str) -> None:
    """:func:`escaped` is the one function that makes a field's text safe: no writer imports ``html`` for its
    ``escape`` or writes an entity of its own."""
    tree = ast.parse((_REPORT / module).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert "html" not in imported, f"{module} escapes by a way of its own"
    entities = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "&lt;" in node.value
    ]
    docstrings = {ast.get_docstring(node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.Module)}
    assert [text for text in entities if text not in docstrings and module != "md_parts.py"] == []


# =============================================================================
# The pages these read
# =============================================================================


def _name_hostile_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every page a goal check whose name, expression and aspect carry markup, the glossary's one source."""
    entries = [(HOSTILE, [(f"{HOSTILE} > 1", "billing"), (f"{HOSTILE} < 2", HOSTILE)])]
    for model in (CampaignReport, ReportModel):
        monkeypatch.setattr(model, "check_glossary", property(lambda self: entries))


def _with_every_fold_and_world(report: Report) -> Report:
    """A hostile report whose blocks also fold under a hostile name, one a ``<details>`` table, and whose trial holds
    a hostile world state: the lines of a report the fixture's own blocks do not reach."""
    table = next(block for block in report.blocks if isinstance(block, TableBlock))
    folded = table.model_copy(update={"name": "folded", "fold": HOSTILE})
    drill = table.model_copy(update={"name": "drill", "details": True})
    blocks = [
        block.model_copy(update={"world": HOSTILE, "world_parts": [HOSTILE, "orders"]})
        if isinstance(block, TrialBlock)
        else block
        for block in report.blocks
    ]
    return report.model_copy(update={"blocks": [*blocks, folded, drill]})


#: The fields that name a kind, a place or a figure rather than carry words, which a hostile string would not be
#: allowed in: a kind, a key, an id, a colour, a tag (one capital letter) and the like.
_STRUCTURE = frozenset(
    {
        "section",
        "anchor",
        "fold",
        "kind",
        "key",
        "color",
        "form",
        "weight",
        "id",
        "ref",
        "status",
        "block",
        "unit",
        "run_date",
        "name",
        "chart",
        "pattern",
        "fmt",
        "level",
        "domain",
        "reference",
        "tag",
    }
)


def _marker(field: str) -> str:
    """The hostile string set in a field, naming the field."""
    return f"<script data-field='{field}'>alert('{field}')</script><img src=x onerror=\"alert('{field}')\">"


def _marked(node: Any, path: str = "", key: str = "") -> Any:
    """``node`` with a marked hostile string before every free-text string in it, the marker naming the field's path."""
    if isinstance(node, str):
        return node if key in _STRUCTURE or not node else f"{_marker(path)} {node}"
    if isinstance(node, dict):
        return {name: _marked(value, f"{path}.{name}".lstrip("."), name) for name, value in node.items()}
    if isinstance(node, list):
        return [_marked(value, f"{path}[{at}]", key) for at, value in enumerate(node)]
    return node


#: The fields a written report's page prints as text, which each must show as the hostile string it holds.
_PRINTED = [
    "masthead.system",
    "headline",
    "lede",
    "flag",
    "confidence.words",
    "confidence.caveats[0].label",
    "confidence.caveats[0].text",
    "decision[0].title",
    "decision[0].text",
    "claims[0].text",
    "claims[0].evidence[0].caption",
    "next[0].text",
    "next[0].settles",
    "arms_key.eyebrow",
    "arms_key.takeaway",
    "full_results[0].title",
    "method[0].term",
    "method[0].text",
]


def _hostile_doc_page() -> DocPage:
    """Kairo's written report with a marked hostile string before every free-text field, and a goal check whose name
    and expression carry markup."""
    name = "prototype3/kairo"
    with frozen_clock():
        page = build(
            ReportDoc.model_validate(converted_doc(name)),
            ReportInput.model_validate(snapshot_of(name).report_input),
            strict=True,
        )
    dump = _marked(page.model_dump(mode="json"))
    dump["checks"] = [{"name": HOSTILE, "expression": f"{HOSTILE} > 1"}]
    return DocPage.model_validate(dump)
