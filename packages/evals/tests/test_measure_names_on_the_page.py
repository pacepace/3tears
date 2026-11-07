"""A goal check is named by a name in every header and every lead, and its expression is the glossary's definition of
that name, once.

The owner's ruling of 2026-11-09 (Q11): a check's expression is code, and a long one (a production campaign's
``call_count(...) >= 1 or ...``, laid out over four lines) made a paragraph of a table header and a lead. So a check is
named by the display name its template declares, else ``Goal check``, numbered where an aspect reads several
(``AnalysisContextBundle.named_checks``), and the page's glossary pairs the name with the expression it stands for.

The shape here is one aspect, production against v2, graded on a four-line check (:data:`LONG`) and a short one, with
the aspect's graders listing the long check first, so the lead reads it, and v2 shown worse on it. Each page is laid
out as ``campaign_report`` lays it out by default (``trials="table"``, no trial's record), so the glossary is the one
place the expression stands.
"""

from __future__ import annotations

import re
import html
from html import escape

import pytest

from threetears.evals.analysis import (
    build_code_only_report,
    build_campaign_report,
    campaign_report_html,
    campaign_report_markdown,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.report.model import DisclosureBlock, ReportBlock, TableBlock
from threetears.evals.analysis.style import CHECK_LINK_TITLE, check_names
from threetears.evals.kernel.metrics import goal_check_measure

from packages.evals.tests.campaign_support import ASSEMBLED_AT
from packages.evals.tests.report_shapes_support import ArmTrials, AspectTrials, Shape, shape_of

#: A check laid out over four lines, as a production template wrote it.
LONG = (
    '(\n    call_count("discord.send_message") >= 1\n    or call_count("music.queue_voice") >= 1\n'
    '    or call_count("discord.add_reaction") >= 1\n)'
)

#: The long check on one line, as the glossary prints it.
LONG_LINE = (
    '( call_count("discord.send_message") >= 1 or call_count("music.queue_voice") >= 1 or '
    'call_count("discord.add_reaction") >= 1 )'
)

#: The aspect's second check.
SHORT = "reply.cites_policy"

#: What marks text as a check's expression rather than a name: a call, a comparison or a dotted path.
_EXPRESSION = re.compile(r"call_count\(|>=|==|reply\.cites")

#: The glossary's entry for each check, in Markdown and in HTML.
_ENTRIES = {
    "markdown": [
        f"- **Goal check 1**: The goal check `{LONG_LINE}`, which code runs on every trial.",
        f"- **Goal check 2**: The goal check `{SHORT}`, which code runs on every trial.",
    ],
    "html": [
        f'<dt id="g-goal-check-1">Goal check 1</dt><dd>The goal check <code>{escape(LONG_LINE)}</code>, which code runs '
        "on every trial.</dd>",
        f'<dt id="g-goal-check-2">Goal check 2</dt><dd>The goal check <code>{escape(SHORT)}</code>, which code runs on '
        "every trial.</dd>",
    ],
}


def _shape() -> Shape:
    """Production passes the long check on eighteen cases, v2 on eight; both pass the short one throughout."""
    shape = shape_of(
        "long check",
        [
            AspectTrials(
                "character",
                [ArmTrials("production", passing=18), ArmTrials("v2", passing=8)],
                expressions=(LONG, SHORT),
                measure_order=(goal_check_measure(LONG),),
            )
        ],
        control=True,
    )
    reports = {
        label: build_code_only_report(
            bundle,
            assembled_at=ASSEMBLED_AT,
            campaign_name=shape.name,
            arm_labels=shape.readings.arm_labels(),
            trials="table",
        )
        for label, bundle in shape.bundles.items()
    }
    return Shape(
        name=shape.name,
        campaign=shape.campaign,
        bundles=shape.bundles,
        reports=reports,
        readings=shape.readings,
        arms=shape.arms,
    )


def _rendered(page: str, form: str) -> str:
    shape = _shape()
    if page == "campaign":
        built = build_campaign_report(shape.campaign, shape.readings, shape.reports)
        return campaign_report_markdown(built) if form == "markdown" else campaign_report_html(built)
    report = shape.reports["character"]
    return report_markdown(report) if form == "markdown" else report_html(report)


def _blocks(page: str) -> list[ReportBlock]:
    """Every block the page lays out, its folds' included."""
    shape = _shape()
    if page == "aspect":
        return list(shape.reports["character"].blocks)
    built = build_campaign_report(shape.campaign, shape.readings, shape.reports)
    return [*built.blocks, *(block for fold in built.aspects for block in fold.report.blocks)]


def _headers(said: str, form: str) -> list[str]:
    """Every column header the rendered page prints: a Markdown table's head row, an HTML ``<th>``."""
    if form == "html":
        return re.findall(r"<th\b[^>]*>(.*?)</th>", said, flags=re.DOTALL)
    lines = said.splitlines()
    return [
        line for line, under in zip(lines, lines[1:], strict=False) if re.fullmatch(r"\|(?:\s*:?-+:?\s*\|)+", under)
    ]


@pytest.mark.parametrize("form", ["markdown", "html"])
@pytest.mark.parametrize("page", ["campaign", "aspect"])
def test_no_header_prints_a_check_s_expression(page: str, form: str) -> None:
    headers = _headers(_rendered(page, form), form)
    assert headers, "the page prints tables"
    assert not [header for header in headers if _EXPRESSION.search(header)]


@pytest.mark.parametrize("page", ["campaign", "aspect"])
def test_every_table_names_each_check_by_its_name(page: str) -> None:
    tables = [block for block in _blocks(page) if isinstance(block, TableBlock)]
    headers = [column.header for table in tables for column in table.columns]
    cells = [
        str(row[key].value) for table in tables for row in table.rows for key in ("reading", "measure") if key in row
    ]
    assert not [text for text in [*headers, *cells] if _EXPRESSION.search(text)]
    assert "Goal check 1" in cells and "Goal check 2" in cells, "the readings table names each check"


@pytest.mark.parametrize(
    ("page", "lead"),
    [
        ("campaign", "Goal check 1: v2 worse than production."),
        ("aspect", "Goal check 1: v2 worse than production; Pass rate: v2 worse than production."),
    ],
)
def test_the_lead_names_the_check_it_reads_by_its_name(page: str, lead: str) -> None:
    (verdict, *_) = [
        block for block in _blocks(page) if isinstance(block, DisclosureBlock) and block.source == "verdict"
    ]
    assert verdict.text.startswith(lead)
    assert not _EXPRESSION.search(verdict.text)


@pytest.mark.parametrize("form", ["markdown", "html"])
@pytest.mark.parametrize("page", ["campaign", "aspect"])
def test_the_glossary_defines_each_check_s_name_by_its_expression_the_one_place_it_stands(page: str, form: str) -> None:
    said = _rendered(page, form)
    for entry in _ENTRIES[form]:
        assert said.count(entry) == 1
    # With the entries aside, no expression is printed, in text or in an attribute.
    rest = said
    for entry in _ENTRIES[form]:
        rest = rest.replace(entry, "")
    assert LONG not in rest and escape(LONG) not in rest
    assert not re.search(r"call_count\(", rest)


def test_a_lead_s_name_links_its_glossary_entry_and_no_attribute_carries_the_expression() -> None:
    said = _rendered("campaign", "html")
    link = re.search(r'<a href="#[^"]+" title="([^"]*)">Goal check 1</a>: v2 worse than production', said)
    assert link is not None
    # The expression is a host's text: the hover says where it is, and no attribute anywhere carries it.
    assert link.group(1) == escape(CHECK_LINK_TITLE)
    assert not [value for value in re.findall(r'="([^"]*)"', said) if "call_count(" in html.unescape(value)]


def test_the_readings_a_campaign_freezes_keep_the_check_s_expression() -> None:
    # The readings are fingerprinted with the campaign's bundle, so no name a page gives a check moves them.
    shape = _shape()
    named = {reading.name for reading in shape.readings.measures if reading.measure == goal_check_measure(LONG)}
    assert named <= {f"`{LONG}`"}


@pytest.mark.parametrize(
    ("checks", "declared", "names"),
    [
        (["b", "a"], {}, {"a": "Goal check 1", "b": "Goal check 2"}),
        (["a", "a"], {}, {"a": "Goal check"}),
        (["b", "a"], {"a": "Total right"}, {"a": "Total right", "b": "Goal check"}),
        (["c", "b", "a"], {"b": "Cites policy"}, {"a": "Goal check 1", "b": "Cites policy", "c": "Goal check 2"}),
    ],
    ids=["two unnamed", "one unnamed", "one named, one not", "numbered past a named one"],
)
def test_an_unnamed_check_is_a_goal_check_numbered_only_beside_another_unnamed_one(
    checks: list[str], declared: dict[str, str], names: dict[str, str]
) -> None:
    assert check_names(checks, declared) == names
