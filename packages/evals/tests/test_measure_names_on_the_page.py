"""A goal check is named by what it tests in every header and every lead, and its expression is the glossary's
definition of that name, once.

The owner's ruling of 2026-11-09 (Q11): a check's expression is code, and a long one (a production campaign's
``call_count(...) >= 1 or ...``, laid out over four lines) made a paragraph of a table header and a lead. So a check is
named by the display name its template declares, else by what it tests, read off its expression alone
(``Refund amount check``, ``Send message or reply check``, :func:`~threetears.evals.analysis.style.check_names`;
a check testing more than two subjects, as the production check does, is a ``Goal check``), and the page's glossary
pairs the name with the expression it stands for.

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
from threetears.evals.analysis.report.md_parts import code_span
from threetears.evals.analysis.report.model import CampaignReport, DisclosureBlock, ReportBlock, TableBlock
from threetears.evals.analysis.style import CHECK_LINK_TITLE, check_names
from threetears.evals.kernel.declaration import BarOverride
from threetears.evals.kernel.metrics import goal_check_measure
from threetears.evals.ops.analysis import report_read

from packages.evals.tests.campaign_support import ASSEMBLED_AT
from packages.evals.tests.report_shapes_support import ArmTrials, AspectTrials, Shape, shape_of
from packages.evals.tests.test_hot_path_oracle import subject

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

#: What the page calls each, by what it tests: the long check tests three calls, more than a name joins, so it is a goal
#: check (the coordinator's ruling on the measure-names verify, N-2).
LONG_NAME, SHORT_NAME = "Goal check", "Reply cites policy check"

#: What marks text as a check's expression rather than a name: a call, a comparison or a dotted path.
_EXPRESSION = re.compile(r"call_count\(|>=|==|reply\.cites")

#: The glossary's entry for each check, in Markdown and in HTML.
_ENTRIES = {
    "markdown": [
        f"- **{LONG_NAME}**: The goal check `{LONG_LINE}`, which code runs on every trial.",
        f"- **{SHORT_NAME}**: The goal check `{SHORT}`, which code runs on every trial.",
    ],
    "html": [
        f'<dt id="g-check-goal-check">{LONG_NAME}</dt><dd>The goal check <code>{escape(LONG_LINE)}</code>, '
        "which code runs on every trial.</dd>",
        f'<dt id="g-check-reply-cites-policy-check">{SHORT_NAME}</dt><dd>The goal check <code>{escape(SHORT)}</code>, '
        "which code runs on every trial.</dd>",
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
    assert LONG_NAME in cells and SHORT_NAME in cells, "the readings table names each check"


@pytest.mark.parametrize(
    ("page", "lead"),
    [
        ("campaign", f"{LONG_NAME}: v2 worse than production."),
        ("aspect", f"{LONG_NAME}: v2 worse than production; Pass rate: v2 worse than production."),
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
    link = re.search(rf'<a href="#[^"]+" title="([^"]*)">{LONG_NAME}</a>: v2 worse than production', said)
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
    ("expression", "subject"),
    [
        (LONG, None),
        ('call_count("discord.send_message") >= 1', "Send message calls"),
        ('call_count("discord.send_message") >= 1 or call_count("discord.reply") >= 1', "Send message or reply"),
        ('call_count("discord.send_message") >= 1 or refused("discord.reply")', "Send message calls or reply refusals"),
        ("state.ticket.status == 'closed' and state.ticket.tags.length >= 1", "Ticket status and ticket tags"),
        ("state.order.total > 0 and state.refund.amount <= state.order.total", "Refund amount"),
        ("state.refund is not None and state.refund.amount > 0", "Refund amount"),
        ('call_count("x.[click](javascript:alert(1)) <img src=x onerror=alert(1)>") >= 1', None),
        ('call_count("") >= 1', None),
        ("state.refund.amount <= state.order.total", "Refund amount"),
        ("state.ticket.tags.length >= 1", "Ticket tags"),
        ("field_accuracy >= 0.92", "Field accuracy"),
        ('all(it.segue_text.length > 0 for it in calls("music.queue_track"))', "Queue track calls"),
        ('any(it.sku == "X" for it in state.inventory.orders)', "Inventory orders"),
        ('not fired("support.escalation")', "Escalation firing"),
        ("passed()", "Deliberate pass"),
        ("1 == 1", None),
        ("answer ==", None),
    ],
)
def test_a_check_s_subject_is_read_off_its_expression_alone(expression: str, subject: str | None) -> None:
    # One check alone: named by its subject, else a goal check where its expression names none.
    assert check_names([expression], {}) == {expression: f"{subject} check" if subject else "Goal check"}


@pytest.mark.parametrize(
    ("checks", "declared", "names"),
    [
        ([LONG, SHORT], {}, {LONG: LONG_NAME, SHORT: SHORT_NAME}),
        (
            ["state.ticket.status == 'open'", "state.ticket.status == 'closed'"],
            {},
            {
                "state.ticket.status == 'closed'": "Ticket status check 1",
                "state.ticket.status == 'open'": "Ticket status check 2",
            },
        ),
        ([SHORT, "1 == 1"], {SHORT: "Cites policy"}, {"1 == 1": "Goal check", SHORT: "Cites policy"}),
        (["1 == 1", "2 == 2"], {}, {"1 == 1": "Goal check 1", "2 == 2": "Goal check 2"}),
        (
            ["state.refund.amount > 0"],
            {"x": "Refund amount check"},
            {"state.refund.amount > 0": "Refund amount check 1", "x": "Refund amount check"},
        ),
    ],
    ids=[
        "each by its subject",
        "a shared subject is numbered",
        "a declared name stands",
        "no subject",
        "beside a declared name",
    ],
)
def test_a_check_is_named_by_what_it_tests_numbered_only_where_two_share_a_name(
    checks: list[str], declared: dict[str, str], names: dict[str, str]
) -> None:
    assert check_names(checks, declared) == names


def _measurement(*aspects: tuple[str, tuple[str] | tuple[str, str]]) -> CampaignReport:
    """One arm measured on each aspect, graded on that aspect's checks."""
    shape = shape_of(
        "checked",
        [
            AspectTrials(label, [ArmTrials("solo", passing=14)], margin=None, expressions=checks)
            for label, checks in aspects
        ],
        control=False,
        question="How often does solo get it right?",
    )
    return shape.page()


def _scorecard_headers(page: CampaignReport) -> list[str]:
    (matrix,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "scorecard"]
    return [column.header for column in matrix.columns if column.key.startswith("measure:goal_state:")]


def test_one_expression_has_one_name_on_every_aspect_and_two_aspects_with_one_check_each_read_apart() -> None:
    refund = "state.refund.amount <= state.order.total"
    page = _measurement(("billing", (LONG,)), ("refunds", (refund,)))
    assert sorted(_scorecard_headers(page)) == [f"{LONG_NAME} (%)", "Refund amount check (%)"]
    alike = _measurement(("billing", (LONG,)), ("refunds", (LONG,)))
    assert _scorecard_headers(alike) == [f"{LONG_NAME} (%)"]
    assert [name for name, _ in alike.check_glossary] == [LONG_NAME]


def test_two_aspects_checks_that_share_a_name_head_their_columns_by_their_aspects() -> None:
    page = _measurement(
        ("billing", ("state.ticket.status == 'open'",)), ("refunds", ("state.ticket.status == 'closed'",))
    )
    assert sorted(_scorecard_headers(page)) == ["Ticket status check, billing (%)", "Ticket status check, refunds (%)"]
    (entry,) = page.check_glossary
    assert entry == (
        "Ticket status check",
        [("state.ticket.status == 'open'", "billing"), ("state.ticket.status == 'closed'", "refunds")],
    )


def test_a_check_s_column_header_links_its_glossary_entry() -> None:
    said = campaign_report_html(_measurement(("billing", (LONG, SHORT))))
    (header,) = re.findall(rf"<th[^>]*>(<a [^>]*>{LONG_NAME} \(%\)</a>)</th>", said)
    assert f'title="{escape(CHECK_LINK_TITLE)}"' in header


def test_a_name_inside_a_longer_one_is_never_linked_as_the_shorter() -> None:
    # Billing reads one check with no subject (Goal check), refunds two (Goal check 1, Goal check 2), each unproven.
    page = _measurement(("billing", ("1 == 1",)), ("refunds", ("2 == 2", "3 == 3")))
    said = campaign_report_html(page)
    assert "Goal check 2 is unproven" in re.sub(r"<[^>]+>", "", said)
    assert not re.search(r">Goal check</a> \d", said)


#: An expression a host could write that parses, so it reaches the naming path, and whose literals are Markdown and
#: markup outside a code span: a name is built from identifiers only, so none of it reaches the check's name.
HOSTILE = 'call_count("x.`y` [click](javascript:alert(1)) <img src=x onerror=alert(1)>") >= 1'


def test_a_hostile_expression_in_the_glossary_is_code_in_both_forms() -> None:
    assert check_names([HOSTILE], {}) == {HOSTILE: "Goal check"}, "the literal is no identifier, so it names nothing"
    page = _measurement(("billing", (HOSTILE, SHORT)))
    markdown = campaign_report_markdown(page)
    assert f"- **Goal check**: The goal check {code_span(HOSTILE)}, which code runs on every trial." in markdown
    unquoted = re.sub(r"(`+)(?!`).+?(?<!`)\1(?!`)", "", markdown)
    assert "javascript:" not in unquoted and "<img" not in unquoted and "](" not in unquoted
    said = campaign_report_html(page)
    assert "<img" not in said and f"<code>{escape(HOSTILE)}</code>" in said
    assert not [value for value in re.findall(r'="([^"]*)"', said) if "javascript:" in html.unescape(value)]


def test_a_check_only_a_bar_names_has_a_name_and_a_glossary_entry_of_its_own() -> None:
    barred = "state.refund.amount > 0"
    bar = BarOverride(measure_id=goal_check_measure(barred), threshold=0.9, direction="higher_is_better")
    shape = shape_of(
        "barred",
        [AspectTrials("billing", [ArmTrials("solo", passing=14)], margin=None, bars=(bar,), expressions=(SHORT,))],
        control=False,
        question="How often?",
    )
    bundle = shape.bundles["billing"]
    assert bundle.named_checks() == {barred: "Refund amount check", SHORT: SHORT_NAME}
    # In the checks' sorted order, as every list of them runs.
    assert [name for name, _ in shape.reports["billing"].check_glossary] == [SHORT_NAME, "Refund amount check"]
    # A check the aspect names nowhere is named among its checks, never by a name another holds.
    unseen = bundle.check_name(goal_check_measure("state.refund.amount < 100"))
    assert unseen == "Refund amount check 1" and unseen not in bundle.named_checks().values()


def test_a_declared_name_that_is_a_term_s_words_takes_an_id_of_its_own() -> None:
    shape = _shape()
    bundle = shape.bundles["character"].model_copy(update={"goal_check_names": {SHORT: "Outcome"}})
    said = report_html(build_code_only_report(bundle, assembled_at=ASSEMBLED_AT, trials="table"))
    ids = re.findall(r'\sid="([^"]+)"', said)
    assert len(ids) == len(set(ids))
    assert 'id="g-check-outcome"' in said


def test_a_written_page_and_its_record_name_a_check_in_one_form() -> None:
    # Until the report writer's judge is recorded again, a report document's tables print an undeclared check's
    # expression, and the evidence record laid out after them prints it so too.
    read = subject("toyhost_doc")
    body = report_read(read.host, read.campaign_id, read.scope_id, format="markdown", static=False, trials="all").body
    assert "| `field_accuracy >= 0.92` |" in body, "the document's table heads the check by its expression"
    assert "check: `field_accuracy >= 0.92`" in body, "the record's trials name it so too"
    assert "Field accuracy check" not in body, "no part of the page names it otherwise"
    assert "The goal check `field_accuracy >= 0.92`" not in body, "a check printed as itself has no glossary entry"
