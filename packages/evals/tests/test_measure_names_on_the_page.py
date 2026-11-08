"""A goal check is named by what it tests in every header and every lead, and its expression is the glossary's
definition of that name, once.

The owner's ruling of 2026-11-07 (Q11): a check's expression is code, and a long one (a production campaign's
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
    campaign_report,
    build_campaign_report,
    campaign_report_html,
    campaign_report_markdown,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.report.md_parts import code_span
from threetears.evals.analysis.report.html_parts import check_header_links, grid_html
from threetears.evals.analysis.report.model import (
    CampaignReport,
    DisclosureBlock,
    ReportBlock,
    TableBlock,
    TableCell,
    TableColumn,
)
from threetears.evals.analysis.report.doc_page.text import doc_page_text
from threetears.evals.analysis.report.parts.full_results import laid_out
from threetears.evals.analysis.style import CHECK_LINK_TITLE, check_names
from threetears.evals.kernel.declaration import BarOverride
from threetears.evals.kernel.metrics import goal_check_measure
from threetears.evals.ops.analysis import report_read
from threetears.evals.schema.goal_grammar import ACTION_BUILTINS, FIRE_PREDICATES, REFUSAL_BUILTINS

from packages.evals.tests.campaign_support import ASSEMBLED_AT
from packages.evals.tests.report_kit_support import report_page
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


def _anchor(name: str) -> str:
    """A check's row id, as the page's glossary gives it."""
    return "g-check-" + re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")


#: The glossary's entry for each check, in Markdown and in HTML.
_ENTRIES = {
    "markdown": [
        f'- <a id="{_anchor(LONG_NAME)}"></a>**{LONG_NAME}**: The goal check `{LONG_LINE}`, which code runs on every trial.',
        f'- <a id="{_anchor(SHORT_NAME)}"></a>**{SHORT_NAME}**: The goal check `{SHORT}`, which code runs on every trial.',
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
        # A case parameter is the expectation the end state is held to, never what the check tests.
        ("variation.expected == state.answer", "Answer"),
        ("state.answer == variation.expected", "Answer"),
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


def _scorecard_checks(page: CampaignReport) -> list[tuple[str, str]]:
    """Each goal check one arm's turned scorecard reads, a row per aspect: the aspect and the name its row is headed
    by (visual pass V2: one arm's matrix is a row per measure, never a column per measure)."""
    (matrix,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "scorecard"]
    defined = {name for name, _ in page.check_glossary}
    assert [column.key for column in matrix.columns][:2] == ["aspect", "measure"], "the control: the matrix is turned"
    return [(row["aspect"].text, row["measure"].text) for row in matrix.rows if row["measure"].text in defined]


def test_one_expression_has_one_name_on_every_aspect_and_two_aspects_with_one_check_each_read_apart() -> None:
    refund = "state.refund.amount <= state.order.total"
    page = _measurement(("billing", (LONG,)), ("refunds", (refund,)))
    assert _scorecard_checks(page) == [("billing", LONG_NAME), ("refunds", "Refund amount check")]
    alike = _measurement(("billing", (LONG,)), ("refunds", (LONG,)))
    assert _scorecard_checks(alike) == [("billing", LONG_NAME), ("refunds", LONG_NAME)]
    assert [name for name, _ in alike.check_glossary] == [LONG_NAME]


def test_two_aspects_checks_that_share_a_name_are_told_apart_by_their_aspects() -> None:
    page = _measurement(
        ("billing", ("state.ticket.status == 'open'",)), ("refunds", ("state.ticket.status == 'closed'",))
    )
    # One arm's matrix is turned: each row is told apart by its aspect, beside the name its aspect reads the check by,
    # and the one glossary entry names each expression with its aspect.
    assert _scorecard_checks(page) == [("billing", "Ticket status check"), ("refunds", "Ticket status check")]
    (entry,) = page.check_glossary
    assert entry == (
        "Ticket status check",
        [("state.ticket.status == 'open'", "billing"), ("state.ticket.status == 'closed'", "refunds")],
    )


def test_a_check_s_row_header_links_its_glossary_entry() -> None:
    said = campaign_report_html(_measurement(("billing", (LONG, SHORT))))
    # One arm's turned scorecard heads each measure's row with its name, which links the check's entry as a column's
    # header does (visual pass V2).
    (header,) = re.findall(rf'<th scope="row"[^>]*>(<a [^>]*>{LONG_NAME}</a>)</th>', said)
    assert f'title="{escape(CHECK_LINK_TITLE)}"' in header
    assert 'href="#g-check-' in header


def test_a_name_inside_a_longer_one_is_never_linked_as_the_shorter() -> None:
    # Billing reads one check with no subject (Goal check), refunds two (Goal check 1, Goal check 2), each unproven.
    page = _measurement(("billing", ("1 == 1",)), ("refunds", ("2 == 2", "3 == 3")))
    said = campaign_report_html(page)
    assert "Goal check 2 is unproven" in re.sub(r"<[^>]+>", "", said)
    assert not re.search(r">Goal check</a> \d", said)


def test_no_glossary_term_is_linked_inside_a_measure_s_name_wherever_a_line_prints_it() -> None:
    """Discodon's re-shoot of host pass 5: the line under a one-arm scorecard linked the glossary's "ran" inside
    "Turns that ran over budget". A term is never linked inside a measure's or a column's name, in the line under a
    table or anywhere else a line prints it; it is linked at its first use that is not a name's word, and a name that
    is the term's own words is the term, linked where it is first read though its unit follows it."""
    scorecard = TableBlock(
        section="scoreboard",
        name="scorecard",
        title="solo by aspect",
        shared_scale=True,
        columns=[
            TableColumn(key="aspect", header="Aspect"),
            TableColumn(key="budget", header="Turns that ran over budget"),
            TableColumn(key="spend", header="Measuring spend, all roles, per 1,000 replies"),
        ],
        rows=[
            {
                "aspect": TableCell(value="Research quality"),
                "budget": TableCell(value=0.0, low=0.0, high=0.0),
                "spend": TableCell(value=2.7, low=2.5, high=3.0, unit="usd", decimals=1),
            }
        ],
        order="aspect by aspect",
        total_rows=1,
    )
    runs = TableBlock(
        section="methods",
        name="runs",
        title="Runs",
        columns=[TableColumn(key="arm", header="Arm"), TableColumn(key="ran", header="Ran")],
        rows=[{"arm": TableCell(value="solo"), "ran": TableCell(value=60)}],
        order="as run",
        total_rows=1,
    )
    lines = [
        DisclosureBlock(section="methods", source="comparisons", text="Turns that ran over budget were none."),
        DisclosureBlock(section="methods", source="comparisons", text="Each arm ran every case once."),
    ]
    said = report_page([scorecard, *lines, runs])
    (axis,) = re.findall(r'<p class="axis">(.*?)</p>', said)
    assert axis.startswith("Turns that ran over budget drawn from "), axis
    assert ">Measuring spend, all roles</a>, per 1,000 replies drawn from" in axis, "the name is the term's words"
    assert "Turns that ran over budget were none." in said, "nor in a line that names the measure"
    assert re.search(r"Each arm <a [^>]*>ran</a> every case once", said), "the term links at its first use as itself"


#: An expression a host could write that parses, so it reaches the naming path, and whose literals are Markdown and
#: markup outside a code span: a name is built from identifiers only, so none of it reaches the check's name.
HOSTILE = 'call_count("x.`y` [click](javascript:alert(1)) <img src=x onerror=alert(1)>") >= 1'


def test_a_hostile_expression_in_the_glossary_is_code_in_both_forms() -> None:
    assert check_names([HOSTILE], {}) == {HOSTILE: "Goal check"}, "the literal is no identifier, so it names nothing"
    page = _measurement(("billing", (HOSTILE, SHORT)))
    markdown = campaign_report_markdown(page)
    assert (
        f'- <a id="g-check-goal-check"></a>**Goal check**: The goal check {code_span(HOSTILE)}, which code runs on every trial.'
        in markdown
    )
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
    # A report document's tables name an undeclared check by what it tests, as the code page does, and the evidence
    # record laid out after them names it so too, through the one call over the one set of checks.
    read = subject("toyhost_doc")
    body = report_read(read.host, read.campaign_id, read.scope_id, format="markdown", static=False, trials="all").body
    # One spelling per figure on the page: the options table and Every arm by measure read the pass rate alike (N-4).
    assert body.count("44% (16 of 36)") >= 2 and "44.4%" not in body, "one figure, one spelling"
    assert "| Field accuracy check | 0% | 44% |" in body, "the document's table names the check by what it tests"
    assert "| failed | Field accuracy check |" in body, "the record's trials name it so too"
    assert "- `Field accuracy check`: `field_accuracy >= 0.92`" in body, "a trial's record prints its expression"
    assert "| `field_accuracy >= 0.92` |" not in body, "no table names it by its expression"
    assert "check: `field_accuracy >= 0.92`" not in body, "no trial names it by its expression"
    assert "The goal check `field_accuracy >= 0.92`" in body, "the glossary defines the name by its expression"


@pytest.mark.parametrize("builtin", sorted(ACTION_BUILTINS | FIRE_PREDICATES))
def test_every_call_predicate_the_grammar_knows_names_its_action(builtin: str) -> None:
    # The namer reads the grammar's own sets: a builtin the grammar gains is named as its set's are.
    word = "refusals" if builtin in REFUSAL_BUILTINS else "firing" if builtin in FIRE_PREDICATES else "calls"
    expression = f'{builtin}("support.send_message")'
    assert check_names([expression], {}) == {expression: f"Send message {word} check"}


def _written(form: str) -> tuple[str, CampaignReport]:
    """toyhost_doc's report document page as a report read serves it by default, and the page it renders."""
    read = subject("toyhost_doc")
    served = report_read(read.host, read.campaign_id, read.scope_id, format=form, static=False, trials="table")
    page = campaign_report(read.host, read.campaign_id, read.scope_id, trials="table")
    return served.body, page


@pytest.mark.parametrize("form", ["markdown", "html"])
def test_a_written_page_defines_each_check_it_names_by_its_expression_once(form: str) -> None:
    body, page = _written(form)
    assert page.doc_page is not None and page.doc_page.checks, "the document names a goal check"
    for check in page.doc_page.checks:
        assert check.name in body
        expression = check.expression if form == "markdown" else escape(check.expression)
        assert body.count(expression) == 1, f"{check.name} is defined by its expression once"
    if form == "markdown":
        assert "### Checks" in body
        assert (
            '- <a id="g-check-field-accuracy-check"></a>**Field accuracy check**: The goal check `field_accuracy >= 0.92`, '
            "which code runs on every trial." in body
        )
    else:
        assert '<dt id="g-check-field-accuracy-check">Field accuracy check</dt>' in body
        # The header naming the check links its row.
        assert re.search(
            r'<th[^>]*><a href="#g-check-field-accuracy-check" title="[^"]*">Field accuracy check</a></th>', body
        )


def _laid_out_table(body: str, name: str) -> str:
    """The table ``name`` as an HTML page lays it out, from its block's opening to its table's end."""
    start = body.index(f'data-table="{name}"')
    return body[start : body.index("</table>", start)]


def test_a_stored_two_arm_document_s_turned_table_heads_each_row_and_links_its_check() -> None:
    # toyhost_doc, read stored as Discodon reads its draft: two arms on more figures than arms, so Every arm by measure
    # is turned a figure per row, and its check is named only as a row's label there. Each label heads its row
    # (scope="row"), so the check's links its Checks entry as a column header naming it does (0.94.0 linked none: the
    # label was a <td>, and the Checks list had no link to it on the stored page).
    body, page = _written("html")
    assert page.doc_page is not None and len(page.doc_page.options.rows) == 2, "the control: two arms"
    table = _laid_out_table(body, "arms_by_measure")
    assert '<th scope="col">Measure</th>' in table, "the control: the table is turned, a figure per row"
    rows = re.findall(r"<tr>(.*?)</tr>", table.split("<tbody>", 1)[1])
    assert rows and all(row.startswith('<th scope="row"') for row in rows), rows
    # A header link after the term's first is plain (``class="seen"``), its link kept.
    assert re.search(
        r'<th scope="row"[^>]*><a (?:class="seen" )?href="#g-check-field-accuracy-check" title="[^"]*">'
        r"Field accuracy check</a></th>",
        table,
    ), table


#: A check named with a hyphenated word, as Discodon's stored draft names "the reply-delivery check".
_HYPHENATED_CHECK = "Reply-delivery check"


def test_a_hyphenated_check_name_links_its_entry_as_a_row_s_label_and_as_a_column_s_header() -> None:
    # A cell holds a hyphenated word whole (``span.id``), in a turned table's row label and in a pinned column's
    # header; the check's link is made on the name through it, as on a header holding no span.
    turned = TableBlock(
        section="contrasts",
        name="arms_by_measure",
        title="Every arm by measure",
        columns=[
            TableColumn(key="measure", header="Measure"),
            TableColumn(key="arm:0", header="P · control"),
            TableColumn(key="arm:1", header="A · candidate"),
        ],
        rows=[
            {
                "measure": TableCell(value=_HYPHENATED_CHECK),
                "arm:0": TableCell(value="70%"),
                "arm:1": TableCell(value="80%"),
            }
        ],
        order="a figure per row",
        total_rows=1,
    )
    headed = TableBlock(
        section="contrasts",
        name="probe",
        title="Probe",
        columns=[
            TableColumn(key="first", header=_HYPHENATED_CHECK),
            TableColumn(key="rate", header=f"{_HYPHENATED_CHECK} (%)"),
        ],
        rows=[{"first": TableCell(value="All cases"), "rate": TableCell(value=80.0)}],
        order="as built",
        total_rows=1,
    )
    markup = grid_html(turned) + grid_html(headed)
    assert '<span class="id">Reply-delivery</span> check' in markup, "the control: the name's word is held whole"
    linked = check_header_links(markup, [_HYPHENATED_CHECK])
    links = re.findall(
        r'<th[^>]*><a (?:class="(seen)" )?href="#g-check-reply-delivery-check" title="[^"]*">(.*?)</a></th>', linked
    )
    # Every header naming the check links it: the first underlined, each later one plain (``class="seen"``).
    assert links == [
        ("", '<span class="id">Reply-delivery</span> check'),
        ("seen", '<span class="id">Reply-delivery</span> check'),
        ("seen", f"{_HYPHENATED_CHECK} (%)"),
    ], links


@pytest.mark.parametrize("form", ["markdown", "html"])
def test_every_measure_a_written_page_s_full_results_name_starts_with_a_capital(form: str) -> None:
    # toyhost_doc's judged dimension has no label, so its name was the humanized key ("extraction layout fidelity")
    # beside "Pass rate" and the engine's labels: every measure name, as a row label or a column header, starts with
    # a capital whatever its source.
    body, page = _written(form)
    assert page.doc_page is not None
    named = [
        name
        for table in page.doc_page.full_results
        for name in [
            *(column.header for column in table.columns if column.key.startswith(("measure:", "judged:"))),
            *(row["measure"].text for row in table.rows if "measure" in row),
        ]
    ]
    named += [
        row["measure"].text for table in page.doc_page.full_results for row in laid_out(table).rows if "measure" in row
    ]
    # A judged score is named with its scale (owner ruling O4).
    assert "extraction layout fidelity (1–5)" in [name.casefold() for name in named], (
        "the control: the judged dimension"
    )
    assert [name for name in named if name[:1].isalpha() and not name[:1].isupper()] == []
    assert "extraction layout fidelity" not in body


def test_the_text_the_judge_reads_carries_no_checks_list() -> None:
    # The report writer's recorded judge reads doc_page_text: the Checks list is the reader's page alone.
    _, page = _written("markdown")
    assert page.doc_page is not None
    said = doc_page_text(page.doc_page)
    assert "Checks" not in said and "field_accuracy >= 0.92" not in said


#: The shapes of four production checks a host seeds (the identifiers are the host's kind, not its text), each with
#: the name the namer gives it: a guard over the end state and a case parameter is passed over, so its consequent
#: names the check; a check testing three or more things falls back to a goal check, which a host names by declaring
#: one.
_PRODUCTION_SHAPES = {
    "three subjects joined by or": (
        'call_count("music.queue_voice") >= 1 or call_count("discord.send_message") >= 1'
        " or state.music.budget.used <= state.music.budget.limit",
        "Goal check",
    ),
    "a contains guard before its consequent": (
        'contains(state.music.queue, variation.stale_track) and state.music.queue.position > 0 and call_count("music.queue_ready") >= 1',
        "Music queue position and queue ready calls check",
    ),
    "a case-parameter condition or two calls, twice": (
        '(not contains(["joke", "chat"], variation.ask) or call_count("research.search") >= 1'
        ' or call_count("research.read") >= 1) and (not contains(["fact"], variation.ask)'
        ' or call_count("research.cite") >= 1 or call_count("research.verify") >= 1)',
        "Goal check",
    ),
    "three not-any clauses": (
        "not any(it.stale for it in state.music.queue) and not any(it.repeat for it in state.music.intros)"
        " and not any(it.broken for it in state.music.links)",
        "Goal check",
    ),
}


@pytest.mark.parametrize(("expression", "name"), list(_PRODUCTION_SHAPES.values()), ids=list(_PRODUCTION_SHAPES))
def test_a_production_check_is_named_by_its_consequent_or_falls_back_to_a_goal_check(
    expression: str, name: str
) -> None:
    assert check_names([expression], {}) == {expression: name}


def _check_hrefs(said: str) -> list[str]:
    """Every link a check's name carries on an HTML page, by the one hover every such link has."""
    return re.findall(rf'<a href="#([^"]+)" title="{re.escape(escape(CHECK_LINK_TITLE))}">', said)


@pytest.mark.parametrize("page", ["campaign", "measurement", "written"])
def test_every_check_link_lands_on_its_own_row(page: str) -> None:
    if page == "written":
        said, _ = _written("html")
    elif page == "measurement":
        said = campaign_report_html(_measurement(("billing", (LONG, SHORT)), ("refunds", (SHORT,))))
    else:
        said = _rendered("campaign", "html")
    hrefs = _check_hrefs(said)
    assert hrefs, "the page links a check's name"
    rows = set(re.findall(r'<dt id="(g-check-[^"]+)"', said))
    assert set(hrefs) <= rows, f"a check's link lands nowhere: {sorted(set(hrefs) - rows)}"
    assert "#glossary" not in hrefs


@pytest.mark.parametrize("page", ["campaign", "written"])
def test_a_markdown_check_row_carries_the_id_its_html_row_has(page: str) -> None:
    said = _written("markdown")[0] if page == "written" else _rendered("campaign", "markdown")
    html_rows = set(
        re.findall(
            r'<dt id="(g-check-[^"]+)"', _written("html")[0] if page == "written" else _rendered("campaign", "html")
        )
    )
    md_rows = set(re.findall(r'^- <a id="(g-check-[^"]+)"></a>\*\*', said, flags=re.M))
    assert md_rows and md_rows == html_rows
