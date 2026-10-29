"""The static page runs and loads nothing, shows every chart as its table, and leaves the default page as it was.

A host whose policy lets a page neither run nor load anything (``default-src 'none'``, no script allowed even were
escaping ever to fail) asks for the static page: ``static=True`` on ``report_html`` and ``campaign_report_html``,
threaded through ``serialize_report``, ``report_read`` and the ``report_read`` action. Held here, on the pages whose
every field carries markup (:mod:`~packages.evals.tests.hostile_page_support`):

- it carries no ``<script>`` and no ``<link>``, not even the package's own, no ``on*`` attribute, no ``javascript:``
  value in any attribute (``data-*`` included: it has no chart data either), and no ``url(`` or ``@import`` in its
  styles;
- every chart is its figure's title and its table of values, and no rule the page ships hides that table: the one
  that does needs a drawn ``div.chart`` beside it, and the static page writes none, so no empty box stands either;
- its text falls back to the system faces the kit's stylesheet names after each Plex face;
- it is the default page less exactly the kit's script, the font links and each ``div.chart``, so the default page,
  which the no-script contract still holds, is the one it always was;
- the package writes no script and no link into Markdown or JSON, and the option changes neither;
- ``report_read`` returns it for a code-only report, a memo analysis and a planned one, and the action takes it.

The first and the fourth are :mod:`~packages.evals.tests.static_page_support`'s, which every static page's test reads.
"""

from __future__ import annotations

import re
from typing import Any

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.actions.engine import ReportReadParams
from threetears.evals.analysis import (
    ChartBlock,
    campaign_report_html,
    report_html,
    serialize_report,
)
from threetears.evals.ops import report_read
from packages.evals.tests.hostile_page_support import HOSTILE, hostile_campaign_page, hostile_report
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture, planned_generation
from packages.evals.tests.page_contract_support import parsed, what_runs
from packages.evals.tests.report_support import stored_memo_analysis, toy_report
from packages.evals.tests.static_page_support import assert_static, less_drawings, less_what_static_leaves_out


def test_the_static_page_of_a_hostile_report_runs_and_loads_nothing_and_shows_every_chart_as_its_table() -> None:
    report = hostile_report()
    page = report_html(report, static=True)

    assert_static(page)
    charts = [block for block in report.blocks if isinstance(block, ChartBlock)]
    assert page.count('<figure class="fig"') == len(charts)
    words = [text for text in parsed(page).text if HOSTILE in text]
    assert len(words) > 2 * len(charts), "each chart's title and the labels in its table survive as words"


async def test_the_static_page_of_a_hostile_campaign_runs_and_loads_nothing_and_shows_every_chart_as_its_table() -> (
    None
):
    assert_static(campaign_report_html(await hostile_campaign_page(), static=True))


def test_the_static_page_falls_back_to_the_system_faces_the_kit_names_after_each_plex_face() -> None:
    css = "".join(parsed(report_html(hostile_report(), static=True)).styles)
    for face in ("display", "body", "mono"):
        declared = re.search(rf"--f-{face}:([^;]+);", css)
        assert declared is not None, f"the kit names no --f-{face}"
        stack = [name.strip().strip('"') for name in declared.group(1).split(",")]
        assert stack[0].startswith("IBM Plex") and len(stack) > 2
        assert stack[-1] in {"sans-serif", "monospace"}, f"--f-{face} ends in no generic family"


async def test_the_default_pages_are_the_static_pages_plus_the_script_the_fonts_and_the_chart_boxes() -> None:
    report, campaign = hostile_report(), await hostile_campaign_page()
    pages = [
        (report_html(report), report_html(report, static=False), report_html(report, static=True)),
        (
            campaign_report_html(campaign),
            campaign_report_html(campaign, static=False),
            campaign_report_html(campaign, static=True),
        ),
    ]
    for default, unasked, static in pages:
        assert unasked == default
        assert less_drawings(static) == less_what_static_leaves_out(default)
        assert parsed(default).tags[-1] == ("script", []), "the default page still ends in the kit's script"


async def test_markdown_and_json_carry_no_script_the_package_writes_and_static_changes_neither() -> None:
    _, _, toy = await toy_report()
    for form in ("markdown", "json"):
        assert what_runs(serialize_report(toy, form)) == [], f"the package writes no script or link into {form}"
    report, campaign = hostile_report(), await hostile_campaign_page()
    for page in (toy, report, campaign):
        for form in ("markdown", "json"):
            assert serialize_report(page, form, static=True) == serialize_report(page, form)
    assert serialize_report(report, "html", static=True) == report_html(report, static=True)
    assert serialize_report(campaign, "html", static=True) == campaign_report_html(campaign, static=True)


def _without_instants(body: str) -> str:
    return re.sub(r"\d{4}-\d{2}-\d{2}T[0-9:.+]+", "", body)


def _read(fixture: Any, **asked: Any) -> Any:
    return off_loop(report_read, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format="html", **asked)


async def test_report_read_returns_the_static_page_for_a_code_only_report() -> None:
    fixture = off_loop(ops_fixture)
    read = _read(fixture, static=True)

    assert read.analysis_id is None
    assert_static(read.body)
    assert _without_instants(less_drawings(read.body)) == _without_instants(
        less_what_static_leaves_out(_read(fixture).body)
    )


async def test_report_read_returns_the_static_page_for_a_memo_analysis() -> None:
    fixture = off_loop(ops_fixture)
    memo, _ = await stored_memo_analysis(fixture.host.eval_host, fixture.campaign)
    read = _read(fixture, static=True)

    assert (read.analysis_id, read.analysis_basis) == (memo.id, "memo")
    assert_static(read.body)
    assert _without_instants(less_drawings(read.body)) == _without_instants(
        less_what_static_leaves_out(_read(fixture).body)
    )


async def test_report_read_returns_the_static_page_for_a_planned_analysis() -> None:
    fixture = off_loop(ops_fixture)
    planned = (await planned_generation(fixture))[0].id
    read = _read(fixture, static=True)

    assert planned is not None and (read.analysis_id, read.analysis_basis) == (planned, "plan")
    assert_static(read.body)
    assert _without_instants(less_drawings(read.body)) == _without_instants(
        less_what_static_leaves_out(_read(fixture).body)
    )


async def test_the_action_takes_static_and_defaults_to_the_page_it_always_read() -> None:
    assert ReportReadParams.model_validate({"campaign_id": "c"}).static is False
    fixture = off_loop(ops_fixture)
    (evals, *_rest) = eval_catalogue().mount_all(standard_tools())
    for asked in ({}, {"static": False}, {"static": True}):
        outcome = await evals.call(
            {"action": "report_read", "campaign_id": fixture.campaign.id, "format": "html", **asked},
            host=fixture.host,
            caller=CALLER,
        )
        assert not outcome.is_error, outcome.text
        assert _without_instants(outcome.text) == _without_instants(_read(fixture, **asked).body)
    static = await evals.call(
        {"action": "report_read", "campaign_id": fixture.campaign.id, "format": "html", "static": True},
        host=fixture.host,
        caller=CALLER,
    )
    assert_static(static.text)
