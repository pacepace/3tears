"""A campaign's page lays out its failed trials as a table by default; their full records are asked for (#951).

Each failed or lost trial's record (what it asked, what the judge saw and said, and why) was most of a large
campaign's page: seven of the eight megabytes one production read returned. By default the trials section is the
census line, one row per failed or lost trial naming its stored result for ``result_get``, and the passed trials'
table; ``trials="all"`` lays the records out as well, exactly as the page did before. Every door reads the same
default, and every form says the same.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals import Eval, Result
from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis import (
    CampaignReport,
    DisclosureBlock,
    TableBlock,
    TrialBlock,
    build_code_only_report,
    campaign_report,
    serialize_report,
)
from threetears.evals.analysis import service as analysis_service
from threetears.evals.ops import report_read
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture

#: The instant every page is assembled at, so two reads can be compared byte for byte.
NOW = "2026-10-14T09:00:00+00:00"

CASES = [{"text": f"refund {n}", "label": "billing"} for n in range(4)] + [
    {"text": f"refund, it crashed {n}", "label": "billing"} for n in range(4)
]


async def _careful(case: Mapping[str, Any]) -> str:
    return "billing" if "refund" in case["text"] else "bug"


async def _hasty(case: Mapping[str, Any]) -> str:
    return "bug" if "crash" in case["text"] else "billing"


@pytest.fixture
async def comparison(monkeypatch: pytest.MonkeyPatch) -> Result:
    monkeypatch.setattr(analysis_service, "utc_now_iso", lambda: NOW)
    return await Eval(CASES, {"hasty": _hasty, "careful": _careful}, ["label"], k=2, name="trials").run(
        store=":memory:"
    )


def _page(comparison: Result, **kwargs: Any) -> CampaignReport:
    return off_loop(campaign_report, comparison.host, comparison.campaign_id, comparison.scope_id, **kwargs)


def _trials(page: CampaignReport) -> list[Any]:
    (fold,) = page.aspects
    return [block for block in fold.report.blocks if block.section == "trials"]


async def test_by_default_the_trials_section_is_the_census_and_a_row_per_failed_trial_naming_its_result(
    comparison: Result,
) -> None:
    page = _page(comparison)
    (fold,) = page.aspects
    bundle = fold.report.bundle
    wrong = [trial for trial in bundle.trials if trial.state != "passed"]
    assert wrong, "the fixture must fail some trials for this to test anything"
    census, table, passed = _trials(page)

    assert isinstance(census, DisclosureBlock) and census.text.startswith(f"{len(bundle.trials)} trials ran.")
    assert isinstance(table, TableBlock) and table.name == "trials"
    assert [column.header for column in table.columns] == ["Trial", "Case", "Arm", "Outcome", "Why", "Stored result"]
    assert [(row["trial"].text, row["trial"].anchor, row["result"].text) for row in table.rows] == [
        (trial.trial_id, trial.trial_id, f"`{trial.result_id}`") for trial in wrong
    ]
    assert isinstance(passed, TableBlock) and passed.name == "passed_trials"
    assert not any(isinstance(block, TrialBlock) for block in fold.report.blocks)


async def test_all_lays_out_the_records_exactly_as_the_aspect_s_own_page_does(comparison: Result) -> None:
    page = _page(comparison, trials="all")
    (fold,) = page.aspects
    whole = build_code_only_report(
        fold.report.bundle,
        assembled_at=NOW,
        campaign_name=fold.report.source.campaign_name,
        arm_labels=page.readings.arm_labels(),
    )
    records = [block for block in _trials(page) if isinstance(block, TrialBlock)]

    assert _trials(page) == [block for block in whole.blocks if block.section == "trials"]
    assert [block.trial_id for block in records] == [
        trial.trial_id for trial in fold.report.bundle.trials if trial.state != "passed"
    ]
    (table,) = [block for block in _trials(page) if isinstance(block, TableBlock) and block.name == "trials"]
    assert [column.header for column in table.columns] == ["Trial", "Case", "Arm", "Outcome", "Why"]


async def test_every_form_says_the_same_and_a_link_to_a_failed_trial_lands_on_its_row(comparison: Result) -> None:
    table, every = _page(comparison), _page(comparison, trials="all")
    (fold_id,) = table.fold_ids
    wrong = [trial for trial in table.aspects[0].report.bundle.trials if trial.state != "passed"]

    html, html_all = serialize_report(table, "html"), serialize_report(every, "html")
    assert '<details class="trial"' not in html and html_all.count('<details class="trial"') == len(wrong)
    for trial in wrong:
        target = f"a-{fold_id}--{trial.trial_id}"
        assert f'<td id="{target}">' in html and f'href="#{target}"' in html
    markdown, markdown_all = serialize_report(table, "markdown"), serialize_report(every, "markdown")
    assert "<details id=" not in markdown and markdown_all.count(f'<details id="a-{fold_id}--T') == len(wrong)
    assert "| Trial | Case | Arm | Outcome | Why | Stored result |" in markdown
    assert len(html) < len(html_all) and len(markdown) < len(markdown_all)
    # The machine form holds no records either way.
    assert json.loads(serialize_report(table, "json")) == json.loads(serialize_report(every, "json"))


async def test_report_read_defaults_to_the_table_and_reads_the_records_when_asked(comparison: Result) -> None:
    for form in ("markdown", "html", "json"):
        default = off_loop(report_read, comparison.host, comparison.campaign_id, comparison.scope_id, format=form)
        asked = off_loop(
            report_read, comparison.host, comparison.campaign_id, comparison.scope_id, format=form, trials="all"
        )
        assert default.body == serialize_report(_page(comparison), form)
        assert asked.body == serialize_report(_page(comparison, trials="all"), form)


async def test_the_action_takes_trials_and_reads_what_the_operation_reads() -> None:
    fixture = off_loop(ops_fixture)
    (evals, *_rest) = eval_catalogue().mount_all(standard_tools())
    for asked in ({}, {"trials": "all"}, {"trials": "table"}):
        outcome = await evals.call(
            {"action": "report_read", "campaign_id": fixture.campaign.id, "format": "html", **asked},
            host=fixture.host,
            caller=CALLER,
        )
        assert not outcome.is_error, outcome.text
        expected = off_loop(
            report_read, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format="html", **asked
        )
        assert re.sub(r"\d{4}-\d{2}-\d{2}T[0-9:.+]+", "", outcome.text) == re.sub(
            r"\d{4}-\d{2}-\d{2}T[0-9:.+]+", "", expected.body
        )
    refused = await evals.call(
        {"action": "report_read", "campaign_id": fixture.campaign.id, "trials": "some"},
        host=fixture.host,
        caller=CALLER,
    )
    assert refused.is_error and "trials" in refused.text
