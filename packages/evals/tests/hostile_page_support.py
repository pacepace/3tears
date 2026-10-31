"""Report pages whose every field a host, an author or a model fills carries markup: what each page test that holds
a page to "nothing written into a field can make it run or load anything" renders.

The markup is a script, a handler, a ``javascript:`` link, a frame, a stylesheet from another host and a copy of the
package's own font links; a line adds an attribute that tries to break out of its quotes and an element claiming one
of the page's own ids. :func:`hostile_report` is one aspect's page with it in every field and every string a chart
of each of the kit's kinds draws; :func:`hostile_campaign_page` is a campaign's page of two real aspects with it in
the campaign's name, its arm and aspect labels, its memo, its fold lines and its second folded page;
:func:`hostile_planned_page` is that campaign's page as a planned analysis lays it out, the page the Kairo reference
plan draws over its input, with markup in every prose field of the plan and every name the input gives a reader, the
hostile folds its appendix.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis import (
    CampaignReport,
    build_campaign_report,
    build_code_only_report,
    campaign_readings,
)
from threetears.evals.analysis.report import Report
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_plan import ReportPlan
from threetears.evals.kernel.authored import AspectSummary, AuthoredCampaignAnalysis, NextStep
from threetears.evals.kernel.declaration import CampaignArm
from packages.evals.tests.campaign_page_support import campaign_fixture, two_aspect_memo
from packages.evals.tests.campaign_support import ASSEMBLED_AT
from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.page_contract_support import FONT_LINK_MARKUP
from packages.evals.tests.report_kit_support import KIT

__all__ = [
    "HOSTILE",
    "HOSTILE_ASPECT",
    "HOSTILE_LINE",
    "hostile_campaign_page",
    "hostile_kit_charts",
    "hostile_memo",
    "hostile_plan_input",
    "hostile_planned_page",
    "hostile_report",
]

#: What an author might write into a field a model fills, to try to make the page run or load something: a script,
#: a handler, a ``javascript:`` link, a frame, a stylesheet from another host, and the package's own font links,
#: which from a field are an author's, not the package's.
HOSTILE = (
    '<script>alert(1)</script><img src=x onerror="alert(2)"><a href="javascript:alert(3)">x</a>'
    '<iframe src=//evil></iframe><link rel="stylesheet" href="https://evil.example/x.css">' + FONT_LINK_MARKUP
)


def hostile_kit_charts() -> list[dict[str, Any]]:
    """A chart of each of the kit's fourteen kinds whose every string the kit draws carries markup: titles, labels,
    short labels, tags, notes, tooltips, legend entries, panel titles, column headings, pill words (a lattice's built
    as markup by ``pill()``), reasons, region labels, a confusion matrix's labels and an example card's every part."""
    h = HOSTILE
    chart = {"kind": "chart", "section": "memo", "finding": 0, "title": h}
    return [
        {
            "kind": "chart",
            "section": "memo",
            "finding": 0,
            "chart": "bars",
            "title": h,
            "data": {
                "domain": [0, 10],
                "fmt": "int",
                "level": "95%",
                "legend": [{"label": h, "color": "--s1"}],
                "reference": {"value": 5, "label": h, "kind": "floor"},
                "rows": [
                    {"label": h, "short_label": h, "value": 4, "lo": 3, "hi": 6, "note": h, "tip": [h, h], "group": h},
                    {
                        "label": h,
                        "tag": h,
                        "segments": [{"value": 2, "color": "--bad", "pattern": "hatch"}],
                        "tip": [h],
                    },
                ],
            },
        },
        {
            **chart,
            "chart": "diverging_bars",
            "data": {
                "domain": [-10, 10],
                "fmt": "+pts",
                "reference": {"value": 0, "label": h, "kind": "no_preference"},
                "legend": [{"label": h, "color": "--bad", "mark": "square", "pattern": "dots"}],
                "rows": [{"label": h, "short_label": h, "value": -4, "note": h, "tip": [h], "group": h}],
            },
        },
        {
            **chart,
            "chart": "forest",
            "data": {
                "domain": [-10, 10],
                "fmt": "+pts",
                "level": h,
                "reference": {"value": 2, "label": h, "kind": "margin"},
                "rows": [
                    {"label": h, "short_label": h, "value": 1, "lo": -2, "hi": 4, "note": h, "tip": [h], "group": h}
                ],
            },
        },
        {
            **chart,
            "chart": "dot_interval",
            "data": {
                "domain": [0, 100],
                "fmt": "pct",
                "level": h,
                "reference": {"value": 80, "label": h, "kind": "floor"},
                "rows": [{"label": h, "value": 85, "lo": 80, "hi": 90, "tip": [h]}, {"label": h, "value": 70}],
            },
        },
        {
            **chart,
            "chart": "range_bars",
            "data": {
                "domain": [0, 10],
                "fmt": "s",
                "reference": {"value": 5, "label": h, "kind": "budget"},
                "legend": [{"label": h}],
                "groups": [{"label": h, "short_label": h, "rows": [{"label": h, "from": 1, "to": 3, "tip": [h]}]}],
            },
        },
        {
            **chart,
            "chart": "gate_scatter",
            "data": {
                "x_domain": [0, 10],
                "y_domain": [0, 100],
                "x_fmt": "usd",
                "y_fmt": "pct",
                "region_label": h,
                "reference": {"value": 80, "label": h, "kind": "floor"},
                "x_reference": {"value": 5, "label": h, "kind": "budget"},
                "points": [{"x": 2, "y": 90, "label": h, "narrow_label": h, "tip": [h]}],
            },
        },
        {
            **chart,
            "chart": "lattice",
            "data": {
                "fmt": "pct",
                "cost_fmt": "usd",
                "cost_unit": h,
                "latency_fmt": "s",
                "bar_domain": [0, 100],
                "columns": [{"label": h}],
                "rows": [
                    {
                        "label": h,
                        "color": "--s1",
                        "cells": [
                            {"value": 80, "cost": 1, "latency": 2, "status": "bad", "status_label": h, "tip": [h]}
                        ],
                    }
                ],
            },
        },
        {
            **chart,
            "chart": "diff_heatmap",
            "data": {
                "fmt": "+pts",
                "detail_fmt": "pct",
                "columns": [{"label": h}],
                "rows": [{"label": h, "cells": [{"value": 3, "detail": 80, "tip": [h]}]}],
            },
        },
        {**chart, "chart": "confusion", "data": {"labels": [h, h], "matrix": [[90, 10], [5, 95]]}},
        {
            "kind": "chart",
            "section": "memo",
            "finding": 0,
            "chart": "scatter_frontier",
            "title": h,
            "data": {
                "x_domain": [0, 10],
                "y_domain": [0, 100],
                "x_fmt": "usd",
                "y_fmt": "pct",
                "x_title": h,
                "y_title": h,
                "price_line": {"value": 2, "label": h, "region_label": h},
                "points": [
                    {"x": 2, "y": 80, "id": "a", "label": h, "narrow_label": h, "tip": [h]},
                    {"x": 6, "y": 90, "id": "b", "label": h, "narrow_label": h, "tip": [h]},
                ],
            },
        },
        {
            "kind": "chart",
            "section": "memo",
            "finding": 0,
            "chart": "gate_matrix",
            "title": h,
            "data": {
                "row_header": h,
                "pick_label": h,
                "cost_fmt": "usd",
                "cost_unit": h,
                "columns": [{"label": h}],
                "rows": [
                    {
                        "label": h,
                        "detail": h,
                        "flag": h,
                        "cells": [{"status": "bad", "label": h, "reasons": [h], "cost": 1, "pick": True}],
                    }
                ],
            },
        },
        {
            "kind": "chart",
            "section": "memo",
            "finding": 0,
            "chart": "heatmap",
            "title": h,
            "data": {
                "domain": [0, 100],
                "fmt": "pct",
                "columns": [{"label": h}],
                "panels": [
                    {"title": h, "subtitle": h, "rows": [{"label": h, "cells": [{"value": 50, "tip": [h]}]}]},
                    {"title": h, "subtitle": h},
                ],
            },
        },
        {
            "kind": "chart",
            "section": "memo",
            "finding": 0,
            "chart": "examples",
            "title": h,
            "data": {
                "cards": [
                    {
                        "meta": [{"text": h, "mono": True}, {"status": "bad", "label": h}, {"tag": h, "label": h}],
                        "question": h,
                        "fields": [{"term": h, "text": h, "tone": "bad"}],
                        "reason": h,
                        "excerpt": h,
                        "note": h,
                    }
                ]
            },
        },
        {
            "kind": "chart",
            "section": "memo",
            "finding": 0,
            "chart": "lines",
            "title": h,
            "data": {
                "categories": [h, h],
                "domain": [0, 100],
                "fmt": "pct",
                "x_title": h,
                "series": [{"name": h, "end_label": h, "values": [40, 60]}],
            },
        },
    ]


def hostile_report() -> Report:
    """A report whose every author-written field, every label a chart draws and every trial's text carries markup."""
    fields: dict[str, Any] = {
        "basis": "analysis",
        "title": HOSTILE,
        "finding_count": 1,
        "source": {
            "analysis_id": "a",
            "campaign_id": "c",
            "scope_id": "s",
            "behavior": "b",
            "generated_at": "2026-10-05T00:00:00+00:00",
            "generator_model": "m",
            "bundle_fingerprint": "f",
        },
        "blocks": [
            {"kind": "disclosure", "section": "header", "source": "verdict", "text": HOSTILE},
            {"kind": "text", "section": "memo", "role": "finding_title", "finding": 0, "body": HOSTILE},
            {
                "kind": "text",
                "section": "memo",
                "role": "finding_body",
                "finding": 0,
                "body": f"- {HOSTILE}\n- **bold** `code`",
            },
            {
                "kind": "text",
                "section": "memo",
                "role": "caveat",
                "finding": 0,
                "body": HOSTILE,
                "facts": [{"name": "Kind", "value": HOSTILE}],
            },
            {
                "kind": "chart",
                "section": "scoreboard",
                "chart": "outcome_bar",
                "title": HOSTILE,
                "bars": [{"arm": HOSTILE, "segments": [{"state": "lost", "label": HOSTILE, "count": 2}]}],
            },
            {
                "kind": "chart",
                "section": "memo",
                "finding": 0,
                "chart": "interval_rows",
                "title": HOSTILE,
                "rows": [{"label": HOSTILE, "value": 1.0, "low": 0.5, "high": 1.5, "n_cases": 6, "verdict": HOSTILE}],
                "unit": HOSTILE,
            },
            *hostile_kit_charts(),
            {
                "kind": "table",
                "section": "cases",
                "name": "cases",
                "title": HOSTILE,
                "columns": [
                    {"key": "case", "header": "Case"},
                    {"key": "label", "header": HOSTILE},
                    {"key": "trials", "header": "Trials"},
                ],
                "rows": [
                    {"case": {"value": "case 1"}, "label": {"value": HOSTILE}, "trials": {"value": f"T1 {HOSTILE}"}}
                ],
                "order": HOSTILE,
                "total_rows": 1,
            },
            {
                "kind": "trial",
                "section": "trials",
                "anchor": "T1",
                "trial_id": "T1",
                "case_label": HOSTILE,
                "arm": HOSTILE,
                "repeat": 1,
                "state": "lost",
                "cause_words": HOSTILE,
                "first_input": HOSTILE,
                "excerpt": HOSTILE,
                "failed_checks": [{"name": HOSTILE, "expression": HOSTILE}],
                "judge": [{"dim": HOSTILE, "criterion": HOSTILE, "score": None, "scale": "ordinal", "reason": HOSTILE}],
                "error_words": HOSTILE,
                "result_id": HOSTILE,
                "open": True,
            },
            {"kind": "disclosure", "section": "methods", "source": "generation", "text": HOSTILE},
        ],
        "bundle": toyhost_bundle(),
    }
    return Report.model_validate(fields)


#: What an author might write where a page prints a label or a line: the hostile markup above, an ``onerror`` and a
#: ``javascript:`` URL in an attribute that tries to break out of its quotes, and an element claiming one of the
#: page's own ids.
HOSTILE_LINE = (
    HOSTILE + '"><img src=x onerror="alert(4)"><a id="glossary" href="javascript:alert(5)">x</a>'
    '<section id="aspect-billing">'
)


#: An aspect label carrying markup. A label never holds a ``/``, so this one has no closing tag: a handler, an
#: element that loads, a ``javascript:`` link and an id the page already holds, all in opening tags.
HOSTILE_ASPECT = '<img src=x onerror="alert(6)"><svg onload="alert(7)"><a id="glossary" href="javascript:alert(8)">'


def hostile_memo(control: str) -> AuthoredCampaignAnalysis:
    """The two-aspect memo with markup in every field its writer fills, recommending ``control``."""
    memo = two_aspect_memo()
    finding = memo.findings[0].model_copy(update={"title": HOSTILE_LINE, "body": f"- {HOSTILE_LINE}\n- **bold**"})
    recommended = memo.recommended()
    assert recommended is not None
    return memo.model_copy(
        update={
            "findings": [finding],
            "recommendation": [recommended.model_copy(update={"arm": control, "reasons": [HOSTILE_LINE]})],
            "aspect_summaries": [
                AspectSummary(aspect=aspect, what_changed=HOSTILE_LINE, how_sure=HOSTILE_LINE, means=HOSTILE_LINE)
                for aspect in ("billing", HOSTILE_ASPECT)
            ],
            "summary": HOSTILE_LINE,
            "next": [NextStep(title=HOSTILE_LINE, why=HOSTILE_LINE, leverage="high", lever="")],
        }
    )


async def hostile_campaign_page() -> CampaignReport:
    """A campaign page of two real aspects, one of them labelled with markup, whose campaign name, arm labels, memo,
    fold lines and second folded page carry markup too: everything on it a host or a model supplies."""
    control_label, candidate_label = f"{HOSTILE_LINE} A", f"{HOSTILE_LINE} B"
    campaign, measured = await campaign_fixture("billing", HOSTILE_ASPECT, name=HOSTILE_LINE)
    control = campaign.control
    assert control is not None
    arms = [
        control.model_copy(update={"label": control_label}),
        CampaignArm(label=candidate_label, role="candidate", variant_key=measured[0].candidate),
    ]
    bundles = {one.bundle.aspect: one.bundle for one in measured}
    read = campaign_readings(campaign, bundles, arms=arms)
    reports = {
        label: build_code_only_report(
            evidence, assembled_at=ASSEMBLED_AT, campaign_name=HOSTILE_LINE, arm_labels=read.arm_labels()
        )
        for label, evidence in bundles.items()
    }
    memo = hostile_memo(control_label)
    page = build_campaign_report(campaign, read, reports, memo, recommended_variant_key=control.variant_key)
    first, second = page.aspects
    folds = [
        first.model_copy(update={"shows": HOSTILE_LINE}),
        second.model_copy(update={"shows": HOSTILE_LINE, "report": hostile_report()}),
    ]
    return page.model_copy(update={"aspects": folds})


#: The plan fields that hold the planner's prose.
_PROSE_FIELDS = frozenset(
    {"headline", "lede", "title", "text", "material_if", "eyebrow", "takeaway", "caption", "term", "recomputes"}
)


def _hostile_prose(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: HOSTILE if key in _PROSE_FIELDS and isinstance(item, str) and item else _hostile_prose(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_hostile_prose(item) for item in value]
    return value


def hostile_plan_input(plan: ReportPlan, report_input: ReportInput) -> tuple[ReportPlan, ReportInput]:
    """A plan with markup in every prose field and its masthead's system, and its input with markup in every name it
    gives a reader: campaign, arms, aspects, strata, tasks, factors, measure and dimension labels, example cards."""
    data = _hostile_prose(plan.model_dump(by_alias=True))
    data["masthead"]["system"] = HOSTILE_LINE
    data["lede"] = HOSTILE.split("<iframe")[0]  # the lede holds two sentences at most
    held = report_input.model_dump(by_alias=True, exclude_none=True)
    held["meta"]["campaign"] = HOSTILE_LINE
    for index, arm in enumerate(held["arms"]):
        arm.update(name=f"{HOSTILE_LINE} {index}")
        arm.pop("short_name", None)
    for part in ("aspects", "strata", "tasks", "factors"):
        for item in held.get(part, []):
            item["name"] = HOSTILE_LINE
    for table in ("measures", "judged"):
        for figures in held[table].values():
            figures["label"] = HOSTILE_LINE
    for example in held.get("examples", []):
        example.update(reason=HOSTILE, excerpt=HOSTILE, input=HOSTILE)
    return ReportPlan.model_validate(data), ReportInput.model_validate(held)


async def hostile_planned_page() -> CampaignReport:
    """The hostile campaign's page as a planned analysis lays it out: the page the Kairo reference plan draws over its
    input, both carrying markup (:func:`hostile_plan_input`), every part of it drawn (its explore part and Full
    results too), the hostile campaign's folds and glossary its appendix."""
    plan, report_input = hostile_plan_input(
        dataset.load_plan(KIT / "references" / "r5-kairo-round2.plan.json"), dataset.report_input_of("r5")
    )
    page = await hostile_campaign_page()
    return CampaignReport.model_validate(
        {
            **dict(page),
            "blocks": [],
            "recommendation": None,
            "recommended_variant_key": None,
            "plan_page": plan_page(plan, report_input, made=HOSTILE_LINE),
        }
    )
