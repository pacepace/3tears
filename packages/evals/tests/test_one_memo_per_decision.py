"""One decision, one memo: a campaign of aspects is read, written and decided as one, through the service.

The owner's ruling (seven memos for one decision is the worst evals pain) as an acceptance test, end to end over real
runs. Seven campaigns stored before aspects existed — v15 documents, one template each — are adopted whole into one
campaign (``aspects_from``): one generation writes one memo over all seven, the campaign's page lays out one row per
aspect, the writer's one recommendation and the person's call recorded through ``campaign_record``. A stored v15
campaign read alone is a campaign of one aspect, whose fold is the aspect's own page. And a campaign of two aspects
that regresses on one (breaking its floor) and improves on the other shows both on its first screen, the broken floor
first; recording a call over it is refused until the call names the broken floor it overrides, and the same for a
missed bar.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

import pytest

from threetears.evals import Answer, Eval, Result
from threetears.evals.analysis import (
    CampaignReport,
    campaign_report,
    campaign_report_markdown,
    create_campaign,
    get_campaign,
    list_analyses,
    prepare_analysis_generation,
    run_analysis_generation,
)
from threetears.evals.analysis.campaign_bundle import writer_view
from threetears.evals.analysis.report import DisclosureBlock, TableBlock, TextBlock
from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.kernel import IDENTITY_VERSION, ConflictError, EvalCampaign, ValidationFailedError
from threetears.evals.kernel.declaration import BarOverride, Floor, OverriddenBar, RateDrop
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.identity import compute_variant_key
from threetears.evals.ops import analyses_list, analysis_archive, campaign_record
from threetears.evals.quick.contract import as_host
from threetears.evals.run.curation import set_analysis_archived
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture
from packages.evals.tests.toyhost_memo import PROMPT, PROMPT_ID, FixturedClient

#: The v15 campaign as v0.70 stored it, the shape every adopted campaign below is written in.
_FROZEN_V15 = Path(__file__).resolve().parent / "fixtures" / "durable_documents" / "v15" / "eval_campaign.json"

SCOPE = "support"

#: The seven campaigns one decision used to need seven memos for: one template, one behavior each.
LABELS = ("billing", "tone", "refunds", "shipping", "returns", "accounts", "outages")

#: When each source campaign declared its design: well before any run.
DECLARED_AT = "2000-01-01T00:00:00+00:00"

#: When each of the seven asked its question: a day apart, all before any run.
ASKED_AT = {label: f"2000-01-{day + 2:02d}T00:00:00+00:00" for day, label in enumerate(LABELS)}

#: The one-arm-pair page v0.70.1 laid out for the billing eval (eight cases, production passing six and the candidate
#: three, two repeats), read off that release's own code: each section's heading, then each block's section and kind,
#: runs of a kind counted. A v15 campaign read now folds this page unchanged, but for one deliberate rename: the
#: section that held the decision rule holds the floors now, and is headed so (``FLOORS_HEADING_SINCE_0_71``).
V0701_SECTION_TITLES = {
    "header": "",
    "scoreboard": "Scoreboard",
    "decision": "Decision",
    "memo": "What it means",
    "contrasts": "Contrasts",
    "cases": "Cases",
    "trials": "Trials",
    "not_tested": "Not tested",
    "methods": "Methods",
    "glossary": "Glossary",
}

#: The one heading the campaign rewrite renamed: a floor warns and never decides, so its section is not headed "Decision".
FLOORS_HEADING_SINCE_0_71 = {"decision": "Floors"}
V0701_BLOCKS = [
    ("header", "disclosure", 2),
    ("scoreboard", "table", 1),
    ("scoreboard", "chart", 1),
    ("contrasts", "table", 3),
    ("cases", "table", 1),
    ("trials", "disclosure", 1),
    ("trials", "table", 1),
    ("trials", "trial", 14),
    ("trials", "table", 1),
    ("not_tested", "disclosure", 3),
    ("not_tested", "table", 1),
    ("methods", "table", 1),
    ("methods", "disclosure", 6),
]

QUESTION = "Does the candidate answer support requests as well as production?"

#: No reply may cost more than a tenth of a cent; the candidate's cost two tenths, so it misses this bar.
COST_BAR = BarOverride(measure_id="cost_usd", threshold=0.001, direction="lower_is_better")


def _first_cases(right: int, cost: float) -> Callable[[Mapping[str, Any]], Coroutine[Any, Any, Answer]]:
    """A runner answering right on every trial of the first ``right`` cases and wrong on the rest, at ``cost``."""

    async def run(case: Mapping[str, Any]) -> Answer:
        index = int(case["text"].rsplit(" ", 1)[1])
        return Answer(
            case["label"] if index < right else "skipped", model="m", input_tokens=100, output_tokens=10, cost_usd=cost
        )

    return run


def _eval(label: str, *, cases: int, production: int, candidate: int, repeats: int) -> Eval:
    """One behavior's eval: production passing its first ``production`` cases, the candidate its first
    ``candidate`` at twice the cost."""
    rows = [{"text": f"{label} request {index}", "label": "queued"} for index in range(cases)]
    arms = {"production": _first_cases(production, 0.0005), "candidate": _first_cases(candidate, 0.002)}
    return Eval(rows, arms, ["label"], k=repeats, name=label)


@dataclass(frozen=True)
class Store:
    """One store holding every eval's runs, a host over all of them, and the writer client it hands out."""

    store: InMemoryDocumentStore
    host: EvalHost
    runs: dict[str, Result]
    clients: list[FixturedClient]


async def _store(evals: Sequence[Eval]) -> Store:
    store = InMemoryDocumentStore()
    runs = {ev.name: await ev.run(store=store, scope=SCOPE) for ev in evals}
    built: list[FixturedClient] = []

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> FixturedClient:
        client = FixturedClient("")
        built.append(client)
        return client

    return Store(store=store, host=as_host(list(evals), store=store, clients=clients), runs=runs, clients=built)


def _stored_v15(
    held: Store,
    label: str,
    *,
    asked_at: str | None = None,
    bars: Sequence[BarOverride] = (),
    stale_control: bool = False,
) -> str:
    """Write the eval ``label``'s campaign again as v0.70 stored it — a v15 document, its design declared before the
    runs and naming production as its control — and return its id.

    ``asked_at`` asks :data:`QUESTION` then, ruled by a 10-point drop on the pass rate; None asks nothing, as the
    quick path's campaign did. ``stale_control`` stores the control's key as one keyed under the identity predicate
    before this one, beside the lever map that re-keys it.
    """
    quick = get_campaign(held.host.storage, held.runs[label].campaign_id, SCOPE)
    (aspect,) = quick.aspects
    control = quick.control
    assert aspect.design is not None and control is not None
    document: dict[str, Any] = json.loads(_FROZEN_V15.read_text(encoding="utf-8"))
    questions = [
        {
            "id": str(uuid.uuid4()),
            "text": QUESTION,
            "decision": {"clauses": [{"kind": "rate_drop", "max_points": 10.0, "reading": None}]},
            "merit_axes": [],
            "asked_at": asked_at,
            "retired_at": None,
            "supersedes": None,
        }
    ]
    document["declared_design"].update(
        {
            "axes": [axis.model_dump(mode="json") for axis in aspect.design.axes],
            "held_fixed": aspect.design.held_fixed.model_dump(mode="json"),
            "bars": [bar.model_dump(mode="json") for bar in bars],
            "control": hashlib.sha256(b"keyed before").hexdigest() if stale_control else control.variant_key,
            "control_levers": {name: value.model_dump(mode="json") for name, value in (control.levers or {}).items()},
            "control_identity_version": control.identity_version - 1 if stale_control else control.identity_version,
            "declared_at": DECLARED_AT,
            "questions": questions if asked_at is not None else [],
        }
    )
    document.update(
        {
            "id": str(uuid.uuid4()),
            "scope_id": SCOPE,
            "name": f"{label} (stored by v0.70)",
            "subject_id": "support",
            "behavior": label,
            "template_id": aspect.template_id,
            "run_ids": list(aspect.run_ids),
        }
    )
    held.store.upsert(document)
    return str(document["id"])


def _memo(arm: str, labels: Sequence[str]) -> dict[str, Any]:
    """The writer's one memo: keep production, resting on no finding, with a summary of every aspect."""
    return {
        "findings": [],
        "recommendation": [
            {
                "arm": arm,
                "reasons": ["No aspect showed the candidate better, and it costs four times as much a reply."],
                "confidence": "medium",
                "rests_on": [],
                "floors_overridden": [],
                "behind_on": [],
                "bars_missed": [],
            }
        ],
        "aspect_summaries": [
            {
                "aspect": label,
                "what_changed": "The candidate moved the pass rate here.",
                "how_sure": "Not sure: eight cases leave it open.",
                "means": "Nothing here argues for switching.",
            }
            for label in labels
        ],
        "summary": "No aspect showed the candidate better, so the control stays.",
        "next": [],
    }


# =============================================================================
# Seven stored campaigns, one decision, one memo
# =============================================================================


_SEVEN: dict[str, tuple[Store, EvalCampaign]] = {}


async def _seven() -> tuple[Store, EvalCampaign]:
    """Seven v15 campaigns adopted into one, measured once for the module."""
    if not _SEVEN:
        evals = [
            _eval(label, cases=8, production=6, candidate=7 if index % 2 else 3, repeats=2)
            for index, label in enumerate(LABELS)
        ]
        held = await _store(evals)
        sources = [off_loop(_stored_v15, held, label, asked_at=ASKED_AT[label]) for label in LABELS]
        one = off_loop(
            create_campaign,
            held.host.storage,
            {"name": "support queue", "subject_id": "support", "behavior": "support", "question": QUESTION},
            scope_id=SCOPE,
            created_by="op",
            profile=held.host.profile,
            aspects_from=sources,
        )
        _SEVEN["one"] = (held, one)
    return _SEVEN["one"]


async def _generated(held: Store, campaign: EvalCampaign) -> str:
    """Generate the campaign's one memo with the writer's canned draft, and return the analysis's id."""

    async def prompt() -> str:
        return PROMPT

    prepared = await prepare_analysis_generation(
        held.host, campaign.id, SCOPE, model=None, resolve_prompt=prompt, out_of_run_cap_usd=None
    )
    control = next(arm.label for arm in prepared.bundle.arms if arm.role == "control")
    held.clients[-1].completion.content = json.dumps(_memo(control, [aspect.label for aspect in campaign.aspects]))
    analysis, _ = await run_analysis_generation(held.host, prepared, prompt_id=PROMPT_ID, max_output_tokens=4000)
    return analysis.id


def _table(page: CampaignReport, name: str) -> TableBlock:
    (table,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == name]
    return table


def _first_screen(page: CampaignReport) -> str:
    """The Markdown above the first fold's heading."""
    return campaign_report_markdown(page).split("\n## ", 1)[0]


async def test_seven_stored_campaigns_adopted_into_one_are_one_memo_one_table_and_one_call() -> None:
    held, one = await _seven()
    assert [aspect.label for aspect in one.aspects] == list(LABELS)
    # Each adopted aspect keeps when its own campaign asked the question, so each reads as asked before its runs.
    assert {aspect.label: aspect.asked_at for aspect in one.aspects} == ASKED_AT
    analysis_id = await _generated(held, one)

    (analysis,) = off_loop(list_analyses, held.host.storage, one.id, SCOPE)
    assert analysis.id == analysis_id and [aspect.label for aspect in analysis.aspects] == list(LABELS)
    page = off_loop(campaign_report, held.host, one.id, SCOPE)
    assert (page.basis, page.analysis_id, page.finding_count) == ("analysis", analysis_id, 0)
    assert [fold.aspect for fold in page.aspects] == list(LABELS)
    assert [row["aspect"].text for row in _table(page, "aspects").rows] == list(LABELS)
    assert page.recommendation is not None and page.recommendation.arm == "control"
    assert {row.aspect for row in page.readings.readings} == set(LABELS)
    assert all(row.confirmatory for row in page.readings.readings), "every aspect was asked before its runs"
    # Each aspect's summary is laid out in its fold, in the writer's words.
    for fold in page.aspects:
        said = [block.body for block in fold.summary if isinstance(block, TextBlock)]
        assert said == [
            "The candidate moved the pass rate here.",
            "Not sure: eight cases leave it open.",
            "Nothing here argues for switching.",
        ], fold.aspect
    first = _first_screen(page)
    assert "**Recommended:** control" in first
    # Code's readings follow the recommendation, in the page's Scorecard.
    markdown = campaign_report_markdown(page)
    assert markdown.index("**Recommended:**") < markdown.index("## Scorecard") < markdown.index("**By aspect**")

    recorded = off_loop(
        campaign_record,
        held.host,
        one.id,
        SCOPE,
        arm="control",
        called_by="brooks",
        reason="Nothing argues for the switch.",
        analysis_id=analysis_id,
    )
    assert recorded.call.variant_key == one.arms[0].variant_key and recorded.call.analysis_id == analysis_id
    called = _first_screen(off_loop(campaign_report, held.host, one.id, SCOPE))
    assert "Called: keep control, by brooks on " in called and 'Reason: "Nothing argues for the switch."' in called


async def test_a_stored_v15_campaign_reads_as_one_aspect_whose_fold_is_the_aspect_s_own_page() -> None:
    held, _ = await _seven()
    source = off_loop(_stored_v15, held, "billing")
    campaign = off_loop(get_campaign, held.host.storage, source, SCOPE)
    (aspect,) = campaign.aspects
    assert aspect.label == "billing" and aspect.run_ids == campaign.run_ids

    page = off_loop(campaign_report, held.host, source, SCOPE)
    (fold,) = page.aspects
    assert fold.report.section_titles == {**V0701_SECTION_TITLES, **FLOORS_HEADING_SINCE_0_71}
    # v0.70.1's layout: a v15 control carries no time it was named, and its aspect is asked no question, so when it
    # was named decides nothing and nothing is said of it.
    assert not [
        block for block in fold.report.blocks if isinstance(block, DisclosureBlock) and block.source == "timing"
    ]
    laid_out = [
        (section, kind, len(list(run)))
        for (section, kind), run in groupby((block.section, block.kind) for block in fold.report.blocks)
    ]
    assert laid_out == V0701_BLOCKS


# =============================================================================
# Two aspects: one regresses past its floor, one improves
# =============================================================================


_TWO: dict[str, tuple[Store, EvalCampaign]] = {}


async def _two() -> tuple[Store, EvalCampaign]:
    """:func:`_build_two`, measured once for the tests that leave its store as they found it."""
    if not _TWO:
        _TWO["two"] = await _build_two()
    return _TWO["two"]


async def _build_two(
    labels: Sequence[str] = ("billing", "tone"),
    *,
    floor: bool = True,
    bar: bool = True,
    asked_at: str = DECLARED_AT,
) -> tuple[Store, EvalCampaign]:
    """Billing regresses (production passes eighteen of twenty cases, the candidate six) under a 10-point floor; tone
    improves (six to nineteen) under the cost bar the candidate misses. Asked before every run.

    ``labels`` keeps those aspects of the two; ``floor`` and ``bar`` declare billing's floor and tone's bar;
    ``asked_at`` asks the question then instead, after the runs where it is later than them."""
    shapes = {"billing": (18, 6), "tone": (6, 19)}
    evals = [
        _eval(label, cases=20, production=shapes[label][0], candidate=shapes[label][1], repeats=3) for label in labels
    ]
    held = await _store(evals)
    quick = {
        label: off_loop(get_campaign, held.host.storage, held.runs[label].campaign_id, SCOPE) for label in held.runs
    }
    control = quick[labels[0]].control
    assert control is not None
    aspects = []
    for label, campaign in quick.items():
        (aspect,) = campaign.aspects
        assert aspect.design is not None
        bars = [COST_BAR] if label == "tone" and bar else []
        design = aspect.design.model_copy(update={"merit_axes": [], "bars": bars})
        aspects.append(
            aspect.model_copy(
                update={
                    "label": label,
                    "margin": 10.0,
                    "design": design,
                    "declared_at": DECLARED_AT,
                    "asked_at": asked_at,
                    "margin_declared_at": DECLARED_AT,
                }
            )
        )
    two = quick[labels[0]].model_copy(
        update={
            "id": str(uuid.uuid4()),
            "name": "support queue",
            "behavior": "support",
            "question": QUESTION,
            "declared_at": DECLARED_AT,
            "arms": [control.model_copy(update={"label": "production", "named_at": DECLARED_AT})],
            "aspects": aspects,
            "floors": [Floor(aspect="billing", clause=RateDrop(max_points=10))]
            if floor and "billing" in labels
            else [],
        }
    )
    off_loop(held.host.storage.save_campaign, EvalCampaign.model_validate(two.model_dump()))
    return held, two


async def test_two_aspects_that_regress_and_improve_are_both_in_the_scorecard_the_broken_floor_first() -> None:
    held, two = await _two()
    page = off_loop(campaign_report, held.host, two.id, SCOPE)
    assert page.basis == "code_only" and [fold.aspect for fold in page.aspects] == ["billing", "tone"]
    readings = {row.aspect: row.reading for row in page.readings.readings}
    assert readings == {"billing": "worse", "tone": "better"}
    first = _first_screen(page)
    floor = first.index("**Adopting candidate overrides the 10-point floor it breaks on billing.**")
    assert first.index("# support queue") < floor
    assert "**Adopting candidate overrides the bar it misses on tone.**" in first
    markdown = campaign_report_markdown(page)
    assert markdown.index("# support queue") < markdown.index("## Scorecard") < markdown.index("**By aspect**")


def _record(held: Store, two: EvalCampaign, **call: Any) -> Any:
    return off_loop(campaign_record, held.host, two.id, SCOPE, called_by="brooks", **call)


async def test_a_call_on_the_candidate_is_refused_until_it_overrides_its_broken_floor_and_its_missed_bar() -> None:
    held, two = await _two()
    page = off_loop(campaign_report, held.host, two.id, SCOPE)
    candidate = next(row for row in page.readings.readings if row.aspect == "tone")
    (floor,) = [status.floor for status in page.readings.broken_floors()]
    (bar,) = [status for status in page.readings.missed_bars() if status.arm == candidate.arm]
    missed = OverriddenBar(aspect=bar.aspect, measure=bar.measure)

    with pytest.raises(
        ValidationFailedError, match=re.escape(f"breaks or misses floor {floor}, the bar on cost_usd on tone")
    ):
        _record(held, two, arm=candidate.arm_label)
    with pytest.raises(ValidationFailedError, match="the bar on cost_usd on tone, and the call does not override"):
        _record(held, two, arm=candidate.arm_label, floors_overridden=[floor])

    recorded = _record(
        held, two, arm=candidate.arm, floors_overridden=[floor], bars_overridden=[missed], reason="Tone matters more."
    )
    assert recorded.call.arm == candidate.arm_label and recorded.call.variant_key == candidate.arm
    first = _first_screen(off_loop(campaign_report, held.host, two.id, SCOPE))
    assert f"Called: adopt {candidate.arm_label}, by brooks on " in first
    assert "The call overrides a broken floor: no pass rate drops by more than 10 points on billing." in first
    assert f"The call overrides a missed bar on tone: {bar.reason}." in first


async def test_a_call_naming_a_warning_its_arm_did_not_raise_or_an_arm_the_campaign_lacks_is_refused() -> None:
    held, two = await _two()
    page = off_loop(campaign_report, held.host, two.id, SCOPE)
    (floor,) = [status.floor for status in page.readings.broken_floors()]
    with pytest.raises(
        ValidationFailedError, match=re.escape(f"overrides floor {floor}, which production does not break")
    ):
        _record(held, two, arm="production", floors_overridden=[floor])
    with pytest.raises(ValidationFailedError, match="has no arm 'v9'") as refused:
        _record(held, two, arm="v9")
    assert "production" in refused.value.remedy


async def test_a_measurement_has_nothing_to_decide_so_a_call_on_it_is_refused() -> None:
    held = await _store([Eval([{"text": "solo request 0", "label": "queued"}], _first_cases(1, 0.001), ["label"])])
    (run,) = held.runs.values()
    with pytest.raises(ValidationFailedError, match="measures one arm, so there is nothing to decide between"):
        off_loop(campaign_record, held.host, run.campaign_id, SCOPE, arm="control", called_by="brooks")


def _warnings(page: CampaignReport) -> list[str]:
    """The bar and floor warnings the page opens with, in its order."""
    return [
        block.text
        for block in page.blocks
        if isinstance(block, DisclosureBlock) and block.section == "warnings" and block.source in ("bar", "floor")
    ]


async def test_an_analysis_of_two_aspects_is_listed_archived_and_restored_by_its_page_s_warnings_and_verdict() -> None:
    """An analysis over both aspects is listed, archived and restored, each line reading as its page opens: the bar
    tone misses and the floor billing breaks, in the page's own words, then the campaign's verdict: what code read on
    each aspect, as the campaign's page says it beside each fold, never one aspect's line."""
    held, two = await _build_two()
    analysis_id = await _generated(held, two)
    page = off_loop(campaign_report, held.host, two.id, SCOPE)
    assert page.analysis_id == analysis_id
    verdict = "; ".join(f"{fold.aspect}: {fold.shows}" for fold in page.aspects)
    assert verdict.startswith("billing: ") and "shown worse; tone: " in verdict and verdict.endswith("shown better")
    bar, floor = _warnings(page)
    assert (bar, floor) == (
        "Adopting candidate overrides the bar it misses on tone.",
        "Adopting candidate overrides the 10-point floor it breaks on billing.",
    )
    expected = f"{bar} {floor} {verdict}"

    (line,) = off_loop(analyses_list, held.host, two.id, SCOPE).analyses
    assert (line.id, line.verdict, line.archived) == (analysis_id, expected, False)
    archived = off_loop(analysis_archive, held.host, analysis_id, SCOPE, archived=True, reason="shown false")
    assert (archived.id, archived.verdict, archived.archived) == (analysis_id, expected, True)
    (listed,) = off_loop(analyses_list, held.host, two.id, SCOPE).analyses
    assert listed.archived
    restored = off_loop(analysis_archive, held.host, analysis_id, SCOPE, archived=False)
    assert (restored.verdict, restored.archived) == (expected, False)


#: An aspect asked long after every result, as one declared over runs whose results are already stored is.
ASKED_AFTER_RUNS = "2099-01-01T00:00:00+00:00"


def _timing(page: CampaignReport) -> list[str]:
    """The first screen's lines naming each criterion set after its readings were visible."""
    return [
        block.text
        for block in page.blocks
        if isinstance(block, DisclosureBlock) and block.section == "warnings" and block.source == "timing"
    ]


#: What :func:`_build_two` says once when its aspects are asked after their readings, its margins and control set
#: before.
_SET_LATE = "Criteria on every aspect were set after the results there existed, so they were not fixed in advance."


def _criteria(page: CampaignReport) -> list[tuple[str, str, str]]:
    """Design & provenance's rows of criteria set late: criterion, when, and the aspects whose results existed."""
    (table,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "criteria_set"]
    return [(row["criterion"].text, row["set_on"].text, row["aspects"].text) for row in table.rows]


async def test_a_campaign_asked_after_its_readings_reads_as_asked_and_names_each_late_criterion_once() -> None:
    """Owner rulings (2026-10-21, 2026-10-23): criteria set once their readings were visible are trusted, not demoted.
    Its rows read exactly as the same campaign set before its readings reads them, every one confirmatory, and every
    surface a reader meets the readings on names each late criterion once, with its time, and never the question or
    when the runs began."""
    timely_store, timely = await _build_two()
    held, two = await _build_two(asked_at=ASKED_AFTER_RUNS)

    def rows(store: Store, campaign: EvalCampaign) -> list[tuple[str, str, str, bool]]:
        page = off_loop(campaign_report, store.host, campaign.id, SCOPE)
        return [(row.aspect, row.arm, row.reading, row.confirmatory) for row in page.readings.readings]

    assert rows(held, two) == rows(timely_store, timely) and len(rows(held, two)) == 2
    assert all(confirmatory for *_, confirmatory in rows(held, two))

    async def prompt() -> str:
        return PROMPT

    prepared = await prepare_analysis_generation(
        held.host, two.id, SCOPE, model=None, resolve_prompt=prompt, out_of_run_cap_usd=None
    )
    late = {part["aspect"]: part["late_criteria"] for part in writer_view(prepared.bundle)["aspects"]}
    # Dated as the page dates them, so the memo and the page say one date; the margins came first, so are not named.
    assert late == {
        "billing": [
            {"criterion": "aspect", "name": None, "set_at": "1 Jan 2099"},
            {"criterion": "floor", "name": "billing/rate_drop/*", "set_at": "1 Jan 2099"},
        ],
        "tone": [
            {"criterion": "aspect", "name": None, "set_at": "1 Jan 2099"},
            {"criterion": "bar", "name": COST_BAR.measure_id, "set_at": "1 Jan 2099"},
        ],
    }
    analysis_id = await _generated(held, two)

    page = off_loop(campaign_report, held.host, two.id, SCOPE)
    # The page JSON carries each aspect's late criteria beside its readings, raw.
    aspects = page.json_document()["aspects"]
    assert isinstance(aspects, list)
    assert [[said["criterion"] for said in aspect["late_criteria"]] for aspect in aspects] == [
        ["aspect", "floor"],
        ["aspect", "bar"],
    ]
    said = _timing(page)
    assert said == [_SET_LATE], "one line for the whole report"
    # Each aspect, billing's floor and tone's bar, all declared with the aspect, each dated once.
    (aspect, floor, bar) = _criteria(page)
    assert aspect == ("aspect", "1 Jan 2099", "every aspect")
    assert floor == ("floor that no pass rate drops by more than 10 points", "1 Jan 2099", "billing")
    assert bar[0].startswith("bar on ") and bar[1:] == ("1 Jan 2099", "tone")
    first = _first_screen(page)
    assert all(first.index(QUESTION) < first.index(line) for line in said), "under the question"
    assert "a lead" not in first and "began" not in first and "The question" not in " ".join(said)
    for fold in page.aspects:
        own = [block.text for block in fold.report.blocks if isinstance(block, DisclosureBlock)]
        assert not [
            line for line in said if line in [block.text for block in fold.head if isinstance(block, DisclosureBlock)]
        ], "the campaign's first screen carries them once"
        assert not [line for line in own if line.startswith("Criteria on ")], "its fold does not repeat the line"
        assert not [b for b in fold.report.blocks if isinstance(b, TableBlock) and b.name == "criteria_set"]
    bar, floor = _warnings(page)
    verdict = "; ".join(f"{fold.aspect}: {fold.shows}" for fold in page.aspects)
    (line,) = off_loop(analyses_list, held.host, two.id, SCOPE).analyses
    assert (line.id, line.verdict) == (analysis_id, " ".join([bar, floor, *said, verdict]))

    timely_page = off_loop(campaign_report, timely_store.host, timely.id, SCOPE)
    assert _timing(timely_page) == [], "set in time: nothing to say"
    timely_aspects = timely_page.json_document()["aspects"]
    assert isinstance(timely_aspects, list)
    assert [aspect["late_criteria"] for aspect in timely_aspects] == [[], []]


#: An ISO-8601 instant, or a page's date of one: what two campaigns read over the same runs may differ by.
_WHEN = re.compile(
    r"\d{4}-\d{2}-\d{2}T[\d:.]+(?:Z|[+-]\d{2}:\d{2})?|\b\d{1,2} [A-Z][a-z]{2} \d{4}(?: at \d{2}:\d{2} UTC\S*)?"
)


#: A bundle's fingerprint, which digests the campaign it was read for.
_FINGERPRINT = re.compile(r"v2:[0-9a-f]{64}")


def _undated(text: str, *ids: str) -> str:
    """``text`` with every instant and date, each of ``ids`` and each bundle fingerprint blanked: what is left is what
    a reader is told."""
    for blank in ids:
        text = text.replace(blank, "<id>")
    return _FINGERPRINT.sub("<fingerprint>", _WHEN.sub("<when>", text))


async def test_a_campaign_adopting_still_valid_runs_reads_exactly_as_one_declared_before_them() -> None:
    """#919: past runs that are still valid count as evidence. A campaign declared before its runs, and one created
    after them adopting its aspects (``aspects_from``), every criterion set before its readings were visible, read the
    same runs identically: the writer's input, the page and its confidence wording differ by nothing but ids and
    dates, and neither names a late criterion, the question's timing or when the runs began."""
    held, declared = await _build_two()
    host = held.host
    adopting = off_loop(
        create_campaign,
        host.storage,
        {"name": declared.name, "subject_id": declared.subject_id, "behavior": declared.behavior},
        scope_id=SCOPE,
        created_by="op",
        profile=host.profile,
        aspects_from=[declared.id],
    )
    stored = off_loop(get_campaign, host.storage, declared.id, SCOPE)
    run_ids = sorted({run_id for aspect in declared.aspects for run_id in aspect.run_ids})
    first_result = min(result.scored_at for result in off_loop(host.storage.query_eval_results_by_runs, run_ids, SCOPE))
    assert adopting.created_at > first_result, "created after the readings were visible"
    assert stored.declared_at is not None and stored.declared_at < first_result, "declared before the runs"

    async def prompt() -> str:
        return PROMPT

    async def told(campaign: EvalCampaign) -> tuple[str, str]:
        prepared = await prepare_analysis_generation(
            host, campaign.id, SCOPE, model=None, resolve_prompt=prompt, out_of_run_cap_usd=None
        )
        view = json.dumps(writer_view(prepared.bundle), sort_keys=True)
        page = campaign_report_markdown(off_loop(campaign_report, host, campaign.id, SCOPE))
        return _undated(view, campaign.id), _undated(page, campaign.id)

    (declared_view, declared_page), (adopted_view, adopted_page) = await told(declared), await told(adopting)
    assert adopted_view == declared_view, "the writer is told the same thing"
    assert adopted_page == declared_page, "the page says the same thing, its confidence wording included"
    assert "late_criteria" not in adopted_view and "after its readings" not in adopted_page
    assert "began" not in adopted_page and "after the runs" not in adopted_page


async def test_an_analysis_s_page_json_names_its_recommended_arm_by_the_key_the_analysis_recorded() -> None:
    """A reader matching a call to the recommendation compares keys, never labels: the page's JSON carries the
    key the analysis stored beside its memo, which is the key a call on that arm records."""
    held, two = await _build_two(floor=False, bar=False)
    analysis_id = await _generated(held, two)
    (analysis,) = off_loop(list_analyses, held.host.storage, two.id, SCOPE)
    assert analysis.id == analysis_id and analysis.recommended_variant_key is not None
    page = off_loop(campaign_report, held.host, two.id, SCOPE)
    assert page.json_document()["recommendation"] == {
        "arm": "production",
        "variant_key": analysis.recommended_variant_key,
        "confidence": "medium",
    }
    recorded = _record(held, two, arm="production", analysis_id=analysis_id)
    assert recorded.call.variant_key == analysis.recommended_variant_key


async def test_the_listing_names_what_was_frozen_with_the_analysis_not_the_campaign_as_it_stands() -> None:
    """The floor is withdrawn from the stored campaign after the analysis was written: the analysis's line still names
    it, worded over the floors frozen with it, as the analysis's page still opens on it."""
    held, two = await _build_two()
    analysis_id = await _generated(held, two)
    off_loop(held.host.storage.save_campaign, EvalCampaign.model_validate({**two.model_dump(), "floors": []}))
    (line,) = off_loop(analyses_list, held.host, two.id, SCOPE).analyses
    assert line.id == analysis_id
    (floor,) = [text for text in _warnings(off_loop(campaign_report, held.host, two.id, SCOPE)) if "floor" in text]
    assert floor == "Adopting candidate overrides the 10-point floor it breaks on billing."
    assert floor in line.verdict


async def test_an_analysis_with_no_missed_bar_and_no_broken_floor_is_listed_by_its_verdict_alone() -> None:
    held, two = await _build_two(floor=False, bar=False)
    await _generated(held, two)
    page = off_loop(campaign_report, held.host, two.id, SCOPE)
    assert _warnings(page) == []
    (line,) = off_loop(analyses_list, held.host, two.id, SCOPE).analyses
    assert line.verdict == "; ".join(f"{fold.aspect}: {fold.shows}" for fold in page.aspects)


async def test_an_analysis_of_one_aspect_is_listed_by_its_page_s_broken_floor_then_the_aspect_s_verdict_line() -> None:
    """One aspect's page is the campaign's page too, so its line leads with the floor that page opens on, then the
    verdict line the aspect's page states; without the floor, the verdict line alone."""
    held, one = await _build_two(("billing",))
    analysis_id = await _generated(held, one)
    page = off_loop(campaign_report, held.host, one.id, SCOPE)
    (floor,) = _warnings(page)
    assert floor == "Adopting candidate overrides the 10-point floor it breaks on billing."
    (fold,) = page.aspects
    (verdict,) = [
        block.text for block in fold.report.blocks if isinstance(block, DisclosureBlock) and block.source == "verdict"
    ]
    (line,) = off_loop(analyses_list, held.host, one.id, SCOPE).analyses
    assert (line.id, line.verdict) == (analysis_id, f"{floor} {verdict}")

    unfloored, plain = await _build_two(("billing",), floor=False)
    await _generated(unfloored, plain)
    (bare,) = off_loop(analyses_list, unfloored.host, plain.id, SCOPE).analyses
    assert bare.verdict == verdict


# =============================================================================
# A call is read against the page it will appear on
# =============================================================================


async def test_a_call_over_an_analysis_is_read_against_its_readings_and_one_without_against_the_evidence_now() -> None:
    """The memo was written while billing's floor stood; the floor is then withdrawn from the stored campaign. A call
    over the analysis is read against the readings frozen with it, which still break the floor; a call on code's
    readings alone is read against the evidence now, which breaks none — and the page, the analysis's, says the floor
    broken there is not overridden."""
    held, two = await _build_two()
    analysis_id = await _generated(held, two)
    off_loop(held.host.storage.save_campaign, EvalCampaign.model_validate({**two.model_dump(), "floors": []}))
    live = off_loop(campaign_report, held.host, two.id, SCOPE).readings
    candidate = next(row for row in live.readings if row.aspect == "tone")
    (bar,) = [status for status in live.missed_bars() if status.arm == candidate.arm]
    missed = OverriddenBar(aspect=bar.aspect, measure=bar.measure)

    with pytest.raises(ValidationFailedError, match=re.escape("breaks or misses floor billing/rate_drop/*")):
        _record(held, two, arm=candidate.arm, bars_overridden=[missed], analysis_id=analysis_id)
    recorded = _record(held, two, arm=candidate.arm, bars_overridden=[missed])

    assert recorded.call.analysis_id is None
    page = off_loop(campaign_report, held.host, two.id, SCOPE)
    assert page.analysis_id == analysis_id
    assert (
        "The call does not name the broken floor as overridden: no pass rate drops by more than 10 points on billing."
        in _first_screen(page)
    )


async def test_a_call_over_an_analysis_the_page_does_not_report_is_refused() -> None:
    """An older analysis, an archived one, and another campaign's are each refused, naming why."""
    held, two = await _build_two()
    older = await _generated(held, two)
    newer = await _generated(held, two)
    with pytest.raises(ValidationFailedError, match=rf"is not campaign {two.id}'s report: its page reports {newer}"):
        _record(held, two, arm="production", analysis_id=older)
    off_loop(set_analysis_archived, held.host.storage, newer, SCOPE, archived=True, reason="shown false")
    with pytest.raises(ValidationFailedError, match=f"analysis {newer} is archived"):
        _record(held, two, arm="production", analysis_id=newer)
    other = off_loop(
        get_campaign, held.host.storage, off_loop(_stored_v15, held, "billing", asked_at=DECLARED_AT), SCOPE
    )
    theirs = await _generated(held, other)
    with pytest.raises(ValidationFailedError, match=f"analysis {theirs} is not an analysis of campaign {two.id}"):
        _record(held, two, arm="production", analysis_id=theirs)

    assert _record(held, two, arm="production", analysis_id=older).call.analysis_id == older


async def test_a_control_stored_under_an_older_identity_predicate_is_kept_and_its_overridden_bar_said() -> None:
    """A v15 control keyed under the predicate before this one is read under its re-keyed key; recording and the page
    both find it by every key it resolves to, so the call keeps it and the page says the bar it overrides."""
    held = await _store([_eval("billing", cases=8, production=6, candidate=3, repeats=2)])
    tight = BarOverride(measure_id="cost_usd", threshold=0.0001, direction="lower_is_better")
    source = off_loop(_stored_v15, held, "billing", asked_at=DECLARED_AT, bars=[tight], stale_control=True)
    campaign = off_loop(get_campaign, held.host.storage, source, SCOPE)
    assert campaign.control is not None and campaign.control.identity_version < IDENTITY_VERSION
    readings = off_loop(campaign_report, held.host, source, SCOPE).readings
    stale, rekeyed = campaign.control.variant_key, compute_variant_key(campaign.control.levers or {})
    assert rekeyed != stale
    # The bundle reads the control's runs under the key this build stamps, never the one stored.
    (bar,) = [status for status in readings.missed_bars() if status.arm == rekeyed]

    recorded = off_loop(
        campaign_record,
        held.host,
        source,
        SCOPE,
        arm="control",
        called_by="brooks",
        bars_overridden=[OverriddenBar(aspect="billing", measure=bar.measure)],
    )

    assert recorded.call.variant_key == campaign.control.variant_key
    first = _first_screen(off_loop(campaign_report, held.host, source, SCOPE))
    assert "Called: keep control, by brooks on " in first
    assert f"The call overrides a missed bar on billing: {bar.reason}." in first


async def test_a_choice_s_call_on_an_arm_shown_behind_is_recorded_and_the_page_warns_of_the_pair() -> None:
    """The person decides: with no control the campaign is a choice, and production, shown behind the candidate on
    tone, is recorded all the same; the page sets the pair beside the call."""
    held, two = await _build_two()
    aspects = [aspect.model_copy(update={"margin": None, "margin_declared_at": None}) for aspect in two.aspects]
    choice = EvalCampaign.model_validate({**two.model_dump(), "arms": [], "floors": [], "aspects": aspects})
    off_loop(held.host.storage.save_campaign, choice)
    page = off_loop(campaign_report, held.host, choice.id, SCOPE)
    assert page.readings.kind == "choice"
    (ahead,) = [pair for pair in page.readings.pairs if pair.aspect == "tone"]
    assert ahead.ahead_label == "candidate"

    recorded = _record(held, choice, arm="production")

    assert recorded.call.arm == "production"
    first = _first_screen(off_loop(campaign_report, held.host, choice.id, SCOPE))
    assert "candidate is shown ahead of production on tone." in first


async def test_a_campaign_changed_while_its_readings_were_read_refuses_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second read — the one under the write lock — finds the floors changed since the first: the call was read
    against another campaign, so it is refused and nothing is written."""
    held, two = await _build_two()
    storage = held.host.storage
    read = storage.load_campaign
    reads: list[str] = []

    def changing(campaign_id: str, scope_id: str) -> EvalCampaign | None:
        reads.append(campaign_id)
        found = read(campaign_id, scope_id)
        return found if found is None or len(reads) == 1 else found.model_copy(update={"floors": []})

    monkeypatch.setattr(storage, "load_campaign", changing)
    with pytest.raises(ConflictError, match="changed while its readings were read"):
        _record(held, two, arm="production")
    monkeypatch.undo()
    stored = off_loop(storage.load_campaign, two.id, SCOPE)
    assert stored is not None and stored.call is None


# =============================================================================
# The action
# =============================================================================


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


async def test_the_action_reads_each_bar_as_aspect_and_measure_and_records_its_caller(evals: MountedTool) -> None:
    """The toy campaign is a choice of two chunk widths, the wider missing a bar: the action parses each
    ``<aspect>/<measure>``, refuses one without a slash, and records the call as made by its caller."""
    fixture = off_loop(ops_fixture)
    page = off_loop(campaign_report, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE)
    (bar,) = [status for status in page.readings.missed_bars() if status.arm_label == "1024tok"]
    call = {"action": "campaign_record", "campaign_id": fixture.campaign.id, "arm": "1024tok"}

    refused = await evals.call({**call, "bars_overridden": ["no slash here"]}, host=fixture.host, caller=CALLER)
    assert (
        refused.is_error and "bars_overridden names 'no slash here', which is not `<aspect>/<measure>`" in refused.text
    )

    named = [f"{bar.aspect}/{bar.measure}"]
    read = await evals.call(
        {**call, "bars_overridden": named, "call_reason": "wider"}, host=fixture.host, caller=CALLER
    )
    assert not read.is_error, read.text
    recorded = read.structured["call"]
    assert recorded["called_by"] == CALLER.identity and recorded["reason"] == "wider"
    assert recorded["bars_overridden"] == [{"aspect": bar.aspect, "measure": bar.measure}]
