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

import json
import re
import uuid
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from threetears.evals import Answer, Eval, Result
from threetears.evals.analysis import (
    CampaignReport,
    assemble_context_bundle,
    build_code_only_report,
    campaign_report,
    campaign_report_markdown,
    create_campaign,
    get_campaign,
    list_analyses,
    prepare_analysis_generation,
    run_analysis_generation,
)
from threetears.evals.analysis.report import TableBlock
from threetears.evals.kernel import EvalCampaign, ValidationFailedError
from threetears.evals.kernel.declaration import BarOverride, Floor, OverriddenBar, RateDrop
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops import campaign_record
from threetears.evals.quick.contract import as_host
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.toyhost_memo import PROMPT, PROMPT_ID, FixturedClient

#: The v15 campaign as v0.70 stored it, the shape every adopted campaign below is written in.
_FROZEN_V15 = Path(__file__).resolve().parent / "fixtures" / "durable_documents" / "v15" / "eval_campaign.json"

SCOPE = "support"

#: The seven campaigns one decision used to need seven memos for: one template, one behavior each.
LABELS = ("billing", "tone", "refunds", "shipping", "returns", "accounts", "outages")

#: When each source campaign declared its design: well before any run.
DECLARED_AT = "2000-01-01T00:00:00+00:00"

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


def _stored_v15(held: Store, label: str, *, bars: Sequence[BarOverride] = ()) -> str:
    """Write the eval ``label``'s campaign again as v0.70 stored it — a v15 document, its design declared before the
    runs and naming production as its control — and return its id."""
    quick = get_campaign(held.host.storage, held.runs[label].campaign_id, SCOPE)
    (aspect,) = quick.aspects
    control = quick.control
    assert aspect.design is not None and control is not None
    document: dict[str, Any] = json.loads(_FROZEN_V15.read_text(encoding="utf-8"))
    document["declared_design"].update(
        {
            "axes": [axis.model_dump(mode="json") for axis in aspect.design.axes],
            "held_fixed": aspect.design.held_fixed.model_dump(mode="json"),
            "bars": [bar.model_dump(mode="json") for bar in bars],
            "control": control.variant_key,
            "control_levers": {name: value.model_dump(mode="json") for name, value in (control.levers or {}).items()},
            "control_identity_version": control.identity_version,
            "declared_at": DECLARED_AT,
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
        sources = [off_loop(_stored_v15, held, label) for label in LABELS]
        one = off_loop(
            create_campaign,
            held.host.storage,
            {
                "name": "support queue",
                "subject_id": "support",
                "behavior": "support",
                "question": QUESTION,
                "margins": {label: 10.0 for label in LABELS},
            },
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
    analysis_id = await _generated(held, one)

    (analysis,) = off_loop(list_analyses, held.host.storage, one.id, SCOPE)
    assert analysis.id == analysis_id and [aspect.label for aspect in analysis.aspects] == list(LABELS)
    page = off_loop(campaign_report, held.host, one.id, SCOPE)
    assert (page.basis, page.analysis_id, page.finding_count) == ("analysis", analysis_id, 0)
    assert [fold.aspect for fold in page.aspects] == list(LABELS)
    assert [row["aspect"].text for row in _table(page, "aspects").rows] == list(LABELS)
    assert page.recommendation is not None and page.recommendation.arm == "control"
    first = _first_screen(page)
    assert "**Recommended:** control" in first
    assert first.index("**Recommended:**") < first.index("**By aspect**")

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
    alone = build_code_only_report(
        off_loop(assemble_context_bundle, campaign, storage=held.host.storage, profile=held.host.profile),
        assembled_at=fold.report.source.generated_at,
        campaign_name=campaign.name,
    )
    assert fold.report.section_titles == alone.section_titles
    assert [block.kind for block in fold.report.blocks] == [block.kind for block in alone.blocks]


# =============================================================================
# Two aspects: one regresses past its floor, one improves
# =============================================================================


_TWO: dict[str, tuple[Store, EvalCampaign]] = {}


async def _two() -> tuple[Store, EvalCampaign]:
    """Billing regresses (production passes eighteen of twenty cases, the candidate six) under a 10-point floor; tone
    improves (six to nineteen) under the cost bar the candidate misses. Asked before every run."""
    if not _TWO:
        evals = [
            _eval("billing", cases=20, production=18, candidate=6, repeats=3),
            _eval("tone", cases=20, production=6, candidate=19, repeats=3),
        ]
        held = await _store(evals)
        quick = {
            label: off_loop(get_campaign, held.host.storage, held.runs[label].campaign_id, SCOPE) for label in held.runs
        }
        control = quick["billing"].control
        assert control is not None
        aspects = []
        for label, campaign in quick.items():
            (aspect,) = campaign.aspects
            assert aspect.design is not None
            design = aspect.design.model_copy(update={"merit_axes": [], "bars": [COST_BAR] if label == "tone" else []})
            aspects.append(
                aspect.model_copy(
                    update={
                        "label": label,
                        "margin": 10.0,
                        "design": design,
                        "declared_at": DECLARED_AT,
                        "asked_at": DECLARED_AT,
                        "margin_declared_at": DECLARED_AT,
                    }
                )
            )
        two = quick["billing"].model_copy(
            update={
                "id": str(uuid.uuid4()),
                "name": "support queue",
                "behavior": "support",
                "question": QUESTION,
                "declared_at": DECLARED_AT,
                "arms": [control.model_copy(update={"label": "production", "named_at": DECLARED_AT})],
                "aspects": aspects,
                "floors": [Floor(aspect="billing", clause=RateDrop(max_points=10))],
            }
        )
        off_loop(held.host.storage.save_campaign, EvalCampaign.model_validate(two.model_dump()))
        _TWO["two"] = (held, two)
    return _TWO["two"]


async def test_two_aspects_that_regress_and_improve_are_both_on_the_first_screen_the_broken_floor_first() -> None:
    held, two = await _two()
    page = off_loop(campaign_report, held.host, two.id, SCOPE)
    assert page.basis == "code_only" and [fold.aspect for fold in page.aspects] == ["billing", "tone"]
    readings = {row.aspect: row.reading for row in page.readings.readings}
    assert readings == {"billing": "worse", "tone": "better"}
    first = _first_screen(page)
    assert first.index("**Floor broken on billing") < first.index("# support queue") < first.index("**By aspect**")
    assert "Bar missed on tone" in first


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
