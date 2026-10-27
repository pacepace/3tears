"""The fixture snapshots every analyst tool is held to, and the calls each tool is held to them with.

The inputs are the reference inputs r1 to r5 (bare report inputs, as a local analyst server seeds them), the misfit
inputs (:mod:`packages.evals.tests.fixtures.report_plans.misfits`), and the Kairo round 2 campaign frozen from its
bundle with the analyst's example pool, so a snapshot holding a bundle is held to the same contract as a bare one.

:data:`ANALYST_CALLS` says how each analyst action is called on one input. An action it does not name is called with
its own example's parameters, the snapshot id put in, so every action a later chunk registers is held to the contract
with nothing added here; a chunk whose action needs arguments read off the input adds its entry.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from functools import cache
from typing import Any

from packages.evals.tests.fixtures.report_plans.dataset import kairo_bundle
from packages.evals.tests.fixtures.report_plans.misfits import MISFITS, NAMES, reference_data
from packages.evals.tests.report_shapes_support import SHAPES
from threetears.evals.analysis.campaign_bundle import CampaignBundle, build_campaign_bundle
from threetears.evals.actions import Action, analyst_actions
from threetears.evals.analysis.analyst.blocks import block_catalog, block_render
from threetears.evals.analysis.analyst.compare import ArmsComparison, arms_compare
from threetears.evals.analysis.analyst.analyses import sensitivity
from threetears.evals.analysis.analyst.levers import lever_effects
from threetears.evals.analysis.analyst.price import PriceTradeoff, price_tradeoff
from threetears.evals.analysis.analyst.settle import settle_cases
from threetears.evals.analysis.analyst.snapshot import analyst_input, report_snapshot
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel import EvalReportSnapshot
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.report_plan import ChartQuery

__all__ = [
    "ANALYST_CALLS",
    "KAIRO",
    "TEXT_BUDGET",
    "WIDE",
    "WIDE_FRONTIER",
    "action_calls",
    "aspect_tagged",
    "budget_inputs",
    "compared_pair",
    "criterion_on_another_aspect",
    "read_actions",
    "fixture_inputs",
    "shape_bundle",
    "fixture_snapshots",
    "frontier_widened",
    "widened",
]

#: The most characters an analyst action's text may run to at its default limits: what one tool result may take of an
#: analyst's context.
TEXT_BUDGET = 12_000

#: The Kairo round 2 snapshot's name among the fixtures.
KAIRO = "kairo round 2"

#: When the Kairo bundle is recorded as assembled: a fixed instant, so its snapshot is the same on every run.
_KAIRO_ASSEMBLED_AT = "2026-10-20T00:00:00+00:00"

#: Who the fixture snapshots are frozen by.
_FROZEN_BY = "agent:test"


def _first[T](values: Iterable[T]) -> T | None:
    return next(iter(values), None)


def _read_in(held: ReportInput, key: str) -> list[str]:
    """What one measure or judged dimension can be read by on an input: arm, and each slice kind holding it."""
    table = held.measures.get(key) or held.judged[key]
    slices = {
        "arm": bool(table.per_arm) or any(table.by_aspect.values()),
        "stratum": any(key in figures for s in held.strata for figures in s.per_arm.values()),
        "task": any(key in figures for t in held.tasks for figures in t.per_arm.values()),
        "cell": any(key in figures for c in held.cells.values() for figures in c.per_arm.values()),
    }
    return [by for by, holds in slices.items() if holds]


def _measure_reads(held: ReportInput) -> list[dict[str, Any]]:
    """For each kind a measure can be read by, the first measure read that way; vs_control within each slice kind
    where a control exists; and the lead measure on its first aspect."""
    keys = [*held.measures, *held.judged]
    calls: list[dict[str, Any]] = []
    for by in ("arm", "stratum", "task", "cell"):
        key = _first(key for key in keys if by in _read_in(held, key))
        if key is None:
            continue
        calls.append({"measure_key": key, "by": by})
        control = any(arm.role == "control" for arm in held.arms)
        if by != "arm" and control:
            calls.append({"measure_key": key, "by": by, "statistics": ["mean", "vs_control", "n_cases"]})
    aspect = _first(aspect.id for aspect in held.aspects)
    if calls and aspect is not None:
        calls.append({"measure_key": calls[0]["measure_key"], "aspect_filter": aspect})
    return calls


def _readings_calls(held: ReportInput) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = [{}, {"arm_filter": held.arms[-1].id}]
    aspect = _first(aspect.id for aspect in held.aspects)
    return calls + ([{"aspect_filter": aspect, "limit": 2}] if aspect is not None else [])


def _contrasts_calls(held: ReportInput) -> list[dict[str, Any]]:
    first = _first(held.contrasts)
    if first is None:
        return [{}]
    return [{}, {"arm_filter": first.arm, "measure_filter": first.measure}, {"offset": 1, "limit": 3}]


def _examples_calls(held: ReportInput) -> list[dict[str, Any]]:
    arm = _first(example.arm for example in held.examples if example.arm is not None)
    return [{}, {"offset": 10}] + ([{"arm_filter": arm}] if arm is not None else [])


def _bars(held: ReportInput) -> list[str]:
    """The bars the input declares as criteria, by id."""
    return [criterion.id for criterion in held.criteria if criterion.kind == "bar"]


def _dominance_calls(held: ReportInput) -> list[dict[str, Any]]:
    """The decision's own axes, then two extra ones where the input holds them, then its bars read as gates where it
    declares one."""
    keys = [*held.measures, *held.judged]
    calls: list[dict[str, Any]] = [{}, {"axes": keys[:2]}] if keys else [{}]
    return calls + ([{"bars_as": "gates"}] if _bars(held) else [])


def _sensitivity_calls(held: ReportInput) -> list[dict[str, Any]]:
    """The first margin, floor and gate the input declares, one sweep each, its first bar read as a gate, and the
    first of those with two flip values or more swept over a range cut short of them (:func:`_ranged_calls`)."""
    margins = [f"margin:{aspect.id}" for aspect in held.aspects if aspect.readings]
    floors = [f"floor:{read.aspect}/{read.floor}" for read in held.floors]
    gates = [f"threshold:{criterion.id}" for criterion in held.criteria if criterion.kind in ("gate", "guardrail")]
    calls = [{"parameter": found[0]} for found in (margins, floors, gates) if found]
    bars = _bars(held)
    calls += [{"parameter": f"threshold:{bars[0]}", "bars_as": "gates"}] if bars else []
    return calls + next((ranged for ranged in (_ranged_calls(held, call["parameter"]) for call in calls) if ranged), [])


def _ranged_calls(held: ReportInput, parameter: str) -> list[dict[str, Any]]:
    """A sweep cut short of its flips, where it has two flip values or more: from midway between its middle two up to
    its last flip value exactly, and from no lower end up to that midway, so ``unswept`` reads both sides; else none."""
    whole = sensitivity(held, parameter, limit=10_000)
    values = sorted({float(flip.at.value) for flip in whole.flips if isinstance(flip.at.value, int | float)})
    if len(values) < 2:
        return []
    middle = (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
    if parameter.partition(":")[0] in ("margin", "floor") and middle < 0:
        return []
    return [
        {"parameter": parameter, "range_low": middle, "range_high": values[-1]},
        {"parameter": parameter, "range_high": middle},
    ]


def _breakdown_calls(held: ReportInput) -> list[dict[str, Any]]:
    """Where the input has a control: per way of slicing it holds, the first measure the control holds there."""
    control = next((arm.id for arm in held.arms if arm.role == "control"), None)
    if control is None:
        return []
    slices = {
        "stratum": [stratum.per_arm for stratum in held.strata],
        "task": [task.per_arm for task in held.tasks],
        "cell": [cell.per_arm for cell in held.cells.values()],
    }
    calls: list[dict[str, Any]] = []
    for slice_by, held_by in slices.items():
        keys = [key for per_arm in held_by for key in per_arm.get(control, {})]
        if keys:
            calls.append({"measure_key": keys[0], "slice_by": slice_by})
    return calls


#: How each analyst action is called on one input, where its example's parameters will not do: the argument sets,
#: without ``action`` or ``snapshot_id``.
ANALYST_CALLS: dict[str, Callable[[ReportInput], list[dict[str, Any]]]] = {
    "readings_list": _readings_calls,
    "contrasts_list": _contrasts_calls,
    "measure_read": _measure_reads,
    "examples_list": _examples_calls,
    "options_dominance": _dominance_calls,
    "decision_sensitivity": _sensitivity_calls,
    "measure_breakdown": _breakdown_calls,
}


@cache
def fixture_inputs() -> Mapping[str, ReportInput]:
    """Every bare fixture input, by name: r1 to r5, then the misfits.

    :return: The inputs.
    :rtype: Mapping[str, ReportInput]
    """
    inputs = {name: ReportInput.model_validate(reference_data(name)) for name in NAMES}
    inputs |= {name: ReportInput.model_validate(build()) for name, build in MISFITS.items()}
    return inputs


@cache
def _kairo_snapshot(scope_id: str) -> EvalReportSnapshot:
    bundle = kairo_bundle()
    return report_snapshot(
        analyst_input(bundle),
        scope_id=scope_id,
        campaign_id=bundle.campaign_id,
        source="campaign",
        created_by=_FROZEN_BY,
        bundle=bundle,
        assembled_at=_KAIRO_ASSEMBLED_AT,
    )


def fixture_snapshots(scope_id: str) -> dict[str, EvalReportSnapshot]:
    """A snapshot of every fixture input in one scope, by name, and of the Kairo round 2 campaign last.

    :param scope_id: The scope they live in.
    :ptype scope_id: str
    :return: The snapshots, unsaved.
    :rtype: dict[str, EvalReportSnapshot]
    """
    snapshots = {
        name: report_snapshot(held, scope_id=scope_id, campaign_id=None, source="report_input", created_by=_FROZEN_BY)
        for name, held in fixture_inputs().items()
    }
    snapshots[KAIRO] = _kairo_snapshot(scope_id)
    return snapshots


#: The shape campaigns a held-out prototype run reads, built from their bundles: s1 (three aspects), s2 (a choice) and
#: s3 (a measurement).
SHAPE_NAMES = ("s1", "s2", "s3")


@cache
def shape_bundle(name: str) -> CampaignBundle:
    """One shape campaign's bundle (:data:`SHAPE_NAMES`), built once.

    :param name: The shape's short name.
    :ptype name: str
    :return: Its campaign bundle.
    :rtype: CampaignBundle
    """
    shape = SHAPES[name]()
    return build_campaign_bundle(shape.campaign, list(shape.bundles.values()), arms=shape.arms, readings=shape.readings)


def shape_snapshots(scope_id: str) -> dict[str, EvalReportSnapshot]:
    """A snapshot of each shape campaign (:data:`SHAPE_NAMES`) in one scope, by name, frozen with its bundle.

    :param scope_id: The scope they live in.
    :ptype scope_id: str
    :return: The snapshots, unsaved.
    :rtype: dict[str, EvalReportSnapshot]
    """
    return {
        name: report_snapshot(
            analyst_input(shape_bundle(name)),
            scope_id=scope_id,
            campaign_id=shape_bundle(name).campaign_id,
            source="campaign",
            created_by=_FROZEN_BY,
            bundle=shape_bundle(name),
            assembled_at=_KAIRO_ASSEMBLED_AT,
        )
        for name in SHAPE_NAMES
    }


#: The wide input's name among the budget inputs.
WIDE = "wide"

#: The reference input whose gates are read in each task, which the wide input is built from.
_GATES_IN_EACH_TASK = "r2 with its gates read in each task"


def _copied_arm(node: Any, source: str, copy: str) -> Any:
    """A document with one arm's figures copied to a new arm id wherever they sit: every mapping keyed by the arm,
    and every row naming it (an example's trial excepted, whose id is its own)."""
    if isinstance(node, dict):
        held = {key: _copied_arm(value, source, copy) for key, value in node.items()}
        if source in held:
            held[copy] = _renamed(_copy(held[source]), source, copy)
        return held
    if isinstance(node, list):
        rows = [_copied_arm(item, source, copy) for item in node]
        copied = [
            _renamed(_copy(row), source, copy)
            for row in rows
            if isinstance(row, dict) and row.get("arm") == source and "trial" not in row
        ]
        return rows + copied
    return node


def _copy(node: Any) -> Any:
    """A deep copy of a JSON document."""
    return json.loads(json.dumps(node))


def _renamed(node: Any, source: str, copy: str) -> Any:
    """A row's ``arm`` field renamed from one arm to another."""
    if isinstance(node, dict) and node.get("arm") == source:
        return {**node, "arm": copy}
    return node


def widened(held: ReportInput, *, arms: int, tasks: int) -> ReportInput:
    """A wide campaign built from an input: each candidate copied ``arms`` times and each task ``tasks`` times, every
    figure of the original carried to each copy, so a result's width grows with the copies and nothing else changes.

    :param held: The input to widen.
    :ptype held: ReportInput
    :param arms: How many copies of each candidate to add.
    :ptype arms: int
    :param tasks: How many copies of each task to add.
    :ptype tasks: int
    :return: The wide input.
    :rtype: ReportInput
    """
    document = held.model_dump(mode="json", by_alias=True)
    candidates = [arm["id"] for arm in document["arms"] if arm["role"] != "control"]
    for source in candidates:
        for number in range(1, arms + 1):
            copy = f"{source}-copy{number}"
            arm = next(row for row in document["arms"] if row["id"] == source)
            document = _copied_arm(document, source, copy)
            document["arms"].append({**_copy(arm), "id": copy, "short_name": None})
    document["tasks"] = [
        *document["tasks"],
        *(
            {**_copy(task), "id": f"{task['id']}-copy{number}", "name": f"{task.get('name') or task['id']} copy"}
            for number in range(1, tasks + 1)
            for task in document["tasks"]
        ),
    ]
    return ReportInput.model_validate(document)


@cache
def budget_inputs() -> Mapping[str, ReportInput]:
    """The inputs every analyst action's default page is held to the budget on, beyond the fixtures: the shape
    campaigns s1 (three aspects), s2 (a choice) and s3 (a measurement), built from their bundles, and a wide campaign
    (:data:`WIDE`): the reference input whose gates are read in each task, its every candidate copied and its tasks
    doubled, so its gates are read per task over many arms; and r1 and Kairo round 2 with each candidate copied
    three times, so a choice's readings, pairs, contrasts and strata run over many arms.

    :return: The inputs, by name.
    :rtype: Mapping[str, ReportInput]
    """
    built: dict[str, ReportInput] = {}
    for name in SHAPE_NAMES:
        built[name] = analyst_input(shape_bundle(name))
    built[WIDE] = widened(fixture_inputs()[_GATES_IN_EACH_TASK], arms=1, tasks=1)
    built[f"{WIDE} choice"] = widened(fixture_inputs()["r1"], arms=3, tasks=1)
    built[f"{WIDE} {KAIRO}"] = widened(ReportInput.model_validate(_kairo_snapshot("s").report_input), arms=3, tasks=0)
    built[WIDE_FRONTIER] = frontier_widened(fixture_inputs()["r2 without its gates"], arms=40)
    return built


#: The budget input whose price frontier is wide: every added arm at its own cost and pass rate, on the frontier.
WIDE_FRONTIER = f"{WIDE} frontier"


def frontier_widened(held: ReportInput, *, arms: int) -> ReportInput:
    """An input whose price frontier on ``pass_rate`` is wide: its first arm copied ``arms`` times, each copy at its
    own cost per 1,000 replies (rising by a sixth each) and its own pass rate (rising by less each time, toward a
    ceiling), so each copy is best at some price per point and nothing else changes.

    :param held: The input to widen; it holds ``pass_rate`` and ``cost_per_1k`` by arm, and no gate.
    :ptype held: ReportInput
    :param arms: How many copies to add.
    :ptype arms: int
    :return: The wide input.
    :rtype: ReportInput
    """
    document = held.model_dump(mode="json", by_alias=True)
    source = document["arms"][0]
    for number in range(arms):
        copy = f"{source['id']}-step{number}"
        document = _copied_arm(document, source["id"], copy)
        document["arms"].append({**_copy(source), "id": copy, "short_name": None})
        document["measures"]["cost_per_1k"]["per_arm"][copy]["mean"] = 0.01 * (7 / 6) ** number
        document["measures"]["pass_rate"]["per_arm"][copy]["mean"] = 99.5 - 9 * 0.9**number
    return ReportInput.model_validate(document)


def aspect_tagged(held: ReportInput) -> tuple[ReportInput, str, str]:
    """An input whose contrasts name their aspect, and whose lead aspect's contrast of one arm on one measure is
    untested while another aspect holds a tested contrast of that arm on the same measure key (its difference and
    interval the lead one's mirrored, so a reading of it is told apart from the arms' own means).

    The lead aspect is the first asked, else the first; every contrast naming no aspect is put on it. The contrast
    made untested is the first that states a difference and an interval.

    :param held: The input: two aspects or more, and a contrast stating a difference and an interval.
    :ptype held: ReportInput
    :return: The tagged input, and the arm and measure whose lead contrast is untested.
    :rtype: tuple[ReportInput, str, str]
    """
    asked = [aspect.id for aspect in held.aspects if aspect.asked]
    lead = asked[0] if asked else held.aspects[0].id
    other = next(aspect.id for aspect in held.aspects if aspect.id != lead)
    document = held.model_dump(mode="json", by_alias=True)
    contrasts = document["contrasts"]
    for contrast in contrasts:
        contrast["aspect"] = contrast["aspect"] or lead
    target = next(c for c in contrasts if c["delta"] is not None and c["interval"] is not None)
    low, high = target["interval"]
    mirrored = {**_copy(target), "aspect": other, "delta": -target["delta"], "interval": [-high, -low]}
    target.update(delta=None, interval=None, verdict="untested", reason="too few shared cases on this aspect")
    contrasts.append(mirrored)
    return ReportInput.model_validate(document), target["arm"], target["measure"]


def criterion_on_another_aspect(held: ReportInput) -> tuple[ReportInput, str, str]:
    """An input whose first criterion read overall on a mean is copied onto another aspect: the copy names the first
    aspect that is not the lead (the first asked, else the first), and that aspect's figures on the criterion's
    measure are the lead's mirrored across the threshold (each mean and interval end reflected in it), so a reader
    taking the lead's figure for the copy's reads the wrong side of the line.

    :param held: The input: two aspects or more, and a criterion read overall on a mean every arm states.
    :ptype held: ReportInput
    :return: The tagged input, the copy's id (``<criterion>@<aspect>``) and its aspect.
    :rtype: tuple[ReportInput, str, str]
    """
    asked = [aspect.id for aspect in held.aspects if aspect.asked]
    lead = asked[0] if asked else held.aspects[0].id
    other = next(aspect.id for aspect in held.aspects if aspect.id != lead)
    document = held.model_dump(mode="json", by_alias=True)
    criterion = next(
        criterion
        for criterion in document["criteria"]
        if criterion["scope"] == "overall" and criterion["statistic"] == "mean" and criterion.get("task") is None
    )
    table = document["measures"].get(criterion["measure"]) or document["judged"][criterion["measure"]]
    line = float(criterion["threshold"])
    mirrored: dict[str, Any] = {}
    for arm, stat in table["per_arm"].items():
        copy = _copy(stat)
        copy["mean"] = 2 * line - stat["mean"]
        if stat.get("ci") is not None:
            low, high = stat["ci"]
            copy["ci"] = [2 * line - high, 2 * line - low]
        mirrored[arm] = copy
    table["by_aspect"][other] = mirrored
    copied = f"{criterion['id']}@{other}"
    document["criteria"].append({**_copy(criterion), "id": copied, "aspect": other})
    return ReportInput.model_validate(document), copied, other


def action_calls(action: Action, held: ReportInput, snapshot_id: str) -> list[dict[str, Any]]:
    """The calls one analyst action is held to on one snapshot: its entry in :data:`ANALYST_CALLS`, else its example.

    :param action: The action.
    :ptype action: Action
    :param held: The snapshot's report input.
    :ptype held: ReportInput
    :param snapshot_id: The snapshot.
    :ptype snapshot_id: str
    :return: Each call's arguments, ``action`` and ``snapshot_id`` among them.
    :rtype: list[dict[str, Any]]
    """
    build = ANALYST_CALLS.get(action.name)
    argument_sets = build(held) if build is not None else [dict(action.example)]
    return [{**arguments, "action": action.name, "snapshot_id": snapshot_id} for arguments in argument_sets]


def read_actions() -> list[Action]:
    """Every analyst action that reads a snapshot: the ones held to the contract on every fixture."""
    return [action for action in analyst_actions() if action.permission == "read"]


def _block_calls(held: ReportInput) -> list[dict[str, Any]]:
    """For each source the input holds rows of, the first kind that draws every row of it; then the first of those
    paged from its second row, two rows to a page."""
    calls: list[dict[str, Any]] = []
    catalog = block_catalog(held, "")
    for source in (row.source for row in catalog.sources if row.holds_rows):
        for kind in (row.kind for row in catalog.kinds if source in row.sources):
            query = ChartQuery.model_validate({**_WHOLE, "source": source})
            try:
                block_render(held, block_kind=kind, query=query)
            except RefusedInput:
                continue
            calls.append({"block_kind": kind, "query": {"source": source}})
            break
    return calls + [{**calls[0], "offset": 1, "limit": 2}] if calls else []


#: A query of every row of its source.
_WHOLE: dict[str, Any] = {
    "aspect": "",
    "measures": [],
    "arms": [],
    "strata": [],
    "limit": 0,
    "transform": "none",
    "small_multiples_by": "none",
}

ANALYST_CALLS["block_render"] = _block_calls


def _price_calls(held: ReportInput) -> list[dict[str, Any]]:
    """On the first gain axis whose frontier has a step (else the first one priced): the trade, its options from the
    second two at a time, the trade priced against a frontier option other than the one it was priced against, within
    the first slice that prices it, and with the first option the gates remove priced as a what-if."""
    priced: list[tuple[str, PriceTradeoff]] = []
    for key in dict.fromkeys([*held.judged, *held.measures]):
        try:
            priced.append((key, price_tradeoff(held, key, limit=len(held.arms))))
        except RefusedInput:
            continue
    if not priced:
        return []
    key, trade = next(((key, trade) for key, trade in priced if trade.steps), priced[0])
    calls: list[dict[str, Any]] = [{"gain_key": key}, {"gain_key": key, "offset": 1, "limit": 2}]
    other = _first(row.arm.removeprefix("arm:") for row in trade.frontier if row.arm != trade.against_arm)
    calls += [{"gain_key": key, "arm_filter": other}] if other is not None else []
    slices = [task.id for task in held.tasks] + [stratum.id for stratum in held.strata]
    slices += [f"{cell.row}.{cell.column}" for cell in held.cells.values()]
    for slice_id in slices:
        try:
            within = price_tradeoff(held, key, slice_filter=slice_id, limit=len(held.arms))
        except RefusedInput:
            continue
        if within.frontier:
            calls.append({"gain_key": key, "slice_filter": slice_id})
            break
    gated = _first(row.arm.removeprefix("arm:") for row in trade.off_frontier if row.reason == "gated")
    return calls + ([{"gain_key": key, "what_if": gated}] if gated is not None else [])


ANALYST_CALLS["price_tradeoff"] = _price_calls


def compared_pair(held: ReportInput) -> tuple[str, str, str, ArmsComparison] | None:
    """The first two arms, in the input's arm order, and the first measure or judged dimension on which arms_compare
    reads them; None where no pair of arms has one.

    :param held: The input.
    :ptype held: ReportInput
    :return: The two arms, the measure, and the comparison; None for none.
    :rtype: tuple[str, str, str, ArmsComparison] | None
    """
    arms = [arm.id for arm in held.arms]
    for index, first in enumerate(arms):
        for second in arms[index + 1 :]:
            for key in dict.fromkeys([*held.measures, *held.judged]):
                try:
                    return first, second, key, arms_compare(held, (first, second), key, limit=200)
                except RefusedInput:
                    continue
    return None


def _compare_calls(held: ReportInput) -> list[dict[str, Any]]:
    """The first pair and measure arms_compare reads, then its places from the second, one to a page."""
    found = compared_pair(held)
    if found is None:
        return []
    first, second, key, comparison = found
    call: dict[str, Any] = {"arm_ids": [first, second], "measure_key": key}
    return [call] + ([{**call, "offset": 1, "limit": 1}] if comparison.total > 1 else [])


def _settled(held: ReportInput, arms: list[str], **named: Any) -> bool:
    try:
        settle_cases(held, arms, **named)
    except RefusedInput:
        return False
    return True


def _settle_calls(held: ReportInput) -> list[dict[str, Any]]:
    """The first arm settled against its criteria, against a hypothetical line a tenth below its first mean, and the
    first pair arms_compare reads settled apart; each where the input lets it answer."""
    calls: list[dict[str, Any]] = []
    first = held.arms[0].id
    if _settled(held, [first]):
        calls.append({"arm_ids": [first]})
    for key in dict.fromkeys([*held.judged, *held.measures]):
        table = held.measures.get(key) or held.judged[key]
        stat = table.per_arm.get(first)
        if stat is None or not stat.mean:
            continue
        line = round(float(stat.mean) * 0.9, 3)
        if _settled(held, [first], measure=key, line=line):
            calls.append({"arm_ids": [first], "measure_key": key, "hypothetical_line": line})
            break
    pair = compared_pair(held)
    if pair is not None and _settled(held, list(pair[:2]), measure=pair[2]):
        calls.append({"arm_ids": list(pair[:2]), "measure_key": pair[2]})
    return calls


ANALYST_CALLS["arms_compare"] = _compare_calls
ANALYST_CALLS["settle_cases"] = _settle_calls


def _levers_calls(held: ReportInput) -> list[dict[str, Any]]:
    """The first measure whose lever effects read overall, the same narrowed to its first setting, and the first
    measure and place whose effects read within a place; none where the arms isolate no setting."""
    places = [
        *(task.id for task in held.tasks),
        *(stratum.id for stratum in held.strata),
        *(f"{cell.row}.{cell.column}" for cell in held.cells.values()),
    ]
    calls: list[dict[str, Any]] = []
    for place in (None, *places):
        for key in dict.fromkeys([*held.measures, *held.judged]):
            try:
                read = lever_effects(held, key, place_filter=place, limit=1)
            except RefusedInput:
                continue
            call: dict[str, Any] = (
                {"measure_key": key} if place is None else {"measure_key": key, "place_filter": place}
            )
            setting = next(row.setting for row in read.settings if row.isolated)
            calls += [call, {**call, "setting_filter": setting}] if place is None else [call]
            break
        if calls and place is not None:
            break
    return calls


ANALYST_CALLS["levers_effects"] = _levers_calls


def _keys_calls(held: ReportInput) -> list[dict[str, Any]]:
    """Keys read off the input as a report would cite them: each arm, the candidates' count, the first measure's mean
    for the first arm holding it, written through a filter; a path naming nothing; then the same keys, a page of two
    from the second."""
    keys = [f"arm:{arm.id}" for arm in held.arms] + ["{n_candidates|int}"]
    measure = _first((key, stat) for key, table in held.measures.items() for stat in table.per_arm.items())
    if measure is not None:
        keys.append(f"measure.{measure[0]}.{measure[1][0]}.mean")
    keys.append("reading.no-such-arm.change|pts")
    return [{"keys": keys}, {"keys": keys, "offset": 1, "limit": 2}]


ANALYST_CALLS["figure_keys_check"] = _keys_calls
