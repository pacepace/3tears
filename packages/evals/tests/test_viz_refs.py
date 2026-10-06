"""The chart compiler — every payload drawn from the surface, every chart it cannot read refused.

Every chart here is built as the model authors it (the unified :class:`Chart`) and compiled the
way the generator compiles it: :func:`reference_from_chart`, then :func:`build_viz_payload`. Two
halves. The first builds each referenceable type over one multi-cell surface and asserts the
payload both PARSES under its type's contract and COMPILES to a chart, then checks the numbers it
carries are the ones :func:`resolve_reading` returns — so a builder that read a figure any other
way, or put one in the wrong slot, goes red. The second asserts one refusal per forbidden shape,
because a guard that has only ever seen valid input has proved it rejects nothing.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from threetears.evals.analysis import stats
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.errors import SoundnessRefusal, UnresolvableReference
from threetears.evals.analysis.references import resolve_reading
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.vega.compiler import compile_chart
from threetears.evals.analysis.viz.payloads import PAYLOAD_MODELS, parse_payload
from threetears.evals.analysis.viz_refs import (
    CHART_READINGS,
    REFERENCEABLE_VIZ_TYPES,
    TIME_VIZ_TYPES,
    build_viz_payload,
    dominated_flags,
    reference_from_chart,
)
from threetears.evals.contracts.analysis_measures import BarAdjudication, BarVerdict, MeasureCollection, MeasureSummary
from threetears.evals.contracts.authored import Chart, MeasureRef
from threetears.evals.contracts.campaign import VariantIndexEntry
from threetears.evals.contracts.host.measures import MeasureRegistry
from threetears.evals.contracts.host.values import IntervalScale, SweepableValue
from threetears.evals.contracts.identity import compute_variant_key
from threetears.evals.contracts.metrics import WITHHELD_NOT_CONTAINED, WITHHELD_PARTITION_INCOMPLETE
from threetears.evals.contracts.surface import (
    CellFacts,
    DecisionSurface,
    JudgedDimensionFacts,
    JudgedReading,
    MeasureFacts,
    TimeAxis,
    TimePosition,
)


RIG = "rig-one-000000000000"
RIG_B = "rig-two-000000000000"


def _levers(model: str, rounds: int) -> dict[str, SweepableValue]:
    return {
        "model": SweepableValue.of(model, display=model),
        "max_rounds": SweepableValue.of(rounds, scale=IntervalScale(value=rounds)),
    }


#: Three arms: A and B share a model and differ on rounds, C changes the model too.
ARMS = {"A": _levers("m1", 2), "B": _levers("m1", 4), "C": _levers("m2", 4)}
KEYS = {name: compute_variant_key(levers) for name, levers in ARMS.items()}
INDEX = [VariantIndexEntry(variant_key=KEYS[name], levers=levers) for name, levers in ARMS.items()]

#: Per arm: total/llm/tool latency (ms), cost (usd), pass rate, tokens, grounding (judged).
VALUES = {
    "A": {"total_ms": 1200.0, "llm_ms": 800.0, "tool_ms": 400.0, "cost_usd": 0.02, "pass_rate": 0.7, "tokens": 900.0},
    "B": {"total_ms": 1000.0, "llm_ms": 650.0, "tool_ms": 350.0, "cost_usd": 0.01, "pass_rate": 0.9, "tokens": 700.0},
    "C": {"total_ms": 1100.0, "llm_ms": 700.0, "tool_ms": 400.0, "cost_usd": 0.05, "pass_rate": 0.95, "tokens": 800.0},
}
GROUNDING = {"A": 3.5, "B": 4.0, "C": 4.5}
POLARITY = {"total_ms": False, "llm_ms": False, "tool_ms": False, "cost_usd": False, "pass_rate": True, "tokens": False}
SCOPE = {"llm_ms": "subsystem", "tool_ms": "subsystem"}
STOPS = {
    "A": {"end_turn": 5, "max_tokens": 2, "tool_use": 1},
    "B": {"end_turn": 8},
    "C": {"end_turn": 6, "max_tokens": 2},
}


def ref(arm: str, rig: str = RIG) -> str:
    return cell_ref(KEYS[arm], rig)


def _numeric(name: str, mean: float, n: int, n_independent: int) -> MeasureSummary:
    sem = abs(mean) * 0.05 if n >= 2 else None
    return MeasureSummary(
        population="scored",
        name=name,
        attribution_scope=SCOPE.get(name, "end_to_end"),
        higher_is_better=POLARITY[name],
        n=n,
        n_independent=n_independent,
        mean=mean,
        p50=mean,
        sem=sem,
        ci_low=mean - 2 * sem if sem is not None else None,
        ci_high=mean + 2 * sem if sem is not None else None,
    )


def _cell(
    arm: str,
    rig: str = RIG,
    *,
    n: int = 8,
    n_independent: int = 8,
    drop: tuple[str, ...] = (),
    scale: float = 1.0,
) -> CellFacts:
    measures = [
        _numeric(name, value * scale, n, n_independent) for name, value in VALUES[arm].items() if name not in drop
    ]
    measures.append(
        MeasureSummary(
            population="scored", name="stop_reason", attribution_scope="end_to_end", n=n, categories=STOPS[arm]
        )
    )
    return CellFacts(
        variant_key=KEYS[arm],
        apparatus_class_id=rig,
        run_ids=[f"run-{arm}-{rig[:7]}"],
        n_observations=n,
        measures=MeasureCollection(measures=sorted(measures, key=lambda m: m.name)),
        judged=[
            JudgedReading(
                dimension="reply.grounding",
                mean=GROUNDING[arm],
                sem=0.2,
                n=n,
                n_independent=n_independent,
                evidence_tier="undetermined",
            )
        ],
    )


def measure_facts(latency_axis: tuple[str, ...] = ("total_ms", "llm_ms", "tool_ms")) -> dict[str, MeasureFacts]:
    facts = {
        "total_ms": MeasureFacts(unit="ms", merit_axis=None, higher_is_better=False),
        "llm_ms": MeasureFacts(unit="ms", merit_axis=None, higher_is_better=False),
        "tool_ms": MeasureFacts(unit="ms", merit_axis=None, higher_is_better=False),
        "cost_usd": MeasureFacts(unit="usd", merit_axis="cost", higher_is_better=False),
        "pass_rate": MeasureFacts(unit=None, merit_axis="quality", higher_is_better=True),
        "tokens": MeasureFacts(unit="tokens", merit_axis=None, higher_is_better=False),
        "stop_reason": MeasureFacts(),
    }
    for name in latency_axis:
        facts[name] = facts[name].model_copy(update={"merit_axis": "latency"})
    return facts


def _bar(cells: list[CellFacts]) -> BarAdjudication:
    verdicts = []
    for cell in cells:
        value = next(m.mean for m in cell.measures.measures if m.name == "pass_rate")
        verdicts.append(
            BarVerdict(
                variant_key=cell.variant_key,
                apparatus_class_id=cell.apparatus_class_id,
                run_ids=cell.run_ids,
                value=value,
                n=8,
                n_independent=8,
                cleared=value >= 0.8,
            )
        )
    return BarAdjudication(
        measure_id="pass_rate",
        threshold=0.8,
        direction="higher_is_better",
        source="declared",
        state="adjudicated",
        verdicts=verdicts,
    )


#: The default surface's time axis: two builds, the second measuring every arm a tenth slower.
BUILDS = ("0.9", "0.10")


def time_axis() -> TimeAxis:
    """Every arm measured at two builds — the arms' own values, then the same values a tenth larger."""
    return TimeAxis(
        basis="release",
        release_label="app_version",
        positions=[
            TimePosition(
                key=build,
                first_run_at=f"2026-10-0{day}T00:00:00+00:00",
                last_run_at=f"2026-10-0{day}T01:00:00+00:00",
                run_ids=[f"run-{build}"],
                cells=sorted(
                    (_cell(arm, scale=scale) for arm in ("A", "B", "C")),
                    key=lambda c: (c.variant_key, c.apparatus_class_id),
                ),
            )
            for day, build, scale in ((1, BUILDS[0], 1.0), (2, BUILDS[1], 1.1))
        ],
    )


def surface(
    cells: list[CellFacts] | None = None,
    *,
    latency_axis: tuple[str, ...] = ("total_ms", "llm_ms", "tool_ms"),
    facts: dict[str, MeasureFacts] | None = None,
    timed: TimeAxis | None | bool = True,
) -> DecisionSurface:
    """Three cells under one rig unless told otherwise: a judged dimension, a categorical measure, a bar.

    ``timed`` gives it :func:`time_axis` by default, a given axis, or none (``None``/``False``).
    """
    cells = cells if cells is not None else [_cell(arm) for arm in ("A", "B", "C")]
    axis = time_axis() if timed is True else (timed or None)
    return DecisionSurface(
        control_variant_key=KEYS["A"],
        cells=sorted(cells, key=lambda c: (c.variant_key, c.apparatus_class_id)),
        bars=[_bar(cells)],
        measures=facts if facts is not None else measure_facts(latency_axis),
        dimensions={"reply.grounding": JudgedDimensionFacts(higher_is_better=True, value_range=(1.0, 5.0))},
        time_axis=axis,
    )


Measure = str | tuple[str, str]


def measures(*names: Measure) -> list[MeasureRef]:
    """Authored readings: a bare name is a measure, a ``(name, reading)`` pair states its namespace."""
    return [
        MeasureRef(measure_id=name, reading="measure")
        if isinstance(name, str)
        else MeasureRef(measure_id=name[0], reading=name[1])
        for name in names
    ]


def chart(
    kind: str, cells: list[str], names: list[Measure], *, axis: str = "", note: str = "", caption: str = ""
) -> Chart:
    """One authored chart, in the one shape the model writes for every type."""
    return Chart(type=kind, cells=cells, measures=measures(*names), axis=axis, note=note, caption=caption)


#: A host that declares no measures of its own: the charts here read the engine's core alone.
_NO_HOST_MEASURES = MeasureRegistry([])


def build(
    authored: Chart,
    on: DecisionSurface | None = None,
    index: list[VariantIndexEntry] | None = None,
    *,
    measures: MeasureRegistry = _NO_HOST_MEASURES,
) -> dict[str, Any]:
    """Compile ``authored`` the way the generator does: read it into its reference, then build the payload."""
    return build_viz_payload(
        reference_from_chart(authored), on or surface(), INDEX if index is None else index, measures=measures
    )


LABEL = {"A": "max_rounds=2, model=m1", "B": "max_rounds=4, model=m1", "C": "max_rounds=4, model=m2"}

#: One valid chart per referenceable type — the population the parse-and-compile gate walks.
VALID: dict[str, Chart] = {
    "delta_table": chart(
        "delta_table",
        [ref("A"), ref("B")],
        ["pass_rate", "total_ms", ("reply.grounding", "judged")],
        caption="More rounds lifted the pass rate.",
    ),
    "distribution": chart("distribution", [ref("A"), ref("B"), ref("C")], ["total_ms"]),
    "null_result": chart(
        "null_result",
        [ref("B"), ref("C")],
        ["total_ms"],
        note="The model swap never reaches the network wait that dominates the turn.",
    ),
    "breakdown": chart("breakdown", [ref("A")], ["stop_reason"]),
    "attribution": chart("attribution", [ref("A"), ref("B")], ["total_ms", "llm_ms"]),
    "frontier": chart("frontier", [], ["pass_rate", "cost_usd", "total_ms"]),
    "sweep_ranking": chart("sweep_ranking", [], ["pass_rate", "cost_usd"]),
    "timeseries": chart("timeseries", [], ["total_ms"]),
}


def valid(kind: str, **update: Any) -> Chart:
    """The valid chart of ``kind`` with some of its authored fields replaced."""
    if "measures" in update:
        update["measures"] = measures(*update["measures"])
    return VALID[kind].model_copy(update=update)


# --- Every type parses and compiles ----------------------------------------------------------------


def test_the_valid_population_is_the_whole_menu():
    # A type on the menu with no entry here would be silently unchecked by the gate below.
    assert set(VALID) == set(REFERENCEABLE_VIZ_TYPES) == set(CHART_READINGS)
    assert TIME_VIZ_TYPES <= REFERENCEABLE_VIZ_TYPES
    assert REFERENCEABLE_VIZ_TYPES <= set(PAYLOAD_MODELS)


@pytest.mark.parametrize("viz_type", sorted(REFERENCEABLE_VIZ_TYPES))
def test_every_referenceable_type_builds_a_payload_that_parses_and_compiles(viz_type: str):
    payload = build(VALID[viz_type])

    assert parse_payload(viz_type, payload) is not None
    compiled = compile_chart(viz_type, payload)
    assert compiled.spec


@pytest.mark.parametrize(
    "authored",
    [
        chart("breakdown", [ref("A")], ["llm_ms", "tool_ms"]),
        chart("attribution", [ref("A"), ref("B")], ["llm_ms", "tool_ms"]),
        chart("distribution", [ref("A"), ref("C")], [("reply.grounding", "judged")]),
        chart("frontier", [ref("A"), ref("B")], ["pass_rate", "cost_usd", "total_ms"]),
        chart("sweep_ranking", [ref("A"), ref("C")], ["pass_rate", "cost_usd"]),
    ],
    ids=[
        "numeric-breakdown",
        "attribution-of-two-components",
        "judged-distribution",
        "named-contestants",
        "named-configs",
    ],
)
def test_the_second_form_of_a_type_parses_and_compiles_too(authored: Chart):
    payload = build(authored)

    assert parse_payload(authored.type, payload) is not None
    assert compile_chart(authored.type, payload).spec


# --- Reading the authored chart by position ---------------------------------------------------------


def test_each_position_is_read_into_the_field_the_type_reads_it_as():
    """The unified chart's lists become the type's reference by position — baseline first, whole before part."""
    attribution = reference_from_chart(
        chart("attribution", [ref("A"), ref("B")], ["total_ms", "llm_ms"], axis="max_rounds")
    )
    assert (attribution.a_cell, attribution.b_cell) == (ref("A"), ref("B"))
    assert (attribution.end_to_end_measure, attribution.subsystem_measure, attribution.lever) == (
        "total_ms",
        "llm_ms",
        "max_rounds",
    )

    frontier = reference_from_chart(VALID["frontier"])
    assert (frontier.quality.measure_id, frontier.cost_measure_id, frontier.latency_measure_id) == (
        "pass_rate",
        "cost_usd",
        "total_ms",
    )
    assert frontier.cells is None, "an empty `cells` means every cell"

    null = reference_from_chart(VALID["null_result"])
    assert null.mechanism == VALID["null_result"].note


def test_an_empty_caption_and_axis_read_as_absent():
    reference = reference_from_chart(VALID["attribution"])
    assert (reference.caption, reference.lever) == (None, None)
    assert reference_from_chart(VALID["delta_table"]).caption == "More rounds lifted the pass rate."


def test_one_measure_reads_as_a_categorical_breakdown_and_several_as_numeric_parts():
    categorical = reference_from_chart(VALID["breakdown"])
    numeric = reference_from_chart(chart("breakdown", [ref("A")], ["llm_ms", "tool_ms"]))

    assert (categorical.measure_id, categorical.part_measure_ids) == ("stop_reason", None)
    assert (numeric.measure_id, numeric.part_measure_ids) == (None, ["llm_ms", "tool_ms"])


def test_a_type_off_the_menu_is_a_programming_error_not_a_refusal():
    """`validate_authored` refuses a type off the menu before this is reached; a caller that skipped it cannot be repaired."""
    with pytest.raises(ValueError, match=r"chart type 'scatter' is not one of") as raised:
        reference_from_chart(chart("scatter", [ref("A"), ref("B")], ["total_ms"]))
    assert not isinstance(raised.value, SoundnessRefusal)


# --- The numbers are the resolver's, in the right slots ---------------------------------------------


def test_a_delta_table_carries_the_resolved_means_and_no_statistic():
    s = surface()
    payload = build(VALID["delta_table"], s)

    assert (payload["a_label"], payload["b_label"]) == (LABEL["A"], LABEL["B"])
    assert payload["caption"] == "More rounds lifted the pass rate."
    rows = {row["metric"]: row for row in payload["rows"]}
    for name, reading in (("pass_rate", "measure"), ("total_ms", "measure"), ("reply.grounding", "judged")):
        a = resolve_reading(s, ref("A"), name, reading)
        b = resolve_reading(s, ref("B"), name, reading)
        assert (rows[name]["a"], rows[name]["b"]) == (a.mean, b.mean)
        assert rows[name]["delta"] == pytest.approx(b.mean - a.mean)
        assert rows[name]["unit"] == a.unit
        assert (rows[name]["d_z"], rows[name]["p"], rows[name]["significant"], rows[name]["paired"]) == (
            None,
            None,
            None,
            False,
        )
    assert rows["total_ms"]["unit"] == "ms"


def test_a_pair_is_counted_by_its_smaller_arm():
    cells = [_cell("A", n=6, n_independent=6), _cell("B"), _cell("C")]
    s = surface(cells)

    delta = build(VALID["delta_table"], s)
    assert {row["n"] for row in delta["rows"]} == {6}
    attribution = build(VALID["attribution"], s)
    assert (attribution["end_to_end"]["n"], attribution["subsystem"]["n"]) == (6, 6)


def test_a_distribution_draws_each_cells_interval_at_the_one_level():
    s = surface()
    payload = build(VALID["distribution"], s)

    assert payload["unit"] == "ms"
    for group, arm in zip(payload["groups"], ("A", "B", "C"), strict=True):
        reading = resolve_reading(s, ref(arm), "total_ms")
        assert group["label"] == LABEL[arm]
        assert group["n"] == reading.n
        assert (group["ci"]["low"], group["ci"]["high"], group["ci"]["mean"]) == (
            reading.ci_low,
            reading.ci_high,
            reading.mean,
        )
        assert group["ci"]["level"] == INTERVAL_LEVEL
    # One variability wording for every unclustered cell, so the compiler reads the widths as comparable.
    assert len({group["ci"]["variability"] for group in payload["groups"]}) == 1


def test_a_drawn_interval_states_the_level_its_width_was_computed_at(monkeypatch):
    """The caption's level is read from the one constant the width function uses, never a copy.

    A judged interval's width is computed at draw time, so moving the constant must move the width
    and the stated level together — a label left at 95% over an 80% width is the mislabelled chart.
    """
    monkeypatch.setattr(stats, "INTERVAL_LEVEL", 0.8)
    s = surface()
    payload = build(valid("distribution", measures=[("reply.grounding", "judged")]), s)

    for group, arm in zip(payload["groups"], ("A", "B", "C"), strict=True):
        reading = resolve_reading(s, ref(arm), "reply.grounding", "judged")
        half = stats.t_critical_two_sided(0.8, reading.n - 1) * reading.sem
        assert group["ci"]["level"] == 0.8
        assert (group["ci"]["low"], group["ci"]["high"]) == (
            pytest.approx(reading.mean - half),
            pytest.approx(reading.mean + half),
        )


def test_clustered_observations_are_named_in_the_variability():
    cells = [_cell("A", n_independent=2), _cell("B"), _cell("C")]
    payload = build(VALID["distribution"], surface(cells))

    by_label = {group["label"]: group["ci"]["variability"] for group in payload["groups"]}
    assert "narrower than the clustering supports" in by_label[LABEL["A"]]
    assert "clustering" not in by_label[LABEL["B"]]


def test_a_null_result_carries_two_intervals_and_the_authored_mechanism():
    payload = build(VALID["null_result"])

    assert [group["label"] for group in payload["groups"]] == [LABEL["B"], LABEL["C"]]
    assert all(group["ci"]["level"] == INTERVAL_LEVEL for group in payload["groups"])
    assert (payload["metric"], payload["unit"]) == ("total_ms", "ms")
    assert payload["mechanism"].startswith("The model swap")


def test_a_categorical_breakdown_counts_its_categories():
    payload = build(VALID["breakdown"])

    assert {part["label"]: part["value"] for part in payload["parts"]} == {
        "end_turn": 5.0,
        "max_tokens": 2.0,
        "tool_use": 1.0,
    }
    assert (payload["unit"], payload["measure"], payload["total"], payload["total_n"]) == (
        "observations",
        "stop_reason",
        8.0,
        8,
    )


def test_a_numeric_breakdown_reads_each_part_and_names_no_whole():
    payload = build(chart("breakdown", [ref("A")], ["llm_ms", "tool_ms"]))

    assert {part["label"]: part["value"] for part in payload["parts"]} == {"llm_ms": 800.0, "tool_ms": 400.0}
    assert payload["unit"] == "ms"
    assert "total" not in payload and "measure" not in payload


def test_an_attribution_withholds_the_remainder_of_one_component_of_a_partition():
    """The pair the prompt steers the model to chart — a whole and the component that carried it.

    `llm_ms` IS declared inside `total_ms`, and that is not enough: `total_ms` partitions into three
    components, so the leftover is the other two's movement, attributed to measures the registry
    names. Stating it as "unattributed" is the conclusion the divergence lens refuses to draw, and a
    compiled payload that states it anyway is one the lens and the chart disagree on. This test once
    asserted that remainder as the honest case.
    """
    partial = build(VALID["attribution"])
    assert partial["contained_by"] == "total_ms"
    assert partial["end_to_end"]["delta"] == pytest.approx(1000.0 - 1200.0)
    assert partial["subsystem"]["delta"] == pytest.approx(650.0 - 800.0)
    assert "unattributed_delta" not in partial
    assert WITHHELD_PARTITION_INCOMPLETE in partial["unattributed_withheld"]
    assert (partial["unit"], partial["a_label"], partial["b_label"]) == ("ms", LABEL["A"], LABEL["B"])
    compile_chart("attribution", partial)


def test_an_attribution_withholds_the_remainder_of_a_measure_not_declared_inside_the_whole():
    # tool_ms is a component of total_ms, not of llm_ms — same unit, not nested, so no remainder.
    withheld = build(chart("attribution", [ref("A"), ref("B")], ["llm_ms", "tool_ms"]))
    assert "unattributed_delta" not in withheld
    assert WITHHELD_NOT_CONTAINED in withheld["unattributed_withheld"]
    compile_chart("attribution", withheld)


def test_an_attribution_states_the_remainder_of_a_sole_component_a_host_declares():
    """The branch the two refusals above cannot reach: a part that EXHAUSTS its whole.

    No core measure is a whole with exactly one component (`total_ms` has three), so the stated
    remainder is only reachable through a host catalogue — which is where the compiler resolves
    descriptors from, and where a second product's `whole`/`part` pair would live. Without this
    the compiler could withhold every remainder and the rule's positive half would stay green in
    `test_metrics.py` while never being honoured on a chart.
    """
    from dataclasses import replace

    from threetears.evals.contracts.metrics import MetricDescriptor
    from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile

    def descriptor(name: str, scope: str, contained_by: str | None = None) -> MetricDescriptor:
        return MetricDescriptor(
            name=name,
            data_type="numeric",
            family="mechanical",
            transferability_class="mechanical",
            attribution_scope=scope,
            higher_is_better=False,
            unit="ms",
            contained_by=contained_by,
            description=f"Synthetic {name}: a whole with one declared component.",
        )

    def reading(name: str, scope: str, mean: float) -> MeasureSummary:
        return MeasureSummary(
            population="scored",
            name=name,
            attribution_scope=scope,
            higher_is_better=False,
            n=8,
            n_independent=8,
            mean=mean,
        )

    means = {"A": (2000.0, 1500.0), "B": (1600.0, 1300.0)}
    cells = [
        _cell(arm).model_copy(
            update={
                "measures": MeasureCollection(
                    measures=sorted(
                        [
                            *_cell(arm).measures.measures,
                            reading("host_part_ms", "subsystem", part),
                            reading("host_whole_ms", "end_to_end", whole),
                        ],
                        key=lambda m: m.name,
                    )
                )
            }
        )
        for arm, (whole, part) in means.items()
    ]
    facts = measure_facts() | {
        name: MeasureFacts(unit="ms", merit_axis=None, higher_is_better=False)
        for name in ("host_whole_ms", "host_part_ms")
    }
    host = replace(
        toyhost_profile(),
        measures=MeasureRegistry(
            [
                *TOYHOST_MEASURES,
                descriptor("host_whole_ms", "end_to_end"),
                descriptor("host_part_ms", "subsystem", contained_by="host_whole_ms"),
            ],
            families=toyhost_profile().measures.families,
        ),
    )
    stated = build(
        chart("attribution", [ref("A"), ref("B")], ["host_whole_ms", "host_part_ms"]),
        surface(cells, facts=facts),
        measures=host.measures,
    )

    assert stated["contained_by"] == "host_whole_ms"
    assert "unattributed_withheld" not in stated
    # (1600 - 2000) - (1300 - 1500): the whole fell 400, the part 200, so 200 is unexplained.
    assert stated["unattributed_delta"] == pytest.approx(-200.0)
    compile_chart("attribution", stated)


def test_a_frontier_places_every_cell_and_computes_domination_in_code():
    payload = build(VALID["frontier"])

    points = {point["label"]: point for point in payload["points"]}
    assert set(points) == set(LABEL.values())
    # B is at least as good as A on both axes and strictly better on both: A is dominated.
    assert points[LABEL["A"]]["dominated"] is True
    assert points[LABEL["B"]]["dominated"] is False
    assert points[LABEL["C"]]["dominated"] is False
    assert points[LABEL["A"]]["latency_ms"] == 1200.0
    assert points[LABEL["C"]]["cost"] == 0.05
    assert all(point["disqualified"] is False for point in payload["points"])
    assert payload["bar"] == 0.8


def test_a_frontiers_axis_titles_are_computed_from_its_measures():
    """The authored chart carries no label, so each title is the measure's name and, when it has one, its unit."""
    payload = build(VALID["frontier"])
    assert (payload["cost_label"], payload["quality_label"]) == ("cost_usd (usd)", "pass_rate")

    tokens = build(chart("frontier", [], ["pass_rate", "tokens"]), surface(latency_axis=()))
    assert tokens["cost_label"] == "tokens (tokens)"


def test_a_frontier_defaults_cost_and_latency_when_one_measure_is_on_each_axis():
    payload = build(chart("frontier", [], ["pass_rate"]), surface(latency_axis=("total_ms",)))

    assert {point["latency_ms"] for point in payload["points"]} == {1200.0, 1000.0, 1100.0}
    assert {point["cost"] for point in payload["points"]} == {0.02, 0.01, 0.05}


def test_a_frontier_with_no_latency_measure_draws_none():
    payload = build(chart("frontier", [], ["pass_rate"]), surface(latency_axis=()))

    assert all(point["latency_ms"] is None for point in payload["points"])


def test_a_cell_that_never_priced_is_unplaced_not_free():
    cells = [_cell("A"), _cell("B"), _cell("C", drop=("cost_usd",))]
    payload = build(VALID["frontier"], surface(cells))

    c = next(point for point in payload["points"] if point["label"] == LABEL["C"])
    assert c["cost"] is None
    assert c["dominated"] is False
    assert parse_payload("frontier", payload) is not None


def test_a_judged_quality_draws_no_bar():
    payload = build(valid("frontier", measures=[("reply.grounding", "judged"), "cost_usd", "total_ms"]))

    assert payload["bar"] is None
    assert payload["quality_label"] == "reply.grounding"


def test_two_thresholds_on_the_quality_measure_draw_no_bar():
    s = surface()
    s.bars.append(s.bars[0].model_copy(update={"threshold": 0.9}))

    assert build(VALID["frontier"], s)["bar"] is None


def test_a_sweep_ranks_descending_and_reads_its_configuration_off_the_variant_index():
    payload = build(VALID["sweep_ranking"])

    assert [row["ranked_value"] for row in payload["rows"]] == [0.95, 0.9, 0.7]
    assert [row["config"] for row in payload["rows"]] == [
        {"max_rounds": "4", "model": "m2"},
        {"max_rounds": "4", "model": "m1"},
        {"max_rounds": "2", "model": "m1"},
    ]
    assert [row["secondary_value"] for row in payload["rows"]] == [0.05, 0.01, 0.02]
    # Declared from each level's scale: an interval lever orders, a nominal one does not.
    assert payload["dimensions"] == [{"name": "max_rounds", "ordered": True}, {"name": "model", "ordered": False}]
    assert (payload["ranked"], payload["secondary"]) == (
        {"measure": "pass_rate", "unit": None},
        {"measure": "cost_usd", "unit": "usd"},
    )
    # Every configuration is drawn, unconstrained: the authored chart names no limit and no held value.
    assert "omitted" not in payload and "held_fixed" not in payload


def test_a_lever_every_row_holds_alike_is_not_a_column():
    payload = build(valid("sweep_ranking", cells=[ref("A"), ref("B")]))

    assert [row["config"] for row in payload["rows"]] == [{"max_rounds": "4"}, {"max_rounds": "2"}]


def test_one_arm_under_two_rigs_is_told_apart_by_its_rig():
    cells = [_cell("A"), _cell("A", RIG_B), _cell("B")]
    s = surface(cells)

    dist = build(chart("distribution", [ref("A"), ref("A", RIG_B), ref("B")], ["total_ms"]), s)
    assert [group["label"] for group in dist["groups"]] == [
        f"{LABEL['A']} @ rig {RIG[:12]}",
        f"{LABEL['A']} @ rig {RIG_B[:12]}",
        LABEL["B"],
    ]
    sweep = build(VALID["sweep_ranking"], s)
    assert {row["config"]["rig"] for row in sweep["rows"]} == {RIG[:12], RIG_B[:12]}
    assert {"name": "rig", "ordered": False} in sweep["dimensions"]
    assert parse_payload("sweep_ranking", sweep) is not None


def test_an_arm_the_index_cannot_describe_is_labelled_as_such():
    stranger = CellFacts(
        variant_key="f" * 64,
        apparatus_class_id=RIG,
        run_ids=["run-x"],
        n_observations=8,
        measures=_cell("A").measures,
    )
    unavailable = VariantIndexEntry(variant_key="e" * 64, levers={}, levels_unavailable="minted under predicate v3")
    orphan = stranger.model_copy(update={"variant_key": "e" * 64})
    s = surface([_cell("A"), stranger, orphan])

    payload = build(
        chart("distribution", [ref("A"), cell_ref("f" * 64, RIG), cell_ref("e" * 64, RIG)], ["total_ms"]),
        s,
        [*INDEX, unavailable],
    )
    labels = [group["label"] for group in payload["groups"]]
    assert labels[1] == f"unplaced ({'f' * 12}) — not in this analysis's variant index"
    assert labels[2] == f"levels unavailable ({'e' * 12}) — minted under predicate v3"


# --- Domination ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([(0.9, 1.0), (0.9, 1.0)], [False, False]),  # identical: neither strictly better on either axis
        ([(0.9, 1.0), (0.9, 2.0)], [False, True]),  # equal quality, cheaper wins
        ([(0.9, 1.0), (0.8, 1.0)], [False, True]),  # equal cost, better quality wins
        ([(0.9, 2.0), (0.8, 1.0)], [False, False]),  # a real trade-off: each better on one axis
        ([(0.9, None), (0.5, 9.0)], [False, False]),  # an unpriced point neither dominates nor is dominated
        ([(0.5, None), (0.9, 1.0)], [False, False]),
    ],
)
def test_domination_needs_at_least_as_good_on_both_and_strictly_better_on_one(points, expected):
    assert dominated_flags(points) == expected


# --- Refusals of the authored chart ------------------------------------------------------------------

#: Every shape `reference_from_chart` refuses, with what its refusal names about the chart. One
#: entry per forbidden shape, because a guard that has only ever seen valid input rejects nothing.
CHART_REFUSALS = [
    pytest.param(
        chart("delta_table", [ref("A")], ["pass_rate"]), r"got 1 cell\(s\) and 1 measure\(s\)", id="delta-one-cell"
    ),
    pytest.param(
        chart("delta_table", [ref("A"), ref("B")], []), r"got 2 cell\(s\) and 0 measure\(s\)", id="delta-no-measure"
    ),
    pytest.param(
        chart("delta_table", [ref("A"), ref("A")], ["pass_rate"]),
        r"`cells` names .* more than once",
        id="delta-same-cell",
    ),
    pytest.param(
        chart("delta_table", [ref("A"), ref("B")], ["pass_rate", "pass_rate"]),
        r"`measures` names pass_rate more than once",
        id="delta-same-measure",
    ),
    pytest.param(chart("distribution", [ref("A")], ["total_ms"]), r"got 1 cell\(s\)", id="distribution-one-cell"),
    pytest.param(
        chart("distribution", [ref("A"), ref("A")], ["total_ms"]),
        r"`cells` names .* more than once",
        id="distribution-same-cell",
    ),
    pytest.param(
        chart("distribution", [ref("A"), ref("B")], ["total_ms", "llm_ms"]),
        r"and 2 measure\(s\)",
        id="distribution-two-measures",
    ),
    pytest.param(
        chart("null_result", [ref("C"), ref("C")], ["total_ms"], note="why"),
        r"`cells` names .* more than once",
        id="null-same-cell",
    ),
    pytest.param(chart("null_result", [ref("B"), ref("C")], ["total_ms"]), r"`note` is empty", id="null-no-mechanism"),
    pytest.param(
        chart("null_result", [ref("B"), ref("C")], ["total_ms"], note="   "),
        r"`note` is empty",
        id="null-blank-mechanism",
    ),
    pytest.param(chart("breakdown", [ref("A")], []), r"got 1 cell\(s\) and 0 measure\(s\)", id="breakdown-no-measure"),
    pytest.param(
        chart("breakdown", [ref("A"), ref("B")], ["stop_reason"]), r"got 2 cell\(s\)", id="breakdown-two-cells"
    ),
    pytest.param(
        chart("breakdown", [ref("A")], ["llm_ms", "llm_ms"]),
        r"`measures` names llm_ms more than once",
        id="breakdown-same-part",
    ),
    pytest.param(
        chart("breakdown", [ref("A")], [("reply.grounding", "judged")]),
        r"not judged dimensions.*grounding",
        id="breakdown-judged",
    ),
    pytest.param(
        chart("attribution", [ref("A"), ref("A")], ["total_ms", "llm_ms"]),
        r"`cells` names .* more than once",
        id="attribution-same-cell",
    ),
    pytest.param(
        chart("attribution", [ref("A"), ref("B")], ["total_ms", "total_ms"]),
        r"end-to-end and subsystem measures are both 'total_ms' — a measure cannot diverge from itself",
        id="attribution-same-measure",
    ),
    pytest.param(
        chart("attribution", [ref("A"), ref("B")], ["total_ms"]), r"and 1 measure\(s\)", id="attribution-one-measure"
    ),
    pytest.param(
        chart("attribution", [ref("A"), ref("B")], ["total_ms", ("reply.grounding", "judged")]),
        r"not judged dimensions.*grounding",
        id="attribution-judged",
    ),
    pytest.param(
        chart("frontier", [ref("A")], ["pass_rate"]), r"got 1 cell — name two or more, or none", id="frontier-one-cell"
    ),
    pytest.param(chart("frontier", [], []), r"got 0 cell\(s\) and 0 measure\(s\)", id="frontier-no-quality"),
    pytest.param(
        chart("frontier", [], ["pass_rate", ("reply.grounding", "judged")]),
        r"not judged dimensions.*grounding",
        id="frontier-judged-cost",
    ),
    pytest.param(
        chart("sweep_ranking", [ref("A")], ["pass_rate", "cost_usd"]),
        r"got 1 cell — name two or more, or none",
        id="sweep-one-cell",
    ),
    pytest.param(chart("sweep_ranking", [], ["pass_rate"]), r"and 1 measure\(s\)", id="sweep-one-measure"),
]

#: The per-type reference's own field names — words the model never wrote, so no refusal may use them.
_REFERENCE_VOCABULARY = (
    "a_cell",
    "b_cell",
    "part_measure_ids",
    "end_to_end_measure",
    "subsystem_measure",
    "mechanism",
    "cost_measure_id",
    "latency_measure_id",
    "it carries",
)


@pytest.mark.parametrize(("authored", "match"), CHART_REFUSALS)
def test_a_chart_that_cannot_be_read_is_refused_in_the_words_of_the_chart_the_model_wrote(authored: Chart, match: str):
    """The refusal the repair round feeds back quotes the type's reading and names what the chart held.

    It never describes the typed reference it was being read into: the model wrote only the unified
    chart, so a refusal naming ``a_cell`` or ``part_measure_ids`` would ask it to fix a shape it has
    never seen.
    """
    with pytest.raises(UnresolvableReference) as raised:
        reference_from_chart(authored)

    message = str(raised.value)
    reading = f"a {authored.type} chart reads {CHART_READINGS[authored.type]}; "
    assert message.startswith(reading), message
    held = message.removeprefix(reading)
    assert re.search(match, held), message
    assert not [word for word in _REFERENCE_VOCABULARY if word in held], message


def test_every_refusal_here_is_repairable():
    assert issubclass(UnresolvableReference, SoundnessRefusal)


# --- Refusals against the surface ---------------------------------------------------------------------


def test_an_unknown_cell_names_the_cells_that_exist():
    with pytest.raises(UnresolvableReference, match=rf"nope:rig.*{KEYS['A']}"):
        build(chart("distribution", [ref("A"), "nope:rig"], ["total_ms"]))


def test_an_empty_cell_names_the_cells_that_exist():
    with pytest.raises(UnresolvableReference, match=rf"names cell ''.*{KEYS['A']}"):
        build(chart("null_result", ["", ref("B")], ["total_ms"], note="why"))


def test_an_unknown_cell_in_a_default_free_reference_is_refused_before_any_reading():
    with pytest.raises(UnresolvableReference, match=r"does not hold"):
        build(chart("breakdown", ["nope:rig"], ["stop_reason"]))


def test_a_judged_dimension_read_as_a_measure_says_which_namespace_it_is_in():
    with pytest.raises(UnresolvableReference, match=r"set reading to 'judged'"):
        build(chart("distribution", [ref("A"), ref("B")], ["reply.grounding"]))


def test_an_ambiguous_latency_default_names_the_candidates():
    with pytest.raises(
        UnresolvableReference, match=r"several measures on the latency axis \(llm_ms, tool_ms, total_ms\)"
    ) as raised:
        build(chart("frontier", [], ["pass_rate"]))
    assert "name one as the chart's third measure" in str(raised.value)


def test_an_ambiguous_cost_default_names_the_candidates():
    facts = measure_facts(("total_ms",))
    facts["tokens"] = facts["tokens"].model_copy(update={"merit_axis": "cost"})

    with pytest.raises(
        UnresolvableReference, match=r"cost axis \(cost_usd, tokens\) — name one as the chart's second measure"
    ):
        build(chart("frontier", [], ["pass_rate"]), surface(facts=facts))


def test_a_frontier_with_no_cost_measure_asks_for_one():
    facts = measure_facts(("total_ms",))
    facts["cost_usd"] = facts["cost_usd"].model_copy(update={"merit_axis": None})

    with pytest.raises(UnresolvableReference, match=r"no measure on the cost axis"):
        build(chart("frontier", [], ["pass_rate"]), surface(facts=facts))


def test_a_named_cost_measure_overrides_an_ambiguous_default():
    facts = measure_facts(("total_ms",))
    facts["tokens"] = facts["tokens"].model_copy(update={"merit_axis": "cost"})

    payload = build(chart("frontier", [], ["pass_rate", "tokens"]), surface(facts=facts))
    assert {point["cost"] for point in payload["points"]} == {900.0, 700.0, 800.0}


def test_a_lower_is_better_ranked_reading_is_refused():
    with pytest.raises(
        UnresolvableReference, match=r"ranked reading must be higher-is-better.*'total_ms' is lower-is-better"
    ):
        build(chart("sweep_ranking", [], ["total_ms", "cost_usd"]))


def test_a_lower_is_better_frontier_quality_is_refused():
    with pytest.raises(UnresolvableReference, match=r"quality reading must be higher-is-better"):
        build(valid("frontier", measures=["tokens", "cost_usd", "total_ms"]))


def test_a_higher_is_better_cost_is_refused():
    with pytest.raises(UnresolvableReference, match=r"lower is better, but it is higher-is-better"):
        build(valid("frontier", measures=["pass_rate", "pass_rate", "total_ms"]))


def test_a_latency_not_in_ms_is_refused():
    with pytest.raises(UnresolvableReference, match=r"states latency in ms, and 'tokens' is in tokens"):
        build(valid("frontier", measures=["pass_rate", "cost_usd", "tokens"]))


def test_a_numeric_measure_named_as_categorical_points_at_the_other_form():
    with pytest.raises(
        UnresolvableReference,
        match=r"it is numeric — break numeric measures down by naming two or more of them as the parts",
    ):
        build(chart("breakdown", [ref("A")], ["total_ms"]))


def test_an_unknown_categorical_measure_names_the_categorical_ones():
    with pytest.raises(UnresolvableReference, match=r"categorical measures are: stop_reason"):
        build(chart("breakdown", [ref("A")], ["outcome"]))


def test_a_single_category_compiles_to_the_one_part_chart_the_drop_removes():
    """Which categories a cell carries is the data's answer, so it is left to the drop, not refused.

    The payload is built as the data has it; it fails the presentation rule, which is what the
    generator's undrawable-chart drop catches — keeping the finding instead of paying for a repair.
    """
    from threetears.evals.analysis.viz.payloads import PayloadError
    from threetears.evals.vega.spec_policy import SpecPolicyError

    payload = build(chart("breakdown", [ref("B")], ["stop_reason"]))
    assert [part["label"] for part in payload["parts"]] == ["end_turn"]
    with pytest.raises((PayloadError, SpecPolicyError)):
        compile_chart("breakdown", payload)


def test_a_categorical_measure_cannot_be_a_numeric_part():
    with pytest.raises(UnresolvableReference, match=r"categorical"):
        build(chart("breakdown", [ref("A")], ["llm_ms", "stop_reason"]))


@pytest.mark.parametrize(
    "authored",
    [
        chart("breakdown", [ref("A")], ["llm_ms", "cost_usd"]),
        chart("attribution", [ref("A"), ref("B")], ["total_ms", "cost_usd"]),
    ],
    ids=["breakdown", "attribution"],
)
def test_mixed_units_are_refused_where_one_unit_is_required(authored: Chart):
    with pytest.raises(UnresolvableReference, match=r"in several \("):
        build(authored)


def test_a_unitless_breakdown_is_refused():
    facts = measure_facts()
    facts["llm_ms"] = facts["llm_ms"].model_copy(update={"unit": None})
    facts["tool_ms"] = facts["tool_ms"].model_copy(update={"unit": None})

    with pytest.raises(UnresolvableReference, match=r"needs a unit and llm_ms, tool_ms carries none"):
        build(chart("breakdown", [ref("A")], ["llm_ms", "tool_ms"]), surface(facts=facts))


def test_a_default_cell_set_on_a_one_cell_surface_is_refused():
    with pytest.raises(UnresolvableReference, match=r"compares at least 2 cells and the decision surface holds 1"):
        build(VALID["frontier"], surface([_cell("A")]))


def test_a_reading_with_no_interval_cannot_be_drawn_as_one():
    cells = [_cell("A", n=1, n_independent=1), _cell("B"), _cell("C")]

    with pytest.raises(UnresolvableReference, match=r"has no interval .* at least 2 observations"):
        build(valid("null_result", cells=[ref("A"), ref("C")]), surface(cells))


def test_a_sweep_cell_whose_levels_are_unknown_is_refused():
    stranger = _cell("A").model_copy(update={"variant_key": "f" * 64})
    s = surface([_cell("A"), _cell("B"), stranger])

    with pytest.raises(UnresolvableReference, match=r"cannot say what its arm ran"):
        build(VALID["sweep_ranking"], s)


def test_a_sweep_cell_the_index_cannot_describe_is_refused():
    unavailable = VariantIndexEntry(variant_key="e" * 64, levers={}, levels_unavailable="minted under predicate v3")
    orphan = _cell("A").model_copy(update={"variant_key": "e" * 64})
    s = surface([_cell("A"), _cell("B"), orphan])

    with pytest.raises(UnresolvableReference, match=r"cannot say what its arm ran"):
        build(VALID["sweep_ranking"], s, [*INDEX, unavailable])


def test_a_sweep_over_cells_at_identical_levels_is_refused():
    # Two stacks that differ in content and render alike: nothing a barcode could draw tells them apart.
    twins = {
        name: {"model": SweepableValue.of(content, display="m1")}
        for name, content in (("X", "m1-build-1"), ("Y", "m1-build-2"))
    }
    keys = {name: compute_variant_key(levers) for name, levers in twins.items()}
    cells = [_cell("A").model_copy(update={"variant_key": keys[name]}) for name in twins]
    index = [VariantIndexEntry(variant_key=keys[name], levers=levers) for name, levers in twins.items()]

    with pytest.raises(UnresolvableReference, match=r"run at identical levels"):
        build(VALID["sweep_ranking"], surface(cells), index)


def test_one_arm_under_two_rigs_ranks_on_its_rig():
    s = surface([_cell("A"), _cell("A", RIG_B), _cell("B")])
    s.cells = [cell for cell in s.cells if cell.variant_key == KEYS["A"]]

    assert build(VALID["sweep_ranking"], s)["dimensions"] == [{"name": "rig", "ordered": False}]


def test_a_measure_absent_at_a_named_cell_is_refused_naming_what_it_holds():
    cells = [_cell("A"), _cell("B", drop=("tokens",)), _cell("C")]

    with pytest.raises(UnresolvableReference, match=r"measured no such measure"):
        build(valid("delta_table", measures=["tokens"]), surface(cells))
