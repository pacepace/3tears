"""The ``timeseries`` chart: a time axis the bundle earns, a payload that draws it, and a gate on its order.

Five parts, each pinned in both directions:

- **The bundle's time axis** (``analysis/bundle.py``). A campaign spanning two builds the host labels, or
  failing that two days, carries one; each position's cells are the decision surface's own algebra over that
  position's runs, so a position reads exactly as a campaign of only its runs would. Positions are ordered
  by when each first ran — never by name, which would put ``0.10`` before ``0.9``.
- **The host's release label** (``kernel/host/profile.py``). It must name a registered ``label``.
- **The builder** (``analysis/viz_refs.py``). A chart over a surface with no time axis is refused; every
  point is the resolver's reading over its position; a position a cell lacks is a stated gap.
- **The payload and the intent** (``analysis/viz/payloads.py``, ``analysis/viz/intents/timeseries.py``).
  Every forbidden shape refused; the values-as-drawn table is the data. How the Vega-Lite renderer draws
  it, and its spec gate's order rule, are ``test_vega_timeseries.py``'s.
- **The generator** (``analysis/generator.py``). The menu offers ``timeseries`` only over a bundle with a
  time axis, so a bundle without one cannot reference the chart — and the policy rule refuses a line
  through categories whose order is unstated.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import uuid
from typing import Any

import pytest

from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis.bundle.surface import cell_dimension_facts, cell_measure_facts
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.errors import SoundnessRefusal, UnresolvableReference
from threetears.evals.analysis.generator import first_request, generate_analysis, prompt_content_version
from threetears.evals.analysis.references import cell_aliases, resolve_reading
from threetears.evals.analysis.viz import chart_intent
from threetears.evals.analysis.viz.payloads import PayloadError, parse_payload
from threetears.evals.analysis.viz_refs import TimeseriesRef, build_viz_payload, reference_from_chart
from threetears.evals.kernel.authored import NO_CHART, Chart, MeasureRef
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.kernel.host.profile import HostProfile, ProfileRegistrationError
from threetears.evals.schema.models import EvalResult, EvalRun, utc_now_iso
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.kernel.surface import DecisionSurface, MeasureFacts, TimeAxis, TimePosition
from packages.evals.tests.fixtures.toyhost.campaign import (
    TOYHOST_NARROW,
    TOYHOST_WIDE,
    toyhost_bundle,
    toyhost_campaign,
)
from packages.evals.tests.fixtures.toyhost.corpus import (
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.chart_examples import timeseries_ci as _ci
from packages.evals.tests.chart_examples import timeseries_payload as _payload
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, memo_payload

_NAMESPACE = uuid.UUID("0b8e6a52-41d7-4f0e-9c3a-7d25e1f4a9b6")

#: Two days, a day apart.
DAY_ONE, DAY_TWO = "2026-03-14T09:30:00+00:00", "2026-03-15T09:30:00+00:00"

#: The base wall-clock per arm, before each position's drift.
_TOTAL_MS = {TOYHOST_NARROW: 900.0, 512: 1100.0, TOYHOST_WIDE: 1400.0}


def _release_profile() -> HostProfile:
    """The toy host, naming its ``batch_label`` as the build that ran."""
    return dataclasses.replace(toyhost_profile(), release_label="batch_label")


def timeseries_batches(
    positions: list[tuple[str, str | None]],
    *,
    profile: HostProfile,
    levels: tuple[int, ...] = (TOYHOST_NARROW, TOYHOST_WIDE),
) -> tuple[list[EvalRun], dict[str, list[EvalResult]]]:
    """One batch per arm at each ``(created_at, label)`` position, each position a tenth slower than the last."""
    runs: list[EvalRun] = []
    results: dict[str, list[EvalResult]] = {}
    for index, (created_at, label) in enumerate(positions):
        for level in levels:
            values: dict[str, Any] = dict(
                chunk_tokens=level,
                retriever_top_k=3,
                extraction_schema="v1",
                ocr_engine_version="tess-5.3.1",
                reviewer_pool="pool-a",
            )
            if label is not None:
                values["batch_label"] = label
            batch = toyhost_batch(**values).model_copy(
                update={"id": str(uuid.uuid5(_NAMESPACE, f"{index}|{level}")), "created_at": created_at}
            )
            runs.append(batch)
            results[batch.id] = toyhost_measurements(
                batch,
                profile=profile,
                cost_usd=0.02,
                total_ms=_TOTAL_MS[level] * (1 + 0.1 * index),
                field_accuracy=0.8,
                layout_fidelity=3,
            )
    return runs, results


def _timed(
    positions: list[tuple[str, str | None]], *, profile: HostProfile | None = None, run_ids: list[str] | None = None
) -> Any:
    """The toy campaign re-run at each position, assembled."""
    host = profile if profile is not None else toyhost_profile()
    campaign, _ = toyhost_campaign(profile=host)
    runs, results = timeseries_batches(positions, profile=host)
    timed = campaign.model_copy(update={"run_ids": run_ids if run_ids is not None else [run.id for run in runs]})
    return assemble_context_bundle(timed, storage=ToyhostStorage(runs, results), profile=host)


def _two_days() -> Any:
    return _timed([(DAY_ONE, None), (DAY_TWO, None)])


# =============================================================================
# The bundle's time axis
# =============================================================================


class TestTheBundleEarnsATimeAxis:
    def test_two_days_give_a_date_axis_in_calendar_order(self) -> None:
        bundle = _two_days()

        assert bundle.time_axis is not None
        assert bundle.time_axis_withheld is None
        assert bundle.time_axis.basis == "date"
        assert bundle.time_axis.release_label is None
        assert [position.key for position in bundle.time_axis.positions] == ["2026-03-14", "2026-03-15"]

    def test_one_day_and_no_build_label_give_no_axis_and_say_why(self) -> None:
        bundle = _timed([(DAY_ONE, None), ("2026-03-14T17:00:00+00:00", None)])

        assert bundle.time_axis is None
        assert bundle.time_axis_withheld == "every run started on one day (2026-03-14) and the host labels no build"

    def test_a_release_label_spanning_two_builds_gives_a_release_axis_on_one_day(self) -> None:
        bundle = _timed([(DAY_ONE, "0.9"), ("2026-03-14T12:00:00+00:00", "0.10")], profile=_release_profile())

        assert bundle.time_axis is not None
        assert (bundle.time_axis.basis, bundle.time_axis.release_label) == ("release", "batch_label")

    def test_builds_are_ordered_by_when_each_first_ran_not_by_name(self) -> None:
        """``0.10`` ran after ``0.9``; sorted by name it would come first, and the line would run backwards."""
        bundle = _timed([(DAY_ONE, "0.9"), ("2026-03-14T12:00:00+00:00", "0.10")], profile=_release_profile())

        assert [position.key for position in bundle.time_axis.positions] == ["0.9", "0.10"]
        assert sorted(["0.9", "0.10"]) == ["0.10", "0.9"], "the fixture must be one that name order gets wrong"

    def test_a_run_with_no_build_recorded_falls_back_to_days_and_names_the_runs_that_lacked_it(self) -> None:
        """The fallback is stated on the axis, not left for a reader to mistake days for builds."""
        bundle = _timed([(DAY_ONE, "0.9"), (DAY_TWO, None)], profile=_release_profile())
        unlabelled = sorted(str(uuid.uuid5(_NAMESPACE, f"1|{level}")) for level in (TOYHOST_NARROW, TOYHOST_WIDE))

        assert bundle.time_axis is not None
        assert bundle.time_axis.basis == "date"
        assert bundle.time_axis.basis_reason == f"2 of 4 runs recorded no batch_label ({', '.join(unlabelled)})"

    def test_a_date_axis_with_no_label_declared_says_the_host_labels_none(self) -> None:
        assert _two_days().time_axis.basis_reason == "the host labels no build"

    def test_a_release_axis_carries_no_fallback_reason(self) -> None:
        bundle = _timed([(DAY_ONE, "0.9"), ("2026-03-14T12:00:00+00:00", "0.10")], profile=_release_profile())

        assert bundle.time_axis.basis_reason is None

    def test_one_build_on_one_day_says_which_build(self) -> None:
        bundle = _timed([(DAY_ONE, "0.9"), ("2026-03-14T12:00:00+00:00", "0.9")], profile=_release_profile())

        assert bundle.time_axis is None
        assert bundle.time_axis_withheld == (
            "every run started on one day (2026-03-14) and every run recorded one batch_label (0.9)"
        )

    def test_labels_differing_only_in_whitespace_are_one_build_not_a_crash(self) -> None:
        """A version read with its trailing newline is the same build; two positions keyed alike were refused."""
        bundle = _timed([(DAY_ONE, "v1"), ("2026-03-14T12:00:00+00:00", "v1\n")], profile=_release_profile())

        assert bundle.time_axis is None
        assert bundle.time_axis_withheld == (
            "every run started on one day (2026-03-14) and every run recorded one batch_label (v1)"
        )

    def test_whitespace_around_a_label_does_not_split_or_name_a_build(self) -> None:
        bundle = _timed(
            [(DAY_ONE, " v1"), ("2026-03-14T11:00:00+00:00", "v1 "), ("2026-03-14T12:00:00+00:00", "v2")],
            profile=_release_profile(),
        )

        assert bundle.time_axis is not None and bundle.time_axis.basis == "release"
        assert [position.key for position in bundle.time_axis.positions] == ["v1", "v2"]

    def test_one_day_with_an_unrecorded_build_says_how_many_runs_lacked_it(self) -> None:
        bundle = _timed([(DAY_ONE, "0.9"), ("2026-03-14T12:00:00+00:00", None)], profile=_release_profile())

        assert bundle.time_axis is None
        unlabelled = ", ".join(
            sorted(str(uuid.uuid5(_NAMESPACE, f"1|{level}")) for level in (TOYHOST_NARROW, TOYHOST_WIDE))
        )
        assert bundle.time_axis_withheld == (
            f"every run started on one day (2026-03-14) and 2 of 4 runs recorded no batch_label ({unlabelled})"
        )

    def test_a_run_that_measured_nothing_has_no_place_in_time(self) -> None:
        host = toyhost_profile()
        campaign, _ = toyhost_campaign(profile=host)
        runs, results = timeseries_batches([(DAY_ONE, None), (DAY_TWO, None)], profile=host)
        for run in runs[2:]:
            results[run.id] = []
        timed = campaign.model_copy(update={"run_ids": [run.id for run in runs]})

        bundle = assemble_context_bundle(timed, storage=ToyhostStorage(runs, results), profile=host)

        assert bundle.time_axis is None
        assert bundle.time_axis_withheld is not None and "2026-03-14" in bundle.time_axis_withheld

    def test_a_campaign_that_measured_nothing_says_so(self) -> None:
        host = toyhost_profile()
        campaign, _ = toyhost_campaign(profile=host)
        runs, results = timeseries_batches([(DAY_ONE, None), (DAY_TWO, None)], profile=host)
        timed = campaign.model_copy(update={"run_ids": [run.id for run in runs]})

        bundle = assemble_context_bundle(
            timed, storage=ToyhostStorage(runs, {run.id: [] for run in runs}), profile=host
        )

        assert bundle.time_axis is None
        assert bundle.time_axis_withheld == "no run produced an observation, so nothing was measured at any time"

    def test_each_position_is_the_surface_a_campaign_of_only_its_runs_would_have(self) -> None:
        """One algebra, two populations: a position's cells ARE a one-day campaign's cells."""
        whole = _two_days()
        for index, position in enumerate(whole.time_axis.positions):
            alone = _timed(
                [(DAY_ONE, None), (DAY_TWO, None)],
                run_ids=[str(uuid.uuid5(_NAMESPACE, f"{index}|{level}")) for level in (TOYHOST_NARROW, TOYHOST_WIDE)],
            )
            assert position.cells == alone.cell_measures
            assert position.run_ids == sorted(alone.run_ids)

    def test_the_positions_differ_by_the_drift_the_runs_carried(self) -> None:
        first, second = _two_days().time_axis.positions

        def total(position: TimePosition) -> list[float]:
            return [next(m.mean for m in cell.measures.measures if m.name == "total_ms") for cell in position.cells]

        # The second day ran a tenth slower per document; each document's own offset rides on top, so the
        # ratio is near 1.1 rather than exactly it — and on neither day is it the pooled campaign's figure.
        assert all(1.05 < b / a < 1.15 for a, b in zip(total(first), total(second), strict=True))

    def test_every_measure_on_the_axis_is_described_on_the_frozen_surface(self) -> None:
        """A surface's facts cover every name in its cells — the axis's included."""
        analysis = _generated(_two_days(), _timeseries_payload)
        surface = analysis.decision_surface
        named = {m.name for p in surface.time_axis.positions for cell in p.cells for m in cell.measures.measures}
        assert named <= set(surface.measures)

    def test_the_bundle_shape_version_moved(self) -> None:
        assert _two_days().schema_version == 47


class TestTheTimeAxisContract:
    def _position(self, key: str) -> TimePosition:
        cells = _two_days().time_axis.positions[0].cells
        return TimePosition(key=key, first_run_at=DAY_ONE, last_run_at=DAY_ONE, run_ids=["r"], cells=cells)

    def test_two_positions_under_one_name_are_refused(self) -> None:
        with pytest.raises(ValueError, match="time positions must be distinct; repeated: a"):
            TimeAxis(basis="date", positions=[self._position("a"), self._position("a")])

    def test_one_position_is_not_an_axis(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            TimeAxis(basis="date", basis_reason="the host labels no build", positions=[self._position("a")])

    @pytest.mark.parametrize("reason", [None, "", "   "])
    def test_a_date_axis_that_does_not_say_why_it_is_not_builds_is_refused(self, reason: str | None) -> None:
        with pytest.raises(ValueError, match="a `date` time axis states its basis_reason"):
            TimeAxis(basis="date", basis_reason=reason, positions=[self._position("a"), self._position("b")])

    def test_a_release_axis_with_a_fallback_reason_is_refused(self) -> None:
        with pytest.raises(ValueError, match="a `release` time axis carries no basis_reason"):
            TimeAxis(
                basis="release",
                release_label="app_version",
                basis_reason="the host labels no build",
                positions=[self._position("a"), self._position("b")],
            )

    def test_a_date_axis_saying_why_is_accepted(self) -> None:
        axis = TimeAxis(
            basis="date", basis_reason="the host labels no build", positions=[self._position("a"), self._position("b")]
        )
        assert axis.basis_reason == "the host labels no build"

    @pytest.mark.parametrize(("basis", "label"), [("release", None), ("date", "app_version")])
    def test_the_basis_and_the_label_agree(self, basis: str, label: str | None) -> None:
        with pytest.raises(ValueError, match="names its release_label"):
            TimeAxis(basis=basis, release_label=label, positions=[self._position("a"), self._position("b")])

    def test_a_release_axis_naming_its_label_is_accepted(self) -> None:
        TimeAxis(basis="release", release_label="app_version", positions=[self._position("a"), self._position("b")])


# =============================================================================
# The host's release label
# =============================================================================


class TestTheReleaseLabelIsARegisteredLabel:
    def test_a_registered_label_is_accepted(self) -> None:
        assert _release_profile().release_label == "batch_label"

    @pytest.mark.parametrize(
        ("name", "why"),
        [
            ("app_version", "is not declared"),
            ("chunk_tokens", "is declared as a lever"),
            ("reviewer_pool", "is declared as a apparatus"),
        ],
    )
    def test_anything_else_is_refused_at_registration(self, name: str, why: str) -> None:
        with pytest.raises(ProfileRegistrationError, match=f"names release_label {name!r}, which {why}"):
            dataclasses.replace(toyhost_profile(), release_label=name)


# =============================================================================
# The builder
# =============================================================================


def _surface(bundle: Any) -> DecisionSurface:
    """The surface a generation freezes from ``bundle`` — built here from the bundle's public facts.

    That a generation freezes exactly this axis is asserted on a real generation below
    (``test_a_bundle_with_one_draws_it_and_freezes_the_axis``).
    """
    return DecisionSurface(
        cells=bundle.cell_measures,
        measures=cell_measure_facts(bundle),
        dimensions=cell_dimension_facts(bundle),
        time_axis=bundle.time_axis,
    )


def _chart(cells: list[str], measure: str = "total_ms", reading: str = "measure") -> Chart:
    return Chart(
        type="timeseries",
        cells=cells,
        measures=[MeasureRef(measure_id=measure, reading=reading)],
        axis="",
        note="",
        caption="",
    )


def _build(bundle: Any, chart: Chart, surface: DecisionSurface | None = None) -> dict[str, Any]:
    return build_viz_payload(
        reference_from_chart(chart),
        surface if surface is not None else _surface(bundle),
        bundle.variant_index,
        measures=MeasureRegistry([]),
    )


class TestTheBuilder:
    def test_a_surface_without_a_time_axis_cannot_draw_one(self) -> None:
        bundle = _two_days()
        flat = _surface(bundle).model_copy(update={"time_axis": None})

        with pytest.raises(UnresolvableReference, match="this analysis has none"):
            _build(bundle, _chart([]), flat)

    def test_every_point_is_the_resolvers_reading_over_its_position(self) -> None:
        bundle = _two_days()
        surface = _surface(bundle)
        payload = _build(bundle, _chart([]))

        assert payload["positions"] == ["2026-03-14", "2026-03-15"]
        assert payload["gaps"] == []
        cells = [cell_ref(c.variant_key, c.apparatus_class_id) for c in surface.cells]
        for cell, line in zip(cells, payload["series"], strict=True):
            for position, point in zip(surface.time_axis.positions, line["points"], strict=True):
                at = DecisionSurface(cells=position.cells, measures=surface.measures, dimensions=surface.dimensions)
                reading = resolve_reading(at, cell, "total_ms")
                # A point's n is its cases, never its observations: the bundle runs each case k times, so the two
                # differ here and a point counting observations would fail.
                assert reading.n_cases is not None and reading.n_cases < reading.n, "the bundle repeats each case"
                assert point["position"] == position.key
                assert (point["ci"]["mean"], point["ci"]["low"], point["ci"]["high"], point["n"]) == (
                    reading.mean,
                    reading.ci_low,
                    reading.ci_high,
                    reading.n_cases,
                )
        assert parse_payload("timeseries", payload) is not None

    def test_builds_whose_runs_interleave_are_named_and_disclosed(self) -> None:
        """v1 ran, then v2, then v1 again: v1's cells pool runs from both sides of the step to v2."""
        bundle = _timed(
            [(DAY_ONE, "v1"), ("2026-03-14T12:00:00+00:00", "v2"), (DAY_TWO, "v1")], profile=_release_profile()
        )
        assert [position.key for position in bundle.time_axis.positions] == ["v1", "v2"]

        payload = _build(bundle, _chart([]))

        assert payload["interleaved"] == ["v1"]
        intent = chart_intent("timeseries", payload)
        assert any(line.startswith("Interleaved builds (v1 into v2)") for line in intent.disclosures)

    def test_builds_run_one_after_another_interleave_with_nothing(self) -> None:
        bundle = _timed([(DAY_ONE, "v1"), (DAY_TWO, "v2")], profile=_release_profile())

        payload = _build(bundle, _chart([]))

        assert payload["interleaved"] == []
        assert not any(line.startswith("Interleaved") for line in chart_intent("timeseries", payload).disclosures)

    def test_one_cell_is_one_line(self) -> None:
        bundle = _two_days()
        first = _surface(bundle).cells[0]
        payload = _build(bundle, _chart([cell_ref(first.variant_key, first.apparatus_class_id)]))

        assert len(payload["series"]) == 1
        assert isinstance(reference_from_chart(_chart(["x"])), TimeseriesRef)

    def test_a_cell_absent_at_a_position_is_a_stated_gap(self) -> None:
        bundle = _two_days()
        surface = _surface(bundle)
        gone = surface.cells[0]
        second = surface.time_axis.positions[1]
        thinned = surface.time_axis.model_copy(
            update={
                "positions": [
                    surface.time_axis.positions[0],
                    second.model_copy(
                        update={
                            "cells": [
                                c
                                for c in second.cells
                                if (c.variant_key, c.apparatus_class_id) != (gone.variant_key, gone.apparatus_class_id)
                            ]
                        }
                    ),
                ]
            }
        )
        payload = _build(bundle, _chart([]), surface.model_copy(update={"time_axis": thinned}))

        assert len(payload["series"][0]["points"]) == 1
        assert payload["gaps"] == [
            {
                "series": payload["series"][0]["label"],
                "position": "2026-03-15",
                "reason": "the cell was not measured there",
            }
        ]

    def test_a_position_below_the_band_floor_is_a_stated_gap(self) -> None:
        """A point is drawn as an interval band, so a position over fewer than 5 cases has none (#677)."""
        bundle = _two_days()
        surface = _surface(bundle)
        second = surface.time_axis.positions[1]
        first_cell = second.cells[0]
        thin = first_cell.model_copy(
            update={
                "measures": first_cell.measures.model_copy(
                    update={
                        "measures": [
                            m.model_copy(update={"n_independent": 3}) if m.name == "total_ms" else m
                            for m in first_cell.measures.measures
                        ]
                    }
                )
            }
        )
        thinned = surface.time_axis.model_copy(
            update={
                "positions": [
                    surface.time_axis.positions[0],
                    second.model_copy(update={"cells": [thin, *second.cells[1:]]}),
                ]
            }
        )
        payload = _build(bundle, _chart([]), surface.model_copy(update={"time_axis": thinned}))

        assert {"position": "2026-03-15", "reason": "3 cases there, fewer than the 5 an interval is drawn from"} in [
            {key: gap[key] for key in ("position", "reason")} for gap in payload["gaps"]
        ]

    def test_no_cell_with_two_points_is_refused(self) -> None:
        bundle = _two_days()
        surface = _surface(bundle)
        first = surface.time_axis.positions[0]
        alone = surface.time_axis.model_copy(
            update={
                "positions": [
                    first,
                    surface.time_axis.positions[1].model_copy(update={"cells": [surface.cells[0]]}),
                ]
            }
        )
        cell = surface.cells[1]

        with pytest.raises(UnresolvableReference, match="none of .* has one at two"):
            _build(
                bundle,
                _chart([cell_ref(cell.variant_key, cell.apparatus_class_id)]),
                surface.model_copy(update={"time_axis": alone}),
            )

    def test_a_reading_no_cell_measured_is_refused_in_the_resolvers_words(self) -> None:
        with pytest.raises(UnresolvableReference, match="which measured no such measure"):
            _build(_two_days(), _chart([], "no_such_measure"))

    def test_a_categorical_reading_is_refused_in_the_resolvers_words(self) -> None:
        bundle = _two_days()
        surface = _surface(bundle)
        cell = surface.cells[0]
        stops = MeasureSummary(
            population="scored", name="stop_reason", attribution_scope="end_to_end", n=4, categories={"end": 4}
        )
        with_stops = cell.model_copy(
            update={"measures": cell.measures.model_copy(update={"measures": [*cell.measures.measures, stops]})}
        )
        surface = surface.model_copy(
            update={
                "cells": [with_stops, *surface.cells[1:]],
                "measures": surface.measures | {"stop_reason": MeasureFacts()},
            }
        )

        with pytest.raises(UnresolvableReference, match="categorical"):
            _build(bundle, _chart([cell_ref(cell.variant_key, cell.apparatus_class_id)], "stop_reason"), surface)

    def test_two_measures_are_not_one_reading(self) -> None:
        chart = _chart([]).model_copy(update={"measures": [MeasureRef(measure_id="a", reading="measure")] * 2})
        with pytest.raises(UnresolvableReference, match="exactly one measure"):
            reference_from_chart(chart)


# =============================================================================
# The payload and the arm
# =============================================================================


def _with_points(series: int, points: list[dict[str, Any]]) -> dict[str, Any]:
    payload = _payload()
    payload["series"] = copy.deepcopy(payload["series"])
    payload["series"][series]["points"] = points
    return payload


def _renamed(position: str, to: str) -> dict[str, Any]:
    """The conforming payload with one position renamed everywhere it appears."""
    renamed: dict[str, Any] = json.loads(json.dumps(_payload()).replace(f'"{position}"', f'"{to}"'))
    return renamed


class TestThePayloadRefusesWhatCannotBeDrawn:
    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            (_payload(positions=["2026-03-14"], series=[], gaps=[]), "at least 2 positions"),
            (_payload(positions=["2026-03-14", "2026-03-14", "2026-03-16"]), "duplicated: 2026-03-14"),
            (_payload(positions=["2026-03-14", " ", "2026-03-16"]), "every position needs a name"),
            (_payload(series=[], gaps=[]), "at least 1 series"),
            (_payload(series=[_payload()["series"][0], _payload()["series"][0]]), "duplicated: narrow"),
            (
                _with_points(
                    0, [{"position": "2026-03-20", "ci": _ci(1.0)}, {"position": "2026-03-14", "ci": _ci(1.0)}]
                ),
                "2026-03-20, which the axis",
            ),
            (
                _with_points(
                    0, [{"position": "2026-03-14", "ci": _ci(1.0)}, {"position": "2026-03-14", "ci": _ci(2.0)}]
                ),
                "more than one point at 2026-03-14",
            ),
            (
                _with_points(
                    0, [{"position": "2026-03-16", "ci": _ci(1.0)}, {"position": "2026-03-14", "ci": _ci(2.0)}]
                ),
                "out of the axis's order",
            ),
            (
                _payload(series=[{"label": "narrow", "points": [{"position": "2026-03-14", "ci": _ci(1.0)}]}], gaps=[]),
                "no series has points at two positions",
            ),
            (
                _payload(gaps=[{"series": "wide", "position": "2026-03-20", "reason": "x"}]),
                "gap names position '2026-03-20'",
            ),
            (_payload(gaps=[]), "series 'wide' has no point at 2026-03-15 and no gap saying why"),
            (
                _payload(gaps=[*_payload()["gaps"], {"series": "absent", "position": "2026-03-15", "reason": "x"}]),
                "gap names series 'absent'",
            ),
            (_renamed("2026-03-16", "zzz"), "'zzz' is not one"),
            (_payload(interleaved=["2026-03-14"]), "only builds interleave"),
            (
                _payload(basis="release", release_label="app_version", interleaved=["2026-03-20"]),
                "interleaved names 2026-03-20",
            ),
            (
                _payload(basis="release", release_label="app_version", interleaved=["2026-03-16"]),
                "the last build has no next build",
            ),
            (_renamed("2026-03-16", "2026-3-16"), "'2026-3-16' is not one"),
            (
                _payload(
                    positions=["2026-03-15", "2026-03-14", "2026-03-16"],
                    series=[
                        {
                            "label": "narrow",
                            "points": [
                                {"position": "2026-03-15", "ci": _ci(1.0)},
                                {"position": "2026-03-14", "ci": _ci(1.0)},
                                {"position": "2026-03-16", "ci": _ci(1.0)},
                            ],
                        }
                    ],
                    gaps=[],
                ),
                "calendar order, earliest first",
            ),
            (
                _payload(gaps=[{"series": "wide", "position": "2026-03-14", "reason": "x"}]),
                "both a point and a gap at '2026-03-14'",
            ),
            (_payload(basis="release"), "names its release_label"),
            (_payload(release_label="app_version"), "names its release_label"),
            (_with_points(1, [{"position": "2026-03-14"}, {"position": "2026-03-16", "ci": _ci(1.0)}]), "ci"),
            (_payload(colour="red"), "not permitted"),
        ],
    )
    def test_each_forbidden_shape_is_refused(self, payload: dict[str, Any], match: str) -> None:
        with pytest.raises(PayloadError, match=match):
            parse_payload("timeseries", payload)

    def test_the_conforming_shape_parses(self) -> None:
        assert parse_payload("timeseries", _payload()) is not None
        assert parse_payload("timeseries", _payload(basis="release", release_label="app_version")) is not None
        renamed = json.loads(
            json.dumps(_payload())
            .replace("2026-03-14", "0.9")
            .replace("2026-03-15", "0.10")
            .replace("2026-03-16", "0.11")
        )
        assert renamed["positions"] == ["0.9", "0.10", "0.11"]
        assert (
            parse_payload("timeseries", renamed | {"basis": "release", "release_label": "app_version"}) is not None
        ), "a release axis is ordered by when each build first ran, never by its name"


class TestTheIntent:
    def test_the_values_as_drawn_table_is_the_data(self) -> None:
        """Every drawn point, in drawn order, restated in the chart's unit — and nothing else."""
        chart = chart_intent("timeseries", _payload())
        expected = [
            {
                "series": line["label"],
                "position": p["position"],
                "mean": p["ci"]["mean"] / 1000,
                "low": p["ci"]["low"] / 1000,
                "high": p["ci"]["high"] / 1000,
                "n": p["n"],
            }
            for line in _payload()["series"]
            for p in line["points"]
        ]
        assert chart.unit == "s"
        assert [row.keys() for row in chart.rows] == [row.keys() for row in expected]
        for row, wanted in zip(chart.rows, expected, strict=True):
            assert row == {
                key: pytest.approx(value) if isinstance(value, float) else value for key, value in wanted.items()
            }
        assert [column.header for column in chart.columns] == [
            "Series",
            "Day (UTC)",
            "Mean (s)",
            "Low (s)",
            "High (s)",
            "Cases",
        ]

    def test_the_table_drawn_from_a_bundle_matches_the_bundles_numbers(self) -> None:
        bundle = _two_days()
        surface = _surface(bundle)
        chart = chart_intent("timeseries", _build(bundle, _chart([])))
        drawn = {(row["series"], row["position"]): row for row in chart.rows}
        labels = {line["label"] for line in _build(bundle, _chart([]))["series"]}

        assert len(drawn) == 2 * len(surface.cells) and {series for series, _ in drawn} == labels
        for cell, line in zip(surface.cells, _build(bundle, _chart([]))["series"], strict=True):
            ref = cell_ref(cell.variant_key, cell.apparatus_class_id)
            for position in surface.time_axis.positions:
                at = DecisionSurface(cells=position.cells, measures=surface.measures, dimensions=surface.dimensions)
                reading = resolve_reading(at, ref, "total_ms")
                row = drawn[(line["label"], position.key)]
                assert reading.n_cases is not None and reading.n_cases < reading.n, "the bundle repeats each case"
                assert (row["mean"], row["low"], row["high"], row["n"]) == pytest.approx(
                    (reading.mean / 1000, reading.ci_low / 1000, reading.ci_high / 1000, reading.n_cases)
                )

    def test_the_disclosures_name_the_order_the_intervals_and_each_gap(self) -> None:
        assert chart_intent("timeseries", _payload()).disclosures == [
            "Days are UTC, in calendar order.",
            "Intervals are 95% CIs.",
            "Intervals span the cell's observations.",
            "Not drawn (the cell was not measured there): wide at 2026-03-15.",
        ]
        release = chart_intent("timeseries", _payload(basis="release", release_label="app_version"))
        assert release.disclosures[0] == "Builds of app_version are in the order each first ran."


# =============================================================================
# The generator
# =============================================================================


def _chart_types(contract: dict[str, Any]) -> set[str]:
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            enum = node.get("enum")
            if isinstance(enum, list) and NO_CHART in enum:
                found.update(enum)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(contract)
    return found


def _timeseries_payload(bundle: Any) -> dict[str, Any]:
    """The toy memo with a timeseries chart over every cell.

    Written over the committed toy bundle: a re-run campaign pools the same two arms under the same rig,
    so its cells, their aliases and its declared question are the committed bundle's.
    """
    payload = memo_payload(toyhost_bundle())
    assert list(cell_aliases(bundle.cell_measures).values()) == list(
        cell_aliases(toyhost_bundle().cell_measures).values()
    )
    aliases = list(cell_aliases(bundle.cell_measures))
    payload["findings"][0]["chart"] = {
        "type": "timeseries",
        "cells": aliases,
        "measures": [{"measure_id": "total_ms", "reading": "measure"}],
        "axis": "",
        "note": "",
        "caption": "",
    }
    return payload


async def _generate(bundle: Any, payload: dict[str, Any]) -> Any:
    client = FixturedClient(json.dumps(payload))
    analysis, _ = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=client,
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=toyhost_profile(),
    )
    return analysis


def _generated(bundle: Any, author: Any) -> Any:
    import asyncio

    return asyncio.run(_generate(bundle, author(bundle)))


class TestTheMenuOffersTimeOnlyWhereThereIsTime:
    def test_a_bundle_without_a_time_axis_is_not_offered_the_chart(self) -> None:
        flat = _timed([(DAY_ONE, None), ("2026-03-14T17:00:00+00:00", None)])
        _, _, contract = first_request(flat, PROMPT, toyhost_profile())

        assert "timeseries" not in _chart_types(contract)
        assert "distribution" in _chart_types(contract), "the walk must find the menu it inspects"

    def test_a_bundle_with_one_is(self) -> None:
        _, _, contract = first_request(_two_days(), PROMPT, toyhost_profile())
        assert "timeseries" in _chart_types(contract)

    async def test_a_bundle_without_a_time_axis_cannot_reference_the_chart(self) -> None:
        flat = _timed([(DAY_ONE, None), ("2026-03-14T17:00:00+00:00", None)])

        with pytest.raises(SoundnessRefusal, match="timeseries"):
            await _generate(flat, _timeseries_payload(flat))

    async def test_a_bundle_with_one_draws_it_and_freezes_the_axis(self) -> None:
        bundle = _two_days()
        analysis = await _generate(bundle, _timeseries_payload(bundle))

        (resolution,) = analysis.resolutions
        assert resolution.chart is not None and resolution.chart.type == "timeseries"
        assert analysis.decision_surface.time_axis == bundle.time_axis
        assert chart_intent("timeseries", resolution.chart.payload).rows

    def test_the_prompt_version_hashes_the_menu_that_was_sent(self) -> None:
        profile = toyhost_profile()
        assert prompt_content_version(PROMPT, profile, time_axis=True) != prompt_content_version(
            PROMPT, profile, time_axis=False
        )

    def test_the_writer_sees_the_axis_by_alias(self) -> None:
        from threetears.evals.analysis.generator import build_user_message

        message = json.loads(build_user_message(_two_days()).split("\n", 1)[1])
        cells = message["time_axis"]["positions"][0]["cells"]
        assert all("cell" in cell and "variant_key" not in cell for cell in cells)


# =============================================================================
# Per-case values below the band floor (#677)
# =============================================================================


def _few_cases(cases: int) -> Any:
    """The toy campaign at one position, each run's results cut to its first ``cases`` test cases."""
    host = toyhost_profile()
    campaign, _ = toyhost_campaign(profile=host)
    runs, results = timeseries_batches([(DAY_ONE, None)], profile=host)
    kept = sorted({result.test_case_id for rows in results.values() for result in rows})[:cases]
    trimmed = {run_id: [r for r in rows if r.test_case_id in kept] for run_id, rows in results.items()}
    scoped = campaign.model_copy(update={"run_ids": [run.id for run in runs]})
    return assemble_context_bundle(scoped, storage=ToyhostStorage(runs, trimmed), profile=host)


class TestASmallCellRecordsItsCases:
    """Below 5 cases a chart draws the cases, so the bundle carries them; at 5 or more it does not."""

    def test_three_cases_carry_three_case_means(self) -> None:
        bundle = _few_cases(3)
        for cell in bundle.cell_measures:
            numeric = next(m for m in cell.measures.measures if m.name == "total_ms")
            assert numeric.n_independent == 3
            assert numeric.case_means is not None and len(numeric.case_means) == 3
            assert numeric.case_means == sorted(numeric.case_means)
            for judged in cell.judged:
                assert judged.n_independent == 3
                assert judged.case_means is not None and len(judged.case_means) == 3
        assert all(arm.case_means is not None for measure in bundle.judged_measures for arm in measure.arms)

    def test_five_cases_carry_none(self) -> None:
        bundle = _few_cases(5)
        for cell in bundle.cell_measures:
            assert all(m.case_means is None for m in cell.measures.measures)
            assert all(j.case_means is None for j in cell.judged)
