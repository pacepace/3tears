"""A time axis the bundle earns: two builds the host labels, or failing that two days, and nothing else.

- **The bundle's time axis** (``analysis/bundle/time_axis.py``). A campaign spanning two builds the host labels, or
  failing that two days, carries one; each position's cells are the decision surface's own algebra over that
  position's runs, so a position reads exactly as a campaign of only its runs would. Positions are ordered
  by when each first ran — never by name, which would put ``0.10`` before ``0.9``.
- **The host's release label** (``kernel/host/profile.py``). It must name a registered ``label``.
- **The generation freezes it** on the analysis's decision surface, and the writer reads it by cell alias.
- **A small cell records its cases**: below five, the bundle carries each case's mean.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from typing import Any

import pytest

from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.analysis.references import cell_aliases
from threetears.evals.kernel.host.profile import HostProfile, ProfileRegistrationError
from threetears.evals.schema.models import EvalResult, EvalRun, utc_now_iso
from threetears.evals.kernel.surface import TimeAxis, TimePosition
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
        analysis = _generated(_two_days())
        surface = analysis.decision_surface
        named = {m.name for p in surface.time_axis.positions for cell in p.cells for m in cell.measures.measures}
        assert named <= set(surface.measures)

    def test_the_bundle_shape_version_moved(self) -> None:
        assert _two_days().schema_version == 57


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
# The generator
# =============================================================================


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


def _generated(bundle: Any) -> Any:
    """A generation over ``bundle`` of the toy memo, written over the committed toy bundle.

    A re-run campaign pools the same two arms under the same rig, so its cells, their aliases and its declared
    question are the committed bundle's.
    """
    import asyncio

    assert list(cell_aliases(bundle.cell_measures).values()) == list(
        cell_aliases(toyhost_bundle().cell_measures).values()
    )
    return asyncio.run(_generate(bundle, memo_payload(toyhost_bundle())))


class TestTheGenerationFreezesTheAxis:
    def test_a_bundle_with_one_freezes_it_on_the_decision_surface(self) -> None:
        bundle = _two_days()
        assert _generated(bundle).decision_surface.time_axis == bundle.time_axis

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
