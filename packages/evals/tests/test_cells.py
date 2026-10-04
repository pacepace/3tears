"""The cell algebra: what pools, what refuses, and what a refusal is worth.

Every test here is about a *merge decision*. The three that matter are the three the design
names — identical apparatus pools, a differing dimension does not, an unrecorded dimension
does not either — and the third is the one a host's own data rarely exercises as a designed
state, which is why the toy host's corpus is built around it.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.cells import (
    ApparatusClass,
    Cell,
    NextExperiment,
    Observation,
    apparatus_class_of,
    pool_observations,
    subject_key_instabilities,
)


_VARIANT = "v" * 64
_OTHER_VARIANT = "w" * 64


def _observation(
    obs_id: str,
    klass: ApparatusClass,
    *,
    variant: str = _VARIANT,
    provenance: str = "declared",
    case_ref: str | None = None,
) -> Observation:
    """One observation at a named class.

    Args:
        obs_id: The observation's id.
        klass: The apparatus class it was measured under.
        variant: Its variant coordinate.
        provenance: Whether the apparatus was declared or witnessed.
        case_ref: The case it exercised, or None for a host with no battery.

    Returns:
        The observation.
    """
    return Observation(
        id=obs_id,
        scope_id="scope-a",
        variant_key=variant,
        apparatus_class_id=klass.apparatus_class_id,
        provenance=provenance,  # type: ignore[arg-type]
        case_ref=case_ref,
    )


def _pool(
    *observations: Observation, classes: tuple[ApparatusClass, ...]
) -> tuple[list[Cell], list, list[NextExperiment]]:
    """Pool observations against the classes they reference.

    Args:
        *observations: The observations.
        classes: Every class referenced.

    Returns:
        The three-part pooling result.
    """
    return pool_observations(list(observations), {c.apparatus_class_id: c for c in classes})


class TestTheMergeRule:
    """Identical apparatus pools; anything else does not, and says why."""

    def test_two_observations_with_the_same_variant_and_identical_apparatus_pool(self) -> None:
        """The base case, and the one that makes k accumulate across batches at all."""
        klass = apparatus_class_of({"judge": "j-1", "template": "t-1"})
        cells, refused, _ = _pool(
            _observation("obs-a", klass),
            _observation("obs-b", klass),
            classes=(klass,),
        )

        assert len(cells) == 1, "one variant, one apparatus class — one cell"
        assert cells[0].n_observations == 2
        assert cells[0].observation_ids == ["obs-a", "obs-b"]
        assert refused == [], "nothing was refused, so nothing should be reported as refused"

    def test_one_differing_dimension_keeps_them_apart_and_the_reason_is_stated(self) -> None:
        """A rival explanation exists, so the two stay separate rather than being averaged."""
        left = apparatus_class_of({"judge": "j-1", "template": "t-1"})
        right = apparatus_class_of({"judge": "j-2", "template": "t-1"})
        cells, refused, _ = _pool(
            _observation("obs-a", left),
            _observation("obs-b", right),
            classes=(left, right),
        )

        assert len(cells) == 2, "the apparatus moved, so these are two cells"
        assert [c.n_observations for c in cells] == [1, 1]
        assert len(refused) == 1
        assert refused[0].reason == "apparatus_differs"
        assert refused[0].dimensions == ["judge"], "the reason names WHICH dimension, not merely that one did"

    def test_one_unknown_dimension_keeps_them_apart_and_reports_an_actionable_gap(self) -> None:
        """``unknown`` blocks the merge — and unlike a difference, recording can fix it.

        The conservative side of the pooling rule, and the half that would be pure refusal without
        the next-experiment: an operator told only "these did not pool" has nothing to do,
        while one told "recording ocr_engine_version on obs-b takes k from 1 to 2" does.
        """
        recorded = apparatus_class_of({"ocr_engine_version": "tess-5.3.1"}, dimensions={"ocr_engine_version"})
        missing = apparatus_class_of({"ocr_engine_version": None}, dimensions={"ocr_engine_version"})

        cells, refused, next_experiments = _pool(
            _observation("obs-a", recorded),
            _observation("obs-b", missing),
            classes=(recorded, missing),
        )

        assert len(cells) == 2, "undecidable is not agreement — the merge is refused"
        assert [r.reason for r in refused] == ["apparatus_unknown"]
        assert refused[0].dimensions == ["ocr_engine_version"]

        assert len(next_experiments) == 1, "the gap must be actionable, not merely reported"
        entry = next_experiments[0]
        assert entry.dimension == "ocr_engine_version"
        assert (entry.n_observations_now, entry.n_observations_if_recorded) == (1, 2)
        assert entry.observation_ids == ["obs-b"], "the recording has to happen where the dimension is missing"

    def test_an_unknown_dimension_is_reported_as_undecidable_rather_than_as_a_difference(self) -> None:
        """Recorded-on-one-side is not a disagreement, and calling it one asserts an observation nobody made."""
        recorded = apparatus_class_of({"judge": "j-1"}, dimensions={"judge"})
        missing = apparatus_class_of({"judge": None}, dimensions={"judge"})

        _, refused, _ = _pool(
            _observation("obs-a", recorded), _observation("obs-b", missing), classes=(recorded, missing)
        )

        assert refused[0].reason == "apparatus_unknown", (
            "one side recorded a value and the other recorded nothing — whether they differ is undecidable, "
            "and reporting it as apparatus_differs would claim a comparison nobody performed"
        )

    def test_declared_and_witnessed_never_pool(self) -> None:
        """The difference between an experiment and a log, which an analysis must not collapse."""
        klass = apparatus_class_of({"judge": "j-1"})
        cells, refused, _ = _pool(
            _observation("obs-a", klass, provenance="declared"),
            _observation("obs-b", klass, provenance="witnessed"),
            classes=(klass,),
        )

        assert len(cells) == 2, "same variant, same apparatus — and still two cells"
        assert [r.reason for r in refused] == ["provenance_differs"]
        assert refused[0].dimensions == [], "provenance names no dimension, and inventing one would be a false lead"


class TestCellsKeyOnTheVariant:
    """A cell is a variant under a rig, never a run — the run-as-cell defect, inverted."""

    def test_two_variants_under_one_rig_are_two_cells_in_one_apparatus_class_with_no_confound(self) -> None:
        """Two arms measured under one rig differ in the variant and in nothing else.

        Under a run-keyed cell, observations were grouped by the batch that produced them, so
        what separated two arms was which run they came from. Keyed on the variant, two arms
        under an identical rig are two cells sharing one apparatus class, whichever runs
        measured them.
        """
        klass = apparatus_class_of({"judge": "j-1", "template": "t-1"})
        cells, refused, next_experiments = _pool(
            _observation("obs-arm-a", klass, variant=_VARIANT),
            _observation("obs-arm-b", klass, variant=_OTHER_VARIANT),
            classes=(klass,),
        )

        assert len(cells) == 2, "two variants under one rig are two cells"
        assert {c.apparatus_class_id for c in cells} == {klass.apparatus_class_id}, "and they share the apparatus class"
        assert refused == [], "nothing was refused: these are different variants, not a blocked merge"
        assert next_experiments == [], "there is no gap to close — the rig was fully recorded"

    def test_a_repeat_of_a_setting_grows_the_cell_rather_than_minting_a_second_one(self) -> None:
        """A re-run is more evidence for the same claim, which a run-keyed cell could not say.

        This used to assert ``k == 3`` of three observations, which is the defect the rename
        removes rather than a behaviour to keep: three observations of one setting are three
        observations, and whether that is one case run three times or three cases run once is
        not something the count can say — the next class is what says it.
        """
        klass = apparatus_class_of({"judge": "j-1"})
        cells, _, _ = _pool(
            _observation("obs-1", klass),
            _observation("obs-2", klass),
            _observation("obs-3", klass),
            classes=(klass,),
        )

        assert len(cells) == 1, "three observations of one setting are one cell, not three that each moved nothing"
        assert cells[0].n_observations == 3
        assert cells[0].n_cases is None, "no observation named a case, so the cell has no case count to state"


class TestACellStatesItsReplicationRatherThanACount:
    """Pooled observations and repeats per case are different numbers, and the cell carries both.

    The failure this pins: the observation count was the field ``k``, a generator read it as
    repeats, and a campaign's arms read as replicated (or as unreplicated) by a number that
    counted cases and repeats together.
    """

    def test_the_same_count_over_one_case_and_over_three_is_two_different_replications(self) -> None:
        """Three observations of one case is ``1 x 3``; three of three cases is ``3 x 1``.

        The two cells are indistinguishable on ``n_observations``, which is exactly why that field
        cannot be the repeat count — an assertion on the count alone passes for both.
        """
        klass = apparatus_class_of({"judge": "j-1"})
        repeated, _, _ = _pool(
            *(_observation(f"obs-{i}", klass, case_ref="case-a") for i in range(3)),
            classes=(klass,),
        )
        spread, _, _ = _pool(
            *(_observation(f"obs-{case}", klass, case_ref=f"case-{case}") for case in "abc"),
            classes=(klass,),
        )

        assert repeated[0].n_observations == spread[0].n_observations == 3, "the fixture must tie on the count"
        assert (repeated[0].n_cases, repeated[0].repeats_per_case_min, repeated[0].repeats_per_case_max) == (1, 3, 3)
        assert (spread[0].n_cases, spread[0].repeats_per_case_min, spread[0].repeats_per_case_max) == (3, 1, 1)

    def test_two_batches_of_one_setting_add_their_repeats(self) -> None:
        """A cell pools runs, so its repeats per case are theirs summed — not any one run's launch count.

        Two batches of the same five cases, each run once, are five cases run twice in the cell,
        which is the reading a run's ``k_runs`` of 1 cannot give.
        """
        klass = apparatus_class_of({"judge": "j-1"})
        observations = [
            Observation(
                id=f"obs-{batch}-{case}",
                scope_id="scope-a",
                variant_key=_VARIANT,
                apparatus_class_id=klass.apparatus_class_id,
                apparatus_ref=batch,
                provenance="declared",
                case_ref=f"case-{case}",
            )
            for batch in ("run-1", "run-2")
            for case in range(5)
        ]

        cells, _, _ = pool_observations(observations, {klass.apparatus_class_id: klass})

        assert len(cells) == 1
        assert (cells[0].n_observations, cells[0].n_cases) == (10, 5)
        assert (cells[0].repeats_per_case_min, cells[0].repeats_per_case_max) == (2, 2)

    def test_an_unevenly_repeated_cell_states_its_fewest_and_its_most(self) -> None:
        """One case lost an observation: two cases, three and two repeats, stated as a range."""
        klass = apparatus_class_of({"judge": "j-1"})
        cells, _, _ = _pool(
            _observation("obs-a1", klass, case_ref="case-a"),
            _observation("obs-a2", klass, case_ref="case-a"),
            _observation("obs-a3", klass, case_ref="case-a"),
            _observation("obs-b1", klass, case_ref="case-b"),
            _observation("obs-b2", klass, case_ref="case-b"),
            classes=(klass,),
        )

        assert (cells[0].n_cases, cells[0].repeats_per_case_min, cells[0].repeats_per_case_max) == (2, 2, 3)

    def test_a_cell_partly_off_the_battery_states_no_case_count(self) -> None:
        """Counting only the observations that named a case would describe evidence the cell does not hold."""
        klass = apparatus_class_of({"judge": "j-1"})
        cells, _, _ = _pool(
            _observation("obs-a", klass, case_ref="case-a"),
            _observation("obs-b", klass, case_ref="case-a"),
            _observation("obs-inline", klass),
            classes=(klass,),
        )

        assert cells[0].n_observations == 3
        assert (cells[0].n_cases, cells[0].repeats_per_case_min, cells[0].repeats_per_case_max) == (None, None, None)


class TestSubjectKeyInstability:
    """One key under two labels, and one label under two keys — both directions warn."""

    def test_one_key_under_two_labels_warns(self) -> None:
        """A rename mid-campaign, or two things collided onto one key."""
        warnings = subject_key_instabilities([("key-1", "Maple"), ("key-1", "Maple (v2)")])

        assert len(warnings) == 1
        assert warnings[0].kind == "one_key_many_labels"
        assert warnings[0].key == "key-1"
        assert warnings[0].counterparts == ["Maple", "Maple (v2)"]

    def test_one_label_under_two_keys_warns(self) -> None:
        """A reader groups on the label, and here the keys do not support that grouping."""
        warnings = subject_key_instabilities([("key-1", "Maple"), ("key-2", "Maple")])

        assert len(warnings) == 1
        assert warnings[0].kind == "one_label_many_keys"
        assert warnings[0].label == "Maple"
        assert warnings[0].counterparts == ["key-1", "key-2"]

    def test_a_stable_population_warns_about_nothing(self) -> None:
        """The ordinary case must be silent, or the warning becomes noise nobody reads."""
        assert subject_key_instabilities([("key-1", "Maple"), ("key-1", "Maple"), ("key-2", "Bea")]) == []


class TestTheModelsRefuseToMisreport:
    """Structural refusals — the states that must be unrepresentable rather than merely unused."""

    def test_a_dimension_cannot_be_both_recorded_and_unknown(self) -> None:
        """Two producers disagreeing about whether something was observed makes the class unreadable."""
        with pytest.raises(ValueError, match="both recorded and unknown"):
            ApparatusClass(apparatus_class_id="x", recorded={"judge": "j-1"}, unknown_dimensions=["judge"])

    def test_the_observation_count_cannot_disagree_with_the_observations_behind_it(self) -> None:
        """A cell reporting a sample size no evidence supports is the failure this whole model fixes."""
        with pytest.raises(ValueError, match="disagrees"):
            Cell(
                variant_key=_VARIANT,
                apparatus_class_id="x",
                provenance="declared",
                n_observations=5,
                observation_ids=["obs-a"],
            )

    @pytest.mark.parametrize(
        ("n_cases", "low", "high", "match"),
        [
            pytest.param(2, None, None, "together or not at all", id="cases-without-repeats"),
            pytest.param(None, 2, 2, "together or not at all", id="repeats-without-cases"),
            pytest.param(2, 3, 2, "exceeds", id="min-above-max"),
            pytest.param(2, 3, 3, "cannot pool", id="more-than-the-sample"),
            pytest.param(1, 2, 2, "cannot pool", id="less-than-the-sample"),
        ],
    )
    def test_a_replication_the_sample_contradicts_is_refused(
        self, n_cases: int | None, low: int | None, high: int | None, match: str
    ) -> None:
        """Every forbidden shape of the replication triple, each against four observations."""
        with pytest.raises(ValueError, match=match):
            Cell(
                variant_key=_VARIANT,
                apparatus_class_id="x",
                provenance="declared",
                n_observations=4,
                n_cases=n_cases,
                repeats_per_case_min=low,
                repeats_per_case_max=high,
                observation_ids=["obs-a", "obs-b", "obs-c", "obs-d"],
            )

    @pytest.mark.parametrize(
        ("n_cases", "low", "high"),
        [
            pytest.param(None, None, None, id="no-battery"),
            pytest.param(2, 2, 2, id="two-by-two"),
            pytest.param(4, 1, 1, id="four-by-one"),
            pytest.param(3, 1, 2, id="uneven"),
        ],
    )
    def test_a_replication_the_sample_supports_is_accepted(
        self, n_cases: int | None, low: int | None, high: int | None
    ) -> None:
        """The acceptance side, on the same four observations, so the refusals above are not a refusal of everything."""
        cell = Cell(
            variant_key=_VARIANT,
            apparatus_class_id="x",
            provenance="declared",
            n_observations=4,
            n_cases=n_cases,
            repeats_per_case_min=low,
            repeats_per_case_max=high,
            observation_ids=["obs-a", "obs-b", "obs-c", "obs-d"],
        )

        assert cell.n_cases == n_cases

    def test_a_next_experiment_promising_no_gain_is_refused(self) -> None:
        """An entry that buys nothing is noise a reader still has to read."""
        with pytest.raises(ValueError, match="does not exceed"):
            NextExperiment(variant_key=_VARIANT, dimension="judge", n_observations_now=3, n_observations_if_recorded=3)

    def test_an_observation_referencing_an_unknown_class_is_refused_not_dropped(self) -> None:
        """Silently dropping it would remove evidence from every cell with nothing marking the loss."""
        klass = apparatus_class_of({"judge": "j-1"})
        stray = Observation(
            id="obs-x",
            scope_id="scope-a",
            variant_key=_VARIANT,
            apparatus_class_id="not-a-class",
            provenance="declared",
        )

        with pytest.raises(KeyError, match="unknown apparatus class"):
            pool_observations([stray], {klass.apparatus_class_id: klass})


class TestTheClassIdCarriesWhatWasNotMeasured:
    """The id is consulted BEFORE any merge rule, so the rule cannot be the only guard."""

    def test_two_classes_agreeing_on_what_they_recorded_differ_when_one_measured_less(self) -> None:
        """A wrong MERGE, and the one outcome nothing downstream can undo.

        `pool_observations` groups by `apparatus_class_id` before `_refusal` is consulted, so
        two classes sharing an id are one group and never reach the rule at all. While the id
        digested `recorded` alone, an observation that also failed to record a dimension hashed
        identically to one that had nothing to record, pooled into a single cell, and emitted no
        refusal — the unknown-blocks-a-merge rule was live and unreachable.
        """
        measured_less = apparatus_class_of({"judge": "j-1", "ocr": None}, dimensions={"judge", "ocr"})
        measured_all = apparatus_class_of({"judge": "j-1"}, dimensions={"judge"})

        assert measured_less.recorded == measured_all.recorded, "the fixture must agree on what was recorded"
        assert measured_less.apparatus_class_id != measured_all.apparatus_class_id, (
            "what a measurement could not speak to is part of what that measurement WAS"
        )

    def test_the_unknown_dimension_actually_reaches_the_refusal(self) -> None:
        """The end-to-end claim: distinct ids, two cells, and the rule fires."""
        measured_less = apparatus_class_of({"judge": "j-1", "ocr": None}, dimensions={"judge", "ocr"})
        measured_all = apparatus_class_of({"judge": "j-1"}, dimensions={"judge"})

        cells, refused, _ = _pool(
            _observation("obs-partial", measured_less),
            _observation("obs-full", measured_all),
            classes=(measured_less, measured_all),
        )

        assert len(cells) == 2, "these must not pool — one of them never measured the OCR build"
        assert [r.reason for r in refused] == ["apparatus_unknown"]
        assert refused[0].dimensions == ["ocr"]

    def test_two_identically_unmeasured_classes_are_still_one_class(self) -> None:
        """Hashing the unknown NAMES must not split observations that were unmeasured alike.

        The guard against over-correcting: including the unknown set in the id is right, but it
        would be wrong if it also split two observations whose measurements were identical —
        including identically silent.
        """
        left = apparatus_class_of({"judge": "j-1", "ocr": None}, dimensions={"judge", "ocr"})
        right = apparatus_class_of({"judge": "j-1", "ocr": None}, dimensions={"judge", "ocr"})

        assert left.apparatus_class_id == right.apparatus_class_id
        cells, refused, _ = _pool(_observation("obs-a", left), _observation("obs-b", right), classes=(left,))
        assert len(cells) == 1 and cells[0].n_observations == 2
        assert refused == [], "one class is one group, and a group has no merge to refuse"


class TestANextExperimentPromisesOnlyWhatRecordingCanDeliver:
    """The optimistic branch is still arithmetic, and it must be arithmetic nobody can falsify."""

    def test_recorders_that_already_disagree_are_not_counted_as_mergeable(self) -> None:
        """Two cells that recorded the dimension at DIFFERENT values can never be joined by recording it.

        The grouping deliberately strips the hypothesised dimension so a recorder and a
        non-recorder can group — that is the hypothesis. Stripping it from two recorders that
        already disagree groups a pair recording cannot join, and the promised k is summed over
        all of them. Left unnarrowed this variant reached the bundle and then the generator.
        """
        recorded_v1 = apparatus_class_of({"ocr": "v1"}, dimensions={"ocr"})
        recorded_v2 = apparatus_class_of({"ocr": "v2"}, dimensions={"ocr"})
        never_recorded = apparatus_class_of({"ocr": None}, dimensions={"ocr"})

        _, _, next_experiments = _pool(
            _observation("obs-v1", recorded_v1),
            _observation("obs-v2", recorded_v2),
            _observation("obs-silent", never_recorded),
            classes=(recorded_v1, recorded_v2, never_recorded),
        )

        assert len(next_experiments) == 1
        entry = next_experiments[0]
        assert entry.n_observations_if_recorded == 2, (
            "recording it on the silent cell can join it to ONE of the two recorders, never both — "
            "they already disagree on the one thing under hypothesis"
        )
        assert entry.observation_ids == ["obs-silent"], "and only the silent cell is where recording happens"

    def test_an_entry_is_never_emitted_with_no_observation_to_record_on(self) -> None:
        """An empty observation list means nothing is missing the dimension, so nothing can be recorded.

        The tell that the arithmetic went wrong, and worth asserting directly: an operator handed
        a next-experiment with no observations named has been told to act and given nowhere to act.
        """
        recorded_v1 = apparatus_class_of({"ocr": "v1"}, dimensions={"ocr"})
        recorded_v2 = apparatus_class_of({"ocr": "v2"}, dimensions={"ocr"})

        _, refused, next_experiments = _pool(
            _observation("obs-v1", recorded_v1),
            _observation("obs-v2", recorded_v2),
            classes=(recorded_v1, recorded_v2),
        )

        assert [r.reason for r in refused] == ["apparatus_differs"], "they differ on a recorded value"
        assert next_experiments == [], "nothing is unrecorded here, so there is nothing to ask anyone to record"
