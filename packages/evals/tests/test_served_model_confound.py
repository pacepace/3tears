"""#684: a candidate launched on a floating alias is compared on the model that answered, not the one asked for.

An arm is keyed by the model id its launch asked for, and a provider resolves a floating alias
(``~vendor/model-latest``) on its side, so two runs of one alias months apart can be answered by different
models under one key. Only the response names the model that answered
(``RoleUsage.served_model``), so the bundle reads that:

* each arm says which models answered it (``arm_served_models``): one, pooled, or not recorded;
* an arm's cells never pool two answering models (owner ruling, 2026-10-23, no pooling by assumption): its results
  under one rig stand in a cell per answering model, those whose responses named none apart from those that named
  one, each compared with the control alone, and the bundle says why they did not pool;
* a comparison ACROSS arms whose runs had one requested id answered by more than one model names the
  ``served_model:candidate`` confound, varied — on the design's contrast and the coverage row of an arm that mixes
  them, and between two arms that asked for the same id;
* a comparison over a response that named no model names it undecided, never as a match with the alias;
* two arms that asked for different ids and were answered by different models are the model lever, not a
  confound of it.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence

from threetears.evals.analysis import (
    AnalysisContextBundle,
    Cell,
    Confound,
    DisclosureBlock,
    assemble_context_bundle,
    build_code_only_report,
    report_markdown,
)
from threetears.evals.kernel import (
    ControlDeclaration,
    SweptAxis,
    resolve_variant_identity,
)
from threetears.evals.schema import EvalResult, EvalRun
from threetears.evals.schema import SweepableValue
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.factories import make_campaign
from packages.evals.tests.v15_shapes import V15Design

#: The alias a launch asks for, and two concrete models a provider could resolve it to.
_ALIAS = "~vendor/extractor-latest"
_MARCH = "vendor/extractor-2026-03"
_JUNE = "vendor/extractor-2026-06"
#: A concrete id a launch asks for by name.
_PINNED = "vendor/extractor-pinned"

_CONFOUND = "served_model:candidate"

#: What a batch's candidate calls said answered them: a model name, or ``None`` for a response that named none.
Served = Callable[[EvalResult], str | None]


def _batch(model: str, salt: str, *, chunk_tokens: int = 512) -> EvalRun:
    """One toy batch asking for ``model``, under an id of its own so two batches can share every setting."""
    batch = toyhost_batch(
        chunk_tokens=chunk_tokens,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
    )
    return batch.model_copy(update={"id": str(uuid.uuid5(uuid.UUID(batch.id), salt)), "candidate_model": model})


def _served(model: str | None) -> Served:
    return lambda _result: model


def _results(batch: EvalRun, served: Served) -> list[EvalResult]:
    """The batch's toy observations, each candidate row naming what ``served`` says answered it."""
    return [
        result.model_copy(
            update={
                "usage": [
                    row.model_copy(update={"served_model": served(result)}) if row.role == "candidate" else row
                    for row in result.usage
                ]
            }
        )
        for result in toyhost_measurements(
            batch, profile=toyhost_profile(), cost_usd=0.02, total_ms=900.0, field_accuracy=0.8
        )
    ]


def _design(control: EvalRun, models: Sequence[str]) -> V15Design:
    return V15Design(
        axes=[
            SweptAxis(
                axis_id="model",
                values=[SweepableValue.of(model, display=model) for model in models],
                rationale="which extractor model",
            )
        ],
        control=resolve_variant_identity(run=control, profile=toyhost_profile()).variant_key,
        held_fixed=ControlDeclaration(stimulus="controlled", apparatus="witnessed"),
        declared_at="2026-03-14T09:30:00+00:00",
    )


def _bundle(batches: Sequence[tuple[EvalRun, Served]], *, design: V15Design | None = None) -> AnalysisContextBundle:
    storage = ToyhostStorage(
        [batch for batch, _ in batches], {batch.id: _results(batch, served) for batch, served in batches}
    )
    campaign = make_campaign(
        id="0b6f2a54-3c1e-4d8a-9e27-5a1c7f3b2d60",
        scope_id=TOYHOST_SCOPE,
        name="served-model",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[batch.id for batch, _ in batches],
        declared_design=design,
        created_by="test:fixture",
    )
    return assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())


def _alias_cells(bundle: AnalysisContextBundle) -> list[Cell]:
    """The cells of every arm but the control, in the bundle's order."""
    assert bundle.design.control_arm is not None
    return [cell for cell in bundle.cells if cell.variant_key != bundle.design.control_arm.variant_key]


def _alias_run(index: int) -> list[str]:
    """The result ids of the alias arm's ``index``-th run in :func:`_alias_against_pinned`, sorted."""
    return sorted(result.id for result in _results(_batch(_ALIAS, f"alias-{index}"), _served(None)))


def _served_confounds(confounds: Sequence[Confound]) -> list[Confound]:
    return [confound for confound in confounds if confound.kind == "served_model"]


def _row(bundle: AnalysisContextBundle, lever: str) -> list[Confound]:
    (row,) = [entry for entry in bundle.coverage if entry.name == lever]
    return _served_confounds(row.confounded_by)


def _alias_against_pinned(alias_runs: Sequence[Served]) -> AnalysisContextBundle:
    """A control asking for the pinned id, and one arm asking for the alias over one run per ``alias_runs`` entry."""
    control = _batch(_PINNED, "control")
    alias = [(_batch(_ALIAS, f"alias-{index}"), served) for index, served in enumerate(alias_runs)]
    return _bundle([(control, _served(_PINNED)), *alias], design=_design(control, [_PINNED, _ALIAS]))


class TestTwoRunsOfOneAliasAnsweredByDifferentModels:
    """The issue's case: one alias, two runs, two models behind it — one arm by key, two models by evidence."""

    def _bundle(self) -> AnalysisContextBundle:
        return _alias_against_pinned([_served(_MARCH), _served(_JUNE)])

    def test_the_two_runs_are_one_arm_by_key(self) -> None:
        bundle = self._bundle()
        assert len(bundle.design.contrasts) == 1
        assert len(bundle.design.contrasts[0].run_ids) == 2

    def test_the_arm_discloses_both_models_and_reads_pooled(self) -> None:
        readings = {reading.served_models[0]: reading for reading in self._bundle().arm_served_models}
        pooled = [reading for reading in readings.values() if reading.state == "pooled"]
        assert len(pooled) == 1
        assert pooled[0].served_models == [_MARCH, _JUNE]
        assert pooled[0].n_unrecorded == 0
        assert readings[_PINNED].state == "one"

    def test_the_design_contrast_names_the_confound(self) -> None:
        (arm,) = self._bundle().design.contrasts
        assert _served_confounds(arm.mechanism_confounds) == [Confound(dimension=_CONFOUND, kind="served_model")]

    def test_each_answering_model_is_a_cell_of_its_own(self) -> None:
        """No pooling by assumption (owner, 2026-10-23): the alias arm's two runs never share a cell."""
        bundle = self._bundle()
        alias = _alias_cells(bundle)
        assert [cell.n_observations for cell in alias] == [36, 36]
        assert len({cell.apparatus_class_id for cell in alias}) == 2
        # Each cell holds one run's results, and so one answering model.
        assert sorted(cell.observation_ids for cell in alias) == sorted([_alias_run(0), _alias_run(1)])
        # The control was answered by one model throughout: one cell, unsplit, under the rig the split cells name.
        (control,) = [cell for cell in bundle.cells if cell.variant_key == bundle.design.control_arm.variant_key]
        assert control.rig_class_id is None
        assert {cell.rig_class_id for cell in alias} == {control.apparatus_class_id}

    def test_the_bundle_says_why_the_two_cells_did_not_pool(self) -> None:
        (refused,) = self._bundle().refused_merges
        assert (refused.reason, refused.dimensions) == ("apparatus_differs", [_CONFOUND])

    def test_no_comparison_pools_the_two_models(self) -> None:
        """Each cell is read against the control alone, over its own run's cases, and neither names a mixture."""
        bundle = self._bundle()
        alias = {cell.apparatus_class_id: set(cell.observation_ids) for cell in _alias_cells(bundle)}
        comparisons = [c for family in bundle.multiple_comparisons.families for c in family.comparisons]
        assert comparisons, "the fixture must produce comparisons, or this asserts nothing"
        assert {c.contrast.apparatus_class_id for c in comparisons} == set(alias)
        # Twelve cases a side: one run's, never the twenty-four trials per case the two runs would pool.
        assert {c.contrast.n_cases for c in comparisons} == {12}
        assert all(_served_confounds(c.mechanism_confounds) == [] for c in comparisons)
        assert {c.contrast.apparatus_class_id for c in bundle.floor_contrasts} == set(alias)

    def test_a_code_only_report_reads_the_arm_once_per_answering_model(self) -> None:
        lines = report_markdown(
            build_code_only_report(self._bundle(), assembled_at="2026-10-09T00:00:00+00:00")
        ).splitlines()
        rows = [line for line in lines if line.startswith("| Field accuracy | extractor-latest (test setup ")]
        assert len(rows) == 4, "the contrast table and the statistics table, a row per answering model in each"
        assert len(set(rows)) == 4

    def test_the_model_rows_comparison_names_it(self) -> None:
        assert _row(self._bundle(), "model") == [Confound(dimension=_CONFOUND, kind="served_model")]

    def test_the_catalog_says_why(self) -> None:
        assert "floating alias" in self._bundle().confound_catalog[_CONFOUND]

    def test_a_code_only_report_says_the_arm_mixes_two_models(self) -> None:
        report = build_code_only_report(self._bundle(), assembled_at="2026-10-09T00:00:00+00:00")
        said = [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]
        assert [text for text in said if _MARCH in text and _JUNE in text and "mix" in text]


class TestOneAliasAnsweredByOneModel:
    """The alias held still: nothing to disclose, nothing confounded, and nothing split."""

    def test_the_two_runs_pool_in_one_cell_under_the_rig_itself(self) -> None:
        bundle = _alias_against_pinned([_served(_MARCH), _served(_MARCH)])
        (alias,) = _alias_cells(bundle)
        (control,) = [cell for cell in bundle.cells if cell.variant_key == bundle.design.control_arm.variant_key]
        assert alias.n_observations == 72
        assert alias.rig_class_id is None
        assert alias.apparatus_class_id == control.apparatus_class_id
        assert "rig_class_id" not in alias.to_dict()
        assert bundle.refused_merges == []

    def test_no_confound_and_the_arm_reads_one(self) -> None:
        bundle = _alias_against_pinned([_served(_MARCH), _served(_MARCH)])
        (arm,) = bundle.design.contrasts
        assert _served_confounds(arm.mechanism_confounds) == []
        assert _row(bundle, "model") == []
        assert {reading.state for reading in bundle.arm_served_models} == {"one"}


class TestAResponseNamingNoModelIsUnknownNotTheAlias:
    """A run whose responses named no model is not recorded — never read as the alias, so never a match."""

    def test_an_unrecorded_run_reads_undecided(self) -> None:
        bundle = _alias_against_pinned([_served(_MARCH), _served(None)])
        (arm,) = bundle.design.contrasts
        assert _served_confounds(arm.mechanism_confounds) == [
            Confound(dimension=_CONFOUND, kind="served_model", status="undecided")
        ]
        (alias,) = [reading for reading in bundle.arm_served_models if reading.state != "one"]
        assert (alias.state, alias.served_models) == ("unrecorded", [_MARCH])
        # One run of two named its model on every result, the other on none.
        assert alias.n_unrecorded * 2 == alias.n_results

    def test_the_unrecorded_run_joins_no_cell_with_the_run_that_named_its_model(self) -> None:
        """A silence is never evidence of a match (owner, 2026-10-23)."""
        bundle = _alias_against_pinned([_served(_MARCH), _served(None)])
        alias = _alias_cells(bundle)
        assert [cell.n_observations for cell in alias] == [36, 36]
        silent = {tuple(cell.observation_ids): cell.unknown_dimensions for cell in alias}
        assert silent == {tuple(_alias_run(0)): [], tuple(_alias_run(1)): [_CONFOUND]}
        (refused,) = bundle.refused_merges
        assert (refused.reason, refused.dimensions) == ("apparatus_unknown", [_CONFOUND])
        # What recording the model would buy is stated, as for any unrecorded dimension.
        (recording,) = bundle.next_experiments
        assert (recording.dimension, recording.n_observations_now, recording.n_observations_if_recorded) == (
            _CONFOUND,
            36,
            72,
        )
        comparisons = [c for family in bundle.multiple_comparisons.families for c in family.comparisons]
        assert {c.contrast.apparatus_class_id for c in comparisons} == {cell.apparatus_class_id for cell in alias}
        assert {c.contrast.n_cases for c in comparisons} == {12}

    def test_two_unrecorded_runs_are_alike_in_what_they_did_not_say_and_stay_one_cell(self) -> None:
        """As for any unrecorded dimension: identically silent observations pool, and the arm reads unrecorded."""
        bundle = _alias_against_pinned([_served(None), _served(None)])
        (alias,) = _alias_cells(bundle)
        assert (alias.n_observations, alias.rig_class_id) == (72, None)
        assert bundle.refused_merges == []


class TestARunTheProviderMovedPartWayThrough:
    """The split is by what answered each result, so one run can hold two cells."""

    def test_its_results_are_read_apart_by_the_model_that_answered_each(self) -> None:
        control = _batch(_PINNED, "control")
        alias = _batch(_ALIAS, "alias")
        moved = _bundle(
            [
                (control, _served(_PINNED)),
                (alias, lambda result: _MARCH if result.test_case_id <= "doc-06" else _JUNE),
            ],
            design=_design(control, [_PINNED, _ALIAS]),
        )
        assert [cell.n_observations for cell in _alias_cells(moved)] == [18, 18]
        assert {cell.n_cases for cell in _alias_cells(moved)} == {6}

    def test_a_result_whose_candidate_made_no_call_stays_with_its_run(self) -> None:
        """A result that failed before its call names no model and is no silence: it belongs with its run's cell."""
        control = _batch(_PINNED, "control")
        march, june = _batch(_ALIAS, "march"), _batch(_ALIAS, "june")

        def uncalled(rows: list[EvalResult]) -> list[EvalResult]:
            first = rows[0].model_copy(update={"usage": [row for row in rows[0].usage if row.role != "candidate"]})
            return [first, *rows[1:]]

        storage = ToyhostStorage(
            [control, march, june],
            {
                control.id: _results(control, _served(_PINNED)),
                march.id: uncalled(_results(march, _served(_MARCH))),
                june.id: _results(june, _served(_JUNE)),
            },
        )
        campaign = make_campaign(
            id="0b6f2a54-3c1e-4d8a-9e27-5a1c7f3b2d61",
            scope_id=TOYHOST_SCOPE,
            name="served-model-no-call",
            subject_id=TOYHOST_SUBJECT.subject_id,
            subject_kind="extractor_config",
            behavior="extract_invoice_fields",
            template_id="",
            run_ids=[control.id, march.id, june.id],
            declared_design=_design(control, [_PINNED, _ALIAS]),
            created_by="test:fixture",
        )
        bundle = assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())
        cells = sorted(cell.observation_ids for cell in _alias_cells(bundle))
        assert cells == sorted(
            [
                sorted(result.id for result in _results(march, _served(None))),
                sorted(result.id for result in _results(june, _served(None))),
            ]
        )

    def test_a_stored_run_that_never_recorded_one_is_not_the_alias(self) -> None:
        bundle = _alias_against_pinned([_served(None)])
        (alias,) = [reading for reading in bundle.arm_served_models if reading.state != "one"]
        assert alias.served_models == []
        assert _ALIAS not in alias.served_models
        assert alias.state == "unrecorded"

    def test_two_named_models_beside_an_unrecorded_run_still_read_varied(self) -> None:
        """Two named models prove the mixture whatever the unrecorded run was."""
        bundle = _alias_against_pinned([_served(_MARCH), _served(_JUNE), _served(None)])
        (arm,) = bundle.design.contrasts
        assert _served_confounds(arm.mechanism_confounds) == [Confound(dimension=_CONFOUND, kind="served_model")]


class TestDifferentIdsAnsweredByDifferentModelsAreTheLever:
    """A model comparison is a comparison of the models that answered: that is the lever, not its confound."""

    def test_no_confound_on_a_plain_model_comparison(self) -> None:
        control = _batch(_PINNED, "control")
        other = _batch(_MARCH, "other")
        bundle = _bundle(
            [(control, _served(_PINNED)), (other, _served(_MARCH))], design=_design(control, [_PINNED, _MARCH])
        )
        (arm,) = bundle.design.contrasts
        assert _served_confounds(arm.mechanism_confounds) == []
        assert _row(bundle, "model") == []


class TestTwoArmsOfOneAliasAnsweredByDifferentModels:
    """Between arms too: a sweep of another knob on one alias whose arms were answered by different models."""

    def test_the_other_levers_comparison_names_it_though_each_arm_is_one_model(self) -> None:
        bundle = _bundle(
            [
                (_batch(_ALIAS, "narrow", chunk_tokens=256), _served(_MARCH)),
                (_batch(_ALIAS, "wide", chunk_tokens=1024), _served(_JUNE)),
            ]
        )
        assert _row(bundle, "chunk_tokens") == [Confound(dimension=_CONFOUND, kind="served_model")]
        assert {reading.state for reading in bundle.arm_served_models} == {"one"}


def test_an_arm_named_by_its_model_is_named_once_by_the_version_that_answered_it_and_by_its_label_in_every_table() -> (
    None
):
    """The subhead names the served version, the name asked by in brackets as its label; every table says the label."""
    control, other = _batch(_PINNED, "control"), _batch(_JUNE, "june")
    bundle = _bundle(
        [(control, _served(f"{_PINNED}-2026-05")), (other, _served(_JUNE))], design=_design(control, [_PINNED, _JUNE])
    )
    lines = report_markdown(build_code_only_report(bundle, assembled_at="2026-10-14T00:00Z")).splitlines()

    assert lines[1].startswith("extractor-pinned-2026-05 (extractor-pinned, control) vs vendor/extractor-2026-06 · ")
    rows = [line for line in lines if line.startswith("| ")]
    assert "| ▸ extractor-pinned | 36 |" in "\n".join(rows)
    assert not any("2026-05" in row for row in rows), "a table names the arm by its label alone"
