"""#684: a candidate launched on a floating alias is compared on the model that answered, not the one asked for.

An arm is keyed by the model id its launch asked for, and a provider resolves a floating alias
(``~vendor/model-latest``) on its side, so two runs of one alias months apart can be answered by different
models and still pool as one arm. Only the response names the model that answered
(``RoleUsage.served_model``), so the bundle reads that:

* each arm says which models answered it (``arm_served_models``): one, pooled, or not recorded;
* a comparison whose runs had one requested id answered by more than one model names the
  ``served_model:candidate`` confound, varied — within an arm or between two arms that asked for the same id;
* a comparison over a response that named no model names it undecided, never as a match with the alias;
* two arms that asked for different ids and were answered by different models are the model lever, not a
  confound of it.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence

from threetears.evals.analysis import (
    AnalysisContextBundle,
    Confound,
    DisclosureBlock,
    assemble_context_bundle,
    build_code_only_report,
)
from threetears.evals.contracts import (
    CampaignDesign,
    ControlDeclaration,
    EvalCampaign,
    EvalResult,
    EvalRun,
    SweptAxis,
    resolve_variant_identity,
)
from threetears.evals.contracts.host import SweepableValue
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

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


def _design(control: EvalRun, models: Sequence[str]) -> CampaignDesign:
    return CampaignDesign(
        axes=[
            SweptAxis(
                axis_id="model",
                values=[SweepableValue.of(model, display=model) for model in models],
                rationale="which extractor model",
            )
        ],
        control=resolve_variant_identity(run=control, profile=toyhost_profile()).variant_key,
        controls=ControlDeclaration(stimulus="controlled", apparatus="witnessed"),
        declared_at="2026-03-14T09:30:00+00:00",
    )


def _bundle(
    batches: Sequence[tuple[EvalRun, Served]], *, design: CampaignDesign | None = None
) -> AnalysisContextBundle:
    storage = ToyhostStorage(
        [batch for batch, _ in batches], {batch.id: _results(batch, served) for batch, served in batches}
    )
    campaign = EvalCampaign(
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

    def test_every_pairwise_comparison_names_it(self) -> None:
        comparisons = [c for family in self._bundle().multiple_comparisons.families for c in family.comparisons]
        assert comparisons, "the fixture must produce comparisons, or this asserts nothing"
        assert all(
            _served_confounds(c.mechanism_confounds) == [Confound(dimension=_CONFOUND, kind="served_model")]
            for c in comparisons
        )

    def test_the_model_rows_comparison_names_it(self) -> None:
        assert _row(self._bundle(), "model") == [Confound(dimension=_CONFOUND, kind="served_model")]

    def test_the_catalog_says_why(self) -> None:
        assert "floating alias" in self._bundle().confound_catalog[_CONFOUND]

    def test_a_code_only_report_says_the_arm_mixes_two_models(self) -> None:
        report = build_code_only_report(
            self._bundle(), measures=toyhost_profile().measures, assembled_at="2026-10-09T00:00:00+00:00"
        )
        said = [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]
        assert [text for text in said if _MARCH in text and _JUNE in text and "mix" in text]


class TestOneAliasAnsweredByOneModel:
    """The alias held still: nothing to disclose, and nothing confounded."""

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
