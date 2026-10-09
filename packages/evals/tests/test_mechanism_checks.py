"""Whether a swept lever took effect, and whether two arms did the same thing — read off what the runs measured.

Two defects of one shape (#577, #576): a comparison's outcome numbers cannot tell a reader why nothing moved,
or whether the arms compared were held alike in what they actually did.

* **A lever that never took effect reads as a clean null.** A lever may declare the measure or covariate it acts
  on (``Sweepable.acts_on``), and each coverage row checks it: ``moved``, ``inert`` (the measure took the same
  mean at every level) or ``unchecked`` with the reason. The toy host's ``chunk_tokens`` acts on the engine's
  ``context_tokens_in`` covariate.
* **One reasoning-effort word does not hold reasoning constant across models.** Each arm's mean reasoning share
  (``reasoning_ratio``) is read onto the bundle, and a comparison whose levels' shares are at least
  ``REASONING_SHARE_DIVERGENCE`` apart names an ``observed_mechanism`` confound carrying the threshold and both
  values. It qualifies the comparison; it suppresses nothing.

Every campaign here is the toy host's, built through its own corpus builders.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import (
    AnalysisContextBundle,
    Confound,
    LeverCoverageInput,
    MechanismCheck,
    assemble_context_bundle,
)
from threetears.evals.analysis.bundle import REASONING_SHARE_DIVERGENCE, observed_mechanism_key
from threetears.evals.contracts import REASONING_RATIO_KEY, EvalCampaign, EvalRun, resolve_variant_identity
from threetears.evals.contracts.host import (
    SHARED_CORE,
    HostProfile,
    ProfileRegistrationError,
    RegistrationError,
    Sweepable,
)
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.sweepables import TOYHOST_ROLES, TOYHOST_SWEEPABLES

_CONTEXT = "context_tokens_in"

#: Two models the toy extractor could run on. Neither name means anything to the engine.
_MODEL_A = "extractor-alpha"
_MODEL_B = "extractor-beta"


def _profile(*, acts_on: str | None = _CONTEXT) -> HostProfile:
    """The toy host's profile with ``chunk_tokens`` declaring ``acts_on`` (``None`` declares no mechanism)."""
    declarations = tuple(
        replace(declared, acts_on=acts_on) if declared.name == "chunk_tokens" else declared
        for declared in TOYHOST_SWEEPABLES
    )
    return replace(toyhost_profile(), host_sweepables=SHARED_CORE.extend(declarations, roles=TOYHOST_ROLES))


def _chunk_batch(chunk_tokens: int) -> EvalRun:
    """One toy batch at a chunk width, its rig held at the fixture's defaults."""
    return toyhost_batch(
        chunk_tokens=chunk_tokens,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
    )


def _model_batch(model: str) -> EvalRun:
    """One toy batch on ``model``, every toy-host setting held — the model is the only lever that moves."""
    batch = _chunk_batch(512)
    return batch.model_copy(update={"candidate_model": model, "id": str(uuid.uuid5(uuid.UUID(batch.id), model))})


def _bundle(
    batches: Sequence[tuple[EvalRun, Mapping[str, float] | None]], *, profile: HostProfile | None = None
) -> AnalysisContextBundle:
    """Assemble a campaign over ``batches``, each with the covariates every one of its observations recorded."""
    host = profile if profile is not None else _profile()
    storage = ToyhostStorage(
        [batch for batch, _covariates in batches],
        {
            batch.id: toyhost_measurements(
                batch, profile=host, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8, covariates=covariates
            )
            for batch, covariates in batches
        },
    )
    campaign = EvalCampaign(
        id="5f0c3b9e-7d61-4a2b-9c84-3e1f6a2d7b90",
        scope_id=TOYHOST_SCOPE,
        name="mechanism",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[batch.id for batch, _covariates in batches],
        created_by="test:fixture",
    )
    return assemble_context_bundle(campaign, storage=storage, profile=host)


def _row(bundle: AnalysisContextBundle, lever: str) -> LeverCoverageInput:
    (row,) = [entry for entry in bundle.coverage if entry.name == lever]
    return row


def _chunk_sweep(
    narrow: Mapping[str, float] | None, wide: Mapping[str, float] | None, *, profile: HostProfile | None = None
) -> AnalysisContextBundle:
    """A chunk-width sweep at 256 and 1024 tokens, each batch recording the given covariates."""
    return _bundle([(_chunk_batch(256), narrow), (_chunk_batch(1024), wide)], profile=profile)


class TestALeverIsCheckedAgainstTheMechanismItDeclares:
    """#577: a lever whose declared mechanism held still never took effect, which is not a null effect."""

    def test_the_declared_campaign_moved_its_mechanism(self) -> None:
        mechanism = _row(toyhost_bundle(), "chunk_tokens").mechanism
        assert mechanism == MechanismCheck(state="moved", measure=_CONTEXT, level_means={"256": 2400.0, "1024": 7600.0})

    def test_a_mechanism_identical_at_every_level_is_inert(self) -> None:
        mechanism = _row(_chunk_sweep({_CONTEXT: 3000}, {_CONTEXT: 3000}), "chunk_tokens").mechanism
        assert mechanism.state == "inert"
        assert mechanism.level_means == {"256": 3000.0, "1024": 3000.0}
        assert mechanism.reason is None

    def test_the_same_sweep_with_the_mechanism_moving_is_moved(self) -> None:
        mechanism = _row(_chunk_sweep({_CONTEXT: 3000}, {_CONTEXT: 3001}), "chunk_tokens").mechanism
        assert mechanism.state == "moved"
        assert mechanism.level_means == {"256": 3000.0, "1024": 3001.0}

    def test_means_that_are_equal_as_written_are_equal(self) -> None:
        # Three observations of 0.1 sum to 0.30000000000000004 in floats; their mean is still 0.1.
        bundle = _bundle(
            [(_chunk_batch(256), {_CONTEXT: 0.1}), (_chunk_batch(1024), {_CONTEXT: 0.1})],
        )
        assert _row(bundle, "chunk_tokens").mechanism.state == "inert"

    def test_a_lever_declaring_no_mechanism_is_unchecked_not_effective(self) -> None:
        bundle = _chunk_sweep({_CONTEXT: 3000}, {_CONTEXT: 9000}, profile=_profile(acts_on=None))
        assert _row(bundle, "chunk_tokens").mechanism == MechanismCheck(state="unchecked", reason="not_declared")

    def test_a_mechanism_observed_at_one_level_only_is_unchecked(self) -> None:
        mechanism = _row(_chunk_sweep({_CONTEXT: 3000}, None), "chunk_tokens").mechanism
        assert mechanism.state == "unchecked"
        assert mechanism.reason == "levels_unobserved"
        assert mechanism.level_means == {"256": 3000.0}
        assert mechanism.unobserved_levels == ["1024"]

    def test_the_candidate_model_lever_declares_no_mechanism(self) -> None:
        bundle = _bundle([(_model_batch(_MODEL_A), None), (_model_batch(_MODEL_B), None)])
        assert _row(bundle, "model").mechanism.reason == "not_declared"


class TestAnUnresolvableMechanismIsRefusedWhereItIsDeclared:
    def test_a_name_no_registry_declares_is_refused_at_the_profile(self) -> None:
        with pytest.raises(ProfileRegistrationError, match="no_such_measure"):
            _profile(acts_on="no_such_measure")

    def test_a_non_numeric_measure_is_refused_at_the_profile(self) -> None:
        with pytest.raises(ProfileRegistrationError, match="execution_mode"):
            _profile(acts_on="execution_mode")

    def test_a_host_measure_is_accepted(self) -> None:
        declared = _profile(acts_on="fields_stripped").sweepables.get("chunk_tokens")
        assert declared is not None
        assert declared.acts_on == "fields_stripped"

    def test_a_blank_name_is_refused_at_registration(self) -> None:
        with pytest.raises(RegistrationError, match="blank acts_on"):
            SHARED_CORE.extend(
                [Sweepable(name="knob", role="lever", read=lambda _r, _s: 1, reader_prose="a knob", acts_on=" ")]
            )

    def test_an_apparatus_input_is_refused_at_registration(self) -> None:
        with pytest.raises(RegistrationError, match="only a swept lever"):
            SHARED_CORE.extend(
                [
                    Sweepable(
                        name="rig",
                        role="apparatus",
                        read=lambda _r, _s: 1,
                        reader_prose="the rig",
                        confounds="a different rig measured it",
                        acts_on=_CONTEXT,
                    )
                ]
            )

    def test_an_open_family_is_refused_at_registration(self) -> None:
        with pytest.raises(RegistrationError, match="open family and declares acts_on"):
            SHARED_CORE.extend(
                [
                    Sweepable(
                        name="knobs",
                        role="lever",
                        read=lambda _r, _s: {},
                        reader_prose="any knob",
                        open_family="a launch may overlay any knob",
                        acts_on=_CONTEXT,
                    )
                ]
            )


class TestAMechanismDeclarationIsNoIdentityInput:
    """What to check is not what a run ran under: declaring it moves no variant key."""

    def test_variant_keys_are_the_same_with_and_without_the_declaration(self) -> None:
        batch = _chunk_batch(256)
        declared = resolve_variant_identity(run=batch, profile=_profile())
        undeclared = resolve_variant_identity(run=batch, profile=_profile(acts_on=None))
        assert declared.variant_key == undeclared.variant_key

    def test_the_bundle_pools_the_same_arms_with_and_without_it(self) -> None:
        with_it = _chunk_sweep({_CONTEXT: 3000}, {_CONTEXT: 9000})
        without = _chunk_sweep({_CONTEXT: 3000}, {_CONTEXT: 9000}, profile=_profile(acts_on=None))
        assert [entry.variant_key for entry in with_it.variant_index] == [
            entry.variant_key for entry in without.variant_index
        ]
        assert [(cell.variant_key, cell.apparatus_class_id) for cell in with_it.cells] == [
            (cell.variant_key, cell.apparatus_class_id) for cell in without.cells
        ]


class TestTheMechanismCheckStatesOnlyWhatItsEvidenceDecides:
    @pytest.mark.parametrize(
        "fields",
        [
            {"state": "unchecked"},
            {"state": "moved", "measure": _CONTEXT, "level_means": {"a": 1.0, "b": 2.0}, "reason": "not_swept"},
            {"state": "moved", "measure": None, "level_means": {"a": 1.0, "b": 2.0}},
            {"state": "unchecked", "measure": None, "reason": "levels_unobserved"},
            {"state": "moved", "measure": _CONTEXT, "level_means": {"a": 1.0}},
            {"state": "inert", "measure": _CONTEXT, "level_means": {"a": 1.0, "b": 1.0}, "unobserved_levels": ["c"]},
        ],
    )
    def test_a_contradictory_check_is_refused(self, fields: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            MechanismCheck.model_validate(fields)


def _model_comparison(share_a: float | None, share_b: float | None) -> AnalysisContextBundle:
    return _bundle(
        [
            (_model_batch(_MODEL_A), None if share_a is None else {REASONING_RATIO_KEY: share_a}),
            (_model_batch(_MODEL_B), None if share_b is None else {REASONING_RATIO_KEY: share_b}),
        ]
    )


def _observed(bundle: AnalysisContextBundle) -> list[Confound]:
    return [confound for confound in _row(bundle, "model").confounded_by if confound.kind == "observed_mechanism"]


class TestAModelComparisonAtOneEffortDisclosesADivergentReasoningShare:
    """#576: two model arms at one reasoning-effort setting can reason very differently."""

    def test_very_different_shares_name_the_confound_with_threshold_and_values(self) -> None:
        bundle = _model_comparison(0.71, 0.30)
        assert _observed(bundle) == [
            Confound(
                dimension=observed_mechanism_key(REASONING_RATIO_KEY),
                kind="observed_mechanism",
                level_values={_MODEL_A: 0.71, _MODEL_B: 0.30},
                threshold=REASONING_SHARE_DIVERGENCE,
            )
        ]
        assert REASONING_SHARE_DIVERGENCE == 0.20
        reason = bundle.confound_catalog[observed_mechanism_key(REASONING_RATIO_KEY)]
        assert "reasoning" in reason

    def test_the_contrast_is_qualified_not_suppressed(self) -> None:
        row = _row(_model_comparison(0.71, 0.30), "model")
        assert row.levels == sorted([_MODEL_A, _MODEL_B])
        assert row.n == 72

    def test_shares_exactly_the_threshold_apart_are_named(self) -> None:
        # 0.7 - 0.5 is 0.19999999999999996 in floats; as written it is 0.2, which is "at least 0.2".
        assert len(_observed(_model_comparison(0.7, 0.5))) == 1

    def test_matched_shares_name_no_confound(self) -> None:
        assert _observed(_model_comparison(0.40, 0.45)) == []

    def test_each_arm_carries_its_share(self) -> None:
        bundle = _model_comparison(0.71, 0.30)
        means = {reading.mean for reading in bundle.arm_mechanisms}
        assert means == {0.71, 0.30}
        assert all(
            (reading.covariate, reading.n_measured, reading.n_results) == (REASONING_RATIO_KEY, 36, 36)
            for reading in bundle.arm_mechanisms
        )

    def test_a_share_never_measured_names_no_confound_and_says_it_is_unmeasured(self) -> None:
        bundle = _model_comparison(None, None)
        assert _observed(bundle) == []
        assert [(r.mean, r.n_measured, r.n_results) for r in bundle.arm_mechanisms] == [(None, 0, 36), (None, 0, 36)]

    def test_one_arm_unmeasured_is_compared_with_nothing(self) -> None:
        bundle = _model_comparison(0.95, None)
        assert _observed(bundle) == []
        assert sorted((r.mean is None, r.n_measured) for r in bundle.arm_mechanisms) == [(False, 36), (True, 0)]

    def test_the_declared_campaign_measured_no_share(self) -> None:
        campaign, _storage = toyhost_campaign()
        bundle = toyhost_bundle()
        assert len(bundle.arm_mechanisms) == len(campaign.run_ids)
        assert all(reading.mean is None for reading in bundle.arm_mechanisms)


def _divergent_model_comparison(share_a: float, share_b: float) -> AnalysisContextBundle:
    """Two model arms whose whole-run time moved while the time inside model calls held — a scope divergence."""
    host = _profile()
    batches = [(_model_batch(_MODEL_A), share_a, 900.0), (_model_batch(_MODEL_B), share_b, 2000.0)]
    results = {
        batch.id: [
            result.model_copy(
                update={"latency": result.latency.model_copy(update={"llm_ms": 500.0}) if result.latency else None}
            )
            for result in toyhost_measurements(
                batch,
                profile=host,
                cost_usd=0.02,
                total_ms=total_ms,
                field_accuracy=0.8,
                covariates={REASONING_RATIO_KEY: share},
            )
        ]
        for batch, share, total_ms in batches
    }
    campaign = EvalCampaign(
        id="9a7e2c14-3b58-4f06-a1d9-6c2b8e0f4d37",
        scope_id=TOYHOST_SCOPE,
        name="divergence",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[batch.id for batch, _share, _total in batches],
        created_by="test:fixture",
    )
    storage = ToyhostStorage([batch for batch, _share, _total in batches], results)
    return assemble_context_bundle(campaign, storage=storage, profile=host)


class TestAScopeDivergenceNamesTheSameConfound:
    """The divergence lens compares the same two levels, so it must name what the coverage row names."""

    def test_a_divergent_share_is_named_on_the_divergence(self) -> None:
        bundle = _divergent_model_comparison(0.71, 0.30)
        divergences = [d for d in bundle.scope_divergences if d.lever == "model"]
        assert divergences, "the fixture must produce a divergence, or this asserts nothing"
        for divergence in divergences:
            observed = [c for c in divergence.confounded_by if c.kind == "observed_mechanism"]
            assert [c.level_values for c in observed] == [{_MODEL_A: 0.71, _MODEL_B: 0.30}]

    def test_a_matched_share_names_nothing_there(self) -> None:
        bundle = _divergent_model_comparison(0.40, 0.45)
        divergences = [d for d in bundle.scope_divergences if d.lever == "model"]
        assert divergences, "the fixture must produce a divergence, or this asserts nothing"
        assert all(c.kind != "observed_mechanism" for d in divergences for c in d.confounded_by)


class TestAnObservedMechanismConfoundCarriesItsEvidence:
    def test_one_without_a_threshold_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Confound(dimension="observed:x", kind="observed_mechanism", level_values={"a": 0.1, "b": 0.9})

    def test_one_with_a_single_level_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Confound(dimension="observed:x", kind="observed_mechanism", level_values={"a": 0.1}, threshold=0.2)

    def test_an_undecided_one_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Confound(
                dimension="observed:x",
                kind="observed_mechanism",
                status="undecided",
                level_values={"a": 0.1, "b": 0.9},
                threshold=0.2,
            )

    @pytest.mark.parametrize("kind", ["swept_lever", "apparatus"])
    def test_another_kind_carries_no_evidence_fields(self, kind: str) -> None:
        with pytest.raises(ValidationError):
            Confound(dimension="x", kind=kind, threshold=0.2)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            Confound(dimension="x", kind=kind, level_values={"a": 1.0, "b": 2.0})  # type: ignore[arg-type]
