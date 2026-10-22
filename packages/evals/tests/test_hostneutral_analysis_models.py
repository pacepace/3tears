"""What a stored analysis records about where its numbers came from, over an analysis with no host vocabulary.

The subject is an invoice extractor swept over two models. The analysis, its surface and its one
finding are built here by hand, over an empty frozen bundle. Nothing is imported from a host adapter, and the toy
corpus is not used, because these are the stored models' own rules.

Pinned here, from ``kernel/campaign.py`` and ``kernel/authored.py``:

- confidence is a tier, on the authored finding and recommendation alike; an evidence row names the
  aspect and the reading it resolved; the analysis keeps each aspect's evidence bundle it was written over.
- a recommendation whose reason opens with a negation is stored as written — which arm it recommends lives in
  ``arm``.
"""

from __future__ import annotations

from typing import Any, get_args

import pytest
from pydantic import ValidationError

from threetears.evals.kernel import authored
from threetears.evals.kernel.analysis_measures import MeasureCollection, MeasureSummary
from threetears.evals.kernel.authored import AuthoredCampaignAnalysis, Finding, Recommendation
from threetears.evals.kernel.campaign import (
    CONFIDENCE_TIERS,
    CampaignDeclaration,
    EvalAnalysis,
    EvidenceRow,
    FindingResolution,
    GenerationProvenance,
)
from threetears.evals.kernel.declaration import Aspect
from threetears.evals.schema.values import SweepableValue
from threetears.evals.kernel.identity import compute_variant_key
from threetears.evals.kernel.surface import CellFacts, DecisionSurface, MeasureFacts
from packages.evals.tests.factories import CURRENT_BUNDLE_SHAPE, analysed_aspect, empty_bundle


AXIS = "extractor_model"
RIG = "r" * 64


def _level(model: str) -> SweepableValue:
    return SweepableValue.of(model, display=model)


def _key(model: str) -> str:
    return compute_variant_key({AXIS: _level(model)})


BASELINE, CHALLENGER = _key("model-a"), _key("model-b")


def _surface() -> DecisionSurface:
    return DecisionSurface(
        control_variant_key=BASELINE,
        cells=[
            CellFacts(
                variant_key=key,
                apparatus_class_id=RIG,
                run_ids=[f"run-{key[:6]}"],
                n_observations=4,
                measures=MeasureCollection(
                    measures=[
                        MeasureSummary(
                            population="scored",
                            name="extract_ms",
                            attribution_scope="end_to_end",
                            n=4,
                            n_independent=4,
                            mean=900.0,
                        )
                    ]
                ),
                n_candidate_failed=0,
                n_no_turn=0,
            )
            for key in sorted([BASELINE, CHALLENGER])
        ],
        measures={"extract_ms": MeasureFacts(unit="ms", merit_axis="latency", higher_is_better=False)},
    )


def _recommendation(reason: str = "model-b is faster.", *, confidence: Any = "high") -> Recommendation:
    return Recommendation(
        arm="model-b",
        reasons=[reason],
        confidence=confidence,
        rests_on=[0],
        floors_overridden=[],
        behind_on=[],
        bars_missed=[],
    )


def _analysis(*, confidence: Any = "high", recommendation: Recommendation | None = None, **fields: Any) -> EvalAnalysis:
    finding = Finding(
        title="model-b extracts faster.",
        body="",
        confidence=confidence,
        axes=[AXIS],
        evidence=[],
        trials=[],
        caveats=[],
        invalidates=[],
        durable="",
    )
    document = AuthoredCampaignAnalysis(
        findings=[finding],
        recommendation=[recommendation or _recommendation()],
        aspect_summaries=[],
        summary="One finding, and nothing yet argues for a switch.",
        next=[],
    )
    aspect = analysed_aspect(
        empty_bundle(campaign_id="camp-invoices", behavior="extraction", aspect="extraction"),
        decision_surface=_surface(),
    )
    return EvalAnalysis(
        scope_id="uni-1",
        campaign_id="camp-invoices",
        subject_id="invoice-extractor",
        subject_kind="extractor_config",
        behavior="extraction",
        generation=GenerationProvenance(
            prompt_id="gen",
            prompt_version="v1",
            generator_model="gen/model",
            bundle_fingerprint="sha256:0",
            generated_at="2026-01-01T00:00:00+00:00",
            token_cost=0.0,
            bundle_assembled_at="2026-01-01T00:00:00+00:00",
            repair_attempts=0,
            repaired_refusal=None,
            cell_model_version=1,
            user_message_digest="sha256:message",
            bundle_schema_version=CURRENT_BUNDLE_SHAPE,
            host_declarations_digest="sha256:declarations",
        ),
        document=document,
        resolutions=[FindingResolution()],
        recommended_variant_key="variant-of-model-b",
        aspects=[aspect],
        campaign_bundle={
            "schema_version": 1,
            "aspects": [{"aspect": "extraction", "fingerprint": aspect.bundle_fingerprint}],
        },
        design_snapshot=CampaignDeclaration(
            question="Which extractor model?",
            declared_at=None,
            declared_by=None,
            arms=[],
            aspects=[Aspect(label="extraction")],
            floors=[],
        ),
        **fields,
    )


class TestConfidenceIsATier:
    @pytest.mark.parametrize("tier", CONFIDENCE_TIERS)
    def test_every_tier_is_accepted(self, tier: str) -> None:
        analysis = _analysis(confidence=tier, recommendation=_recommendation(confidence=tier))
        assert analysis.document.findings[0].confidence == tier
        assert analysis.document.recommendation[0].confidence == tier

    def test_the_tiers_are_the_four_named_ones_in_order(self) -> None:
        assert CONFIDENCE_TIERS == ("very_high", "high", "medium", "low")

    def test_the_authored_contract_offers_exactly_the_stored_tiers(self) -> None:
        """The contract sent to the generator and the tiers every surface ranks by are one set."""
        assert set(get_args(authored.Confidence)) == set(CONFIDENCE_TIERS)

    @pytest.mark.parametrize(
        "bad", ["certain", 0.8, 1.5], ids=["a-word-that-is-no-tier", "a-probability", "out-of-range"]
    )
    def test_anything_but_a_tier_is_refused(self, bad: Any) -> None:
        with pytest.raises(ValidationError):
            _analysis(confidence=bad)
        with pytest.raises(ValidationError):
            _recommendation(confidence=bad)


class TestTheReferenceFieldsAreStored:
    def test_an_evidence_row_says_which_reading_it_is(self) -> None:
        # No default: the finding's evidence tier is read off this field, so a default would decide it.
        row = {
            "aspect": "extraction",
            "cell_ref": f"{BASELINE}:{RIG}",
            "measure_id": "extract_ms",
            "value": 900.0,
            "n": 4,
            "dispersion": "sem 5",
        }

        with pytest.raises(ValidationError, match="reading"):
            EvidenceRow.model_validate(row)
        assert EvidenceRow.model_validate({**row, "reading": "measure"}).reading == "measure"
        judged = {**row, "reading": "judged", "judged_tier": "undetermined"}
        assert EvidenceRow.model_validate(judged).reading == "judged"
        with pytest.raises(ValidationError):
            EvidenceRow.model_validate({**row, "reading": "guessed"})

    def test_the_analysis_keeps_each_aspect_s_bundle_it_was_written_over(self) -> None:
        analysis = _analysis()
        reread = EvalAnalysis.model_validate(analysis.model_dump(mode="json"))
        assert reread.aspects[0].bundle == analysis.aspects[0].bundle
        assert reread.campaign_bundle == analysis.campaign_bundle
        # Every stored analysis was written over its aspects' bundles, so an aspect without one is refused.
        without = analysis.model_dump(mode="json")
        del without["aspects"][0]["bundle"]
        with pytest.raises(ValidationError, match=r"bundle\s+Field required"):
            EvalAnalysis.model_validate(without)


class TestARecommendationsWordingIsNotRefused:
    """Which arm is recommended lives in ``arm``; what its reasons say in prose is the reporter eval's to judge."""

    @pytest.mark.parametrize(
        "reason", ["Do not keep model-a.", "Never ship model-a to the intake queue.", "Don't widen the sweep."]
    )
    def test_a_reason_that_opens_with_a_negation_is_stored_as_written(self, reason: str) -> None:
        analysis = _analysis(recommendation=_recommendation(reason))
        assert analysis.document.recommendation[0].reasons == [reason]
