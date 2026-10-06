"""What a stored analysis records about where its numbers came from, over an analysis with no host vocabulary.

The subject is an invoice extractor swept over two models. The analysis, its surface and its one
finding are built here by hand. Nothing is imported from a host adapter, and the toy corpus is not
used, because these are the stored models' own rules and none of them reads a bundle.

Pinned here, from ``contracts/campaign.py`` and ``contracts/authored.py``:

- confidence is a tier, on the authored finding and decision alike; an evidence row names the
  reading it resolved; a chart keeps the reference it was compiled from.
- a decision whose proposal opens with a negation is stored as written — polarity lives in
  ``disposition``.
"""

from __future__ import annotations

from typing import Any, get_args

import pytest
from pydantic import ValidationError

from threetears.evals.contracts import authored
from threetears.evals.contracts.analysis_measures import MeasureCollection, MeasureSummary
from threetears.evals.contracts.authored import AuthoredAnalysis, Chart, Decision, Finding
from threetears.evals.contracts.campaign import (
    CONFIDENCE_TIERS,
    EvalAnalysis,
    EvidenceRow,
    FindingResolution,
    GenerationProvenance,
    Viz,
)
from threetears.evals.contracts.declaration import CampaignDesign, ControlDeclaration, SweptAxis
from threetears.evals.contracts.host.values import SweepableValue
from threetears.evals.contracts.identity import compute_variant_key
from threetears.evals.contracts.surface import CellFacts, DecisionSurface, MeasureFacts


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
            )
            for key in sorted([BASELINE, CHALLENGER])
        ],
        measures={"extract_ms": MeasureFacts(unit="ms", merit_axis="latency", higher_is_better=False)},
    )


def _decision(proposal: str = "Adopt model-b.", *, confidence: Any = "high", disposition: str = "adopted") -> Decision:
    return Decision(
        proposal=proposal,
        disposition=disposition,
        cells=[f"{CHALLENGER}:{RIG}"],
        confidence=confidence,
        rests_on=[0],
        revisit_when="",
    )


def _analysis(*, confidence: Any = "high", decision: Decision | None = None, **fields: Any) -> EvalAnalysis:
    finding = Finding(
        title="model-b extracts faster.",
        body="",
        confidence=confidence,
        axes=[AXIS],
        evidence=[],
        chart=Chart(type="none", cells=[], measures=[], axis="", note="", caption=""),
        caveats=[],
        invalidates=[],
        durable="",
    )
    document = AuthoredAnalysis(
        headline="Adopt model-b.",
        summary="",
        findings=[finding],
        decisions=[decision or _decision()],
        questions=[],
        next=[],
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
        ),
        document=document,
        resolutions=[FindingResolution()],
        design_snapshot=CampaignDesign(
            axes=[SweptAxis(axis_id=AXIS, values=[_level("model-a"), _level("model-b")])],
            control=BASELINE,
            controls=ControlDeclaration(stimulus="controlled", apparatus="commissioned"),
        ),
        **{"decision_surface": DecisionSurface(), **fields},
    )


class TestConfidenceIsATier:
    @pytest.mark.parametrize("tier", CONFIDENCE_TIERS)
    def test_every_tier_is_accepted(self, tier: str) -> None:
        analysis = _analysis(confidence=tier, decision=_decision(confidence=tier))
        assert analysis.document.findings[0].confidence == tier
        assert analysis.document.decisions[0].confidence == tier

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
            _decision(confidence=bad)


class TestTheReferenceFieldsAreStored:
    def test_an_evidence_row_says_which_reading_it_is(self) -> None:
        # No default: the finding's evidence tier is read off this field, so a default would decide it.
        row = {
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

    def test_a_chart_keeps_the_reference_it_was_compiled_from(self) -> None:
        ref = {
            "a_cell": f"{BASELINE}:{RIG}",
            "b_cell": f"{CHALLENGER}:{RIG}",
            "measures": [{"measure_id": "extract_ms"}],
        }
        viz = Viz.model_validate({"type": "delta_table", "ref": ref, "payload": {"rows": []}})

        assert Viz.model_validate(viz.model_dump(mode="json")).ref == ref
        # Every stored chart was compiled from a reference, so a chart without one is refused.
        with pytest.raises(ValidationError, match=r"ref\s+Field required"):
            Viz(type="delta_table")


class TestADecisionsWordingIsNotRefused:
    """Polarity lives in ``disposition``; what the proposal says in prose is the reporter eval's to judge."""

    @pytest.mark.parametrize(
        "proposal", ["Do not adopt model-b.", "Never ship model-a to the intake queue.", "Don't widen the sweep."]
    )
    def test_a_proposal_that_opens_with_a_negation_is_stored_as_written(self, proposal: str) -> None:
        analysis = _analysis(decision=_decision(proposal, disposition="rejected"))
        assert analysis.document.decisions[0].proposal == proposal
