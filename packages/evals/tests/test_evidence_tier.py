"""A finding's evidence tier is code's, read off the evidence rows its readings resolved to."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from threetears.evals.contracts.campaign import EvidenceRow, FindingResolution, evidence_tier_of


def _row(reading: str) -> EvidenceRow:
    return EvidenceRow(cell_ref="k1:a1", measure_id="total_ms", reading=reading, value=1.0, n=5, dispersion="sd 0.1")


class TestTheTierIsReadOffTheRows:
    def test_measures_alone_are_mechanical(self):
        assert FindingResolution(evidence=[_row("measure"), _row("measure")]).evidence_tier == "mechanical"

    def test_any_judged_reading_makes_the_finding_directional(self):
        # Order-free: the weakest reading decides wherever it sits.
        assert FindingResolution(evidence=[_row("measure"), _row("judged")]).evidence_tier == "directional"
        assert FindingResolution(evidence=[_row("judged"), _row("measure")]).evidence_tier == "directional"

    def test_no_reading_stands_on_nothing(self):
        assert FindingResolution(evidence=[]).evidence_tier == "none"

    def test_the_derivation_names_no_tier_it_cannot_compute(self):
        import typing

        from threetears.evals.contracts.campaign import EvidenceTier

        assert set(typing.get_args(EvidenceTier)) == {"mechanical", "directional", "none"}
        assert {evidence_tier_of(kinds) for kinds in ([], ["measure"], ["judged"], ["measure", "judged"])} == set(
            typing.get_args(EvidenceTier)
        )


class TestTheTierCannotDisagreeWithItsRows:
    def test_a_stored_tier_is_re_derived_on_read_not_believed(self):
        # A stored document carries the tier it was serialised with; reading it back derives the tier
        # again from the rows, so a tampered or stale value cannot reach a reader.
        stored = json.loads(FindingResolution(evidence=[_row("judged")]).model_dump_json())
        assert stored["evidence_tier"] == "directional"
        stored["evidence_tier"] = "mechanical"
        assert FindingResolution.model_validate(stored).evidence_tier == "directional"

    def test_a_row_must_say_which_kind_of_reading_it_is(self):
        # The tier is decided by `reading`, so a default would decide it: a judged row that lost its
        # kind would read as mechanical.
        with pytest.raises(ValidationError, match="reading"):
            EvidenceRow(cell_ref="k1:a1", measure_id="total_ms", value=1.0, n=5, dispersion="sd 0.1")
