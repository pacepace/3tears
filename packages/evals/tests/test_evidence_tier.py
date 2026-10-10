"""A finding's evidence tier is code's, read off the evidence rows its readings resolved to."""

from __future__ import annotations

import json
import typing

import pytest
from pydantic import ValidationError

from threetears.evals.kernel.campaign import EvidenceRow, EvidenceTier, FindingResolution, evidence_tier_of
from threetears.evals.kernel.evidence_tiers import JUDGED_TIERS_WEAKEST_FIRST, JudgedEvidenceTier


def _row(reading: str, tier: JudgedEvidenceTier | None = None) -> EvidenceRow:
    return EvidenceRow(
        cell_ref="k1:a1",
        measure_id="total_ms",
        reading=reading,
        value=1.0,
        n=5,
        dispersion="sd 0.1",
        judged_tier=tier,
    )


class TestTheTierIsReadOffTheRows:
    def test_measures_alone_are_mechanical(self):
        assert FindingResolution(evidence=[_row("measure"), _row("measure")]).evidence_tier == "mechanical"

    @pytest.mark.parametrize("tier", typing.get_args(JudgedEvidenceTier))
    def test_a_judged_reading_brings_its_own_tier(self, tier: JudgedEvidenceTier):
        # Order-free: the judged row decides wherever it sits.
        assert FindingResolution(evidence=[_row("measure"), _row("judged", tier)]).evidence_tier == tier
        assert FindingResolution(evidence=[_row("judged", tier), _row("measure")]).evidence_tier == tier

    def test_several_judged_readings_stand_on_the_weakest(self):
        rows = [_row("judged", "calibrated"), _row("judged", "separation"), _row("measure")]
        assert FindingResolution(evidence=rows).evidence_tier == "separation"
        rows.append(_row("judged", "incidental"))
        assert FindingResolution(evidence=rows).evidence_tier == "incidental"

    def test_an_undetermined_reading_beside_a_stronger_one_is_undetermined(self):
        # Undetermined could still be incidental, so a composite holding it cannot claim more.
        assert evidence_tier_of(["calibrated", "undetermined"]) == "undetermined"
        assert evidence_tier_of(["separation", "undetermined"]) == "undetermined"

    def test_a_known_incidental_reading_beside_an_undetermined_one_is_incidental(self):
        # Incidental is the weakest judged tier, so it is what the pair bears whatever the other turns out to be.
        assert evidence_tier_of(["undetermined", "incidental"]) == "incidental"

    def test_no_reading_stands_on_nothing(self):
        assert FindingResolution(evidence=[]).evidence_tier == "none"

    def test_the_derivation_reaches_every_tier_it_names(self):
        reached = {evidence_tier_of([]), evidence_tier_of([None])} | {
            evidence_tier_of([tier]) for tier in JUDGED_TIERS_WEAKEST_FIRST
        }
        assert reached == set(typing.get_args(EvidenceTier))


class TestTheTierCannotDisagreeWithItsRows:
    def test_a_stored_tier_is_re_derived_on_read_not_believed(self):
        # A stored document carries the tier it was serialised with; reading it back derives the tier
        # again from the rows, so a tampered or stale value cannot reach a reader.
        stored = json.loads(FindingResolution(evidence=[_row("judged", "incidental")]).model_dump_json())
        assert stored["evidence_tier"] == "incidental"
        stored["evidence_tier"] = "mechanical"
        assert FindingResolution.model_validate(stored).evidence_tier == "incidental"

    def test_a_row_must_say_which_kind_of_reading_it_is(self):
        # The tier is decided by `reading`, so a default would decide it: a judged row that lost its
        # kind would read as mechanical.
        with pytest.raises(ValidationError, match="reading"):
            EvidenceRow(cell_ref="k1:a1", measure_id="total_ms", value=1.0, n=5, dispersion="sd 0.1")

    def test_a_judged_row_without_its_tier_is_refused(self):
        # It would compose as mechanical — a finding leaning on a judge reading as one that needs none.
        with pytest.raises(ValidationError, match="judged_tier"):
            _row("judged")

    def test_a_measure_row_carrying_a_tier_is_refused(self):
        # It would weaken a finding no judge touched.
        with pytest.raises(ValidationError, match="judged_tier"):
            _row("measure", "incidental")
