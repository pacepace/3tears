"""What an evidence row carries, and the numbers a reading resolves to against the campaign that reported it.

An evidence row (``EvidenceRow``) names a cell and a reading and carries the number code filled, its sample size and
its spread; a reading resolves only where the campaign's decision surface holds it. The authored finding these rows
once backed is the retired memo writer's, held as JSON and never read, so nothing here builds one.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.errors import UnresolvableReference
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, toyhost_bundle
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION
from packages.evals.tests.toyhost_cells import cell_at
from packages.evals.tests.writer_host_support import doc_analysis_over
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.references import ResolvedReading, resolve_reading
from threetears.evals.kernel.campaign import EvidenceRow, ReadingKind


class TestEvidenceRowsCarryTheirBasis:
    def test_a_row_with_no_cell_is_refused(self):
        with pytest.raises(ValidationError, match="cell_ref"):
            EvidenceRow(
                aspect="conversation", measure_id="cost_usd", reading="measure", value=0.5, n=3, dispersion="±0.1"
            )

    def test_a_cell_keyed_row_is_legitimate(self):
        row = EvidenceRow(
            aspect="conversation",
            cell_ref="v:a",
            measure_id="cost_usd",
            reading="measure",
            value=0.5,
            n=3,
            dispersion="±0.1",
        )
        assert row.cell_ref == "v:a"

    def test_an_empty_cell_ref_is_refused_rather_than_passing_for_a_coordinate(self):
        with pytest.raises(ValidationError):
            EvidenceRow(
                aspect="conversation",
                cell_ref="",
                measure_id="cost_usd",
                reading="measure",
                value=0.5,
                n=3,
                dispersion="±0.1",
            )

    @pytest.mark.parametrize("missing", ["n", "dispersion", "reading"])
    def test_n_dispersion_and_the_reading_kind_are_required(self, missing):
        fields = {
            "cell_ref": "v:a",
            "measure_id": "cost_usd",
            "reading": "measure",
            "value": 0.5,
            "n": 3,
            "dispersion": "±0.1",
        }
        del fields[missing]
        with pytest.raises(ValidationError):
            EvidenceRow(**fields)


class TestEvidenceRowsAreCheckedAgainstTheCampaignThatReportedThem:
    """An evidence row is the number a reader checks the prose against.

    The row names a cell and a reading and code fills its number, so "does the campaign report
    this?" is answered by whether the reading RESOLVES — against the decision surface, which holds
    every measure and judged dimension the bundle measured in each cell. A reading naming anything else
    has no number to fill and is refused, naming what the cell does hold.

    Resolved through :func:`~threetears.evals.analysis.references.resolve_reading` over the surface a stored analysis
    of the toy host's campaign freezes (:func:`~threetears.evals.analysis.doc_analysis.aspect_analysis`).
    """

    @staticmethod
    def _bundle() -> Any:
        profile = toyhost_profile()
        return toyhost_bundle(profile=profile)

    @staticmethod
    def _cell(bundle: Any) -> Any:
        """The facts of the cell the readings below name, read off the bundle the analysis froze."""
        ref = cell_at(bundle, TOYHOST_NARROW)
        (cell,) = [cell for cell in bundle.cell_measures if cell_ref(cell.variant_key, cell.apparatus_class_id) == ref]
        return cell

    def _resolve(self, measure: str, reading: ReadingKind = "measure") -> ResolvedReading:
        bundle = self._bundle()
        analysis, _ = doc_analysis_over(campaign_bundle(bundle))
        (aspect,) = analysis.aspects
        return resolve_reading(aspect.decision_surface, cell_at(bundle, TOYHOST_NARROW), measure, reading)

    def test_a_measure_the_campaign_reported_survives_with_its_number_filled(self):
        bundle = self._bundle()
        (reported,) = [m for m in self._cell(bundle).measures.measures if m.name == "total_ms"]

        row = self._resolve("total_ms")

        assert (row.cell_ref, row.measure_id, row.reading, row.mean, row.n) == (
            cell_at(bundle, TOYHOST_NARROW),
            "total_ms",
            "measure",
            reported.mean,
            reported.n,
        )
        assert row.dispersion, "a resolved number carries its spread"

    def test_a_measure_the_campaign_never_reported_is_refused(self):
        with pytest.raises(UnresolvableReference, match="no such measure"):
            self._resolve("invented_metric")

    def test_a_judged_dimension_the_campaign_scored_survives_as_evidence(self):
        """Quality is reportable: a judged score is a measurement the bundle carries, off the ranking surface."""
        (scored,) = [j for j in self._cell(self._bundle()).judged if j.dimension == TOYHOST_JUDGED_DIMENSION]

        row = self._resolve(TOYHOST_JUDGED_DIMENSION, "judged")

        assert (row.mean, row.reading) == (scored.mean, "judged")

    def test_a_judged_dimension_the_campaign_never_scored_is_refused(self):
        with pytest.raises(UnresolvableReference, match="no such dimension"):
            self._resolve("tone", "judged")
