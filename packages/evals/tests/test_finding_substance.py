"""What a finding IS — what it invalidates, what qualifies it, and the numbers its evidence resolves to.

These pin the half of the authored document that carries the substance: the claim, what it
invalidates, the class of what qualifies it, and the readings code turns into numbers. Where each
declared question stands is a separate surface with its own tests; nothing here asserts about it.
"""

from __future__ import annotations

import itertools
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
from threetears.evals.kernel.authored import Caveat, Finding, OffVocabulary, validate_authored
from threetears.evals.kernel.campaign import ENGINE_CAVEAT_KINDS, EvalAnalysis, EvidenceRow, ReadingKind
from packages.evals.tests.factories import empty_document, make_analysis


def _finding(*, invalidates: list[int] | None = None, **overrides) -> Finding:
    """A schema-valid authored finding, with every required field supplied."""
    defaults = {
        "title": "rev2 holds tone at k=3.",
        "body": "",
        "confidence": "high",
        "axes": ["candidate_model"],
        "evidence": [],
        "trials": [],
        "caveats": [],
        "invalidates": invalidates or [],
        "durable": "",
    }
    return Finding(**{**defaults, **overrides})


def _analysis(findings: list[Finding]) -> EvalAnalysis:
    """An analysis carrying ``findings`` and nothing that links to them."""
    return make_analysis(document=empty_document(findings=findings))


class TestGatingIsAPartitionAndSoNeedsAStrictOrder:
    """``invalidates`` orders the reading: a finding is read before those it invalidates."""

    def test_an_invalidation_naming_no_finding_is_refused(self):
        with pytest.raises(ValidationError, match=r"findings\[0\]\.invalidates names finding position\(s\) \[4\]"):
            _analysis([_finding(invalidates=[4]), _finding()])

    def test_a_two_node_cycle_is_refused(self):
        with pytest.raises(ValidationError, match="cycle"):
            _analysis([_finding(invalidates=[1]), _finding(invalidates=[0])])

    def test_a_longer_cycle_is_refused(self):
        with pytest.raises(ValidationError, match="cycle"):
            _analysis([_finding(invalidates=[1]), _finding(invalidates=[2]), _finding(invalidates=[0])])

    def test_a_diamond_is_not_a_cycle(self):
        analysis = _analysis(
            [_finding(invalidates=[1, 2]), _finding(invalidates=[3]), _finding(invalidates=[3]), _finding()]
        )
        assert analysis.document.findings[0].invalidates == [1, 2]

    def test_a_finding_invalidating_itself_is_refused(self):
        with pytest.raises(ValidationError, match="invalidates itself"):
            _analysis([_finding(invalidates=[0])])


class TestTheGateGraphIsCheckedExhaustivelyRatherThanByExample:
    """The cycle check decides whether a paid generation is kept, in both directions.

    A false refusal discards an analysis that was already billed; a missed cycle stores a gate
    graph that orders nothing, and every consumer downstream assumes the order exists. Neither
    direction is something a handful of hand-written shapes covers, and the space is small enough
    to close: four nodes has 64 distinct acyclic graphs and three nodes has 64 directed graphs.
    """

    @staticmethod
    def _analysis_over(nodes: int, edges: dict[int, list[int]]) -> EvalAnalysis:
        return _analysis([_finding(invalidates=edges.get(node, [])) for node in range(nodes)])

    def _refuses(self, nodes: int, edges: dict[int, list[int]]) -> bool:
        try:
            self._analysis_over(nodes, edges)
        except ValidationError as exc:
            return "cycle" in str(exc)
        return False

    def test_no_acyclic_gate_graph_is_ever_refused(self):
        """Every DAG on four nodes. Edges point forward only, so acyclicity is by construction."""
        forward = list(itertools.combinations(range(4), 2))
        for mask in range(1 << len(forward)):
            edges: dict[int, list[int]] = {}
            for bit, (a, b) in enumerate(forward):
                if mask >> bit & 1:
                    edges.setdefault(a, []).append(b)
            assert not self._refuses(4, edges), f"an acyclic gate graph was refused as a cycle: {edges}"

    def test_no_acyclic_gate_graph_is_refused_when_its_edges_point_backward_either(self):
        """The same DAGs reversed, so an order the check happens to walk in cannot hide a false refusal."""
        forward = list(itertools.combinations(range(4), 2))
        for mask in range(1 << len(forward)):
            edges: dict[int, list[int]] = {}
            for bit, (a, b) in enumerate(forward):
                if mask >> bit & 1:
                    edges.setdefault(b, []).append(a)
            assert not self._refuses(4, edges), f"an acyclic gate graph was refused as a cycle: {edges}"

    def test_every_cyclic_gate_graph_is_refused(self):
        """Every directed graph on three nodes that contains a cycle, found independently."""
        directed = [(a, b) for a in range(3) for b in range(3) if a != b]
        checked = 0
        for mask in range(1 << len(directed)):
            edges: dict[int, list[int]] = {}
            for bit, (a, b) in enumerate(directed):
                if mask >> bit & 1:
                    edges.setdefault(a, []).append(b)
            if not _contains_a_cycle(edges):
                continue
            checked += 1
            assert self._refuses(3, edges), f"a cyclic gate graph was accepted: {edges}"
        assert checked, "the cycle enumeration found nothing to check, so this asserts nothing"


def _contains_a_cycle(edges: dict[int, list[int]]) -> bool:
    """Whether ``edges`` has a cycle — written independently of the validator under test.

    A recursive walk over at most three nodes, deliberately not sharing an implementation with
    :meth:`EvalAnalysis.check_positions`: an oracle that reuses the code it judges agrees with it
    by construction and proves nothing.
    """
    finished: set[int] = set()
    on_path: set[int] = set()

    def walk(node: int) -> bool:
        if node in on_path:
            return True
        if node in finished:
            return False
        on_path.add(node)
        found = any(walk(nxt) for nxt in edges.get(node, []))
        on_path.discard(node)
        finished.add(node)
        return found

    return any(walk(node) for node in list(edges))


class TestACaveatSaysWhichClassOfQualificationItIs:
    """A caveat is a class and a sentence, and the class is one the host offered."""

    def test_a_caveat_with_no_text_is_refused(self):
        with pytest.raises(ValidationError, match="text"):
            Caveat.model_validate({"kind": "sampling"})

    def test_a_caveat_with_no_kind_is_refused(self):
        """An unclassified caveat is the state the classification exists to make unrepresentable."""
        with pytest.raises(ValidationError, match="kind"):
            Caveat.model_validate({"text": "k=1"})

    @staticmethod
    def _payload(kind: str) -> dict:
        finding = _finding(caveats=[Caveat(kind=kind, text="Two raters only.")])
        return {
            "findings": [finding.model_dump()],
            "recommendation": [
                {
                    "arm": "control",
                    "reasons": [],
                    "confidence": "low",
                    "rests_on": [0],
                    "floors_overridden": [],
                    "behind_on": [],
                    "bars_missed": [],
                }
            ],
            "aspect_summaries": [],
            "summary": "Nothing here moves the control.",
            "next": [],
        }

    def test_a_caveats_kind_is_open_to_a_host_beyond_the_engines_four(self):
        """A host registers its own kinds; one it offered is accepted like the engine's."""
        offered = {*ENGINE_CAVEAT_KINDS, "rater_pool"}
        document = validate_authored(self._payload("rater_pool"), offered)
        assert document.findings[0].caveats[0].kind == "rater_pool"

    def test_a_kind_nobody_offered_is_refused_and_named(self):
        with pytest.raises(OffVocabulary, match=r"findings\[0\]\.caveats\[0\]\.kind 'rater_pool'"):
            validate_authored(self._payload("rater_pool"), ENGINE_CAVEAT_KINDS)


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
