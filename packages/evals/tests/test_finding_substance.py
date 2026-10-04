"""What a finding IS — what it invalidates, what qualifies it, and the numbers its evidence resolves to.

These pin the half of the authored document that carries the substance: the claim, what it
invalidates, the class of what qualifies it, and the readings code turns into numbers. Where each
declared question stands is a separate surface with its own tests; nothing here asserts about it.
"""

from __future__ import annotations

import copy
import itertools
import json
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.errors import GenerationError, SoundnessRefusal, UnresolvableReference
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, toyhost_bundle
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION
from packages.evals.tests.toyhost_memo import (
    MODEL,
    PROMPT,
    PROMPT_ID,
    FixturedClient,
    alias_at,
    cell_at,
    memo_payload,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.contracts.authored import (
    AuthoredAnalysis,
    Caveat,
    Chart,
    Finding,
    OffVocabulary,
    validate_authored,
)
from threetears.evals.contracts.campaign import ENGINE_CAVEAT_KINDS, EvalAnalysis, EvidenceRow, GenerationProvenance
from threetears.evals.contracts.models import utc_now_iso
from threetears.evals.contracts.surface import DecisionSurface


_NO_CHART = Chart(type="none", cells=[], measures=[], axis="", note="", caption="")


def _finding(*, invalidates: list[int] | None = None, **overrides) -> Finding:
    """A schema-valid authored finding, with every required field supplied."""
    defaults = {
        "title": "rev2 holds tone at k=3.",
        "body": "",
        "confidence": "high",
        "axes": ["candidate_model"],
        "evidence": [],
        "chart": _NO_CHART,
        "caveats": [],
        "invalidates": invalidates or [],
        "durable": "",
    }
    return Finding(**{**defaults, **overrides})


def _analysis(findings: list[Finding]) -> EvalAnalysis:
    """An analysis carrying ``findings`` and nothing that links to them."""
    return EvalAnalysis(
        scope_id="uni-1",
        campaign_id="campaign-1",
        subject_id="ent-maple",
        subject_kind="agent",
        behavior="conversation",
        generation=GenerationProvenance(
            prompt_id="eval_analysis_gen",
            prompt_version="v1",
            generator_model="anthropic/claude-opus",
            bundle_fingerprint="sha256:abc",
            generated_at="2026-08-22T00:00:00+00:00",
            token_cost=0.0,
            bundle_assembled_at="2026-01-01T00:00:00+00:00",
            repair_attempts=0,
            repaired_refusal=None,
            cell_model_version=1,
            user_message_digest="sha256:message",
        ),
        document=AuthoredAnalysis(headline="h", summary="", findings=findings, decisions=[], questions=[], next=[]),
        decision_surface=DecisionSurface(),
    )


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
            "headline": "h",
            "summary": "",
            "findings": [finding.model_dump()],
            "decisions": [],
            "questions": [],
            "next": [],
        }

    def test_a_caveats_kind_is_open_to_a_host_beyond_the_engines_four(self):
        """A host registers its own kinds; one it offered is accepted like the engine's."""
        offered = {*ENGINE_CAVEAT_KINDS, "rater_pool"}
        document = validate_authored(self._payload("rater_pool"), offered, [])
        assert document.findings[0].caveats[0].kind == "rater_pool"

    def test_a_kind_nobody_offered_is_refused_and_named(self):
        with pytest.raises(OffVocabulary, match=r"findings\[0\]\.caveats\[0\]\.kind 'rater_pool'"):
            validate_authored(self._payload("rater_pool"), ENGINE_CAVEAT_KINDS, [])


class TestEvidenceRowsCarryTheirBasis:
    def test_a_row_with_no_cell_is_refused(self):
        with pytest.raises(ValidationError, match="cell_ref"):
            EvidenceRow(measure_id="cost_usd", reading="measure", value=0.5, n=3, dispersion="±0.1")

    def test_a_cell_keyed_row_is_legitimate(self):
        row = EvidenceRow(cell_ref="v:a", measure_id="cost_usd", reading="measure", value=0.5, n=3, dispersion="±0.1")
        assert row.cell_ref == "v:a"

    def test_an_empty_cell_ref_is_refused_rather_than_passing_for_a_coordinate(self):
        with pytest.raises(ValidationError):
            EvidenceRow(cell_ref="", measure_id="cost_usd", reading="measure", value=0.5, n=3, dispersion="±0.1")

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
    every measure and judged dimension the bundle measured in each cell. A row naming anything else
    has no number to fill and is refused, repairably, naming which row it was.

    Driven through :func:`~threetears.evals.analysis.generator.generate_analysis` over the toy
    host's campaign, with a fixtured completion: the surface is the one the generator builds from
    the bundle, and a refusal is the one a finished generation raises after its one repair.
    """

    @staticmethod
    def _bundle() -> Any:
        profile = toyhost_profile()
        return toyhost_bundle(profile=profile)

    @staticmethod
    def _cell(bundle: Any) -> Any:
        """The facts of the cell the rows below name, read off the bundle the generator was handed."""
        ref = cell_at(bundle, TOYHOST_NARROW)
        (cell,) = [cell for cell in bundle.cell_measures if cell_ref(cell.variant_key, cell.apparatus_class_id) == ref]
        return cell

    async def _generate(self, bundle: Any, *rows: dict[str, str]) -> EvalAnalysis:
        """Generate over a memo whose SECOND finding carries ``rows`` as its evidence and no chart."""
        payload = memo_payload(bundle)
        (finding,) = payload["findings"]
        payload["findings"] = [finding, {**copy.deepcopy(finding), "evidence": list(rows)}]
        profile = toyhost_profile()
        analysis, _insights = await generate_analysis(
            bundle,
            prompt=PROMPT,
            model=MODEL,
            client=FixturedClient(json.dumps(payload)),
            prompt_id=PROMPT_ID,
            bundle_assembled_at=utc_now_iso(),
            profile=profile,
        )
        return analysis

    async def _resolve(self, measure: str, reading: str = "measure") -> EvidenceRow:
        bundle = self._bundle()
        row = {"cell": alias_at(bundle, TOYHOST_NARROW), "measure_id": measure, "reading": reading}
        analysis = await self._generate(bundle, row)
        return analysis.resolutions[1].evidence[0]

    async def test_a_measure_the_campaign_reported_survives_with_its_number_filled(self):
        bundle = self._bundle()
        (reported,) = [m for m in self._cell(bundle).measures.measures if m.name == "total_ms"]

        row = await self._resolve("total_ms")

        assert (row.cell_ref, row.measure_id, row.reading, row.value, row.n) == (
            cell_at(bundle, TOYHOST_NARROW),
            "total_ms",
            "measure",
            reported.mean,
            reported.n,
        )
        assert row.dispersion, "a resolved number carries its spread"

    async def test_a_measure_the_campaign_never_reported_is_refused(self):
        with pytest.raises(GenerationError, match="no such measure"):
            await self._resolve("invented_metric")

    async def test_the_refusal_names_the_finding_and_the_row(self):
        """An operator deciding whether to re-run a billed generation needs to know which row."""
        with pytest.raises(GenerationError) as excinfo:
            await self._resolve("invented_metric")
        assert "findings[1].evidence[0]" in str(excinfo.value)

    async def test_a_judged_dimension_the_campaign_scored_survives_as_evidence(self):
        """Quality is reportable: a judged score is a measurement the bundle carries, off the ranking surface."""
        (scored,) = [j for j in self._cell(self._bundle()).judged if j.dimension == TOYHOST_JUDGED_DIMENSION]

        row = await self._resolve(TOYHOST_JUDGED_DIMENSION, "judged")

        assert (row.value, row.reading) == (scored.mean, "judged")

    async def test_a_judged_dimension_the_campaign_never_scored_is_refused(self):
        with pytest.raises(SoundnessRefusal, match="no such dimension"):
            await self._resolve("tone", "judged")

    async def test_the_refusal_is_repairable_rather_than_terminal(self):
        """The class is the contract, so assert it rather than its superclass.

        ``SoundnessRefusal`` subclasses ``GenerationError``, so the tests above would pass for a
        refusal on the terminal side of the repair boundary too. A row naming a reading the
        campaign never measured is something a finished call DID, and the refusal names the row,
        so it is feedable.
        """
        with pytest.raises(UnresolvableReference):
            await self._resolve("invented_metric")

    async def test_a_finding_with_no_chart_resolves_to_none_and_no_note(self):
        analysis = await self._generate(self._bundle())
        resolution = analysis.resolutions[1]
        assert (resolution.evidence, resolution.chart, resolution.chart_note) == ([], None, "")
