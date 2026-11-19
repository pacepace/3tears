"""The set grader: recall over a case's key and its core, the share of what was named that the key holds, and the
measures a kind lands them as.

A named entity outside the key is unverified, never wrong: a key may be non-exhaustive, so no reading here may count
one against the output. A reading with no value on a cell is left off it, so the cell is outside that measure's n,
which is how an empty output reads differently from a perfect one, and the runner-level test holds that through a
real run's summary.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import (
    DOCUMENT_PARAM,
    TOY_EXTRACTOR_KIND,
    TOYHOST_EXTRACTION_FAMILY,
    ScriptedExtractionClient,
    ToyExtractorInstance,
    ToyExtractorKind,
)
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import (
    RUN_K,
    RUN_MODELS,
    toyhost_run,
    toyhost_template,
    toyhost_test_cases,
)
from packages.evals.tests.off_loop import off_loop
from threetears.evals.analysis import summarize_run
from threetears.evals.kernel import (
    EXTRACTED_SET_FAMILY,
    CandidateOutput,
    CellSink,
    EntityExtractor,
    FoldingMatcher,
    NamedEntity,
    SetGrade,
    grade_set,
    set_grade_descriptors,
)
from threetears.evals.kernel.host import MeasureRegistrationError, MeasureRegistry
from threetears.evals.kernel.metrics import measure_definition_digests
from threetears.evals.run import RunnerOptions, execute_run, run_blocking
from threetears.evals.schema import EvalTestCase
from threetears.evals.schema.reference import ReferenceKey

_PREFIX = "picks"

#: The key every test below grades against: two core entities, two acceptable ones, one alias.
_KEY = ReferenceKey.model_validate(
    {
        "core": [{"name": "Band A", "aliases": ["The Band A"]}, {"name": "Band B"}],
        "acceptable": [{"name": "Band C"}, {"name": "Band D"}],
    }
)


def _named(*names: str) -> list[NamedEntity]:
    return [NamedEntity(name=name) for name in names]


def _measures(grade: SetGrade) -> dict[str, float]:
    return grade.host_measures(_PREFIX)


# =============================================================================
# Done-when 1: one fixture counts both directions
# =============================================================================


def test_one_fixture_counts_recall_over_the_key_and_the_core_and_the_share_the_key_holds() -> None:
    # "the band a" and "THE BAND A" fold alike, so they are one named entity; "The Band A" is an alias of Band A.
    grade = grade_set(_named("the band a", "Band C", "A deep cut", "THE BAND A"), _KEY)

    assert grade.named == 3
    assert grade.verified == 2
    assert grade.found == ("Band A", "Band C")
    assert grade.core_missed == ("Band B",)
    assert grade.unverified_names == ("A deep cut",)
    assert grade.unverified == 1
    assert grade.recall == pytest.approx(2 / 4)
    assert grade.core_recall == pytest.approx(1 / 2)
    assert grade.verified_share == pytest.approx(2 / 3)
    assert _measures(grade) == {
        "picks_recall": pytest.approx(0.5),
        "picks_core_recall": pytest.approx(0.5),
        "picks_verified_share": pytest.approx(2 / 3),
        "picks_named": 3.0,
        "picks_unverified": 1.0,
    }


def test_an_unverified_name_is_listed_and_never_counted_against_recall() -> None:
    without = grade_set(_named("Band A", "Band C"), _KEY)
    with_more = grade_set(_named("Band A", "Band C", "A deep cut", "Another deep cut"), _KEY)

    assert with_more.unverified_names == ("A deep cut", "Another deep cut")
    # Unverified names change what was named and nothing found: recall over the key and the core stand.
    assert (with_more.recall, with_more.core_recall) == (without.recall, without.core_recall)
    assert with_more.found == without.found
    assert with_more.core_missed == without.core_missed
    # No reading names them wrong: the measures are the five, and the unverified count is a diagnostic.
    assert set(_measures(with_more)) == {
        "picks_recall",
        "picks_core_recall",
        "picks_verified_share",
        "picks_named",
        "picks_unverified",
    }
    (unverified,) = [d for d in set_grade_descriptors(_PREFIX, reader_label="Picks") if d.name == "picks_unverified"]
    assert unverified.diagnostic is True
    assert unverified.higher_is_better is None


# =============================================================================
# Done-when 2: the empty, perfect and no-core states
# =============================================================================


def test_an_empty_output_names_nothing_and_has_no_verified_share() -> None:
    assert _measures(grade_set([], _KEY)) == {
        "picks_recall": 0.0,
        "picks_core_recall": 0.0,
        "picks_named": 0.0,
        "picks_unverified": 0.0,
    }


def test_a_perfect_output_reads_recall_one_and_verified_share_one() -> None:
    grade = grade_set(_named("Band D", "The Band A", "Band B", "band c"), _KEY)

    assert grade.found == ("Band A", "Band B", "Band C", "Band D")
    assert _measures(grade) == {
        "picks_recall": 1.0,
        "picks_core_recall": 1.0,
        "picks_verified_share": 1.0,
        "picks_named": 4.0,
        "picks_unverified": 0.0,
    }


def test_a_key_with_no_core_carries_no_core_recall() -> None:
    key = ReferenceKey.model_validate({"acceptable": [{"name": "Band C"}, {"name": "Band D"}]})

    grade = grade_set(_named("Band C", "Band E"), key)

    assert grade.core_recall is None
    assert _measures(grade) == {
        "picks_recall": 0.5,
        "picks_verified_share": 0.5,
        "picks_named": 2.0,
        "picks_unverified": 1.0,
    }


def test_a_grade_whose_counts_contradict_its_names_is_refused() -> None:
    with pytest.raises(ValueError, match="unverified names listed"):
        SetGrade(named=2, verified=1, core_size=1, key_size=2, found=("A",), core_missed=(), unverified_names=())


def test_a_prefix_that_is_no_measure_key_is_refused() -> None:
    with pytest.raises(ValueError, match="prefix"):
        grade_set([], _KEY).host_measures("Picks ")
    with pytest.raises(ValueError, match="prefix"):
        set_grade_descriptors("2picks", reader_label="Picks")


# =============================================================================
# Done-when 3: a host matcher replaces the default, and its identity is the definition
# =============================================================================


class _ArticleDroppingMatcher:
    """A host's own normaliser: drops a leading English article before folding, which the default must never do."""

    def __init__(self, identity: str = "articles-v1") -> None:
        self._identity = identity

    @property
    def identity(self) -> str:
        return self._identity

    def match(self, named: NamedEntity, key: ReferenceKey) -> str | None:
        said = named.name.casefold().removeprefix("the ").strip()
        for entity in key.entities:
            if said in {name.casefold().removeprefix("the ").strip() for name in (entity.name, *entity.aliases)}:
                return entity.name
        return None


class _NamesWhatTheKeyLacks:
    """A defective matcher: it answers a name the key does not hold."""

    @property
    def identity(self) -> str:
        return "defective-v1"

    def match(self, named: NamedEntity, key: ReferenceKey) -> str | None:
        return "Band Z"


def test_a_host_matcher_replaces_the_default() -> None:
    named = _named("The Band B", "The Band D")

    assert grade_set(named, _KEY).found == ()
    assert grade_set(named, _KEY, matcher=_ArticleDroppingMatcher()).found == ("Band B", "Band D")


def test_a_matcher_that_names_an_entity_the_key_lacks_is_refused() -> None:
    with pytest.raises(ValueError, match="'Band Z', which the key does not hold"):
        grade_set(_named("Band A"), _KEY, matcher=_NamesWhatTheKeyLacks())


def test_a_changed_matcher_identity_changes_every_measure_definition() -> None:
    default = measure_definition_digests(set_grade_descriptors(_PREFIX, reader_label="Picks"))
    host = measure_definition_digests(
        set_grade_descriptors(_PREFIX, reader_label="Picks", matcher=_ArticleDroppingMatcher())
    )
    revised = measure_definition_digests(
        set_grade_descriptors(_PREFIX, reader_label="Picks", matcher=_ArticleDroppingMatcher("articles-v2"))
    )

    assert set(default) == set(host) == set(revised)
    assert all(default[name] != host[name] != revised[name] != default[name] for name in default)


def test_renaming_the_set_for_a_reader_moves_no_definition() -> None:
    picks = measure_definition_digests(set_grade_descriptors(_PREFIX, reader_label="Picks"))
    chosen = measure_definition_digests(set_grade_descriptors(_PREFIX, reader_label="Chosen"))

    assert picks == chosen


def test_the_formula_names_the_matcher_and_the_source_of_the_set() -> None:
    (recall, *_) = set_grade_descriptors(_PREFIX, reader_label="Picks")

    assert recall.formula == "set grade v1 | matcher fold-v1 | extractor structured"
    assert recall.family == "mechanical"


# parity-with: threetears.evals.schema.CompletionClient
class _FakeNamedClient:
    """A client that names its model and is never called: what an extractor's identity is read off."""

    model_name = "scripted/extractor"

    async def generate(self, *, system: str, user: str, response_format: dict[str, object] | None = None) -> object:
        raise AssertionError("an extractor's identity makes no call")

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> _FakeNamedClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


def test_an_extracted_set_is_its_own_family_and_definition() -> None:
    extractor = EntityExtractor(client=_FakeNamedClient(), entity_kind="bands")
    extracted = set_grade_descriptors(_PREFIX, reader_label="Picks", extractor=extractor)
    structured = set_grade_descriptors(_PREFIX, reader_label="Picks")

    assert {d.family for d in extracted} == {EXTRACTED_SET_FAMILY.name}
    assert extracted[0].formula == f"set grade v1 | matcher fold-v1 | extractor {extractor.identity}"
    digests = measure_definition_digests(extracted)
    assert all(digests[name] != digest for name, digest in measure_definition_digests(structured).items())


def test_a_host_registers_the_measures_and_declares_the_extracted_family() -> None:
    extractor = EntityExtractor(client=_FakeNamedClient(), entity_kind="bands")

    MeasureRegistry(set_grade_descriptors(_PREFIX, reader_label="Picks"))
    MeasureRegistry(
        set_grade_descriptors(_PREFIX, reader_label="Picks", extractor=extractor), families=(EXTRACTED_SET_FAMILY,)
    )
    with pytest.raises(MeasureRegistrationError, match="extracted_set"):
        MeasureRegistry(set_grade_descriptors(_PREFIX, reader_label="Picks", extractor=extractor))


# =============================================================================
# Done-when 4: the default folds case and Unicode forms only
# =============================================================================


@pytest.mark.parametrize(
    ("written", "named", "matches"),
    [
        ("Émile", "émile", True),
        ("ﬁre", "fire", True),
        ("Grand  Band", " grand band ", True),
        ("émile", "emile", False),
    ],
)
def test_the_default_folds_case_and_compatibility_forms_and_keeps_accents(
    written: str, named: str, matches: bool
) -> None:
    key = ReferenceKey.model_validate({"core": [{"name": written}]})

    hit = FoldingMatcher().match(NamedEntity(name=named), key)

    assert hit == (written if matches else None)
    assert FoldingMatcher().identity == "fold-v1"


# =============================================================================
# Done-when 6: through the runner, a cell without a verified share is outside that measure's n
# =============================================================================

#: What the output names on each toy document: two (one outside the key), nothing, and the whole key.
_PICKS: dict[str, tuple[str, ...]] = {
    "doc-01": ("Band A", "A deep cut"),
    "doc-02": (),
    "doc-03": ("Band A", "Band B", "Band C", "Band D"),
}


class _PickingKind(ToyExtractorKind):
    """The toy extractor, whose every cell also lands a set grade of the picks its document names."""

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        output = await super().invoke(instance, test_case, sink)
        picks = _PICKS[test_case.variation_params[DOCUMENT_PARAM]]
        grade = grade_set(_named(*picks), _KEY)
        return output.model_copy(update={"host_measures": {**output.host_measures, **grade.host_measures(_PREFIX)}})


async def test_a_cell_naming_nothing_is_outside_the_verified_share_n_of_a_run_summary() -> None:
    profile = replace(
        toyhost_profile(),
        measures=MeasureRegistry(
            (*TOYHOST_MEASURES, *set_grade_descriptors(_PREFIX, reader_label="Picks")),
            families=(TOYHOST_EXTRACTION_FAMILY,),
        ),
    )
    host = toyhost_host(profile=profile)
    world = host.profile.world
    assert world is not None, "the toy profile registers a world"
    template = toyhost_template()
    kind = _PickingKind(client=ScriptedExtractionClient(), world=world, goal_checks=tuple(template.goal_state_checks))
    test_cases = toyhost_test_cases(template)
    assert sorted(case.variation_params[DOCUMENT_PARAM] for case in test_cases) == sorted(_PICKS)
    run = toyhost_run(model=RUN_MODELS[0], template=template, kind=kind, world=world)
    await run_blocking(host.blocking_executor, host.storage.save_eval_run, run)

    await execute_run(
        host,
        run=run,
        template=template,
        test_cases=test_cases,
        judge_service=None,
        options=RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}),
    )

    results = await run_blocking(host.blocking_executor, host.storage.query_eval_results_by_run, run.id, run.scope_id)
    cells = len(_PICKS) * RUN_K
    assert len(results) == cells
    assert sum("picks_verified_share" not in result.host_measures for result in results) == RUN_K, (
        "the runner accepted every cell, the ones naming nothing without a verified share"
    )
    summary = off_loop(summarize_run, host, run.id, run.scope_id)
    by_name = {measure.name: measure for measure in summary.measures}
    assert (by_name["picks_verified_share"].n, by_name["picks_verified_share"].mean) == (
        cells - RUN_K,
        pytest.approx((0.5 + 1.0) / 2),
    )
    assert (by_name["picks_recall"].n, by_name["picks_recall"].mean) == (cells, pytest.approx((0.25 + 0.0 + 1.0) / 3))
    assert (by_name["picks_named"].n, by_name["picks_named"].mean) == (cells, pytest.approx((2 + 0 + 4) / 3))
