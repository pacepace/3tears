"""The set grader: recall over a case's key and its core, the share of what was named that the key holds, and the
measures a kind lands them as.

A named entity outside the key is unverified, never wrong: a key may be non-exhaustive. Only recall and core recall
judge quality; the verified share, the named count and the unverified count are diagnostics with no better direction,
so an unlisted pick never counts against an output. One grader declares the measures and grades the set, so a grade
and its measures always name one matcher and one extractor. A reading with no value on a cell is left off it, so the
cell is outside that measure's n, which is how an empty output reads differently from a perfect one, and the
runner-level test holds that through a real run's summary.
"""

from __future__ import annotations

from dataclasses import replace

import inspect

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
    RoleUsageLedger,
    EntityMatcher,
    SetGrade,
    SetGrader,
)
from threetears.evals.kernel.host import MeasureRegistrationError, MeasureRegistry
from threetears.evals.kernel.metrics import MetricDescriptor, measure_definition_digests
from threetears.evals.run import RunnerOptions, execute_run, run_blocking
from threetears.evals.schema import EvalTestCase
from threetears.evals.schema.reference import ReferenceKey

_PREFIX = "picks"

#: The key every test below grades against: two core entities, two acceptable ones, one alias.
_KEY = ReferenceKey.model_validate(
    {
        "core": [{"name": "Frankenstein", "aliases": ["The Modern Prometheus"]}, {"name": "Odyssey"}],
        "acceptable": [{"name": "Beloved"}, {"name": "Iliad"}],
    }
)


#: The structured-set grader most tests grade with: the default matcher, no extractor.
_GRADER = SetGrader(_PREFIX, reader_label="Picks")


def _named(*names: str) -> list[NamedEntity]:
    return [NamedEntity(name=name) for name in names]


def _grade(*names: str, key: ReferenceKey = _KEY, matcher: EntityMatcher | None = None) -> SetGrade:
    grader = _GRADER if matcher is None else SetGrader(_PREFIX, reader_label="Picks", matcher=matcher)
    return grader.grade(_named(*names), key)


def _measures(grade: SetGrade) -> dict[str, float]:
    return grade.host_measures()


def _descriptors(
    *, matcher: EntityMatcher | None = None, extractor: EntityExtractor | None = None
) -> tuple[MetricDescriptor, ...]:
    return SetGrader(_PREFIX, reader_label="Picks", matcher=matcher, extractor=extractor).descriptors()


# =============================================================================
# Done-when 1: one fixture counts both directions
# =============================================================================


def test_one_fixture_counts_recall_over_the_key_and_the_core_and_the_share_the_key_holds() -> None:
    # "the modern prometheus" and "THE MODERN PROMETHEUS" fold alike, so they are one named entity; "The Modern Prometheus" is an alias of Frankenstein.
    grade = _grade("the modern prometheus", "Beloved", "An unlisted novel", "THE MODERN PROMETHEUS")

    assert grade.named == 3
    assert grade.verified == 2
    assert grade.found == ("Frankenstein", "Beloved")
    assert grade.core_missed == ("Odyssey",)
    assert grade.unverified_names == ("An unlisted novel",)
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


def test_two_spellings_of_one_key_entity_are_one_pick() -> None:
    # "Frankenstein" and its alias "The Modern Prometheus" fold differently and match one key entity; "an UNLISTED novel" folds as "An unlisted novel".
    grade = _grade("Frankenstein", "The Modern Prometheus", "An unlisted novel", "an UNLISTED novel")

    assert (grade.named, grade.verified, grade.unverified) == (2, 1, 1)
    assert grade.found == ("Frankenstein",)
    assert grade.unverified_names == ("An unlisted novel",)
    assert grade.verified_share == pytest.approx(1 / 2)


def test_an_unverified_name_is_listed_and_never_counted_against_recall() -> None:
    without = _grade("Frankenstein", "Beloved")
    with_more = _grade("Frankenstein", "Beloved", "An unlisted novel", "Another unlisted novel")

    assert with_more.unverified_names == ("An unlisted novel", "Another unlisted novel")
    # Unverified names change what was named and nothing found: recall over the key and the core stand.
    assert (with_more.recall, with_more.core_recall) == (without.recall, without.core_recall)
    assert with_more.found == without.found
    assert with_more.core_missed == without.core_missed
    # The verified share falls, and that reading is a diagnostic: nothing that judges quality reads it.
    assert with_more.verified_share is not None and without.verified_share is not None
    assert with_more.verified_share < without.verified_share
    judged = {d.name for d in _GRADER.descriptors() if d.higher_is_better is not None}
    assert judged == {"picks_recall", "picks_core_recall"}


def test_a_context_entity_never_counts_in_a_set_grade_named_or_missed() -> None:
    """The key's context tier carries facts, never answers: a set grade reads the answer set alone, so a key with
    context grades every output exactly as the same key without it, and a context entity named is unverified."""
    with_context = ReferenceKey.model_validate(
        {
            **_KEY.model_dump(mode="json"),
            "context": [{"name": "Mary Shelley", "aliases": ["Mary Wollstonecraft Shelley"]}, {"name": "Homer"}],
        }
    )
    assert [e.name for e in with_context.entities] == [e.name for e in _KEY.entities], "context is outside the set"

    for names in [(), ("Frankenstein", "Beloved"), ("Frankenstein", "Mary Shelley", "Homer", "An unlisted novel")]:
        assert _grade(*names, key=with_context) == _grade(*names), names

    grade = _grade("Frankenstein", "Mary Shelley", "Homer", key=with_context)
    assert (grade.named, grade.verified, grade.key_size) == (3, 1, 4)
    assert grade.found == ("Frankenstein",)
    assert grade.unverified_names == ("Mary Shelley", "Homer")
    assert FoldingMatcher().match(NamedEntity(name="mary shelley"), with_context) is None


def test_only_recall_and_core_recall_judge_quality_and_the_rest_are_diagnostics() -> None:
    by_name = {d.name: d for d in _GRADER.descriptors()}

    for name in ("picks_recall", "picks_core_recall"):
        assert (by_name[name].higher_is_better, by_name[name].merit_axis, by_name[name].diagnostic) == (
            True,
            "quality",
            False,
        )
    for name in ("picks_verified_share", "picks_named", "picks_unverified"):
        assert (by_name[name].higher_is_better, by_name[name].merit_axis, by_name[name].diagnostic) == (
            None,
            None,
            True,
        )
    assert by_name["picks_verified_share"].value_range == (0.0, 1.0)


# =============================================================================
# Done-when 2: the empty, perfect and no-core states
# =============================================================================


def test_an_empty_output_names_nothing_and_has_no_verified_share() -> None:
    assert _measures(_grade()) == {
        "picks_recall": 0.0,
        "picks_core_recall": 0.0,
        "picks_named": 0.0,
        "picks_unverified": 0.0,
    }


def test_a_perfect_output_reads_recall_one_and_verified_share_one() -> None:
    grade = _grade("Iliad", "The Modern Prometheus", "Odyssey", "beloved")

    assert grade.found == ("Frankenstein", "Odyssey", "Beloved", "Iliad")
    assert _measures(grade) == {
        "picks_recall": 1.0,
        "picks_core_recall": 1.0,
        "picks_verified_share": 1.0,
        "picks_named": 4.0,
        "picks_unverified": 0.0,
    }


def test_a_key_with_no_core_carries_no_core_recall() -> None:
    key = ReferenceKey.model_validate({"acceptable": [{"name": "Beloved"}, {"name": "Iliad"}]})

    grade = _grade("Beloved", "Walden", key=key)

    assert grade.core_recall is None
    assert _measures(grade) == {
        "picks_recall": 0.5,
        "picks_verified_share": 0.5,
        "picks_named": 2.0,
        "picks_unverified": 1.0,
    }


def test_a_grade_whose_counts_contradict_its_names_is_refused() -> None:
    with pytest.raises(ValueError, match="unverified names listed"):
        SetGrade(
            prefix=_PREFIX,
            matcher_identity="fold-v1",
            extractor_identity="structured",
            named=2,
            verified=1,
            core_size=1,
            key_size=2,
            found=("A",),
            core_missed=(),
            unverified_names=(),
        )


def test_a_prefix_that_is_no_measure_key_is_refused() -> None:
    with pytest.raises(ValueError, match="prefix"):
        SetGrader("Picks ", reader_label="Picks")
    with pytest.raises(ValueError, match="prefix"):
        SetGrader("2picks", reader_label="Picks")
    with pytest.raises(ValueError, match="reader label"):
        SetGrader(_PREFIX, reader_label="  ")


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
        return "Ulysses"


def test_a_host_matcher_replaces_the_default() -> None:
    assert _grade("The Odyssey", "The Iliad").found == ()
    assert _grade("The Odyssey", "The Iliad", matcher=_ArticleDroppingMatcher()).found == ("Odyssey", "Iliad")


def test_a_matcher_that_names_an_entity_the_key_lacks_is_refused() -> None:
    with pytest.raises(ValueError, match="'Ulysses', which the key does not hold"):
        _grade("Frankenstein", matcher=_NamesWhatTheKeyLacks())


def test_a_matcher_that_answers_a_context_entity_is_refused_since_context_is_no_answer() -> None:
    key = ReferenceKey.model_validate({**_KEY.model_dump(mode="json"), "context": [{"name": "Ulysses"}]})

    with pytest.raises(ValueError, match="'Ulysses', which the key does not hold"):
        _grade("Frankenstein", key=key, matcher=_NamesWhatTheKeyLacks())


def test_a_changed_matcher_identity_changes_every_measure_definition() -> None:
    default = measure_definition_digests(_GRADER.descriptors())
    host = measure_definition_digests(_descriptors(matcher=_ArticleDroppingMatcher()))
    revised = measure_definition_digests(_descriptors(matcher=_ArticleDroppingMatcher("articles-v2")))

    assert set(default) == set(host) == set(revised)
    assert all(default[name] != host[name] != revised[name] != default[name] for name in default)


def test_renaming_the_set_for_a_reader_moves_no_definition() -> None:
    picks = measure_definition_digests(_GRADER.descriptors())
    chosen = measure_definition_digests(SetGrader(_PREFIX, reader_label="Chosen").descriptors())

    assert picks == chosen


def test_the_formula_names_the_matcher_and_the_source_of_the_set() -> None:
    (recall, *_) = _GRADER.descriptors()

    assert recall.formula == "set grade v1 | matcher fold-v1 | extractor structured"
    assert recall.family == "mechanical"


@pytest.mark.parametrize("matcher", [None, _ArticleDroppingMatcher(), _ArticleDroppingMatcher("articles-v2")])
def test_a_grade_and_its_measures_name_one_matcher(matcher: EntityMatcher | None) -> None:
    grader = SetGrader(_PREFIX, reader_label="Picks", matcher=matcher)

    grade = grader.grade(_named("The Odyssey", "An unlisted novel"), _KEY)

    assert grade.matcher_identity == grader.matcher.identity
    assert grade.extractor_identity == "structured"
    assert {descriptor.formula for descriptor in grader.descriptors()} == {grade.formula}
    assert set(grade.host_measures()) <= {descriptor.name for descriptor in grader.descriptors()}
    # The grader is the only place a matcher is chosen: grading takes the set and the key, and nothing else.
    assert list(inspect.signature(SetGrader.grade).parameters) == ["self", "named", "key"]


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
    extractor = EntityExtractor(client=_FakeNamedClient(), entity_kind="books")
    extracted = _descriptors(extractor=extractor)
    structured = _GRADER.descriptors()

    assert {d.family for d in extracted} == {EXTRACTED_SET_FAMILY.name}
    assert extracted[0].formula == f"set grade v1 | matcher fold-v1 | extractor {extractor.identity}"
    digests = measure_definition_digests(extracted)
    assert all(digests[name] != digest for name, digest in measure_definition_digests(structured).items())


def test_a_host_registers_the_measures_and_declares_the_extracted_family() -> None:
    extractor = EntityExtractor(client=_FakeNamedClient(), entity_kind="books")

    MeasureRegistry(_GRADER.descriptors())
    MeasureRegistry(_descriptors(extractor=extractor), families=(EXTRACTED_SET_FAMILY,))
    with pytest.raises(MeasureRegistrationError, match="extracted_set"):
        MeasureRegistry(_descriptors(extractor=extractor))


def test_a_grader_bound_to_an_extractor_refuses_a_set_listed_another_way() -> None:
    grader = SetGrader(
        _PREFIX, reader_label="Picks", extractor=EntityExtractor(client=_FakeNamedClient(), entity_kind="books")
    )

    with pytest.raises(ValueError, match="grade_text"):
        grader.grade(_named("Frankenstein"), _KEY)


async def test_a_grader_with_no_extractor_refuses_prose() -> None:
    with pytest.raises(ValueError, match="no extractor"):
        await _GRADER.grade_text("Frankenstein, then Beloved.", _KEY, ledger=RoleUsageLedger(role="judge"))


# =============================================================================
# Done-when 4: the default folds case and Unicode forms only
# =============================================================================


@pytest.mark.parametrize(
    ("written", "named", "matches"),
    [
        ("Émile", "émile", True),
        ("ﬁre", "fire", True),
        ("Grand  Canyon", " grand canyon ", True),
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
    "doc-01": ("Frankenstein", "An unlisted novel"),
    "doc-02": (),
    "doc-03": ("Frankenstein", "Odyssey", "Beloved", "Iliad"),
}


class _PickingKind(ToyExtractorKind):
    """The toy extractor, whose every cell also lands a set grade of the picks its document names."""

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        output = await super().invoke(instance, test_case, sink)
        picks = _PICKS[test_case.variation_params[DOCUMENT_PARAM]]
        grade = _GRADER.grade(_named(*picks), _KEY)
        return output.model_copy(update={"host_measures": {**output.host_measures, **grade.host_measures()}})


async def test_a_cell_naming_nothing_is_outside_the_verified_share_n_of_a_run_summary() -> None:
    profile = replace(
        toyhost_profile(),
        measures=MeasureRegistry(
            (*TOYHOST_MEASURES, *_GRADER.descriptors()),
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
