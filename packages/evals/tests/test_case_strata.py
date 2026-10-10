"""Results by kind of case: a test case declares a stratum, and the analysis reads every cell again per stratum.

One run over a mixed case set — plain cases, lookalikes, and a case that declares no stratum — is
assembled through a real store and read back:

- **Each cell is summarised per stratum beside its pooled figure**, with each stratum's cases and
  observations, by the walk that summarised the cell: accuracy, the confusion matrix and each label's
  precision and recall come from the stratum's own results, and a judged dimension is scored over them too.
- **A run whose cases declare no stratum reads as it did before strata existed**: no cell is broken
  down, and its report has no strata table and no strata disclosure.
- **The report carries it**: a ``strata`` table, a column per stratum after the pooled one, each
  stratum's case count first, and a disclosure naming every stratum under
  :data:`~threetears.evals.contracts.STRATUM_MIN_CASES` cases — in the code-only report and in an
  analysis's, in JSON the published schema accepts, Markdown and HTML.
- **The stored shapes refuse what cannot be true**: strata that do not add up to their cell, a stratum
  listed twice, a breakdown made only of undeclared cases, two nominated stratum axes, an ``llm`` one.
- **Generation stamps it**: a case generated from a template that nominates an axis takes that axis's
  value as its stratum, and the store reads strata back without the rest of the case.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError

from threetears.evals.analysis import (
    AnalysisContextBundle,
    DisclosureBlock,
    Report,
    TableBlock,
    assemble_context_bundle,
    build_code_only_report,
    build_report,
    published_report_schema,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.bundle import bundle_decision_surface
from threetears.evals.analysis.report.build import NO_STRATUM, TOO_FEW_CASES
from threetears.evals.analysis.surface_table import SURFACE_ORDER_NO_CONTROL
from threetears.evals.contracts import (
    STRATUM_MIN_CASES,
    CellFacts,
    EvalResult,
    EvalStorage,
    EvalTestCase,
    RubricScore,
    StratumFacts,
    classifier_label_measure,
    confusion_cell,
)
from threetears.evals.contracts.analysis_measures import MeasureSummary
from threetears.evals.contracts.host import SHARED_CORE, HostProfile, MeasureRegistry
from threetears.evals.contracts.models import VariationAxis
from threetears.evals.gen import generate_variations
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import (
    fixture_variant_key,
    make_analysis,
    make_campaign,
    make_eval_result,
    make_eval_run,
    make_template,
)

_SCOPE = "uni-1"
_PROFILE = HostProfile(host_id="strata", host_sweepables=SHARED_CORE, measures=MeasureRegistry(()))
_TONE = "conversation.tone"

#: Twelve plain cases, at or above the floor, and three lookalikes, below it.
_PLAIN = [f"plain-{index:02d}" for index in range(12)]
_LOOKALIKE = ["lookalike-0", "lookalike-1", "lookalike-2"]
_UNDECLARED = "undeclared-0"


def _case(case_id: str, stratum: str | None) -> EvalTestCase:
    return EvalTestCase(id=case_id, scope_id=_SCOPE, template_id="tpl-1", stratum=stratum)


def _classified(
    run_id: str, model: str, case_id: str, expected: str, predicted: str, *, k: int = 1, tone: int = 4
) -> EvalResult:
    """One classification of one case, as a classifier kind lands it: ``match`` and its confusion cell."""
    return make_eval_result(
        scope_id=_SCOPE,
        eval_run_id=run_id,
        test_case_id=case_id,
        model=model,
        k_iteration=k,
        variant_key=fixture_variant_key(model),
        host_measures={"match": expected == predicted, "confusion_cell": confusion_cell(expected, predicted)},
        goal_state_outcomes=[],
        rubric_scores=[RubricScore(dim=_TONE, score=tone, scale="ordinal")],
    )


def _store(cases: list[EvalTestCase], arms: dict[str, list[EvalResult]]) -> tuple[EvalStorage, Any]:
    """A real store holding the cases, one run per arm with its results, and a campaign over the runs."""
    storage = EvalStorage(InMemoryDocumentStore())
    for case in cases:
        storage.save_test_case(case)
    run_ids = []
    for model, results in arms.items():
        run = make_eval_run(
            id=f"run-{model}",
            scope_id=_SCOPE,
            candidate_model=model,
            k_runs=max(result.k_iteration for result in results),
            test_case_ids=sorted({result.test_case_id for result in results}),
            rubric_scales={_TONE: "ordinal"},
            status="completed",
        )
        storage.save_eval_run(run)
        for result in results:
            storage.save_eval_result(result.model_copy(update={"eval_run_id": run.id}))
        run_ids.append(run.id)
    campaign = make_campaign(scope_id=_SCOPE, run_ids=run_ids)
    storage.save_campaign(campaign)
    return storage, campaign


def _mixed() -> tuple[EvalStorage, Any]:
    """Two arms over twelve plain cases, three lookalikes and one case declaring no stratum.

    ``sonnet`` gets every plain case right, one lookalike of three (the other two read as ``DIRECT``), and the
    undeclared case right. ``haiku`` gets every case right. Every plain case is ``NONE``; every lookalike is
    ``NONE`` dressed as ``DIRECT``; the undeclared one is ``DIRECT``.
    """
    cases = [
        *(_case(case, "plain") for case in _PLAIN),
        *(_case(case, "lookalike") for case in _LOOKALIKE),
        _case(_UNDECLARED, None),
    ]
    sonnet = [
        *(_classified("run-sonnet", "sonnet", case, "NONE", "NONE", tone=4) for case in _PLAIN),
        _classified("run-sonnet", "sonnet", "lookalike-0", "NONE", "NONE", tone=2),
        _classified("run-sonnet", "sonnet", "lookalike-1", "NONE", "DIRECT", tone=2),
        _classified("run-sonnet", "sonnet", "lookalike-2", "NONE", "DIRECT", tone=2),
        _classified("run-sonnet", "sonnet", _UNDECLARED, "DIRECT", "DIRECT", tone=4),
    ]
    haiku = [
        *(_classified("run-haiku", "haiku", case, "NONE", "NONE") for case in [*_PLAIN, *_LOOKALIKE]),
        _classified("run-haiku", "haiku", _UNDECLARED, "DIRECT", "DIRECT"),
    ]
    return _store(cases, {"sonnet": sonnet, "haiku": haiku})


def _bundle(storage: EvalStorage, campaign: Any) -> AnalysisContextBundle:
    return assemble_context_bundle(campaign, storage=storage, profile=_PROFILE)


def _cell(bundle: AnalysisContextBundle, model: str) -> CellFacts:
    return next(cell for cell in bundle.cell_measures if cell.variant_key == fixture_variant_key(model))


def _summaries(stratum: StratumFacts | CellFacts) -> dict[str, MeasureSummary]:
    return {summary.name: summary for summary in stratum.measures.measures}


def _strata(cell: CellFacts) -> dict[str | None, StratumFacts]:
    return {stratum.stratum: stratum for stratum in cell.strata}


def _code_only(bundle: AnalysisContextBundle) -> Report:
    return build_code_only_report(bundle, measures=_PROFILE.measures, assembled_at="2026-10-08T00:00:00+00:00")


def _strata_table(report: Report) -> TableBlock:
    (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "strata"]
    return table


def _row(table: TableBlock, model: str, reading: str) -> dict[str, Any]:
    """The row of one arm — named, in this fixture, by its variant key's digest — and one reading."""
    digest = short_digest(fixture_variant_key(model))
    return next(row for row in table.rows if digest in str(row["arm"]) and row["reading"] == reading)


def _column(table: TableBlock, header: str) -> str:
    return next(column.key for column in table.columns if column.header == header)


# =============================================================================
# The bundle: each cell per stratum, beside its pool
# =============================================================================


class TestEachCellIsReadPerStratum:
    def test_a_mixed_run_reports_accuracy_per_stratum_with_its_cases(self) -> None:
        cell = _cell(_bundle(*_mixed()), "sonnet")
        strata = _strata(cell)

        assert list(strata) == ["lookalike", "plain", None], "named strata in name order, then the undeclared"
        assert (strata["plain"].n_cases, strata["plain"].n_observations) == (12, 12)
        assert (strata["lookalike"].n_cases, strata["lookalike"].n_observations) == (3, 3)
        assert (strata[None].n_cases, strata[None].n_observations) == (1, 1)
        assert _summaries(strata["plain"])["accuracy"].mean == 1.0
        assert _summaries(strata["lookalike"])["accuracy"].mean == pytest.approx(1 / 3)
        assert _summaries(strata[None])["accuracy"].mean == 1.0
        # The pool is the cell's own figure, untouched: 14 right of 16.
        assert _summaries(cell)["accuracy"].mean == pytest.approx(14 / 16)
        assert _summaries(cell)["accuracy"].n == 16

    def test_each_stratum_carries_its_own_confusion_matrix_and_per_label_statistics(self) -> None:
        strata = _strata(_cell(_bundle(*_mixed()), "sonnet"))
        lookalike, plain = _summaries(strata["lookalike"]), _summaries(strata["plain"])

        assert lookalike["confusion_cell"].categories == {"NONE → NONE": 1, "NONE → DIRECT": 2}
        assert plain["confusion_cell"].categories == {"NONE → NONE": 12}
        # NONE's recall on the lookalikes is one of three; on the plain cases, twelve of twelve.
        recall = classifier_label_measure("recall", "NONE")
        assert (lookalike[recall].rate, lookalike[recall].n) == (pytest.approx(1 / 3), 3)
        assert (plain[recall].rate, plain[recall].n) == (1.0, 12)
        # DIRECT was predicted twice among the lookalikes and right neither time; the plain cases never predicted it.
        precision = classifier_label_measure("precision", "DIRECT")
        assert (lookalike[precision].rate, lookalike[precision].n) == (0.0, 2)
        assert precision not in plain

    def test_a_judged_dimension_is_scored_per_stratum(self) -> None:
        cell = _cell(_bundle(*_mixed()), "sonnet")
        strata = _strata(cell)

        (pooled,) = cell.judged
        (lookalike,) = strata["lookalike"].judged
        (plain,) = strata["plain"].judged
        assert (pooled.mean, pooled.n) == (pytest.approx((12 * 4 + 3 * 2 + 4) / 16), 16)
        assert (lookalike.mean, lookalike.n, lookalike.n_independent) == (2.0, 3, 3)
        assert (plain.mean, plain.n) == (4.0, 12)
        assert lookalike.evidence_tier == pooled.evidence_tier

    def test_repeats_count_as_observations_and_cases_as_cases(self) -> None:
        cases = [_case("a", "plain"), _case("b", "plain"), _case("c", "lookalike")]
        results = [
            _classified("run-sonnet", "sonnet", case, "NONE", "NONE", k=k) for case in ("a", "b", "c") for k in (1, 2)
        ]
        cell = _cell(_bundle(*_store(cases, {"sonnet": results})), "sonnet")
        strata = _strata(cell)

        assert (strata["plain"].n_cases, strata["plain"].n_observations) == (2, 4)
        assert (strata["lookalike"].n_cases, strata["lookalike"].n_observations) == (1, 2)
        assert None not in strata, "every case declares a stratum, so there is no undeclared entry"

    def test_a_faulted_observation_is_counted_in_its_stratum_and_left_out_of_its_figures(self) -> None:
        cases = [_case("a", "plain"), _case("b", "lookalike")]
        faulted = _classified("run-sonnet", "sonnet", "b", "NONE", "DIRECT").model_copy(
            update={"infra_error": "apparatus: the rig broke"}
        )
        results = [_classified("run-sonnet", "sonnet", "a", "NONE", "NONE"), faulted]
        cell = _cell(_bundle(*_store(cases, {"sonnet": results})), "sonnet")
        lookalike = _strata(cell)["lookalike"]

        assert (lookalike.n_observations, lookalike.n_infra_excluded) == (1, 1)
        assert "accuracy" not in _summaries(lookalike), "the one lookalike was faulted, so nothing scored it"

    def test_a_case_that_no_longer_resolves_reads_as_declaring_none(self) -> None:
        storage, campaign = _mixed()
        storage.delete_test_case("lookalike-0", _SCOPE)
        strata = _strata(_cell(_bundle(storage, campaign), "sonnet"))

        assert strata["lookalike"].n_cases == 2
        assert strata[None].n_cases == 2


class TestNoStrataReadsAsBefore:
    def _unstratified(self) -> tuple[EvalStorage, Any]:
        cases = [_case(case, None) for case in [*_PLAIN, *_LOOKALIKE]]
        results = [_classified("run-sonnet", "sonnet", case, "NONE", "NONE") for case in [*_PLAIN, *_LOOKALIKE]]
        return _store(cases, {"sonnet": results})

    def test_no_cell_is_broken_down(self) -> None:
        bundle = _bundle(*self._unstratified())
        assert [cell.strata for cell in bundle.cell_measures] == [[]]

    def test_the_report_has_no_strata_table_and_no_strata_disclosure(self) -> None:
        report = _code_only(_bundle(*self._unstratified()))
        assert not [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "strata"]
        assert not [block for block in report.blocks if isinstance(block, DisclosureBlock) and block.source == "strata"]
        assert "By stratum" not in report_markdown(report)

    def test_a_cell_whose_cases_declare_none_is_not_broken_down_beside_one_whose_cases_do(self) -> None:
        """Strata are per cell: an arm that ran only undeclared cases is read as it always was."""
        cases = [_case("a", "plain"), _case("b", None)]
        storage, campaign = _store(
            cases,
            {
                "sonnet": [_classified("run-sonnet", "sonnet", "a", "NONE", "NONE")],
                "haiku": [_classified("run-haiku", "haiku", "b", "NONE", "NONE")],
            },
        )
        bundle = _bundle(storage, campaign)
        assert [stratum.stratum for stratum in _cell(bundle, "sonnet").strata] == ["plain"]
        assert _cell(bundle, "haiku").strata == []


# =============================================================================
# The report: a strata table and the too-small disclosure, on both kinds of report
# =============================================================================


class TestTheReportCarriesStrata:
    def test_the_code_only_report_lays_out_each_stratum_beside_the_pool(self) -> None:
        table = _strata_table(_code_only(_bundle(*_mixed())))

        headers = [column.header for column in table.columns]
        assert headers == ["Arm", "Reading", "All cases", "lookalike", "plain", NO_STRATUM]
        accuracy = _row(table, "sonnet", "Accuracy")
        assert accuracy["all"].startswith("0.875 ± ") and accuracy["all"].endswith("(n=16)")
        assert accuracy[_column(table, "lookalike")] == "0.3333 ± 0.3333 (n=3)"
        assert accuracy[_column(table, "plain")] == "1 ± 0 (n=12)"
        # A rate carries its Wilson interval, which a stratum this small makes wide.
        assert _row(table, "sonnet", "Label matched")[_column(table, "lookalike")] == "0.3333 [0.06149, 0.7923] (n=3)"

    def test_the_arms_follow_the_decision_surfaces_row_order_and_the_table_states_it(self) -> None:
        """No control here, so no reference row: the arms alphabetically by name (#645)."""
        table = _strata_table(_code_only(_bundle(*_mixed())))

        arms = [str(row["arm"]) for row in table.rows if row["reading"] == "cases"]
        assert arms == sorted(arms, key=str.casefold) and len(arms) == 2
        assert table.order.startswith(SURFACE_ORDER_NO_CONTROL)

    def test_each_arm_opens_with_the_cases_each_figure_rests_on_and_a_small_stratum_says_so(self) -> None:
        table = _strata_table(_code_only(_bundle(*_mixed())))
        cases = _row(table, "sonnet", "cases")

        assert cases["all"] == "16 cases, 16 obs"
        assert cases[_column(table, "lookalike")] == f"3 cases, 3 obs — {TOO_FEW_CASES}"
        assert cases[_column(table, "plain")] == "12 cases, 12 obs"
        assert table.rows.index(cases) < table.rows.index(_row(table, "sonnet", "Accuracy"))

    def test_the_confusion_matrix_and_per_label_rows_are_per_stratum(self) -> None:
        table = _strata_table(_code_only(_bundle(*_mixed())))

        matrix = _row(table, "sonnet", "Confusion-matrix cell")
        assert matrix[_column(table, "lookalike")] == "NONE → DIRECT: 2; NONE → NONE: 1 (n=3)"
        assert matrix[_column(table, "plain")] == "NONE → NONE: 12 (n=12)"
        recall = _row(table, "sonnet", "Recall of NONE")
        assert recall[_column(table, "lookalike")] == "0.3333 [0.06149, 0.7923] (n=3)"
        assert recall[_column(table, "plain")] == "1 [0.7575, 1] (n=12)"
        assert _row(table, "sonnet", f"{_TONE} (judged)")[_column(table, "lookalike")] == "2 ± 0 (n=3)"

    def test_a_stratum_below_the_floor_is_disclosed_by_name_and_count(self) -> None:
        report = _code_only(_bundle(*_mixed()))
        (disclosure,) = [
            block for block in report.blocks if isinstance(block, DisclosureBlock) and block.source == "strata"
        ]

        assert f"at least {STRATUM_MIN_CASES} cases" in disclosure.text
        assert "lookalike in " in disclosure.text and "(3 cases)" in disclosure.text
        assert f"{NO_STRATUM} in " in disclosure.text and "(1 case)" in disclosure.text
        assert "plain in " not in disclosure.text, "twelve cases is enough, so plain is not named"

    def test_a_stratum_at_the_floor_is_read_alone(self) -> None:
        cases = [_case(f"p{index}", "plain") for index in range(STRATUM_MIN_CASES)] + [_case("x", "lookalike")]
        results = [_classified("run-sonnet", "sonnet", case.id, "NONE", "NONE") for case in cases]
        report = _code_only(_bundle(*_store(cases, {"sonnet": results})))
        table = _strata_table(report)

        assert TOO_FEW_CASES not in _row(table, "sonnet", "cases")[_column(table, "plain")]
        assert TOO_FEW_CASES in _row(table, "sonnet", "cases")[_column(table, "lookalike")]

    def test_the_report_validates_against_the_published_schema_and_serializes(self) -> None:
        report = _code_only(_bundle(*_mixed()))

        document = json.loads(report.to_canonical_json())
        jsonschema.Draft202012Validator(published_report_schema()).validate(document)
        assert Report.model_validate(document) == report
        markdown = report_markdown(report)
        assert "**By stratum**" in markdown and "| lookalike |" in markdown and TOO_FEW_CASES in markdown
        html = report_html(report)
        assert "By stratum" in html and 'data-source="strata"' in html

    def test_an_analysis_report_lays_out_the_strata_its_surface_froze(self) -> None:
        bundle = _bundle(*_mixed())
        analysis = make_analysis(decision_surface=bundle_decision_surface(bundle), variant_index=bundle.variant_index)
        report = build_report(analysis)

        table = _strata_table(report)
        assert [column.header for column in table.columns][2:] == ["All cases", "lookalike", "plain", NO_STRATUM]
        assert [block for block in report.blocks if isinstance(block, DisclosureBlock) and block.source == "strata"]
        jsonschema.Draft202012Validator(published_report_schema()).validate(json.loads(report.to_canonical_json()))

    def test_the_stored_surface_round_trips_its_strata(self) -> None:
        bundle = _bundle(*_mixed())
        analysis = make_analysis(decision_surface=bundle_decision_surface(bundle), variant_index=bundle.variant_index)
        storage = EvalStorage(InMemoryDocumentStore())
        storage.save_analysis(analysis)

        loaded = storage.load_analysis(analysis.id, analysis.scope_id)
        assert loaded is not None
        assert loaded.decision_surface.cells == analysis.decision_surface.cells


# =============================================================================
# The stored shapes refuse what cannot be true
# =============================================================================


def _cell_facts(strata: list[StratumFacts], **update: Any) -> CellFacts:
    fields: dict[str, Any] = dict(
        variant_key="v", apparatus_class_id="r", run_ids=["run"], n_observations=4, n_cases=4, strata=strata
    )
    return CellFacts(**(fields | update))


def _stratum(name: str | None, n: int, **update: Any) -> StratumFacts:
    return StratumFacts(**({"stratum": name, "n_observations": n, "n_cases": n} | update))


class TestTheShapeRefusesWhatCannotBeTrue:
    def test_strata_that_add_up_are_accepted(self) -> None:
        assert len(_cell_facts([_stratum("a", 3), _stratum(None, 1)]).strata) == 2

    def test_a_stratum_left_out_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="every one of its observations"):
            _cell_facts([_stratum("a", 3)])

    def test_cases_that_do_not_add_up_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="every one of its cases"):
            _cell_facts([_stratum("a", 3, n_cases=2), _stratum("b", 1)])

    def test_faulted_observations_that_do_not_add_up_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="every one of its faulted observations"):
            _cell_facts([_stratum("a", 3, n_infra_excluded=1), _stratum("b", 1)])

    def test_a_stratum_listed_twice_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="each stratum once"):
            _cell_facts([_stratum("a", 2), _stratum("a", 2)])

    def test_a_breakdown_of_only_undeclared_cases_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="carries no strata"):
            _cell_facts([_stratum(None, 4)])

    def test_a_blank_stratum_on_a_case_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _case("a", "   ")

    def test_two_nominated_axes_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="at most one variation axis"):
            make_template(
                variation_axes=[
                    VariationAxis(name="kind", generator="enum", values=["plain"], stratum=True),
                    VariationAxis(name="tone", generator="enum", values=["casual"], stratum=True),
                ]
            )

    def test_an_llm_axis_cannot_be_the_stratum(self) -> None:
        with pytest.raises(ValidationError, match="written by a model"):
            make_template(variation_axes=[VariationAxis(name="kind", generator="llm", stratum=True)])


# =============================================================================
# Generation stamps the stratum; the store reads strata back alone
# =============================================================================


class _KeepRecordingStore(InMemoryDocumentStore):
    """The reference store, noting the fields each batch read asked to keep."""

    def __init__(self) -> None:
        super().__init__()
        self.kept: list[tuple[str, tuple[str, ...]]] = []

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        self.kept.append((doc_type, tuple(keep)))
        return super().get_many(doc_type, doc_ids, scope_id, exclude=exclude, keep=keep)


class TestGenerationAndTheStore:
    async def test_a_generated_case_takes_the_nominated_axis_value_as_its_stratum(self) -> None:
        template = make_template(
            variation_axes=[
                VariationAxis(name="kind", generator="enum", values=["plain", "lookalike"], stratum=True),
                VariationAxis(name="tone", generator="enum", values=["casual", "cocky"]),
            ]
        )
        storage = EvalStorage(InMemoryDocumentStore())
        generated = await generate_variations(template, 4, storage=storage, scope_id=_SCOPE, blocking_executor=None)

        assert len(generated.cases) == 4
        assert all(case.stratum == case.variation_params["kind"] for case in generated.cases)
        stored = storage.query_test_cases(_SCOPE, template_id=template.id)
        assert sorted(case.stratum or "" for case in stored) == ["lookalike", "lookalike", "plain", "plain"]

    async def test_a_template_nominating_no_axis_generates_cases_declaring_none(self) -> None:
        template = make_template(variation_axes=[VariationAxis(name="tone", generator="enum", values=["casual"])])
        storage = EvalStorage(InMemoryDocumentStore())
        generated = await generate_variations(template, 1, storage=storage, scope_id=_SCOPE, blocking_executor=None)
        assert [case.stratum for case in generated.cases] == [None]

    def test_the_store_reads_back_each_cases_stratum_and_skips_an_absent_one(self) -> None:
        storage = EvalStorage(InMemoryDocumentStore())
        storage.save_test_case(_case("a", "plain"))
        storage.save_test_case(_case("b", None))

        read = {case.id: case.stratum for case in storage.load_case_strata(["a", "b", "missing"], _SCOPE)}
        assert read == {"a": "plain", "b": None}
        assert storage.load_case_strata(["a"], "another-scope") == []
        assert storage.load_case_strata([], _SCOPE) == []

    def test_assembly_reads_only_each_cases_id_and_stratum(self) -> None:
        """A case carries the host's whole stimulus; assembling a bundle ships none of it."""
        store = _KeepRecordingStore()
        storage = EvalStorage(store)
        source, campaign = _mixed()
        for case in source.load_test_cases_by_ids([*_PLAIN, *_LOOKALIKE, _UNDECLARED], _SCOPE):
            storage.save_test_case(case.model_copy(update={"host_payload": {"messages": ["the whole stimulus"]}}))
        for run in source.load_eval_runs(campaign.run_ids, _SCOPE):
            storage.save_eval_run(run)
            for result in source.query_eval_results_by_run(run.id, _SCOPE):
                storage.save_eval_result(result)

        bundle = _bundle(storage, campaign)
        assert [stratum.stratum for stratum in _cell(bundle, "sonnet").strata] == ["lookalike", "plain", None]
        assert [keep for doc_type, keep in store.kept if doc_type == "eval_test_case"] == [("id", "stratum")]
