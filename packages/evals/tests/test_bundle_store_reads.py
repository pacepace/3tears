"""Assembling a campaign's bundle asks the store a fixed number of questions, but for its results on the engine path.

A host's store may open a connection per call, so a read per run, per template, per insight or per trial is a round
trip each, and a 16-run campaign of about 1,800 results ran past a host's 120s call budget that way. Every read the
assembler makes is a batch: the runs, the cases' strata, the ratings, the templates, the judge profiles, the insight
ledger, the archived flag of each analysis an insight names, the kept trials' judge evidence, and the kept trials'
results read whole (for the judge's reasons).

The results themselves (archived members' included, when a declared control needs them) are read as their readings
(``EvalStorage.iter_result_readings``), and how many calls that takes is the path's (owner ruling D7, 2026-10-30):

- **the store's path**, over a store that narrows its own rows (``ResultReadings``): one call, however large the
  campaign;
- **the engine's path**, over a store behind the document-store port alone: one key read per run read, then one read by
  id per ``READINGS_PAGE`` results.

These tests count the store calls of an assembly over a small campaign and over a larger one, through the real
``EvalStorage``, on both paths.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.kernel import CampaignArm
from threetears.evals.kernel.storage import READINGS_PAGE, EvalStorage
from threetears.evals.schema import GroupMeasure, JudgedArtifact, JudgeEvidence
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import (
    DEFAULT_SUBJECT,
    fixture_variant_key,
    make_calibration_rating,
    make_aspect_campaign,
    make_eval_result,
    make_eval_run,
    make_eval_trace,
    make_insight,
    make_template,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.read_path_support import EngineOnlyStore

_SCOPE = "uni-1"

#: The results each run stores.
_CASES = 3


class _CallCounter(InMemoryDocumentStore):
    """The reference store, counting every read by method and ``doc_type`` (``get`` by id has none)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: Counter[tuple[str, str]] = Counter()

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        self.calls["get", "-"] += 1
        return super().get(doc_id, scope_id)

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        self.calls["get_with_etag", "-"] += 1
        return super().get_with_etag(doc_id, scope_id)

    def get_many(self, doc_type: str, doc_ids: Sequence[str], scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls["get_many", doc_type] += 1
        return super().get_many(doc_type, doc_ids, scope_id, **kwargs)

    def by_doc_type(
        self, doc_type: str, scope_id: str, *, field_in: Mapping[str, Sequence[Any]] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.calls["by_doc_type", doc_type] += 1
        return super().by_doc_type(doc_type, scope_id, field_in=field_in, **kwargs)

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        self.calls["iter_by_doc_type", doc_type] += 1
        return super().iter_by_doc_type(doc_type, scope_id)

    def iter_narrowed(self, doc_type: str, scope_id: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        self.calls["iter_narrowed", doc_type] += 1
        return super().iter_narrowed(doc_type, scope_id, **kwargs)

    def grouped(
        self, doc_type: str, scope_id: str, *, by: Sequence[str], measures: Mapping[str, GroupMeasure], **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.calls["grouped", doc_type] += 1
        return super().grouped(doc_type, scope_id, by=by, measures=measures, **kwargs)


def _assembly_reads(
    *, runs: int, archived: int, templates: int, insights: int, capable: bool
) -> Counter[tuple[str, str]]:
    """The store calls one assembly makes over a campaign of this size, whose declared control needs its archived runs.

    :param runs: Member runs, each of ``templates`` in turn, with three traced, judged and rated results apiece.
    :param archived: How many of them are archived.
    :param templates: Distinct templates the runs ran.
    :param insights: Prior insights of the subject, each minted by an analysis of its own.
    :param capable: Read through the store as it is (the store's path), or behind the port alone (the engine's).
    """
    store = _CallCounter()
    storage = EvalStorage(store if capable else EngineOnlyStore(store))
    for index in range(templates):
        storage.save_template(make_template(id=f"tpl-{index}", name=f"template-{index}"))
    run_ids = []
    for index in range(runs):
        run = make_eval_run(
            id=f"run-{index}", template_id=f"tpl-{index % templates}", status="completed", archived=index < archived
        )
        storage.save_eval_run(run)
        run_ids.append(run.id)
        for case in range(_CASES):
            result = make_eval_result(
                id=f"res-{index}-{case}", eval_run_id=run.id, test_case_id=f"tc-{case}", scope_id=_SCOPE
            )
            storage.save_eval_result(
                result,
                make_eval_trace(
                    result_id=result.id,
                    eval_run_id=run.id,
                    judge_evidence=JudgeEvidence(case_material=f"case {case}", artifact="reply"),
                    judged_artifact=JudgedArtifact.TRANSCRIPT,
                ),
            )
            storage.save_calibration_rating(make_calibration_rating(run_id=run.id, result_id=result.id))
    for index in range(insights):
        storage.save_insight(
            make_insight(
                id=f"ins-{index}", subject_id=DEFAULT_SUBJECT.subject_id, source_analysis_id=f"analysis-{index}"
            )
        )
    campaign = make_aspect_campaign(
        run_ids=run_ids,
        subject_id=DEFAULT_SUBJECT.subject_id,
        control=CampaignArm(label="control", role="control", variant_key=fixture_variant_key("sonnet")),
    )
    store.calls.clear()

    bundle = assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())

    assert len(bundle.run_ids) == runs - archived
    assert len(bundle.trials) == _CASES * (runs - archived)
    return store.calls


#: Every read but the results', one batch call each, on either path. The last is the kept trials' results read whole.
_BATCHES = Counter(
    {
        ("get_many", "eval_run"): 1,
        ("get_many", "eval_template"): 1,
        ("by_doc_type", "calibration_rating"): 1,
        ("by_doc_type", "eval_judge_profile"): 1,
        ("by_doc_type", "eval_insight"): 1,
        ("get_many", "eval_analysis"): 1,
        ("get_many", "eval_test_case"): 1,
        ("get_many", "eval_trace"): 1,
        ("get_many", "eval_result"): 1,
    }
)


def test_on_the_stores_path_the_store_calls_do_not_grow_with_the_campaign() -> None:
    small = _assembly_reads(runs=2, archived=1, templates=1, insights=1, capable=True)
    large = _assembly_reads(runs=16, archived=3, templates=4, insights=6, capable=True)

    assert large == small, f"reads that grew with the campaign: {sorted((large - small).items())}"


def test_on_the_stores_path_every_read_is_one_batch_call() -> None:
    reads = _assembly_reads(runs=6, archived=2, templates=3, insights=4, capable=True)

    assert reads == _BATCHES + Counter({("iter_narrowed", "eval_result"): 1})


def test_on_the_engines_path_the_results_take_a_key_read_per_run_and_a_read_per_page() -> None:
    for runs, archived in ((2, 1), (16, 3)):
        reads = _assembly_reads(runs=runs, archived=archived, templates=2, insights=3, capable=False)
        # Every member run's results are read, the archived ones' too: the declared control may have been curated out.
        pages = math.ceil(runs * _CASES / READINGS_PAGE)

        assert reads == _BATCHES + Counter({("by_doc_type", "eval_result"): runs, ("get_many", "eval_result"): pages})
