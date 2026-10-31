"""A campaign's bundle reads its results without the verbatim fields, which no reading of it reads.

A stored result carries what its cell did verbatim: a world's event log (``world_events``) and its kind's own payload
(``kind_payload``). They are most of a result's bytes, and a 41-run campaign's bundle read whole held them all at once
(a 2 GB pod killed on 2026-10-30). Every read of results the bundle makes (its readings, by type or narrowed, and
its kept trials, by id) asks the store to leave them out, so:

- **the bundle is the one a whole read gives**: assembled over a store that ships every field anyway, it is equal, a
  host measure named as a world event's field (``turn``) included: the measure walk reads no event as a measurement;
- **the read names both fields** for the store to leave out, so a store that projects never ships them;
- **its peak memory does not grow with them**: a campaign whose results carry tens of kilobytes verbatim each is
  assembled in about the memory of one whose results carry none.

Each holds on both read paths: the store's own narrowed read, and the engine's reads by key and by id over a store
behind the document-store port alone (``EngineOnlyStore``), the path a host's store without ``ResultReadings`` takes.
"""

from __future__ import annotations

import tracemalloc
from dataclasses import replace
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import pytest

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.host import MeasureRegistry
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.kernel.run_pages import VERBATIM_RESULT_FIELDS
from threetears.evals.schema import EvalResult, RubricScore
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.schema import WorldEvent
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.factories import fixture_variant_key, make_campaign, make_eval_result, make_eval_run
from packages.evals.tests.read_path_support import EngineOnlyStore

_SCOPE = "uni-1"
#: A host measure named as a world event's field is, which these results do not report: a whole read's walk pooled each
#: event's turn into it.
_TURN = MetricDescriptor(
    name="turn",
    reader_name="Turns",
    data_type="numeric",
    family=next(descriptor.family for descriptor in TOYHOST_MEASURES if descriptor.name == "field_accuracy"),
    transferability_class="mechanical",
    attribution_scope="end_to_end",
    description="How many turns the cell took.",
    higher_is_better=False,
    merit_axis="quality",
)
_PROFILE = replace(
    toyhost_profile(),
    measures=MeasureRegistry((*TOYHOST_MEASURES, _TURN), families=toyhost_profile().measures.families),
)
_TONE = "conversation.tone"
_CASES = [f"case-{index:02d}" for index in range(12)]


class _ShipsEveryField(InMemoryDocumentStore):
    """A store that ignores ``exclude`` on every read of rows, shipping every field: what a whole read of a result gives."""

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        exclude: Sequence[str] = (),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        return super().by_doc_type(
            doc_type, scope_id, order_by=order_by, descending=descending, limit=limit, field_in=field_in, **field_eq
        )

    def iter_narrowed(
        self, doc_type: str, scope_id: str, *, exclude: Sequence[str] = (), **kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        return super().iter_narrowed(doc_type, scope_id, **kwargs)

    def get_many(
        self, doc_type: str, doc_ids: Sequence[str], scope_id: str, *, exclude: Sequence[str] = (), **kwargs: Any
    ) -> list[dict[str, Any]]:
        return super().get_many(doc_type, doc_ids, scope_id, **kwargs)


class _RecordsExclude(InMemoryDocumentStore):
    """A store that records what each read of results asked it to leave out."""

    def __init__(self) -> None:
        super().__init__()
        self.excluded: list[tuple[str, ...]] = []

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        exclude: Sequence[str] = (),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        if doc_type == "eval_result":
            self.excluded.append(tuple(exclude))
        return super().by_doc_type(
            doc_type,
            scope_id,
            order_by=order_by,
            descending=descending,
            limit=limit,
            exclude=exclude,
            field_in=field_in,
            **field_eq,
        )

    def iter_narrowed(
        self, doc_type: str, scope_id: str, *, exclude: Sequence[str] = (), **kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        if doc_type == "eval_result":
            self.excluded.append(tuple(exclude))
        return super().iter_narrowed(doc_type, scope_id, exclude=exclude, **kwargs)

    def get_many(
        self, doc_type: str, doc_ids: Sequence[str], scope_id: str, *, exclude: Sequence[str] = (), **kwargs: Any
    ) -> list[dict[str, Any]]:
        if doc_type == "eval_result":
            self.excluded.append(tuple(exclude))
        return super().get_many(doc_type, doc_ids, scope_id, exclude=exclude, **kwargs)


def _result(model: str, case_id: str, k: int, *, verbatim: int) -> EvalResult:
    """One classification of one case, carrying ``verbatim`` characters of event log and payload when more than none."""
    return make_eval_result(
        id=f"{model}-{case_id}-{k}",
        scope_id=_SCOPE,
        eval_run_id=f"run-{model}",
        test_case_id=case_id,
        model=model,
        k_iteration=k,
        scored_at=f"2026-10-01T00:{_CASES.index(case_id):02d}:0{k}+00:00",
        variant_key=fixture_variant_key(model),
        host_measures={
            "match": (len(case_id) + k + len(model)) % 3 != 0,
            "fields_stripped": float(k + len(case_id)),
        },
        rubric_scores=[
            RubricScore(
                dim=_TONE, score=1 + (k + len(model)) % 5, scale="ordinal", axis="capability", judge_temperature=0.0
            )
        ],
        world_events=(
            [
                WorldEvent(
                    kind="event",
                    dimension="payment_hold",
                    condition=f"posted-{index}",
                    caused_by="world",
                    event="x" * 400,
                    turn=index + k,
                )
                for index in range(verbatim // 800)
            ]
            if verbatim
            else None
        ),
        kind_payload={"transcript": "y" * (verbatim // 2)} if verbatim else None,
    )


def _store(store: InMemoryDocumentStore, *, verbatim: int, engine: bool) -> tuple[EvalStorage, EvalCampaign]:
    """Two arms over twelve cases, three repeats each, and a campaign over them, read through ``store`` as it is, or
    behind the port alone (``engine``)."""
    storage = EvalStorage(EngineOnlyStore(store) if engine else store)
    run_ids = []
    for model in ("sonnet", "haiku"):
        run = make_eval_run(
            id=f"run-{model}",
            scope_id=_SCOPE,
            candidate_model=model,
            k_runs=3,
            test_case_ids=_CASES,
            rubric_scales={_TONE: "ordinal"},
            status="completed",
            created_at="2026-10-01T00:00:00+00:00",
        )
        storage.save_eval_run(run)
        for case_id in _CASES:
            for k in (1, 2, 3):
                storage.save_eval_result(_result(model, case_id, k, verbatim=verbatim))
        run_ids.append(run.id)
    campaign = make_campaign(id="campaign-verbatim", scope_id=_SCOPE, run_ids=run_ids)
    storage.save_campaign(campaign)
    return storage, campaign


def _bundle(storage: EvalStorage, campaign: EvalCampaign) -> AnalysisContextBundle:
    return assemble_context_bundle(campaign, storage=storage, profile=_PROFILE)


def _peak(read: Callable[[], object]) -> int:
    """The peak memory ``read`` allocates above what was held before it, in bytes."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        read()
        return tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()


def _without_volatile(bundle: AnalysisContextBundle) -> Mapping[str, Any]:
    """The bundle as data, less what differs between two assemblies of one campaign (when each was assembled)."""
    return {name: value for name, value in bundle.model_dump(mode="json").items() if name != "assembled_at"}


#: The two read paths: the store's own narrowed read, and the engine's behind the port alone.
_PATHS = pytest.mark.parametrize("engine", (False, True), ids=("store", "engine"))


@_PATHS
def test_the_bundle_is_the_one_a_whole_read_of_its_results_gives(engine: bool) -> None:
    slim = _bundle(*_store(InMemoryDocumentStore(), verbatim=4_000, engine=engine))
    whole = _bundle(*_store(_ShipsEveryField(), verbatim=4_000, engine=engine))

    assert _without_volatile(slim) == _without_volatile(whole)


@_PATHS
def test_the_read_asks_the_store_to_leave_both_verbatim_fields_out(engine: bool) -> None:
    store = _RecordsExclude()
    _bundle(*_store(store, verbatim=0, engine=engine))

    assert store.excluded, "the bundle read its results"
    assert all(set(VERBATIM_RESULT_FIELDS) <= set(excluded) for excluded in store.excluded), store.excluded
    assert set(VERBATIM_RESULT_FIELDS) == {"world_events", "kind_payload"}


@_PATHS
def test_its_peak_memory_does_not_grow_with_what_its_results_carry_verbatim(engine: bool) -> None:
    bare = _store(InMemoryDocumentStore(), verbatim=0, engine=engine)
    heavy = _store(InMemoryDocumentStore(), verbatim=32_000, engine=engine)
    _bundle(*bare)
    _bundle(*heavy)

    alone = _peak(lambda: _bundle(*bare))
    carrying = _peak(lambda: _bundle(*heavy))

    # 72 results × 32 KB verbatim is 2.3 MB of what a whole read would have held; leaving it out holds none of it.
    assert carrying < alone * 1.25, (alone, carrying)
