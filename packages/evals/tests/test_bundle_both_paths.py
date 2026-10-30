"""A campaign's bundle is the same bundle whichever path its results are read on (W4b P6).

Assembly reads a campaign's results as their readings (``EvalStorage.iter_result_readings``): on the store's own path
(``store``) when the store narrows its rows itself (``ResultReadings``), else on the engine's (``engine``: each run's
keys, then the results by id a page at a time). The trials the bundle keeps are then read whole, by id, in one more
call, for the judge's reasons a reading leaves out. Checked over the enriched production-shaped campaign (every record
list filled: re-judges, repeats, second judgings, background deliveries, latency) held in one reference store, read
once through the store as it is and once through the same store behind the document-store port alone:

- the bundle's JSON and its fingerprint are byte-identical;
- the kept trial rows are identical, and show the judge's reasons, which only the whole results carry;
- a kept trial deleted between the two reads is a conflict (409) the caller reads again, never a row without its
  reasons.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
import pytest

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.analysis import service as analysis_service
from threetears.evals.kernel.errors import ConflictError
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.schema.models import EvalResult
from threetears.evals.schema.store_port import DocumentStore, ResultReadings
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.hot_path_support import HotPathCorpus, hot_path_corpus
from packages.evals.tests.read_path_support import EngineOnlyStore
from packages.evals.tests.test_result_reading_fields import ENRICHED_CASES, REPORT_NOW, enrich

#: The scope the campaign is held in, its own so no other test's corpus shares its documents.
_SCOPE = "bundle-both-paths"


@functools.cache
def _enriched() -> tuple[HotPathCorpus, InMemoryDocumentStore]:
    """The enriched campaign, built once, with the reference store it is held in."""
    store = InMemoryDocumentStore()
    corpus = hot_path_corpus(scale=1, scope_id=_SCOPE, cases_per_run=ENRICHED_CASES, store=store)
    enrich(corpus.host, _SCOPE)
    return corpus, store


def _assembled(store: DocumentStore) -> AnalysisContextBundle:
    """The enriched campaign's bundle, its results read through ``store``."""
    corpus, _ = _enriched()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(analysis_service, "utc_now_iso", lambda: REPORT_NOW)
        return assemble_context_bundle(corpus.campaign, storage=EvalStorage(store), profile=corpus.host.profile)


@functools.cache
def _both() -> tuple[AnalysisContextBundle, AnalysisContextBundle]:
    """The bundle read on the store's path, then on the engine's."""
    _, store = _enriched()
    return _assembled(store), _assembled(EngineOnlyStore(store))


def _json(bundle: AnalysisContextBundle) -> str:
    return json.dumps(bundle.to_dict(), sort_keys=True, ensure_ascii=False)


def test_the_two_stores_take_the_two_paths() -> None:
    _, store = _enriched()

    assert isinstance(store, ResultReadings), "the reference store offers its own narrowed reads"
    assert not isinstance(EngineOnlyStore(store), ResultReadings), "behind the port alone it offers none"


def test_the_bundle_and_its_fingerprint_are_byte_identical_on_both_paths() -> None:
    on_store, on_engine = _both()

    assert on_store.trials, "the campaign keeps no trial, so the comparison checks no kept row"
    assert _json(on_store) == _json(on_engine)
    assert on_store.fingerprint() == on_engine.fingerprint()


def test_the_kept_trial_rows_are_identical_and_show_the_judges_reasons() -> None:
    on_store, on_engine = _both()

    assert [row.model_dump() for row in on_store.trials] == [row.model_dump() for row in on_engine.trials]
    reasons = [judgement.reason for row in on_store.trials for judgement in row.judge]
    assert any(reasons), "no kept row shows a reason the judge gave: the rows were not built from whole results"


class _TrialGoneBeforeItsRow(EvalStorage):
    """The campaign's storage, missing the first kept trial when the kept trials are read whole: deleted between the
    read of the readings and the read of the rows."""

    def load_eval_results(
        self, result_ids: Sequence[str], scope_id: str, /, *, elide: Sequence[str] = ()
    ) -> list[EvalResult]:
        return super().load_eval_results(result_ids[1:], scope_id, elide=elide)


def test_a_kept_trial_gone_before_its_row_is_read_is_a_conflict_to_read_again() -> None:
    corpus, store = _enriched()

    with pytest.raises(ConflictError) as refused:
        assemble_context_bundle(corpus.campaign, storage=_TrialGoneBeforeItsRow(store), profile=corpus.host.profile)

    assert refused.value.status_code == 409
    assert refused.value.remedy == "a member run changed while the report was read; read it again"
