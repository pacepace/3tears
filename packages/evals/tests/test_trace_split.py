"""The eval_result / eval_trace split — the seam that makes a result row narrow.

In an illustrative 120 kB result, the turn-by-turn ``trace`` at 90 kB and the OTel spans at
21 kB are 92% payload that no list, aggregate or cost path reads. Those two fields now
live in a sibling ``eval_trace`` document, and this file pins the properties that make the
split safe rather than merely smaller.

Asserted against a recording repository rather than a live database, because every claim
here is about **which documents are written and read** — the one thing a round-trip through
real storage would confirm but not localize.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.contracts.models import EvalResult, EvalTrace, eval_trace_doc_id
from threetears.evals.run.reads import get_result_trace
from threetears.evals.contracts.errors import StorageError
from threetears.evals.contracts.storage import EvalStorage
from threetears.evals.contracts.store_port import omit_paths
from packages.evals.tests.factories import make_eval_result, make_eval_trace


class _RecordingRepo:
    """A repository double that remembers every call, and can be told to fail one write or delete."""

    def __init__(
        self, *, fail_doc_types: frozenset[str] = frozenset(), fail_delete_ids: frozenset[str] = frozenset()
    ) -> None:
        self.container = "documents"
        self.written: list[dict[str, Any]] = []
        self.reads: list[str] = []
        self.deletes: list[str] = []
        self.queried_doc_types: list[str] = []
        self._fail_doc_types = fail_doc_types
        self._fail_delete_ids = fail_delete_ids

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None) -> dict[str, Any]:
        if document.get("doc_type") in self._fail_doc_types:
            raise RuntimeError("storage refused this write")
        self.written.append(document)
        return document

    def get(self, item_id: str, scope_id: str) -> dict[str, Any] | None:
        self.reads.append(item_id)
        return next((d for d in self.written if d["id"] == item_id), None)

    def by_doc_type(
        self, doc_type: str, scope_id: str, *, exclude: list[str] | None = None, **field_eq: Any
    ) -> list[dict[str, Any]]:
        self.queried_doc_types.append(doc_type)
        return [omit_paths(d, exclude or ()) for d in self.written if d["doc_type"] == doc_type]

    def delete(self, item_id: str, scope_id: str) -> bool:
        self.deletes.append(item_id)
        if item_id in self._fail_delete_ids:
            return False
        before = len(self.written)
        self.written = [d for d in self.written if d["id"] != item_id]
        return len(self.written) != before


def _storage(repo: _RecordingRepo) -> EvalStorage:
    return EvalStorage(repo)  # type: ignore[arg-type]


def _result_and_trace() -> tuple[EvalResult, EvalTrace]:
    result = make_eval_result(id="res-1", scope_id="uni-1", eval_run_id="run-1")
    trace = make_eval_trace(result_id="res-1", trace=[{"turn": 1, "role": "candidate", "content": "hi"}])
    return result, trace


class TestTheWriteSplitsAndTheMarkerRecordsWhatHappened:
    def test_a_cell_with_a_trace_writes_two_documents_payload_first(self):
        """Payload before result, so no reader can meet a marker whose document is not there yet."""
        repo = _RecordingRepo()
        result, trace = _result_and_trace()

        _storage(repo).save_eval_result(result, trace)

        assert [d["doc_type"] for d in repo.written] == ["eval_trace", "eval_result"]
        assert repo.written[0]["id"] == eval_trace_doc_id("res-1")
        assert repo.written[1]["has_trace"] is True

    def test_the_result_document_carries_neither_payload(self):
        """The whole point: what makes a query over thousands of these affordable."""
        repo = _RecordingRepo()
        result, trace = _result_and_trace()

        _storage(repo).save_eval_result(result, trace)
        stored_result = repo.written[1]

        assert "trace" not in stored_result
        assert "otel_trace" not in stored_result

    def test_an_empty_payload_writes_no_second_document(self):
        """A row per empty payload is the row count this split exists to keep down."""
        repo = _RecordingRepo()
        result = make_eval_result(id="res-1")

        _storage(repo).save_eval_result(result, make_eval_trace(result_id="res-1", trace=[], otel_trace=[]))

        assert [d["doc_type"] for d in repo.written] == ["eval_result"]
        assert repo.written[0]["has_trace"] is False

    def test_a_failed_payload_write_still_persists_the_measurement(self):
        """The runner counts a False here against run completeness.

        Failing the whole write because a debug payload did not persist would trade the
        measurement for the thing that exists to explain it — and the marker would then be
        promising detail no fetch could find.
        """
        repo = _RecordingRepo(fail_doc_types=frozenset({"eval_trace"}))
        result, trace = _result_and_trace()

        _storage(repo).save_eval_result(result, trace)

        assert [d["doc_type"] for d in repo.written] == ["eval_result"]
        assert repo.written[0]["has_trace"] is False, "the marker must record the write, not the intent"

    def test_the_caller_s_result_is_not_mutated(self):
        """The marker is applied to a copy, so a caller holding this object is not surprised."""
        repo = _RecordingRepo()
        result, trace = _result_and_trace()

        _storage(repo).save_eval_result(result, trace)

        assert result.has_trace is False


class TestReadsPayForWhatTheyAsked:
    def test_a_list_query_never_reads_a_trace_document(self):
        """Asserted on the repository, not inferred from the returned rows.

        A query that returned trace-free results would look identical whether or not it
        had fetched them, which is exactly the assumption worth refusing to make.
        """
        repo = _RecordingRepo()
        result, trace = _result_and_trace()
        store = _storage(repo)
        store.save_eval_result(result, trace)

        rows = store.query_eval_results("uni-1")

        assert [r.id for r in rows] == ["res-1"]
        assert repo.queried_doc_types == ["eval_result"], "a list read touched the payload container"
        assert repo.reads == [], "a list read issued a point read it did not need"

    def test_the_detail_read_composes_the_payload_back(self):
        """Round-trips both fields — the drill-down is the one path that pays for them."""
        repo = _RecordingRepo()
        result, trace = _result_and_trace()
        store = _storage(repo)
        store.save_eval_result(
            result,
            make_eval_trace(
                result_id="res-1",
                trace=[{"turn": 1, "role": "candidate", "content": "hi"}],
                otel_trace=[{"name": "gen_ai.agent.invoke"}],
            ),
        )

        loaded = store.load_eval_result("res-1", "uni-1")
        payload = store.load_eval_trace("res-1", "uni-1")

        assert loaded is not None and loaded.has_trace is True
        assert payload is not None
        assert payload.trace == [{"turn": 1, "role": "candidate", "content": "hi"}]
        assert payload.otel_trace == [{"name": "gen_ai.agent.invoke"}]
        assert payload.result_id == "res-1"

    def test_a_result_with_no_payload_reads_as_none_not_empty(self):
        """``None`` says no document was written; an empty ``EvalTrace`` would say one was."""
        repo = _RecordingRepo()
        store = _storage(repo)
        store.save_eval_result(make_eval_result(id="res-1"), None)

        assert store.load_eval_trace("res-1", "uni-1") is None


class TestDeletesTakeThePayloadWithThem:
    def test_deleting_a_result_deletes_its_trace(self):
        """An orphan payload is unreachable — nothing queries these by anything but a result id."""
        repo = _RecordingRepo()
        result, trace = _result_and_trace()
        store = _storage(repo)
        store.save_eval_result(result, trace)

        assert store.delete_eval_result("res-1", "uni-1") is True

        assert repo.deletes == [eval_trace_doc_id("res-1"), "res-1"]
        assert repo.written == []

    def test_deleting_a_result_that_stored_no_trace_still_reports_the_result(self):
        """The absent payload must not make a successful delete look like a failed one."""
        repo = _RecordingRepo()
        store = _storage(repo)
        store.save_eval_result(make_eval_result(id="res-1"), None)

        assert store.delete_eval_result("res-1", "uni-1") is True

    def test_a_trace_that_survives_its_delete_keeps_the_result_and_raises(self):
        """#650: the trace is ~95% of a result's bytes and only its result's id finds it.

        Deleting the result after a trace delete that did not take would orphan it for good, and
        the caller would hear a clean delete. ``has_trace`` says a trace should be there, so the
        delete is refused and the result survives for a retry.
        """
        repo = _RecordingRepo(fail_delete_ids=frozenset({eval_trace_doc_id("res-1")}))
        result, trace = _result_and_trace()
        store = _storage(repo)
        store.save_eval_result(result, trace)

        with pytest.raises(StorageError, match="left intact"):
            store.delete_eval_result("res-1", "uni-1")

        assert "res-1" not in repo.deletes
        assert store.load_eval_result("res-1", "uni-1") is not None
        assert store.load_eval_trace("res-1", "uni-1") is not None

    def test_a_result_stamped_without_a_trace_ignores_an_empty_trace_delete(self):
        """``has_trace=False``: a trace delete that removes nothing is the expected case."""
        repo = _RecordingRepo(fail_delete_ids=frozenset({eval_trace_doc_id("res-1")}))
        store = _storage(repo)
        store.save_eval_result(make_eval_result(id="res-1"), None)

        assert store.delete_eval_result("res-1", "uni-1") is True
        assert repo.written == []

    def test_a_marker_whose_trace_is_already_gone_does_not_make_the_result_undeletable(self, caplog):
        """``has_trace=True`` with no document: nothing to orphan, so refusing would strand the result."""
        import logging

        repo = _RecordingRepo()
        store = _storage(repo)
        store.replace_eval_result(make_eval_result(id="res-1", has_trace=True), if_match=None)

        with caplog.at_level(logging.WARNING, logger="threetears.evals.contracts.storage"):
            assert store.delete_eval_result("res-1", "uni-1") is True

        assert repo.written == []
        assert any("has no trace document" in r.getMessage() for r in caplog.records)

    def test_an_id_naming_another_document_type_deletes_nothing(self):
        """A run's id is not a result's: the delete must not remove the run document."""
        repo = _RecordingRepo()
        repo.written.append({"id": "run-1", "doc_type": "eval_run", "scope_id": "uni-1"})
        store = _storage(repo)

        assert store.delete_eval_result("run-1", "uni-1") is False
        assert [d["id"] for d in repo.written] == ["run-1"]


class TestTheMarkerAndTheDocumentAreCheckedAgainstEachOther:
    def test_a_marker_with_no_document_is_logged_not_silently_dropped(self, caplog):
        """`has_trace=True` with no document means a payload was lost, not that there is none.

        The marker is written from the payload write's own OUTCOME, so the two disagreeing
        is an apparatus inconsistency — a payload deleted without its result, or a write
        that landed half. Both read surfaces render it identically to an ordinary absence,
        which is the one reading that is certainly wrong, so it has to reach an error query.
        """
        import logging

        repo = _RecordingRepo()
        store = _storage(repo)
        result = make_eval_result(id="res-1", has_trace=True)

        with caplog.at_level(logging.WARNING, logger="threetears.evals.run.reads"):
            assert get_result_trace(store, result) is None

        assert any("claims a stored trace" in r.getMessage() for r in caplog.records)

    def test_an_honest_absence_is_not_logged(self, caplog):
        """`has_trace=False` and no document agree — nothing happened worth a warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="threetears.evals.run.reads"):
            assert get_result_trace(_storage(_RecordingRepo()), make_eval_result(id="res-1")) is None

        assert not caplog.records
