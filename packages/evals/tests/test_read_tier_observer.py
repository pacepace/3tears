"""The read tier's observer: ``pivot`` and ``export_results`` each log their size and wall time (#652).

The read tier projects and aggregates a whole scope in Python on every call, calibrated to a corpus of
dozens-to-hundreds of cells. These pin that a host can see that premise being crossed: each call logs one
``eval.read_tier`` line naming what it read (``runs_in``, ``results_in``), what it projected
(``records_out``), the pivot's ``cells_out`` and ``elapsed_ms``, and a call whose projection is past
``READ_TIER_ROW_BUDGET`` logs it at WARNING, naming the constant.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``reads._observe_read``: logging at INFO regardless of the budget; comparing ``>=`` rather than ``>``;
  dropping the ``over READ_TIER_ROW_BUDGET`` suffix.
- ``reads.pivot`` / ``reads.export_results``: not calling the observer; counting the status-narrowed runs
  as ``runs_in``; counting results as ``records_out``.
"""

from __future__ import annotations

import logging
import re

import pytest

from threetears.evals.analysis import READ_TIER_ROW_BUDGET, export_results, pivot, reads
from threetears.evals.analysis.reporting import project_score_records
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.run.reads import list_runs
from packages.evals.tests.factories import make_eval_result, make_eval_run, memory_storage
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

PROFILE = toyhost_profile(every_seat=True)
LOGGER = "threetears.evals.analysis.reads"
LINE = re.compile(r"eval\.read_tier lens=(?P<lens>\S+) scope_id=(?P<scope>\S+) (?P<fields>.*)")


def _toy_scope() -> tuple[EvalStorage, str]:
    """Two completed runs of two models over three cases, and one failed run the default status filter drops."""
    storage, _ = memory_storage()
    scope_id = "uni-observer"
    for model, status in (("sonnet", "completed"), ("haiku", "completed"), ("opus", "failed")):
        run = make_eval_run(scope_id=scope_id, candidate_model=model, status=status, template_id="tpl-1")
        storage.save_eval_run(run)
        for case in range(3):
            storage.save_eval_result(
                make_eval_result(
                    id=f"{model}-{case}",
                    eval_run_id=run.id,
                    scope_id=scope_id,
                    model=model,
                    test_case_id=f"tc-{case}",
                )
            )
    return storage, scope_id


def _lister(storage: EvalStorage):  # type: ignore[no-untyped-def]
    return lambda scope, **kwargs: list_runs(toyhost_host(storage=storage), scope, **kwargs)


def _expected_records(storage: EvalStorage, scope_id: str) -> int:
    """The completed runs' score records, projected independently of the lens."""
    every = list_runs(toyhost_host(storage=storage), scope_id)
    completed = [run for run in every if run.status == "completed"]
    projection = project_score_records(
        completed,
        storage.query_eval_results(scope_id),
        known_run_ids={run.id for run in every},
        archived_run_ids=set(),
        profile=PROFILE,
    )
    return len(projection.records)


def _pivot(storage: EvalStorage, scope_id: str, **overrides: str):  # type: ignore[no-untyped-def]
    arguments = {"row_factor": "test_case_id", "column_factor": "model", **overrides}
    return pivot(storage, scope_id, list_runs=_lister(storage), profile=PROFILE, **arguments)


def _the_line(caplog: pytest.LogCaptureFixture) -> tuple[logging.LogRecord, str, str, dict[str, str]]:
    """The one ``eval.read_tier`` record, its lens, its scope and its ``key=value`` fields."""
    (record,) = [r for r in caplog.records if r.name == LOGGER and r.getMessage().startswith("eval.read_tier")]
    match = LINE.match(record.getMessage())
    assert match, record.getMessage()
    fields = dict(pair.split("=", 1) for pair in match["fields"].split() if "=" in pair)
    return record, match["lens"], match["scope"], fields


class TestEachCallReportsWhatItReadAndHowLongItTook:
    def test_a_pivot_logs_its_counts_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        storage, scope_id = _toy_scope()
        with caplog.at_level(logging.INFO, logger=LOGGER):
            table = _pivot(storage, scope_id)

        record, lens, scope, fields = _the_line(caplog)
        assert record.levelno == logging.INFO
        assert (lens, scope) == ("pivot", scope_id)
        assert fields["runs_in"] == "3", "every run the scope lists, the failed one included: it was read"
        assert fields["results_in"] == "9"
        records = _expected_records(storage, scope_id)
        assert records > 0
        assert fields["records_out"] == str(records)
        assert fields["cells_out"] == str(len(table.cells)) == "6", "three cases by two completed models"
        assert float(fields["elapsed_ms"]) >= 0.0
        assert "READ_TIER_ROW_BUDGET" not in record.getMessage()

    def test_an_export_logs_its_counts_at_info_and_has_no_cells(self, caplog: pytest.LogCaptureFixture) -> None:
        storage, scope_id = _toy_scope()
        with caplog.at_level(logging.INFO, logger=LOGGER):
            export = export_results(storage, scope_id, list_runs=_lister(storage), profile=PROFILE)

        record, lens, scope, fields = _the_line(caplog)
        assert record.levelno == logging.INFO
        assert (lens, scope) == ("export_results", scope_id)
        assert (fields["runs_in"], fields["results_in"]) == ("3", "9")
        assert fields["records_out"] == str(_expected_records(storage, scope_id)) == str(export.n_records)
        assert "cells_out" not in fields
        assert float(fields["elapsed_ms"]) >= 0.0

    def test_a_refused_pivot_still_reports_the_read_it_paid_for(self, caplog: pytest.LogCaptureFixture) -> None:
        storage, scope_id = _toy_scope()
        with caplog.at_level(logging.INFO, logger=LOGGER), pytest.raises(ValidationFailedError):
            _pivot(storage, scope_id, row_factor="no_such_axis")

        _, lens, _, fields = _the_line(caplog)
        assert lens == "pivot"
        assert fields["cells_out"] == "refused"
        assert fields["results_in"] == "9"


class TestPastTheBudgetTheLineIsAWarning:
    def test_the_budget_is_stated(self) -> None:
        assert READ_TIER_ROW_BUDGET == reads.READ_TIER_ROW_BUDGET == 50_000

    def test_a_pivot_over_the_budget_warns_and_names_it(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        storage, scope_id = _toy_scope()
        budget = _expected_records(storage, scope_id) - 1
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", budget)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            _pivot(storage, scope_id)

        record, lens, _, _ = _the_line(caplog)
        assert lens == "pivot"
        assert record.levelno == logging.WARNING
        assert record.getMessage().endswith(f"over READ_TIER_ROW_BUDGET={budget}")

    def test_an_export_over_the_budget_warns_and_names_it(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        storage, scope_id = _toy_scope()
        budget = _expected_records(storage, scope_id) - 1
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", budget)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            export_results(storage, scope_id, list_runs=_lister(storage), profile=PROFILE)

        record, lens, _, _ = _the_line(caplog)
        assert lens == "export_results"
        assert record.levelno == logging.WARNING
        assert record.getMessage().endswith(f"over READ_TIER_ROW_BUDGET={budget}")

    def test_a_call_exactly_at_the_budget_is_still_info(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        storage, scope_id = _toy_scope()
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", _expected_records(storage, scope_id))
        with caplog.at_level(logging.INFO, logger=LOGGER):
            _pivot(storage, scope_id)

        record, _, _, _ = _the_line(caplog)
        assert record.levelno == logging.INFO
