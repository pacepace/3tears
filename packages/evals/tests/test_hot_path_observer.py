"""Where a bundle assembly's and a report read's time went: one ``eval.read_tier`` line each, by phase.

A slow report read in production is answered by its log line, not a guess: ``lens=assemble_context_bundle`` for
each campaign bundle assembled, ``lens=report_read`` for each read, each with ``store_ms``, ``hydrate_ms``,
``fold_ms``, ``render_ms``, ``elapsed_ms``, ``runs_in``, ``results_in``, ``path`` and ``store_calls``. A phase not
measured is empty, never zero; the phases add up to the elapsed time, never counting a millisecond twice.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``phase_log.charged``: charging a block's whole time rather than less its nested charges; not reaching the
  enclosing call's line, from its own thread or from a ``run_blocking`` worker.
- ``phase_log.phase_logged``: not charging the residual phase; writing ``0.0`` for a phase nothing charged; not
  logging when the block raised, or swallowing what it raised; leaving the call open in the context after it ends.
- ``phase_log.took_in``: not reaching the enclosing call's line.
- ``service._assembled_campaign_bundle``: no line; counting one aspect's runs and results rather than every aspect's
  (a run belongs to one aspect at most, so the aspects' counts add).
- ``ops.analysis.report_read``: no line; serializing outside ``charged("render")``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time

import pytest

from threetears.evals.analysis import assemble_campaign_bundle
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.kernel.phase_log import charged, phase_logged, took_in
from threetears.evals.ops import report_read
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.factories import analysed_aspect, empty_bundle, make_analysis
from packages.evals.tests.hot_path_support import HOT_PATH_SCOPE, hot_path_corpus
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, OpsFixture, ops_fixture

LOGGER = "threetears.evals.kernel.phase_log"
LINE = re.compile(r"eval\.read_tier lens=(?P<lens>\S+) (?P<fields>.*)")
PHASE_FIELDS = ("store_ms", "hydrate_ms", "fold_ms", "render_ms")
#: Every field the line carries, in its order: the names a host's log queries are written against.
FIELD_ORDER = (
    "scope_id",
    "campaign_id",
    "runs_in",
    "results_in",
    *PHASE_FIELDS,
    "elapsed_ms",
    "path",
    "store_calls",
)
#: Each phase prints to a tenth of a millisecond, so a sum of four can round away from the whole by this much.
ROUNDING_MS = 0.25
NAP_S = 0.02


def _lines(caplog: pytest.LogCaptureFixture) -> list[tuple[logging.LogRecord, str, dict[str, str]]]:
    """Every phase line logged, in order: its record, its lens and its ``key=value`` fields."""
    found = []
    for record in caplog.records:
        if record.name != LOGGER:
            continue
        match = LINE.match(record.getMessage())
        assert match, record.getMessage()
        pairs = [pair.split("=", 1) for pair in match["fields"].split()]
        found.append((record, match["lens"], {key: value for key, value in pairs}))
    return found


def _ms(fields: dict[str, str], name: str) -> float:
    return float(fields[name])


def _phases_add_up(fields: dict[str, str]) -> None:
    measured = sum(float(fields[name]) for name in PHASE_FIELDS if fields[name])
    assert abs(measured - _ms(fields, "elapsed_ms")) <= ROUNDING_MS, fields


class TestTheLine:
    def test_an_uncharged_call_is_all_residual_and_leaves_what_it_did_not_measure_empty(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=LOGGER), phase_logged("probe", scope_id="s1", campaign_id="c1"):
            time.sleep(NAP_S)

        ((record, lens, fields),) = _lines(caplog)
        assert record.levelno == logging.INFO
        assert lens == "probe"
        assert tuple(fields) == FIELD_ORDER
        assert (fields["scope_id"], fields["campaign_id"]) == ("s1", "c1")
        assert fields["fold_ms"] == f"{_ms(fields, 'elapsed_ms'):.1f}"
        assert _ms(fields, "elapsed_ms") >= NAP_S * 1000
        for empty in ("store_ms", "hydrate_ms", "render_ms", "runs_in", "results_in", "path", "store_calls"):
            assert fields[empty] == "", f"{empty} was not measured, so it is empty rather than zero"

    def test_the_residual_phase_is_the_call_s_to_name(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            with phase_logged("probe", scope_id="s", campaign_id="c", residual="render"):
                pass

        ((_, _, fields),) = _lines(caplog)
        assert fields["fold_ms"] == "" and fields["render_ms"] != ""

    def test_a_charged_block_is_its_phase_and_the_rest_is_the_residual(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=LOGGER), phase_logged("probe", scope_id="s", campaign_id="c"):
            time.sleep(NAP_S)
            with charged("render"):
                time.sleep(NAP_S)

        ((_, _, fields),) = _lines(caplog)
        assert _ms(fields, "render_ms") >= NAP_S * 1000
        assert _ms(fields, "fold_ms") >= NAP_S * 1000
        _phases_add_up(fields)

    def test_a_phase_charged_inside_another_is_not_counted_twice(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=LOGGER), phase_logged("probe", scope_id="s", campaign_id="c"):
            with charged("render"):
                with charged("store"):
                    time.sleep(2 * NAP_S)
                time.sleep(NAP_S / 4)

        ((_, _, fields),) = _lines(caplog)
        assert _ms(fields, "store_ms") >= 2 * NAP_S * 1000
        assert _ms(fields, "render_ms") < 2 * NAP_S * 1000, "the store time inside the render is the store's alone"
        _phases_add_up(fields)

    def test_a_nested_call_logs_first_and_its_charges_and_intake_reach_the_enclosing_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            with phase_logged("outer", scope_id="s", campaign_id="c"):
                with phase_logged("inner", scope_id="s", campaign_id="c"):
                    with charged("store"):
                        time.sleep(NAP_S)
                    took_in(runs=2, results=10)
                took_in(runs=1, results=4)

        (_, first, inner), (_, second, outer) = _lines(caplog)
        assert (first, second) == ("inner", "outer")
        assert (inner["runs_in"], inner["results_in"]) == ("2", "10")
        assert (outer["runs_in"], outer["results_in"]) == ("3", "14"), "the outer line sums what its calls read"
        assert _ms(outer, "store_ms") >= _ms(inner, "store_ms") >= NAP_S * 1000
        _phases_add_up(inner)
        _phases_add_up(outer)

    def test_a_call_that_raised_is_logged_naming_what_it_raised_and_the_exception_reaches_the_caller(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=LOGGER), pytest.raises(LookupError, match="gone"):
            with phase_logged("probe", scope_id="s", campaign_id="c"):
                raise LookupError("gone")

        ((record, _, fields),) = _lines(caplog)
        assert fields["raised"] == "LookupError"
        assert record.getMessage().endswith(" raised=LookupError")

    def test_outside_a_call_charging_and_intake_reach_no_line(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            with phase_logged("before", scope_id="s", campaign_id="c"):
                pass
            with charged("store"):
                pass
            took_in(runs=5, results=50)
            with phase_logged("after", scope_id="s", campaign_id="c"):
                pass

        lines = _lines(caplog)
        assert [lens for _, lens, _ in lines] == ["before", "after"]
        assert lines[1][2]["runs_in"] == "" and lines[1][2]["store_ms"] == "", "a closed call leaves nothing open"

    def test_a_run_blocking_worker_charges_the_caller_s_open_line(self, caplog: pytest.LogCaptureFixture) -> None:
        def read() -> None:
            with charged("store"):
                time.sleep(NAP_S)
            took_in(runs=1, results=3)

        async def offloaded() -> None:
            await run_blocking(None, read)

        with caplog.at_level(logging.INFO, logger=LOGGER), phase_logged("probe", scope_id="s", campaign_id="c"):
            asyncio.run(offloaded())

        ((_, _, fields),) = _lines(caplog)
        assert _ms(fields, "store_ms") >= NAP_S * 1000, "the worker's read reached the line its caller opened"
        assert (fields["runs_in"], fields["results_in"]) == ("1", "3")
        _phases_add_up(fields)

    def test_threads_each_log_their_own_call(self, caplog: pytest.LogCaptureFixture) -> None:
        start = threading.Barrier(2)

        def call(name: str, runs: int) -> None:
            with phase_logged(name, scope_id="s", campaign_id="c"):
                start.wait()
                took_in(runs=runs, results=runs * 10)
                start.wait()

        with caplog.at_level(logging.INFO, logger=LOGGER):
            workers = [threading.Thread(target=call, args=(name, runs)) for name, runs in (("one", 1), ("two", 2))]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

        intake = {lens: (fields["runs_in"], fields["results_in"]) for _, lens, fields in _lines(caplog)}
        assert intake == {"one": ("1", "10"), "two": ("2", "20")}


def _counts_in_store(storage: EvalStorage, campaign: EvalCampaign, scope_id: str) -> tuple[int, int]:
    """The campaign's member runs that resolve, and the results stored for them, read from the store directly."""
    runs = [run for run_id in campaign.run_ids if (run := storage.load_eval_run(run_id, scope_id))]
    results = sum(len(storage.query_eval_results_by_run(run.id, scope_id)) for run in runs)
    return len(runs), results


def _member_counts(fixture: OpsFixture) -> tuple[int, int]:
    return _counts_in_store(fixture.host.eval_host.storage, fixture.campaign, TOYHOST_SCOPE)


class TestTheCallSites:
    def test_a_bundle_assembly_logs_one_line_with_what_it_summarised(self, caplog: pytest.LogCaptureFixture) -> None:
        fixture = off_loop(ops_fixture)
        host = fixture.host.eval_host
        runs, results = _member_counts(fixture)
        assert runs > 0 and results > runs

        with caplog.at_level(logging.INFO, logger=LOGGER):
            bundle = assemble_campaign_bundle(host, fixture.campaign.id, TOYHOST_SCOPE)

        ((record, lens, fields),) = _lines(caplog)
        assert record.levelno == logging.INFO
        assert lens == "assemble_context_bundle"
        assert (fields["scope_id"], fields["campaign_id"]) == (TOYHOST_SCOPE, fixture.campaign.id)
        assert (fields["runs_in"], fields["results_in"]) == (str(runs), str(results))
        assert len(bundle.aspects) >= 1
        # The results are read through the reading seam, which charges its store call and its hydration apart from
        # the folds, and says the path that served it: the in-memory store offers the narrowed read.
        assert fields["store_ms"] != "" and fields["hydrate_ms"] != "", "the readings' read is timed apart"
        assert fields["render_ms"] == ""
        assert fields["path"] == "store" and int(fields["store_calls"]) >= 1, fields
        _phases_add_up(fields)

    def test_a_two_aspect_assembly_counts_every_aspect_s_runs_and_results(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        corpus = hot_path_corpus(scale=1, two_aspects=True)
        runs, results = _counts_in_store(corpus.host.storage, corpus.campaign, HOT_PATH_SCOPE)
        assert results == corpus.n_results

        with caplog.at_level(logging.INFO, logger=LOGGER):
            bundle = assemble_campaign_bundle(corpus.host, corpus.campaign.id, HOT_PATH_SCOPE)

        assert len(bundle.aspects) == 2
        first = len(bundle.aspects[0].run_summaries)
        assert 0 < first < runs, "each aspect holds some of the runs, so one aspect's count is not the whole"
        ((_, lens, fields),) = _lines(caplog)
        assert lens == "assemble_context_bundle"
        assert (fields["runs_in"], fields["results_in"]) == (str(runs), str(results))

    @pytest.mark.parametrize("format", ["markdown", "html", "json"])
    def test_a_code_only_report_read_logs_its_bundle_s_line_then_its_own(
        self, format: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        fixture = off_loop(ops_fixture)
        host = fixture.host.eval_host
        runs, results = _member_counts(fixture)

        with caplog.at_level(logging.INFO, logger=LOGGER):
            read = report_read(host, fixture.campaign.id, TOYHOST_SCOPE, format=format)

        assert read.basis == "code_only"
        (_, first, bundle), (record, second, page) = _lines(caplog)
        assert (first, second) == ("assemble_context_bundle", "report_read")
        assert record.levelno == logging.INFO
        assert (page["scope_id"], page["campaign_id"]) == (TOYHOST_SCOPE, fixture.campaign.id)
        assert (page["runs_in"], page["results_in"]) == (bundle["runs_in"], bundle["results_in"])
        assert (page["runs_in"], page["results_in"]) == (str(runs), str(results))
        assert page["render_ms"] != "", "serializing the page is its render"
        # The assembly's store and hydration time is charged to those phases of the read too, so the read's fold
        # holds the rest of the assembly: its elapsed time less what it charged to store and hydrate.
        charged_inside = sum(_ms(bundle, name) for name in ("store_ms", "hydrate_ms") if bundle[name])
        assert _ms(page, "fold_ms") >= _ms(bundle, "elapsed_ms") - charged_inside - 2 * ROUNDING_MS, (
            "the assembly's folding is in the read's fold"
        )
        assert (page["store_ms"], page["hydrate_ms"]) == (bundle["store_ms"], bundle["hydrate_ms"]) != ("", ""), (
            "the assembly's readings read reaches the read's line"
        )
        assert "raised" not in page
        _phases_add_up(page)

    def test_a_report_read_that_raises_is_logged_naming_the_refusal(self, caplog: pytest.LogCaptureFixture) -> None:
        fixture = off_loop(ops_fixture)
        host = fixture.host.eval_host
        campaign_id = fixture.campaign.id
        # A memo laid out over another campaign's readings: its page refuses, and the read raises that refusal.
        other = empty_bundle(campaign_id="another-campaign")
        held = make_analysis(
            campaign_id=campaign_id,
            scope_id=TOYHOST_SCOPE,
            aspects=[analysed_aspect(other)],
            campaign_bundle=campaign_bundle(other).frame(),
        )
        off_loop(host.storage.save_analysis, held)

        with caplog.at_level(logging.INFO, logger=LOGGER), pytest.raises(ValidationFailedError):
            report_read(host, campaign_id, TOYHOST_SCOPE, format="markdown")

        (_, lens, fields) = _lines(caplog)[-1]
        assert lens == "report_read"
        assert fields["raised"] == "ValidationFailedError"
        assert fields["render_ms"] == "", "it raised before rendering"
