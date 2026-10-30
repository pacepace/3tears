"""The aggregates, the planning block and the regression watch's judged tiers answer alike on either read path (W4b P5).

Every aggregate reads through the reading seams (``EvalStorage.iter_result_readings`` and ``result_groups``), which
take the store's path when the store offers its narrowed and grouped reads
(:class:`~threetears.evals.schema.store_port.ResultReadings`) and the engine's otherwise. Each family is read over the
same documents twice, through the reference store as it is and through the same store with those reads hidden
(:class:`~packages.evals.tests.read_path_support.EngineOnlyStore`), and the outputs are compared byte for byte:

- **the aggregates:** ``pivot`` (every measure), ``history`` (every measure), ``frontier`` (with and without a bar),
  ``program_budget``, ``orphaned_runs`` and ``comparison_sets`` (whole and narrowed to a campaign);
- **the regression watch** on a step that regressed, so its judged tiers are read;
- **the planning block** a launch estimate carries (``detectable_effects``).

The read path is the one each ``eval.read_tier`` line names: always ``engine`` with the reads hidden.

The spend lenses fold figures read off a grouped read's values rather than hydrated results; their answers are held,
byte for byte, to the same lens folded over the scope's results read whole. Their grouped read asks for one group (no
``by``), so its values come in the read's order and no group order reaches the fold.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from collections.abc import Mapping, Sequence
from itertools import zip_longest
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.analysis import comparison_sets, orphaned_runs, program_budget
from threetears.evals.analysis import service as analysis_service
from threetears.evals.analysis.bundle.comparisons import planning_readings
from threetears.evals.analysis.lenses.comparison_sets import compute_comparison_sets
from threetears.evals.analysis.lenses.orphaned_runs import compute_orphaned_runs
from threetears.evals.analysis.lenses.program_budget import compute_program_budget
from threetears.evals.kernel.result_reading import GroupsRead
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.run.reads import list_runs
from threetears.evals.schema.store_port import DocumentStore, GroupMeasure
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.hot_path_support import (
    ARM_MODELS,
    CASES_PER_RUN,
    HOT_PATH_SCOPE,
    K_RUNS,
    hot_path_corpus,
)
from packages.evals.tests.read_path_support import EngineOnlyStore
from packages.evals.tests.test_result_reading_fields import (
    ENRICHED_CASES,
    ENRICHED_SCOPE,
    REGRESSED_SCOPE,
    REPORT_NOW,
    Campaign,
    add_a_regressed_step,
    enrich,
    read_aggregates,
    read_planning,
    regressed_steps,
)

#: The logger the read-tier lines are written to.
LOGGER = "threetears.evals.analysis.reads"

#: A read-tier line's lens and its path.
PATH = re.compile(r"eval\.read_tier lens=(?P<lens>\S+) .* path=(?P<path>\S+) store_calls=(?P<calls>\d+)")


@functools.cache
def built(name: str) -> tuple[Campaign, InMemoryDocumentStore]:
    """The campaign ``name`` and the reference store holding it, built once per process."""
    store = InMemoryDocumentStore()
    if name == "scaled":
        corpus = hot_path_corpus(scale=1, store=store)
        scope_id, cases = HOT_PATH_SCOPE, CASES_PER_RUN
    elif name == "regressed":
        corpus = hot_path_corpus(scale=1, scope_id=REGRESSED_SCOPE, store=store)
        add_a_regressed_step(corpus.host, REGRESSED_SCOPE)
        scope_id, cases = REGRESSED_SCOPE, CASES_PER_RUN
    elif name == "enriched":
        corpus = hot_path_corpus(scale=1, scope_id=ENRICHED_SCOPE, cases_per_run=ENRICHED_CASES, store=store)
        enrich(corpus.host, ENRICHED_SCOPE)
        scope_id, cases = ENRICHED_SCOPE, ENRICHED_CASES
    else:
        raise KeyError(name)
    read = Campaign(corpus.host, corpus.campaign.id, scope_id, corpus.template_ids[0], cases, K_RUNS, len(ARM_MODELS))
    return read, store


def on(read: Campaign, store: DocumentStore) -> Campaign:
    """``read`` with its host reading through ``store``."""
    return replace(read, host=replace(read.host, storage=EvalStorage(store)))


def both_paths(name: str) -> tuple[Campaign, Campaign]:
    """The campaign ``name`` read through its store as it is, and through the same store with its narrowed and grouped
    reads hidden (the engine's path)."""
    read, store = built(name)
    return on(read, store), on(read, EngineOnlyStore(store))


def paths_logged(caplog: pytest.LogCaptureFixture) -> set[str]:
    """Every read path the read-tier lines logged so far name."""
    return {match["path"] for record in caplog.records if (match := PATH.match(record.getMessage()))}


@pytest.fixture(autouse=True)
def _pinned_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis_service, "utc_now_iso", lambda: REPORT_NOW)


class TestEveryFamilyAnswersAlikeOnEitherPath:
    @pytest.mark.parametrize("name", ("scaled", "enriched", "regressed"))
    def test_the_aggregates(self, name: str, caplog: pytest.LogCaptureFixture) -> None:
        as_is, hidden = both_paths(name)
        steps = regressed_steps(as_is)
        lenses = name != "regressed"

        with caplog.at_level(logging.INFO, logger=LOGGER):
            engine = read_aggregates(hidden, steps, lenses=lenses)
        engine_paths = paths_logged(caplog)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=LOGGER):
            stored = read_aggregates(as_is, steps, lenses=lenses)

        assert engine == stored
        assert engine_paths == {"engine"}, "with the store's narrowed reads hidden, every read is the engine's"
        assert len(paths_logged(caplog)) == 1, "one store, one path"
        if name == "regressed":
            assert steps, "a step regressed, so the watch read its judged tiers on it"

    @pytest.mark.parametrize("name", ("scaled", "enriched"))
    def test_the_planning_block(self, name: str) -> None:
        as_is, hidden = both_paths(name)

        engine = read_planning(hidden)

        assert json.loads(engine[0])["run_ids"], "the block read earlier runs"
        assert engine == read_planning(as_is)


class TestThePlanningReadDoesNotDependOnTheOrderAcrossRuns:
    """A store's own read may hand a set of runs' results out in its order, runs interleaved; the planning read
    groups by run itself, so the readings are the ones a read run by run gives."""

    @pytest.mark.parametrize("name", ("scaled", "enriched"))
    def test_interleaved_runs_plan_as_runs_read_one_after_another(self, name: str) -> None:
        read, _ = built(name)
        runs = [run for run in list_runs(read.host, read.scope_id) if run.template_id == read.template_id]
        readings = list(read.host.storage.iter_result_readings([run.id for run in runs], read.scope_id))
        by_run: dict[str, list[Any]] = {}
        for reading in readings:
            by_run.setdefault(reading.eval_run_id, []).append(reading)
        # Round robin over the runs, each run's own results kept in their order; and the runs named last first.
        interleaved = [reading for column in zip_longest(*by_run.values()) for reading in column if reading is not None]
        reversed_runs = [reading for run in reversed(runs) for reading in by_run.get(run.id, [])]

        expected = planning_readings(runs, readings, profile=read.host.profile)

        assert len({reading.eval_run_id for reading in interleaved[: len(by_run)]}) == len(by_run) > 1
        assert planning_readings(runs, interleaved, profile=read.host.profile) == expected
        assert planning_readings(runs, reversed_runs, profile=read.host.profile) == expected


def _lister(read: Campaign) -> Any:
    return lambda scope_id, **kwargs: list_runs(read.host, scope_id, **kwargs)


class TestTheSpendLensesFoldTheFiguresAsTheResultsFoldThem:
    """The spend lenses read figures off the store's rows, never hydrated, and answer exactly as the same lens folded
    over the results read whole."""

    @pytest.mark.parametrize("name", ("scaled", "enriched"))
    def test_program_budget(self, name: str) -> None:
        for read in both_paths(name):
            runs = list_runs(read.host, read.scope_id, include_archived=True)
            whole = compute_program_budget(runs, read.host.storage.query_eval_results(read.scope_id))

            answered = program_budget(read.host.storage, read.scope_id, list_runs=_lister(read))

            assert json.dumps(answered) == json.dumps(whole.model_dump(mode="json"))

    @pytest.mark.parametrize("name", ("scaled", "enriched"))
    def test_orphaned_runs(self, name: str) -> None:
        for read in both_paths(name):
            storage = read.host.storage
            runs = list_runs(read.host, read.scope_id, include_archived=True)
            claimed = [campaign.run_ids for campaign in storage.list_campaigns(read.scope_id)]
            whole = compute_orphaned_runs(runs, storage.query_eval_results(read.scope_id), claimed)

            answered = orphaned_runs(storage, read.scope_id, list_runs=_lister(read))

            assert json.dumps(answered) == json.dumps(whole.model_dump(mode="json"))

    def test_the_enriched_campaign_carries_rejudge_spend(self) -> None:
        read, _ = built("enriched")
        results = read.host.storage.query_eval_results(read.scope_id)

        assert any(rescore.cost_usd for result in results for rescore in result.judge_rescores), (
            "a re-judge's spend is folded, so the figures' re-judge column is read"
        )

    def test_the_spend_read_asks_for_one_group(self) -> None:
        read, store = built("enriched")
        storage = _RecordsGroupedReads(EvalStorage(store))

        program_budget(storage, read.scope_id, list_runs=_lister(read))  # type: ignore[arg-type]

        assert storage.groupings == [((), ("eval_run_id", "cost_usd", "judge_rescores", "cost_roles"))]


class TestComparisonSetsReadTheirWindowsAndBadgesOffReadings:
    @pytest.mark.parametrize("name", ("scaled", "enriched"))
    def test_as_from_the_results_read_whole(self, name: str) -> None:
        for read in both_paths(name):
            runs = list_runs(read.host, read.scope_id, status="completed")
            whole = compute_comparison_sets(
                runs, results=read.host.storage.query_eval_results(read.scope_id), profile=read.host.profile
            )

            answered = comparison_sets(
                read.host.storage, read.scope_id, list_runs=_lister(read), profile=read.host.profile
            )

            assert json.dumps(answered) == json.dumps(whole.model_dump(mode="json"))


class _RecordsGroupedReads:
    """The engine's storage, recording each grouped read's grouping and measure names; every other read passes."""

    def __init__(self, inner: EvalStorage) -> None:
        self._inner = inner
        self.groupings: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def result_groups(
        self,
        run_ids: Sequence[str] | None,
        scope_id: str,
        /,
        *,
        by: Sequence[str],
        measures: Mapping[str, GroupMeasure],
    ) -> GroupsRead:
        self.groupings.append((tuple(by), tuple(measures)))
        return self._inner.result_groups(run_ids, scope_id, by=by, measures=measures)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
