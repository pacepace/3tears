"""The engine's take-up under the null, when a cell's latency follows its answer: how often an engine comparison whose
launches an operator interrupts, and which the take-up then finishes, separates two arms of equal accuracy.

The engine path, as a host or an agent drives it (no quick path anywhere):

- **launch**: ``run_launch`` starts both arms as one interleaved launch group, its cells ordered by a digest of the
  group's id, its repeat and its case (``runner.cell_execution_order`` with ``seed``: the whole matrix shuffled, one
  order shared by the arms), four at a time under one shared width (``LaunchSettings.max_concurrent_cells``, taken in
  turn by ``arm_schedule.CellInterleaver``);
- **interrupt**: ``job_cancel`` on the launch's job cancels every run's job (``launch_job._launch_work``), and each run
  records the cells it had in flight as cut (termination ``cancelled``, ``runner.execute_run``); nothing drains;
- **take-up**: ``job_resume`` takes the launch up again (``resume.resume_launch``), and each run runs, in its own order,
  every cell with no row and every row that runs again (``kernel.result_condition.runs_again``), a cut row among them,
  written over its row;
- **read**: a campaign over the runs, control ``a``, read by ``assemble_context_bundle``: the paired t test at α at a
  fixed count, or, under a plan of ``L`` looks declared before the first (``AspectDesign.planned_looks``), each look at
  α/L. A take-up of a launch is that launch's occasion, so it is no look of its own (``significance.looks_taken``).

A cell in flight when the cut lands is drawn afresh by the take-up, and a cell in flight is a long one: where one arm's
wrong answers are its slow ones, a cut takes its wrong answers away more often than its right ones, and the arm reads
better than it is. The simulations here measure how much that moves the engine's two readings, against a control whose
arms share one latency law (both slower when wrong), interrupted alike.

Setup, as the quick path's watched simulations had it (``test_rereadable_bet.py``): arms of accuracy 0.5 each, k = 1;
arm ``a`` 0.4 ms whatever it answers, arm ``b`` 2 ms when right and four or twenty times that when wrong; the control's
arms both 2 ms right and 40 ms wrong. Each holding (the launch and each take-up) is interrupted in four of five at a
point drawn apart from any answer (after a uniform count of the answers left to land), at most five holdings, the last
run to its end.

Measured, not asserted against α where the reading's own rate sits at α: the t test at a fixed count errs at α itself
on data with no difference (``test_simulated_looks.py``, "one look on the exact t test"), so its share's upper bound
lies above α at any finite count, cut or not. Each long simulation states its share with its 95% Clopper-Pearson
interval, and asserts what holds of it (see each test).

The long simulations are ``long_simulation``, one pool request each (``.testament.toml``'s ``simulation`` suite, named
by node id); the unit suite keeps the smoke check.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.actions.catalogue import Caller
from threetears.evals.analysis import ComparisonFamily, assemble_context_bundle, get_campaign
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA, exact_rate_interval
from threetears.evals.kernel import ValidationFailedError, default_job_timeout
from threetears.evals.kernel.declaration import AspectDesign, ControlDeclaration, SweptAxis
from threetears.evals.kernel.host.profile import CANDIDATE_MODEL_LEVER
from threetears.evals.kernel.result_condition import runs_again
from threetears.evals.ops import LaunchArguments, OpsHost, job_cancel, job_poll, job_resume, parse_job_id, run_launch
from threetears.evals.quick.one_call import CALLABLE_KIND, callable_host, callable_kind
from threetears.evals.run import (
    EvalLaunch,
    KindWiring,
    LaunchableKind,
    LaunchHost,
    LaunchRequest,
    LaunchSettings,
    launch_run,
    list_results,
)
from threetears.evals.schema import EvalTemplate, EvalTestCase, SubjectSnapshot
from threetears.evals.schema.values import SweepableValue
from threetears.evals.storage import InMemoryDocumentStore

from packages.evals.tests.fixtures.toyhost.host import with_registry
from packages.evals.tests.off_loop import off_loop

#: The scope, template and subject every simulated comparison lives in (each in a store of its own).
_SCOPE, _TEMPLATE, _SUBJECT = "engine", "coin", "coin-subject"

#: How long each arm's answer takes, as a law of the answer: both arms 2 ms right and 40 ms wrong (the control, one law
#: for both); or arm ``a`` 0.4 ms whatever it answers and arm ``b`` 2 ms right and four or twenty times that wrong.
_Law = Literal["alike-20x", "b-slower-4x", "b-slower-20x"]

#: The share of holdings interrupted, and the most holdings one launch has (the last runs to its end).
_INTERRUPTED, _HOLDINGS = 0.8, 5

#: The cases one comparison read at a fixed count runs; the cases each launch of a planned comparison adds, one look
#: each (20, 40 and 60 cases at its three looks), and the looks it plans.
_FIXED_CASES, _LOOK_CASES, _PLANNED = 40, 20, 3

#: Comparisons in each long simulation at a fixed count, in each under a plan of looks, and in each smoke check: each
#: long one a pool request of its own, inside its window.
_FIXED_REPLICATES, _LOOKS_REPLICATES, _SMOKE = 800, 500, 3

_CALLER = Caller(scope_id=_SCOPE, identity="engine-operator")

#: The launch settings: the engine's own width (four cells at once), nothing capped that a coin could reach.
_SETTINGS = LaunchSettings(
    max_launch_arms=2,
    max_admitted_runs=4,
    judge_concurrency=1,
    enforcement_enabled=False,
    max_cost_usd=1.0,
    max_metered_calls=None,
    max_out_of_run_cost_usd=1.0,
)


def correct(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the coin landed right."""
    return answer == "right"


def _seconds(law: _Law, arm: str, right: bool) -> float:
    """How long ``arm``'s answer takes under ``law``."""
    if law == "alike-20x":
        return 0.002 if right else 0.040
    if arm == "a":
        return 0.0004
    return 0.002 if right else 0.002 * (4 if law == "b-slower-4x" else 20)


@dataclass
class _Landings:
    """The answers landed in the holding under way, and the count at which it is interrupted (``None``: never)."""

    landed: int = 0
    stop_at: int | None = None
    reached: asyncio.Event = field(default_factory=asyncio.Event)
    #: Every answer drawn and every answer landed, by arm and whether it was right, over the comparison: a cell cut while
    #: its answer was on its way drew one that never landed.
    drawn: Counter[tuple[str, bool]] = field(default_factory=Counter)
    arrived: Counter[tuple[str, bool]] = field(default_factory=Counter)

    def land(self) -> None:
        self.landed += 1
        if self.stop_at is not None and self.landed >= self.stop_at:
            self.reached.set()


@dataclass
class _Engine:
    """One comparison's engine: a host of the callable kind over a store of its own, its two arms' coins, and the
    action catalogue an operator drives it through."""

    ops: OpsHost
    tool: Any
    batch: list[EvalTestCase]
    landings: _Landings


def _engine(rng: random.Random, law: _Law) -> _Engine:
    """A fresh engine host: the callable kind (``quick.one_call.callable_kind``) wired by a launcher of its own through
    the engine's ``launch_run``, each arm a coin that lands right half the time and takes as long as ``law`` says."""
    host = callable_host([correct], store=InMemoryDocumentStore())
    off_loop(
        host.storage.save_template,
        EvalTemplate(id=_TEMPLATE, scope_id=_SCOPE, name="a coin", intent="Land right.", candidate_kind=CALLABLE_KIND),
    )
    landings = _Landings()

    def coin(arm: str) -> Callable[[Mapping[str, Any]], Any]:
        async def run(case: Mapping[str, Any]) -> str:
            right = rng.random() < 0.5
            landings.drawn[arm, right] += 1
            await asyncio.sleep(_seconds(law, arm, right))
            landings.arrived[arm, right] += 1
            landings.land()
            return "right" if right else "wrong"

        run.__name__ = arm
        return run

    kinds = {arm: callable_kind(coin(arm), [correct]) for arm in "ab"}
    batch: list[EvalTestCase] = []
    subject = SubjectSnapshot(subject_id=_SUBJECT, subject_label=_SUBJECT, state={})

    async def launch(request: LaunchRequest) -> Any:
        kind = kinds[request.candidate_model]
        # A launch runs the batch it was asked for; a take-up runs the run's own cases (``LaunchRequest.cases_or``).
        wiring = KindWiring(
            kind_factory=lambda _cell: kind,
            subject=subject,
            test_cases=request.cases_or(batch),
            judge=None,
            max_concurrent_cells=None,
        )
        return await launch_run(launching, request, wiring)

    launching = LaunchHost(
        eval_host=with_registry(
            host,
            {
                CALLABLE_KIND: LaunchableKind(
                    launch=launch,
                    resolve_subject=lambda _subject, _scope: None,
                    unhonoured_launch_arguments=frozenset(
                        {"simulator_model", "judge_config_ids", "n_variations", "judge_model", "cassette_mode"}
                    ),
                )
            },
        ),
        settings=lambda: _SETTINGS,
        job_timeout_factory=default_job_timeout,
    )
    return _Engine(
        ops=OpsHost(launch=launching),
        tool=eval_catalogue().mount_all(standard_tools())[0],
        batch=batch,
        landings=landings,
    )


async def _act(engine: _Engine, action: str, **arguments: Any) -> Any:
    done = await engine.tool.call({"action": action, **arguments}, host=engine.ops, caller=_CALLER)
    assert not done.is_error, done.text
    return done.structured


def _cases(start: int, count: int) -> list[EvalTestCase]:
    return [
        EvalTestCase(
            id=f"case-{n:04d}",
            scope_id=_SCOPE,
            template_id=_TEMPLATE,
            variation_params={"n": str(n)},
            host_payload={"case": {"n": n}},
        )
        for n in range(start, start + count)
    ]


@dataclass
class _Cuts:
    """The rows each arm's take-ups ran again, by arm, and the holdings the comparison's launches took."""

    again: dict[str, int] = field(default_factory=lambda: {"a": 0, "b": 0})
    holdings: int = 0
    #: The answers drawn by cells that were cut before they landed, by arm and whether each was right.
    lost: Counter[tuple[str, bool]] = field(default_factory=Counter)


async def _launched(engine: _Engine, rng: random.Random, cases: list[EvalTestCase], cuts: _Cuts) -> list[str]:
    """Launch both arms over ``cases`` and hold the launch to its end as an operator would: each holding interrupted in
    :data:`_INTERRUPTED` of them after a uniform count of the answers left to land (``job_cancel``), then taken up again
    (``job_resume``), at most :data:`_HOLDINGS` holdings, the last run to its end. Returns the launch's runs, a then b."""
    for case in cases:
        off_loop(engine.ops.launch.eval_host.storage.save_test_case, case)
    engine.batch[:] = cases
    manager, storage = engine.ops.launch.job_manager, engine.ops.launch.eval_host.storage
    started = await run_launch(
        engine.ops,
        LaunchArguments(template_id=_TEMPLATE, subject_id=_SUBJECT, models=["a", "b"], k_runs=1),
        _SCOPE,
    )
    (handle,) = started.jobs
    _, launch_id, _ = parse_job_id(handle.job_id)
    left = 2 * len(cases)
    for holding in range(1, _HOLDINGS + 1):
        cuts.holdings += 1
        landings = engine.landings
        landings.landed, landings.reached = 0, asyncio.Event()
        interrupt = holding < _HOLDINGS and rng.random() < _INTERRUPTED
        landings.stop_at = rng.randint(1, max(left, 1)) if interrupt else None
        finished = asyncio.ensure_future(manager.wait_for([launch_id]))
        if interrupt:
            reached = asyncio.ensure_future(landings.reached.wait())
            await asyncio.wait({finished, reached}, return_when=asyncio.FIRST_COMPLETED)
            reached.cancel()
            if not finished.done():
                try:
                    await job_cancel(engine.ops, handle.job_id, _SCOPE, reason="interrupted")
                except ValidationFailedError:
                    # The launch ended between its last answer landing and the cancel: nothing was left to interrupt.
                    assert finished.done() or (await job_poll(engine.ops, handle.job_id, _SCOPE)).done
        await finished
        status = await job_poll(engine.ops, handle.job_id, _SCOPE)
        if status.state == "completed":
            break
        assert status.state == "cancelled", status
        runs = off_loop(storage.load_eval_runs, off_loop(_run_ids, engine, launch_id), _SCOPE)
        if all(run.status == "completed" for run in runs):
            # The cancel reached the launch after its runs had ended: nothing is left to take up.
            break
        rows = [
            row
            for run_id in off_loop(_run_ids, engine, launch_id)
            for row in off_loop(list_results, storage, run_id, _SCOPE)
        ]
        recorded = [row for row in rows if not runs_again(row)]
        for row in rows:
            if runs_again(row):
                cuts.again[row.model] += 1
        left = 2 * len(cases) - len(recorded)
        await job_resume(engine.ops, handle.job_id, _SCOPE)
    else:
        raise AssertionError(f"launch {launch_id} did not complete within {_HOLDINGS} holdings")
    runs = sorted(
        off_loop(storage.load_eval_runs, off_loop(_run_ids, engine, launch_id), _SCOPE),
        key=lambda run: run.candidate_model,
    )
    # The take-up finished the launch as one uninterrupted would have: every cell one row, none left to run again.
    for run in runs:
        assert run.status == "completed", run.status
        rows = off_loop(list_results, storage, run.id, _SCOPE)
        assert sorted(row.test_case_id for row in rows) == sorted(case.id for case in cases)
        assert not any(runs_again(row) for row in rows)
    return [run.id for run in runs]


def _run_ids(engine: _Engine, launch_id: str) -> list[str]:
    """The runs a launch started, in arm order, as its record names them. Reads the store."""
    launch, _ = engine.ops.launch.eval_host.storage.load_job_record(EvalLaunch, launch_id, _SCOPE)
    assert launch is not None
    return list(launch.run_ids)


# --- the reading -----------------------------------------------------------------------------------------------------


def _design(planned: int | None) -> dict[str, Any]:
    """The comparison's declared design, a plan of ``planned`` looks in it where one is given."""
    return AspectDesign(
        axes=[
            SweptAxis(
                axis_id=CANDIDATE_MODEL_LEVER,
                values=[SweepableValue.of(arm, display=arm) for arm in "ab"],
                rationale="does b land right more often than a",
            )
        ],
        held_fixed=ControlDeclaration(stimulus="controlled", apparatus="commissioned"),
        intended_repetitions=1,
        measure_order=["correct"],
        planned_looks=planned,
    ).model_dump(mode="json", exclude_none=True)


def _family(engine: _Engine, campaign_id: str) -> ComparisonFamily:
    """The campaign's one family, as the engine reads it."""
    host = engine.ops.launch.eval_host
    campaign = off_loop(get_campaign, host.storage, campaign_id, _SCOPE)
    bundle = off_loop(assemble_context_bundle, campaign, storage=host.storage, profile=host.profile)
    (family,) = bundle.multiple_comparisons.families
    return family


def _separated(family: ComparisonFamily) -> bool:
    """Whether the family separates the arms on ``correct``, read by the paired t test (never the bet)."""
    (compared,) = [comparison for comparison in family.comparisons if comparison.name == "correct"]
    assert compared.basis in (None, "t", "identical"), compared.basis
    return compared.verdict in ("improved", "regressed")


@dataclass(frozen=True)
class _Outcome:
    """One comparison: whether it separated, the looks it read, and its take-ups' cut rows by arm."""

    separated: bool
    looks: int
    cuts: _Cuts


async def _at_a_fixed_count(tag: str, law: _Law) -> _Outcome:
    """One comparison read once by the engine's default reading: :data:`_FIXED_CASES` cases launched, interrupted and
    taken up to their end, then filed as a campaign (control ``a``) and read at α."""
    rng = random.Random(tag)
    engine = _engine(rng, law)
    cuts = _Cuts()
    a, b = await _launched(engine, rng, _cases(0, _FIXED_CASES), cuts)
    filed = await _act(
        engine,
        "campaign_create",
        name="fixed",
        subject_id=_SUBJECT,
        behavior="coin",
        question="does b land right more often than a?",
        aspects=[{"label": "coin", "run_ids": [a, b], "design": _design(None)}],
        campaign_arms=[{"label": "a", "role": "control", "from_run_ids": [a]}],
    )
    family = _family(engine, str(filed["id"]))
    assert family.look_alpha is None and family.alpha == SIGNIFICANCE_ALPHA
    cuts.lost = engine.landings.drawn - engine.landings.arrived
    return _Outcome(_separated(family), 1, cuts)


async def _watched_over_planned_looks(tag: str, law: _Law) -> _Outcome:
    """One comparison under a plan of :data:`_PLANNED` looks, declared before its first: each look launches
    :data:`_LOOK_CASES` new cases on both arms, held to its end as :func:`_launched` holds it, joins the campaign and is
    read at α/L; the watcher stops at the first look that separates."""
    rng = random.Random(tag)
    engine = _engine(rng, law)
    cuts = _Cuts()
    planned = await _act(
        engine,
        "campaign_create",
        name="planned",
        subject_id=_SUBJECT,
        behavior="coin",
        question="does b land right more often than a?",
        aspects=[{"label": "coin", "template_id": _TEMPLATE, "run_ids": [], "design": _design(_PLANNED)}],
    )
    campaign_id = str(planned["id"])
    for look in range(1, _PLANNED + 1):
        a, b = await _launched(engine, rng, _cases((look - 1) * _LOOK_CASES, _LOOK_CASES), cuts)
        await _act(engine, "campaign_runs_add", campaign_id=campaign_id, run_ids=[a, b])
        if look == 1:
            await _act(engine, "campaign_update", campaign_id=campaign_id, control_from_run_id=a)
        family = _family(engine, campaign_id)
        assert family.look_alpha == pytest.approx(SIGNIFICANCE_ALPHA / _PLANNED), family.look_alpha
        if _separated(family):
            break
    cuts.lost = engine.landings.drawn - engine.landings.arrived
    return _Outcome(_separated(family), look, cuts)


_Reading = Literal["fixed", "looks"]


async def _simulate(reading: _Reading, law: _Law, count: int, tag: str) -> tuple[int, _Cuts]:
    """How many of ``count`` comparisons separated, and their take-ups' cut rows by arm, summed."""
    simulate = _at_a_fixed_count if reading == "fixed" else _watched_over_planned_looks
    separated, total = 0, _Cuts()
    for index in range(count):
        outcome = await simulate(f"{tag}-{reading}-{law}-{index}", law)
        separated += outcome.separated
        total.holdings += outcome.cuts.holdings
        for arm, again in outcome.cuts.again.items():
            total.again[arm] += again
        total.lost += outcome.cuts.lost
    return separated, total


def _said(reading: _Reading, law: _Law, separated: int, count: int, cuts: _Cuts) -> str:
    """The share, its 95% Clopper-Pearson interval, and what the take-ups ran again, for a report or a failure."""
    return (
        f"{reading} {law}: {separated}/{count} separated, {exact_rate_interval(separated, count)}; "
        f"holdings {cuts.holdings}; rows run again a={cuts.again['a']} b={cuts.again['b']}; answers the cuts lost, "
        f"right/wrong: a {cuts.lost['a', True]}/{cuts.lost['a', False]}, b {cuts.lost['b', True]}/{cuts.lost['b', False]}"
    )


# --- the simulations -------------------------------------------------------------------------------------------------


async def test_a_few_engine_comparisons_interrupted_and_taken_up_finish_whole_and_read_by_the_t_test() -> None:
    """The unit suite's smoke check of both simulations: a few comparisons each, every launch held through its
    interrupts and take-ups to an end where every cell has one row and none runs again (inside :func:`_launched`), each
    read by the paired t test, at α at a fixed count and at α/L under the plan (inside :func:`_family`'s callers); and
    the reading separating at most one of each few, which at a rate of α happens with probability above 0.99."""
    for reading in ("fixed", "looks"):
        separated, cuts = await _simulate(reading, "b-slower-20x", _SMOKE, "smoke")
        assert separated <= 1, _said(reading, "b-slower-20x", separated, _SMOKE, cuts)
        assert cuts.holdings >= _SMOKE


def _held(reading: _Reading, law: _Law, separated: int, count: int, cuts: _Cuts) -> None:
    """Print what the simulation drew (``-rP`` shows it on a pass), and hold its 95% Clopper-Pearson upper bound at α."""
    said = _said(reading, law, separated, count, cuts)
    print(said)
    bound = exact_rate_interval(separated, count)
    assert bound is not None and bound[1] <= SIGNIFICANCE_ALPHA, said


@pytest.mark.long_simulation
@pytest.mark.timeout(1700)
@pytest.mark.parametrize("law", ["alike-20x", "b-slower-4x", "b-slower-20x"])
async def test_under_the_null_an_engine_comparison_interrupted_and_taken_up_read_at_a_fixed_count_separates_at_most_alpha(
    law: _Law,
) -> None:
    """:data:`_FIXED_REPLICATES` comparisons of :data:`_FIXED_CASES` cases (:func:`_at_a_fixed_count`), each launch
    interrupted and taken up as :func:`_launched` holds it, read once by the engine's default reading: the share that
    separates has a 95% Clopper-Pearson upper bound at most α, for arms of one latency law and for arms of different
    laws whose slow arm is slower again when wrong."""
    separated, cuts = await _simulate("fixed", law, _FIXED_REPLICATES, "engine")
    _held("fixed", law, separated, _FIXED_REPLICATES, cuts)


@pytest.mark.long_simulation
@pytest.mark.timeout(1700)
@pytest.mark.parametrize("law", ["alike-20x", "b-slower-4x", "b-slower-20x"])
async def test_under_the_null_an_engine_comparison_interrupted_and_taken_up_over_planned_looks_separates_at_most_alpha(
    law: _Law,
) -> None:
    """:data:`_LOOKS_REPLICATES` comparisons under a plan of :data:`_PLANNED` looks
    (:func:`_watched_over_planned_looks`), each look's launch interrupted and taken up as :func:`_launched` holds it, the
    watcher stopping at the first look that separates: the share that separates has a 95% Clopper-Pearson upper bound
    at most α."""
    separated, cuts = await _simulate("looks", law, _LOOKS_REPLICATES, "engine")
    _held("looks", law, separated, _LOOKS_REPLICATES, cuts)
