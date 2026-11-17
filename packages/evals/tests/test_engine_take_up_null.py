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

Two graders, since the engine's reading of each differs: a check (``correct``) is read in a family beside the pass rate
(``passed``) it equals, so Holm halves α in effect; a grade on the range 0 to 1 (``grade``) is alone in its family and
read at α. **What the simulations show** (:data:`_SHOWN` has the shares): read by a check, the shares hold at most α at a
fixed count, because the doubled family leaves the lean room, and over the plan's looks all but one do: arms of
different laws at 20x drew 4.2% on the pool, an upper bound above α; read by a grade, the control sits at
α, as the t test does with no cut at all (``test_simulated_looks.py``, "one look on the exact t test"), and arms of
different laws separate past it: about 7% at a fixed count, and 7% and 13% over the plan's three looks, almost all
toward the slow arm, whose wrong answers the cuts took (in the 20x rows, about seven of its wrong answers redrawn for
each right one). So the engine's take-up does push false separations above α where latency tracks the answer and the
reading spends α. Each long simulation asserts what its share shows; the reported ones are said there.

The long simulations are ``long_simulation``, one pool request each (``.testament.toml``'s ``simulation`` suite, named
by node id); the unit suite keeps the smoke check.
"""

from __future__ import annotations

import asyncio
import functools
import math
import random
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.actions.catalogue import Caller
from threetears.evals.analysis import ComparisonFamily, assemble_context_bundle, get_campaign
from threetears.evals.analysis.bundle.comparisons import redrawn_sentence
from threetears.evals.analysis.bundle.trials import redrawn_cells
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA, exact_rate_interval, separation_test
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
_LAWS: tuple[_Law, ...] = ("alike-20x", "b-slower-4x", "b-slower-20x")

#: The share of holdings interrupted, and the most holdings one launch has (the last runs to its end).
_INTERRUPTED, _HOLDINGS = 0.8, 5

#: How an operator interrupts a holding: ``drain``, one cancel (the engine lets the cells in flight finish and record);
#: ``second``, one cancel and, in :data:`_SECOND` of the holdings interrupted, a second 5 ms after it, which cuts the
#: cells still in flight; ``cut``, two cancels at once, so every interrupt cuts what is in flight, as the engine did
#: before it drained (kept as the comparison that shows why it drains).
_Stop = Literal["drain", "second", "cut"]

#: The share of interrupted holdings an operator interrupts a second time, and how long after the first.
_SECOND, _SECOND_AFTER_S = 0.3, 0.005

#: The cases one comparison read at a fixed count runs; the cases each launch of a planned comparison adds, one look
#: each (20, 40 and 60 cases at its three looks), and the looks it plans.
_FIXED_CASES, _LOOK_CASES, _PLANNED = 40, 20, 3

#: The reading's own rate under the plan of three looks, uninterrupted, as :func:`_exact_rate` computes it.
_EXACT_LOOKS = 0.04485

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
    """Whether the coin landed right, as a check."""
    return answer == "right"


def grade(case: Mapping[str, Any], answer: str) -> float:
    """Whether the coin landed right, as a grade on its declared range."""
    return 1.0 if answer == "right" else 0.0


#: How the answer is graded: by a check (``correct``), whose family the engine reads beside the pass rate it equals, so
#: Holm reads two equal p values and separates where p < α/2; or by a grade on the range 0 to 1 (``grade``), alone in
#: its family, so the paired t test separates where p < α.
_Grader = Literal["check", "grade"]

_GRADERS: dict[_Grader, tuple[Callable[[Mapping[str, Any], str], Any], dict[str, tuple[float, float]]]] = {
    "check": (correct, {}),
    "grade": (grade, {"grade": (0.0, 1.0)}),
}


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

    #: Where set, every coin waits on it once its answer is drawn, so a test holds cells in flight; and how many wait.
    held: asyncio.Event | None = None
    holding: int = 0

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


def _engine(rng: random.Random, law: _Law, grader: _Grader, *, lease_s: float | None = None) -> _Engine:
    """A fresh engine host: the callable kind (``quick.one_call.callable_kind``) wired by a launcher of its own through
    the engine's ``launch_run``, each arm a coin that lands right half the time and takes as long as ``law`` says; its
    job leases last ``lease_s`` where given, the engine's default otherwise."""
    scorer, ranges = _GRADERS[grader]
    host = callable_host([scorer], ranges=ranges, store=InMemoryDocumentStore())
    off_loop(
        host.storage.save_template,
        EvalTemplate(id=_TEMPLATE, scope_id=_SCOPE, name="a coin", intent="Land right.", candidate_kind=CALLABLE_KIND),
    )
    landings = _Landings()

    def coin(arm: str) -> Callable[[Mapping[str, Any]], Any]:
        async def run(case: Mapping[str, Any]) -> str:
            right = rng.random() < 0.5
            landings.drawn[arm, right] += 1
            if landings.held is not None:
                landings.holding += 1
                try:
                    await landings.held.wait()
                finally:
                    landings.holding -= 1
            await asyncio.sleep(_seconds(law, arm, right))
            landings.arrived[arm, right] += 1
            landings.land()
            return "right" if right else "wrong"

        run.__name__ = arm
        return run

    kinds = {arm: callable_kind(coin(arm), [scorer], ranges=ranges) for arm in "ab"}
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
        **({} if lease_s is None else {"job_lease_s": lease_s}),
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


async def _interrupt(engine: _Engine, job_id: str, finished: asyncio.Future[None]) -> None:
    """One operator's ``job_cancel`` of a launch still running; a cancel that finds it ended is no interrupt."""
    if finished.done():
        return
    try:
        await job_cancel(engine.ops, job_id, _SCOPE, reason="interrupted")
    except ValidationFailedError:
        # The launch ended between its last answer landing and the cancel: nothing was left to interrupt.
        assert finished.done() or (await job_poll(engine.ops, job_id, _SCOPE)).done


async def _launched(
    engine: _Engine, rng: random.Random, cases: list[EvalTestCase], cuts: _Cuts, stop: _Stop
) -> list[str]:
    """Launch both arms over ``cases`` and hold the launch to its end as an operator would: each holding interrupted in
    :data:`_INTERRUPTED` of them after a uniform count of the answers left to land (``job_cancel``, as ``stop`` says),
    then taken up again (``job_resume``), at most :data:`_HOLDINGS` holdings, the last run to its end. Returns the
    launch's runs, a then b."""
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
            await _interrupt(engine, handle.job_id, finished)
            if stop == "cut":
                await _interrupt(engine, handle.job_id, finished)
            elif stop == "second" and rng.random() < _SECOND:
                await asyncio.wait({finished}, timeout=_SECOND_AFTER_S)
                await _interrupt(engine, handle.job_id, finished)
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
        if stop == "drain":
            # One cancel drains: every cell the launch started finished and recorded its answer.
            assert not any(runs_again(row) for row in rows), "a drained launch left a cell cut"
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


def _design(planned: int | None, grader: _Grader) -> dict[str, Any]:
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
        measure_order=[_GRADERS[grader][0].__name__],
        planned_looks=planned,
    ).model_dump(mode="json", exclude_none=True)


def _family(engine: _Engine, campaign_id: str) -> ComparisonFamily:
    """The campaign's one family, as the engine reads it."""
    host = engine.ops.launch.eval_host
    campaign = off_loop(get_campaign, host.storage, campaign_id, _SCOPE)
    bundle = off_loop(assemble_context_bundle, campaign, storage=host.storage, profile=host.profile)
    (family,) = bundle.multiple_comparisons.families
    return family


def _separated(family: ComparisonFamily, grader: _Grader) -> bool:
    """Whether the family separates the arms on the grader's reading, by the paired t test (never the bet), read as a
    family of that one reading: at the family's level (α, or α/L under the plan), uncorrected.

    A grade's family holds it alone. A check's holds it beside the pass rate (``passed``), which an eval of one check
    has case for case: one hypothesis counted twice, so Holm reads each at half the level. That double count is removed
    from the family (the same ``multiple_comparisons`` reads it for every campaign); until a family read here drops it,
    the read here is the corrected one: the check's own p at the family's level, asserted to be the pass rate's too."""
    name = _GRADERS[grader][0].__name__
    names = sorted(comparison.name for comparison in family.comparisons)
    assert names in ([name], [name, "passed"]) if grader == "check" else names == [name], names
    by_name = {comparison.name: comparison for comparison in family.comparisons}
    compared = by_name[name]
    assert compared.basis in (None, "t", "identical"), compared.basis
    if "passed" in by_name:
        assert by_name["passed"].p_raw == compared.p_raw, "the pass rate is the check, case for case"
    level = family.look_alpha if family.look_alpha is not None else family.alpha
    return compared.p_raw is not None and compared.p_raw < level


@dataclass(frozen=True)
class _Outcome:
    """One comparison: whether it separated, the looks it read, and its take-ups' cut rows by arm."""

    separated: bool
    looks: int
    cuts: _Cuts


async def _at_a_fixed_count(tag: str, law: _Law, grader: _Grader, stop: _Stop) -> _Outcome:
    """One comparison read once by the engine's default reading: :data:`_FIXED_CASES` cases launched, interrupted and
    taken up to their end, then filed as a campaign (control ``a``) and read at α."""
    rng = random.Random(tag)
    engine = _engine(rng, law, grader)
    cuts = _Cuts()
    a, b = await _launched(engine, rng, _cases(0, _FIXED_CASES), cuts, stop)
    filed = await _act(
        engine,
        "campaign_create",
        name="fixed",
        subject_id=_SUBJECT,
        behavior="coin",
        question="does b land right more often than a?",
        aspects=[{"label": "coin", "run_ids": [a, b], "design": _design(None, grader)}],
        campaign_arms=[{"label": "a", "role": "control", "from_run_ids": [a]}],
    )
    family = _family(engine, str(filed["id"]))
    assert family.look_alpha is None and family.alpha == SIGNIFICANCE_ALPHA
    cuts.lost = engine.landings.drawn - engine.landings.arrived
    return _Outcome(_separated(family, grader), 1, cuts)


async def _watched_over_planned_looks(tag: str, law: _Law, grader: _Grader, stop: _Stop) -> _Outcome:
    """One comparison under a plan of :data:`_PLANNED` looks, declared before its first: each look launches
    :data:`_LOOK_CASES` new cases on both arms, held to its end as :func:`_launched` holds it, joins the campaign and is
    read at α/L; the watcher stops at the first look that separates."""
    rng = random.Random(tag)
    engine = _engine(rng, law, grader)
    cuts = _Cuts()
    planned = await _act(
        engine,
        "campaign_create",
        name="planned",
        subject_id=_SUBJECT,
        behavior="coin",
        question="does b land right more often than a?",
        aspects=[{"label": "coin", "template_id": _TEMPLATE, "run_ids": [], "design": _design(_PLANNED, grader)}],
    )
    campaign_id = str(planned["id"])
    for look in range(1, _PLANNED + 1):
        a, b = await _launched(engine, rng, _cases((look - 1) * _LOOK_CASES, _LOOK_CASES), cuts, stop)
        await _act(engine, "campaign_runs_add", campaign_id=campaign_id, run_ids=[a, b])
        if look == 1:
            await _act(engine, "campaign_update", campaign_id=campaign_id, control_from_run_id=a)
        family = _family(engine, campaign_id)
        assert family.look_alpha == pytest.approx(SIGNIFICANCE_ALPHA / _PLANNED), family.look_alpha
        if _separated(family, grader):
            break
    cuts.lost = engine.landings.drawn - engine.landings.arrived
    return _Outcome(_separated(family, grader), look, cuts)


_Reading = Literal["fixed", "looks"]


async def _simulate(
    reading: _Reading, law: _Law, grader: _Grader, stop: _Stop, count: int, tag: str
) -> tuple[int, _Cuts]:
    """How many of ``count`` comparisons separated, and their take-ups' cut rows by arm, summed."""
    simulate = _at_a_fixed_count if reading == "fixed" else _watched_over_planned_looks
    separated, total = 0, _Cuts()
    for index in range(count):
        outcome = await simulate(f"{tag}-{reading}-{law}-{grader}-{stop}-{index}", law, grader, stop)
        separated += outcome.separated
        total.holdings += outcome.cuts.holdings
        for arm, again in outcome.cuts.again.items():
            total.again[arm] += again
        total.lost += outcome.cuts.lost
    return separated, total


def _said(reading: _Reading, law: _Law, grader: _Grader, stop: _Stop, separated: int, count: int, cuts: _Cuts) -> str:
    """The share, its 95% Clopper-Pearson interval, and what the take-ups ran again, for a report or a failure."""
    return (
        f"{reading} {law} {grader} {stop}: {separated}/{count} separated, {exact_rate_interval(separated, count)}; "
        f"holdings {cuts.holdings}; rows run again a={cuts.again['a']} b={cuts.again['b']}; answers the cuts lost, "
        f"right/wrong: a {cuts.lost['a', True]}/{cuts.lost['a', False]}, b {cuts.lost['b', True]}/{cuts.lost['b', False]}"
    )


# --- the interrupt rule, held cell by cell -------------------------------------------------------------------------


async def _held_launch(engine: _Engine, cases: int) -> tuple[str, str]:
    """Launch both arms over ``cases`` cases with every coin held once drawn, and wait until the run's width of cells is
    in flight. Returns the launch's job id and its id."""
    engine.landings.held = asyncio.Event()
    batch = _cases(0, cases)
    for case in batch:
        off_loop(engine.ops.launch.eval_host.storage.save_test_case, case)
    engine.batch[:] = batch
    started = await run_launch(
        engine.ops, LaunchArguments(template_id=_TEMPLATE, subject_id=_SUBJECT, models=["a", "b"], k_runs=1), _SCOPE
    )
    (handle,) = started.jobs
    async with asyncio.timeout(10):
        while engine.landings.holding < _SETTINGS.max_concurrent_cells:
            await asyncio.sleep(0.005)
    return handle.job_id, parse_job_id(handle.job_id)[1]


def _rows_of(engine: _Engine, launch_id: str) -> list[Any]:
    storage = engine.ops.launch.eval_host.storage
    return [row for run_id in _run_ids(engine, launch_id) for row in list_results(storage, run_id, _SCOPE)]


async def _taken_up_and_read(engine: _Engine, job_id: str, launch_id: str) -> ComparisonFamily:
    """Take the launch up to its end and read its runs as an engine campaign, control ``a``."""
    engine.landings.held = None
    await job_resume(engine.ops, job_id, _SCOPE)
    await engine.ops.launch.job_manager.wait_for([launch_id])
    assert (await job_poll(engine.ops, job_id, _SCOPE)).state == "completed"
    a, b = sorted(
        off_loop(engine.ops.launch.eval_host.storage.load_eval_runs, off_loop(_run_ids, engine, launch_id), _SCOPE),
        key=lambda run: run.candidate_model,
    )
    filed = await _act(
        engine,
        "campaign_create",
        name="held",
        subject_id=_SUBJECT,
        behavior="coin",
        question="does b land right more often than a?",
        aspects=[{"label": "coin", "run_ids": [a.id, b.id], "design": _design(None, "grade")}],
        campaign_arms=[{"label": "a", "role": "control", "from_run_ids": [a.id]}],
    )
    return _family(engine, str(filed["id"]))


async def test_an_operator_s_first_cancel_of_a_launch_lets_its_cells_in_flight_finish_and_the_take_up_runs_the_rest() -> (
    None
):
    """One ``job_cancel`` of an engine launch starts no further cell: the cells in flight, held, keep it running, and
    once they finish each records its answer; the launch ends cancelled with no cell cut, its take-up runs only the cells
    never started, and the family says nothing ran again."""
    engine = _engine(random.Random("drained"), "alike-20x", "grade")
    job_id, launch_id = await _held_launch(engine, 8)
    await job_cancel(engine.ops, job_id, _SCOPE, reason="shipping a fix")
    await asyncio.sleep(0.05)
    assert (await job_poll(engine.ops, job_id, _SCOPE)).state == "running", "the cells in flight keep it running"
    assert engine.landings.holding == _SETTINGS.max_concurrent_cells
    assert engine.landings.held is not None
    engine.landings.held.set()
    await engine.ops.launch.job_manager.wait_for([launch_id])
    assert (await job_poll(engine.ops, job_id, _SCOPE)).state == "cancelled"
    rows = off_loop(_rows_of, engine, launch_id)
    assert len(rows) == _SETTINGS.max_concurrent_cells, "each cell in flight recorded, none started after"
    assert {row.termination for row in rows} == {"completed"} and not any(runs_again(row) for row in rows)

    family = await _taken_up_and_read(engine, job_id, launch_id)
    rows = off_loop(_rows_of, engine, launch_id)
    assert len(rows) == 16 and not any(row.written_over for row in rows)
    assert redrawn_cells(rows) == (0, 0)
    assert "recorded no answer" not in family.disclosure


async def test_a_second_cancel_cuts_the_cells_in_flight_and_the_family_says_how_many_ran_again() -> None:
    """A second ``job_cancel`` while the first lets the cells finish cuts them: each is recorded cut, the take-up runs it
    again written over its row (``written_over``: ``cut``), and the engine family says how many cells ran again, the cut
    ones among them, in the one sentence that says it (``redrawn_sentence``)."""
    engine = _engine(random.Random("cut"), "alike-20x", "grade")
    job_id, launch_id = await _held_launch(engine, 8)
    await job_cancel(engine.ops, job_id, _SCOPE, reason="shipping a fix")
    await asyncio.sleep(0.05)
    await job_cancel(engine.ops, job_id, _SCOPE, reason="shipping a fix")
    await engine.ops.launch.job_manager.wait_for([launch_id])
    rows = off_loop(_rows_of, engine, launch_id)
    cut = [row for row in rows if row.termination == "cancelled"]
    assert len(cut) == _SETTINGS.max_concurrent_cells, "every cell in flight was cut"

    family = await _taken_up_and_read(engine, job_id, launch_id)
    rows = off_loop(_rows_of, engine, launch_id)
    assert len(rows) == 16 and {row.id for row in rows if row.written_over == "cut"} == {row.id for row in cut}
    assert redrawn_cells(rows) == (len(cut), len(cut))
    assert redrawn_sentence(len(cut), len(cut)) in family.disclosure


async def test_a_cancel_the_lease_then_finds_on_the_record_is_the_same_cancel_and_does_not_cut() -> None:
    """An operator's ``job_cancel`` cancels the launch here at once and writes the request on its record; the launch's
    lease, renewing, finds that request and does not deliver it again, so the launch drains rather than being cut by
    its own first cancel."""
    engine = _engine(random.Random("leased"), "alike-20x", "grade", lease_s=0.3)
    job_id, launch_id = await _held_launch(engine, 8)
    await job_cancel(engine.ops, job_id, _SCOPE, reason="shipping a fix")
    await asyncio.sleep(0.5)  # several renewals, each reading the request
    assert (await job_poll(engine.ops, job_id, _SCOPE)).state == "running", "the lease delivered no second cancel"
    assert engine.landings.held is not None
    engine.landings.held.set()
    await engine.ops.launch.job_manager.wait_for([launch_id])
    rows = off_loop(_rows_of, engine, launch_id)
    assert rows and {row.termination for row in rows} == {"completed"}, "nothing was cut"


# --- the simulations -------------------------------------------------------------------------------------------------


@functools.cache
def _separates_at(plus: int, minus: int, cases: int, level: float) -> bool:
    """Whether the engine's reading separates coin pairs where ``plus`` cases went to ``b`` alone, ``minus`` to ``a``
    alone, and the rest tied: the paired t test on the 0-1 range the family reads (``separation_test``), at ``level``."""
    a = [0.0] * plus + [1.0] * minus + [0.0] * (cases - plus - minus)
    b = [1.0] * plus + [0.0] * minus + [0.0] * (cases - plus - minus)
    p = separation_test(a, b, paired=True, value_range=(0.0, 1.0)).p
    return p is not None and p < level


@functools.cache
def _exact_rate(reading: _Reading) -> float:
    """The rate the engine's reading separates arms of equal accuracy (0.5) at, uninterrupted, computed exactly: each
    case goes to ``b`` alone, to ``a`` alone or ties with chances 1/4, 1/4 and 1/2, and the reading is read at the
    fixed count (α), or after each look's cases with the watcher stopping at the first separation (α/L). What a
    simulation with no lean draws around; the t test on such pairs is discrete, so this is not α itself."""
    sizes, level = (
        ([_FIXED_CASES], SIGNIFICANCE_ALPHA)
        if reading == "fixed"
        else ([_LOOK_CASES] * _PLANNED, SIGNIFICANCE_ALPHA / _PLANNED)
    )
    alive: dict[tuple[int, int], float] = {(0, 0): 1.0}
    cases, separated = 0, 0.0
    for size in sizes:
        batch = {
            (plus, minus): math.comb(size, plus)
            * math.comb(size - plus, minus)
            * 0.25 ** (plus + minus)
            * 0.5 ** (size - plus - minus)
            for plus in range(size + 1)
            for minus in range(size + 1 - plus)
        }
        grown: dict[tuple[int, int], float] = {}
        for (plus, minus), weight in alive.items():
            for (more_plus, more_minus), chance in batch.items():
                key = (plus + more_plus, minus + more_minus)
                grown[key] = grown.get(key, 0.0) + weight * chance
        cases += size
        alive = {}
        for (plus, minus), weight in grown.items():
            if _separates_at(plus, minus, cases, level):
                separated += weight
            else:
                alive[plus, minus] = weight
    return separated


def _tail(separated: int, count: int, rate: float) -> float:
    """The chance of ``separated`` or more separations in ``count`` comparisons at ``rate``: the exact binomial tail."""
    return sum(math.comb(count, k) * rate**k * (1 - rate) ** (count - k) for k in range(separated, count + 1))


async def test_the_reading_s_own_rate_is_exact_and_above_alpha_at_a_fixed_count() -> None:
    """The engine's paired t test on coin pairs, at a fixed count of 40, separates at 5.28% with no interrupt at all:
    the test is discrete on such pairs, so an at-most-α claim on it fails before any cut. Under the plan of three looks
    at α/3 it separates at the rate below. Held to the digits, so a change to the reading moves this first."""
    assert _exact_rate("fixed") == pytest.approx(0.05276, abs=5e-5)
    assert _exact_rate("looks") == pytest.approx(_EXACT_LOOKS, abs=5e-5)


async def test_a_few_engine_comparisons_interrupted_and_taken_up_finish_whole_and_read_by_the_t_test() -> None:
    """The unit suite's smoke check of every simulation: a few comparisons each, every launch held through its
    interrupts and take-ups to an end where every cell has one row and none runs again (inside :func:`_launched`; a
    drained launch's take-ups run no cell again), each read by the paired t test, at α at a fixed count and at α/L under
    the plan; and the reading separating at most one of each few, which at a rate of α happens with probability above
    0.99."""
    for reading in ("fixed", "looks"):
        for stop in ("drain", "second", "cut"):
            separated, cuts = await _simulate(reading, "b-slower-20x", "check", stop, _SMOKE, "smoke")
            assert separated <= 1, _said(reading, "b-slower-20x", "check", stop, separated, _SMOKE, cuts)
            assert cuts.holdings >= _SMOKE
            if stop == "drain":
                assert cuts.again == {"a": 0, "b": 0} and not cuts.lost


#: What a simulation's share shows, against the reading's own exact rate (:func:`_exact_rate`) and its binomial tail
#: (:func:`_tail`, one-sided, at :data:`_SURE`, so twenty-odd rows run together fail by chance about once in forty):
#:
#: - ``no_lean``: its count is not above the exact rate at that level: the interrupts leaned nothing;
#: - ``leans``: its count is above it at that level: the redrawn cells lean the reading.
_Shown = Literal["no_lean", "leans"]

#: The one-sided level each row's lean is decided at.
_SURE = 0.001

#: Each simulation, as (reading, latency law, grader, interrupt), and what it shows.
_SHOWN: dict[tuple[_Reading, _Law, _Grader, _Stop], _Shown] = {
    # The engine as it is: a first interrupt drains, so no cell is cut and none drawn afresh, whatever the latency law.
    **{
        (reading, law, grader, "drain"): "no_lean"
        for reading in ("fixed", "looks")
        for law in _LAWS
        for grader in ("check", "grade")
    },
    # A second interrupt in three of ten interrupted holdings, 5 ms after the first, cuts what is still in flight: the
    # slow arm's wrong answers, and nothing else, where its wrong answers are its slow ones. A check read as a family of
    # one is the grade's reading on the same values, so the grade's alone are drawn.
    **{(reading, law, "grade", "second"): "no_lean" for reading in ("fixed", "looks") for law in _LAWS},
    # The engine before it drained, every interrupt cutting: the comparison that shows why it drains.
    ("looks", "b-slower-20x", "check", "cut"): "leans",
    ("looks", "b-slower-20x", "grade", "cut"): "leans",
}


def _shown(reading: _Reading, law: _Law, grader: _Grader, stop: _Stop, separated: int, count: int, cuts: _Cuts) -> None:
    """Hold a simulation's share to what :data:`_SHOWN` says it shows."""
    exact = _exact_rate(reading)
    tail = _tail(separated, count, exact)
    said = f"{_said(reading, law, grader, stop, separated, count, cuts)}; exact rate {exact:.4f}, tail {tail:.2g}"
    shown = _SHOWN[reading, law, grader, stop]
    assert (tail >= _SURE) if shown == "no_lean" else (tail < _SURE), f"{shown}: {said}"


_CELLS = sorted(_SHOWN)


@pytest.mark.long_simulation
@pytest.mark.timeout(1700)
@pytest.mark.parametrize(("reading", "law", "grader", "stop"), _CELLS)
async def test_under_the_null_an_interrupted_engine_comparison_taken_up_and_read(
    reading: _Reading, law: _Law, grader: _Grader, stop: _Stop
) -> None:
    """Comparisons of arms of equal accuracy (:func:`_at_a_fixed_count`, :data:`_FIXED_REPLICATES` of them, or
    :func:`_watched_over_planned_looks`, :data:`_LOOKS_REPLICATES`), each launch interrupted as ``stop`` says and taken up
    as :func:`_launched` holds it; the share that separates as :data:`_SHOWN` states it."""
    count = _FIXED_REPLICATES if reading == "fixed" else _LOOKS_REPLICATES
    separated, cuts = await _simulate(reading, law, grader, stop, count, "engine")
    _shown(reading, law, grader, stop, separated, count, cuts)
