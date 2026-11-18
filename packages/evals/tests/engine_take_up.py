"""The engine's take-up under the null, when a cell's latency follows its answer: an engine comparison whose launches an
operator interrupts, and which the take-up then finishes, over two arms of equal accuracy.

The engine path, as a host or an agent drives it (no quick path anywhere):

- **launch**: ``run_launch`` starts both arms as one interleaved launch group, its cells ordered by a digest of the
  group's id, its repeat and its case (``runner.cell_execution_order`` with ``seed``: the whole matrix shuffled, one
  order shared by the arms), four at a time under one shared width (``LaunchSettings.max_concurrent_cells``, taken in
  turn by ``arm_schedule.CellInterleaver``);
- **interrupt**: ``job_cancel`` on the launch's job cancels every run's job (``launch_job._launch_work``), and the
  job manager reads a run's first cancel as a drain (``EvalJobManager.cancel_job``): the run starts no further cell,
  and the cells in flight finish and record. A further cancel (``kernel.offload.drain_then_cut``) cuts what is still in
  flight, recorded cut (termination ``cancelled``, ``runner.execute_run``);
- **take-up**: ``job_resume`` takes the launch up again (``resume.resume_launch``), and each run runs, in its own order,
  every cell with no row and every row that runs again (``kernel.result_condition.runs_again``), a cut row among them,
  written over its row, which then records what it replaced (``EvalResult.written_over``);
- **read**: a campaign over the runs, control ``a``, read by ``assemble_context_bundle``: on the range 0 to 1 the bounded
  test (the mixture bet over the pairs in the order fixed by their count), at α at a fixed count, or, under a plan of
  ``L`` looks declared before the first (``AspectDesign.planned_looks``), each look at α/L. A take-up of a launch is that
  launch's occasion, so it is no look of its own (``significance.looks_taken``).

A cell cut in flight is drawn afresh by the take-up, and a cell in flight is a long one: where one arm's wrong answers
are its slow ones, a cut takes its wrong answers away more often than its right ones, and the arm reads better than it
is. So the engine drains on a first interrupt, as the quick path does. The simulations read the comparisons three ways
(:data:`Stop`): drained (one cancel), drained with a second cancel 5 ms later in three of ten interrupted holdings, and
cut at every interrupt (the engine before it drained, kept as the comparison that shows why it drains).

Setup: arms of accuracy 0.5 each, k = 1; arm ``a`` 0.4 ms whatever it answers, arm ``b`` 2 ms when right and four or
twenty times that when wrong; the control's arms both 2 ms right and 40 ms wrong. Each holding (the launch and each
take-up) is interrupted in four of five at a point drawn apart from any answer (after a uniform count of the answers left
to land), at most five holdings, the last run to its end.

**Against what.** α. The reading's own rate on such pairs, uninterrupted, is computed exactly (:func:`exact_rate`): the
bounded test's capital on coin pairs is a function of how many cases went each way, so the chance it ever reaches 1/level
is a sum over those counts, 2.85% at a fixed count of 40 read at α; over three looks read at α/3, each look's own rate
summed bounds the watcher's at 2.04%. The paired t test the engine read by before separated 5.28% and 4.49% of the time
(:func:`t_test_rate`). A drained comparison is held exactly to the uninterrupted reading of its first draws
(:func:`drained_as_uninterrupted`), and every simulation's count to α by its Clopper-Pearson bound (:data:`SHOWN`). The
family is read as the product builds it (:func:`reading_separates`): an eval of one check holds that check alone, its
pass rate being the check case for case, and a grade on 0 to 1 is alone in its family too.

The statistics certificate runs these (``stats_certificate.py``); ``test_engine_take_up_null.py`` holds the interrupt
rule cell by cell, the exact rates and a smoke check of every simulation.
"""

from __future__ import annotations

import asyncio
import functools
import math
import random
from collections import Counter
from fractions import Fraction
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.actions.catalogue import Caller
from threetears.evals.analysis import ComparisonFamily, assemble_context_bundle
from threetears.evals.analysis.campaigns import get_campaign
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA, composite_significance, exact_rate_interval
from threetears.evals.kernel import ValidationFailedError, default_job_timeout
from threetears.evals.kernel.declaration import AspectDesign, ControlDeclaration, SweptAxis
from threetears.evals.kernel.host.profile import CANDIDATE_MODEL_LEVER
from threetears.evals.kernel.result_condition import runs_again
from threetears.evals.ops import LaunchArguments, OpsHost, job_cancel, job_poll, job_resume, run_launch
from threetears.evals.ops.jobs import parse_job_id
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
from packages.evals.tests.simulation_support import family_verdicts

#: The scope, template and subject every simulated comparison lives in (each in a store of its own).
SCOPE, TEMPLATE, SUBJECT = "engine", "coin", "coin-subject"

#: How long each arm's answer takes, as a law of the answer: both arms 2 ms right and 40 ms wrong (the control, one law
#: for both); or arm ``a`` 0.4 ms whatever it answers and arm ``b`` 2 ms right and four or twenty times that wrong.
Law = Literal["alike-20x", "b-slower-4x", "b-slower-20x"]
LAWS: tuple[Law, ...] = ("alike-20x", "b-slower-4x", "b-slower-20x")

#: The share of holdings interrupted, and the most holdings one launch has (the last runs to its end).
INTERRUPTED, HOLDINGS = 0.8, 5

#: How an operator interrupts a holding: ``drain``, one cancel (the engine lets the cells in flight finish and record);
#: ``second``, one cancel and, in :data:`SECOND` of the holdings interrupted, a second 5 ms after it, which cuts the
#: cells still in flight; ``cut``, two cancels at once, so every interrupt cuts what is in flight, as the engine did
#: before it drained (kept as the comparison that shows why it drains).
Stop = Literal["drain", "second", "cut"]

#: The share of interrupted holdings an operator interrupts a second time, and how long after the first.
SECOND, SECOND_AFTER_S = 0.3, 0.005

#: The cases one comparison read at a fixed count runs; the cases each launch of a planned comparison adds, one look
#: each (20, 40 and 60 cases at its three looks), and the looks it plans.
FIXED_CASES, LOOK_CASES, PLANNED = 40, 20, 3

#: The bound on the reading's own rate under the plan of three looks, uninterrupted, as :func:`exact_rate` computes it:
#: each look's exact rate at α/3, summed.
EXACT_LOOKS = 0.02037

#: Comparisons in each long simulation at a fixed count, in each under a plan of looks, and in each smoke check: each
#: long one a pool request of its own, inside its window.
FIXED_REPLICATES, LOOKS_REPLICATES, SMOKE = 800, 500, 3

CALLER = Caller(scope_id=SCOPE, identity="engine-operator")

#: The launch settings: the engine's own width (four cells at once), nothing capped that a coin could reach.
SETTINGS = LaunchSettings(
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


#: How the answer is graded: by a check (``correct``), whose family holds it alone, the pass rate being that check case
#: for case; or by a grade on the range 0 to 1 (``grade``), alone in its family. Either is read by the bounded test.
Grader = Literal["check", "grade"]

GRADERS: dict[Grader, tuple[Callable[[Mapping[str, Any], str], Any], dict[str, tuple[float, float]]]] = {
    "check": (correct, {}),
    "grade": (grade, {"grade": (0.0, 1.0)}),
}


def _seconds(law: Law, arm: str, right: bool) -> float:
    """How long ``arm``'s answer takes under ``law``."""
    if law == "alike-20x":
        return 0.002 if right else 0.040
    if arm == "a":
        return 0.0004
    return 0.002 if right else 0.002 * (4 if law == "b-slower-4x" else 20)


@dataclass
class Landings:
    """The answers landed in the holding under way, and the count at which it is interrupted (``None``: never)."""

    landed: int = 0
    stop_at: int | None = None
    reached: asyncio.Event = field(default_factory=asyncio.Event)
    #: Every answer drawn and every answer landed, by arm and whether it was right, over the comparison: a cell cut while
    #: its answer was on its way drew one that never landed.
    drawn: Counter[tuple[str, bool]] = field(default_factory=Counter)
    arrived: Counter[tuple[str, bool]] = field(default_factory=Counter)
    #: Each cell's draws in the order drawn, by arm and case: one where nothing was cut, each cut one more.
    draws: dict[tuple[str, int], list[bool]] = field(default_factory=dict)

    #: Where set, every coin waits on it once its answer is drawn, so a test holds cells in flight; and how many wait.
    held: asyncio.Event | None = None
    holding: int = 0

    def land(self) -> None:
        self.landed += 1
        if self.stop_at is not None and self.landed >= self.stop_at:
            self.reached.set()


@dataclass
class Engine:
    """One comparison's engine: a host of the callable kind over a store of its own, its two arms' coins, and the
    action catalogue an operator drives it through."""

    ops: OpsHost
    tool: Any
    batch: list[EvalTestCase]
    landings: Landings


def make_engine(rng: random.Random, law: Law, grader: Grader, *, lease_s: float | None = None) -> Engine:
    """A fresh engine host: the callable kind (``quick.one_call.callable_kind``) wired by a launcher of its own through
    the engine's ``launch_run``, each arm a coin that lands right half the time and takes as long as ``law`` says; its
    job leases last ``lease_s`` where given, the engine's default otherwise."""
    scorer, ranges = GRADERS[grader]
    host = callable_host([scorer], ranges=ranges, store=InMemoryDocumentStore())
    off_loop(
        host.storage.save_template,
        EvalTemplate(id=TEMPLATE, scope_id=SCOPE, name="a coin", intent="Land right.", candidate_kind=CALLABLE_KIND),
    )
    landings = Landings()

    def coin(arm: str) -> Callable[[Mapping[str, Any]], Any]:
        async def run(case: Mapping[str, Any]) -> str:
            right = rng.random() < 0.5
            landings.drawn[arm, right] += 1
            landings.draws.setdefault((arm, int(case["n"])), []).append(right)
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
    subject = SubjectSnapshot(subject_id=SUBJECT, subject_label=SUBJECT, state={})

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
        settings=lambda: SETTINGS,
        job_timeout_factory=default_job_timeout,
        **({} if lease_s is None else {"job_lease_s": lease_s}),
    )
    return Engine(
        ops=OpsHost(launch=launching),
        tool=eval_catalogue().mount_all(standard_tools())[0],
        batch=batch,
        landings=landings,
    )


async def act(engine: Engine, action: str, **arguments: Any) -> Any:
    done = await engine.tool.call({"action": action, **arguments}, host=engine.ops, caller=CALLER)
    assert not done.is_error, done.text
    return done.structured


def engine_cases(start: int, count: int) -> list[EvalTestCase]:
    return [
        EvalTestCase(
            id=f"case-{n:04d}",
            scope_id=SCOPE,
            template_id=TEMPLATE,
            variation_params={"n": str(n)},
            host_payload={"case": {"n": n}},
        )
        for n in range(start, start + count)
    ]


@dataclass
class Cuts:
    """The rows each arm's take-ups ran again, by arm, and the holdings the comparison's launches took."""

    again: dict[str, int] = field(default_factory=lambda: {"a": 0, "b": 0})
    holdings: int = 0
    #: The answers drawn by cells that were cut before they landed, by arm and whether each was right.
    lost: Counter[tuple[str, bool]] = field(default_factory=Counter)


async def _interrupt(engine: Engine, job_id: str, finished: asyncio.Future[None]) -> None:
    """One operator's ``job_cancel`` of a launch still running; a cancel that finds it ended is no interrupt."""
    if finished.done():
        return
    try:
        await job_cancel(engine.ops, job_id, SCOPE, reason="interrupted")
    except ValidationFailedError:
        # The launch ended between its last answer landing and the cancel: nothing was left to interrupt.
        assert finished.done() or (await job_poll(engine.ops, job_id, SCOPE)).done


async def _launched(engine: Engine, rng: random.Random, cases: list[EvalTestCase], cuts: Cuts, stop: Stop) -> list[str]:
    """Launch both arms over ``cases`` and hold the launch to its end as an operator would: each holding interrupted in
    :data:`INTERRUPTED` of them after a uniform count of the answers left to land (``job_cancel``, as ``stop`` says),
    then taken up again (``job_resume``), at most :data:`HOLDINGS` holdings, the last run to its end. Returns the
    launch's runs, a then b."""
    for case in cases:
        off_loop(engine.ops.launch.eval_host.storage.save_test_case, case)
    engine.batch[:] = cases
    manager, storage = engine.ops.launch.job_manager, engine.ops.launch.eval_host.storage
    started = await run_launch(
        engine.ops,
        LaunchArguments(template_id=TEMPLATE, subject_id=SUBJECT, models=["a", "b"], k_runs=1),
        SCOPE,
    )
    (handle,) = started.jobs
    _, launch_id, _ = parse_job_id(handle.job_id)
    left = 2 * len(cases)
    for holding in range(1, HOLDINGS + 1):
        cuts.holdings += 1
        landings = engine.landings
        landings.landed, landings.reached = 0, asyncio.Event()
        interrupt = holding < HOLDINGS and rng.random() < INTERRUPTED
        landings.stop_at = rng.randint(1, max(left, 1)) if interrupt else None
        finished = asyncio.ensure_future(manager.wait_for([launch_id]))
        if interrupt:
            reached = asyncio.ensure_future(landings.reached.wait())
            await asyncio.wait({finished, reached}, return_when=asyncio.FIRST_COMPLETED)
            reached.cancel()
            await _interrupt(engine, handle.job_id, finished)
            if stop == "cut":
                await _interrupt(engine, handle.job_id, finished)
            elif stop == "second" and rng.random() < SECOND:
                await asyncio.wait({finished}, timeout=SECOND_AFTER_S)
                await _interrupt(engine, handle.job_id, finished)
        await finished
        status = await job_poll(engine.ops, handle.job_id, SCOPE)
        if status.state == "completed":
            break
        assert status.state == "cancelled", status
        runs = off_loop(storage.load_eval_runs, off_loop(_run_ids, engine, launch_id), SCOPE)
        if all(run.status == "completed" for run in runs):
            # The cancel reached the launch after its runs had ended: nothing is left to take up.
            break
        rows = [
            row
            for run_id in off_loop(_run_ids, engine, launch_id)
            for row in off_loop(list_results, storage, run_id, SCOPE)
        ]
        recorded = [row for row in rows if not runs_again(row)]
        for row in rows:
            if runs_again(row):
                cuts.again[row.model] += 1
        if stop == "drain":
            # One cancel drains: every cell the launch started finished and recorded its answer.
            assert not any(runs_again(row) for row in rows), "a drained launch left a cell cut"
        left = 2 * len(cases) - len(recorded)
        await job_resume(engine.ops, handle.job_id, SCOPE)
    else:
        raise AssertionError(f"launch {launch_id} did not complete within {HOLDINGS} holdings")
    runs = sorted(
        off_loop(storage.load_eval_runs, off_loop(_run_ids, engine, launch_id), SCOPE),
        key=lambda run: run.candidate_model,
    )
    # The take-up finished the launch as one uninterrupted would have: every cell one row, none left to run again.
    for run in runs:
        assert run.status == "completed", run.status
        rows = off_loop(list_results, storage, run.id, SCOPE)
        assert sorted(row.test_case_id for row in rows) == sorted(case.id for case in cases)
        assert not any(runs_again(row) for row in rows)
    return [run.id for run in runs]


def _run_ids(engine: Engine, launch_id: str) -> list[str]:
    """The runs a launch started, in arm order, as its record names them. Reads the store."""
    launch, _ = engine.ops.launch.eval_host.storage.load_job_record(EvalLaunch, launch_id, SCOPE)
    assert launch is not None
    return list(launch.run_ids)


# --- the reading -----------------------------------------------------------------------------------------------------


def design(planned: int | None, grader: Grader) -> dict[str, Any]:
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
        measure_order=[GRADERS[grader][0].__name__],
        planned_looks=planned,
    ).model_dump(mode="json", exclude_none=True)


def family_of(engine: Engine, campaign_id: str) -> ComparisonFamily:
    """The campaign's one family, as the engine reads it."""
    host = engine.ops.launch.eval_host
    campaign = off_loop(get_campaign, host.storage, campaign_id, SCOPE)
    bundle = off_loop(assemble_context_bundle, campaign, storage=host.storage, profile=host.profile)
    (family,) = bundle.multiple_comparisons.families
    return family


def reading_separates(family: ComparisonFamily, grader: Grader) -> bool:
    """Whether the family separates the arms, as the product reads it: its verdict on the grader's reading.

    The family holds that one reading: a grade alone, and a check alone, the pass rate of an eval of one check being
    that check case for case. It is read by the bounded test on 0 to 1 (never the bet read again: no evidence is
    graded), at the family's level, α or α/L under the plan."""
    name = GRADERS[grader][0].__name__
    names = sorted(comparison.name for comparison in family.comparisons)
    assert names == [name], names
    (compared,) = family.comparisons
    assert compared.basis in (None, "bounded", "identical"), compared.basis
    assert compared.evidence_better is None, "read once at its count, not by the bet read again"
    return compared.verdict in ("improved", "regressed")


@dataclass(frozen=True)
class Outcome:
    """One comparison: whether it separated, the looks it read, its take-ups' cut rows by arm, each look's reading (the
    cases read, the grader's raw p and the level it is read at), and every cell's draws (:attr:`Landings.draws`)."""

    separated: bool
    looks: int
    cuts: Cuts
    reads: tuple[tuple[int, float | None, float], ...] = ()
    draws: dict[tuple[str, int], list[bool]] = field(default_factory=dict)


def _grader_p(family: ComparisonFamily, grader: Grader) -> tuple[float | None, float]:
    """The grader's raw p in ``family``, and the level the engine reads it at."""
    name = GRADERS[grader][0].__name__
    (compared,) = [comparison for comparison in family.comparisons if comparison.name == name]
    return compared.p_raw, family.look_alpha if family.look_alpha is not None else family.alpha


def drained_as_uninterrupted(outcome: Outcome) -> str | None:
    """Where a drained comparison's reading is not the uninterrupted one on the same answers, as a sentence; ``None``
    where it is. Drained, every cell drew once, so each value read is a fresh fair coin assigned before it was seen and
    the values are those of an uninterrupted launch; and each look's p and verdict are the family's rule on those first
    draws (``simulation_support.family_verdicts``, which reads 0 to 1 by the bounded test), so the comparison separates
    exactly where the uninterrupted reading does, and its rate is at most :func:`exact_rate`'s."""
    redrawn = sorted(cell for cell, draws in outcome.draws.items() if len(draws) != 1)
    if redrawn:
        return f"cells drew more than once: {redrawn[:5]}"
    restated_separated, restated_looks = False, 0
    for cases, p, level in outcome.reads:
        control = {f"case-{n:04d}": float(outcome.draws["a", n][0]) for n in range(cases)}
        contrast = {f"case-{n:04d}": float(outcome.draws["b", n][0]) for n in range(cases)}
        (rule,) = family_verdicts([(control, contrast, True)], value_ranges=[(0.0, 1.0)], alpha=level)
        recomputed = rule.p_raw
        if (p is None) != (recomputed is None) or (
            p is not None and recomputed is not None and not math.isclose(p, recomputed, rel_tol=1e-9, abs_tol=1e-12)
        ):
            return f"at {cases} cases the product read p {p}, the first draws {recomputed}"
        restated_looks += 1
        if rule.verdict in ("improved", "regressed"):
            restated_separated = True
            break
    if (restated_separated, restated_looks) != (outcome.separated, outcome.looks):
        return (
            f"the product separated {outcome.separated} at look {outcome.looks}; the first draws read "
            f"{restated_separated} at look {restated_looks}"
        )
    return None


async def at_a_fixed_count(tag: str, law: Law, grader: Grader, stop: Stop) -> Outcome:
    """One comparison read once by the engine's default reading: :data:`FIXED_CASES` cases launched, interrupted and
    taken up to their end, then filed as a campaign (control ``a``) and read at α."""
    rng = random.Random(tag)
    engine = make_engine(rng, law, grader)
    cuts = Cuts()
    a, b = await _launched(engine, rng, engine_cases(0, FIXED_CASES), cuts, stop)
    filed = await act(
        engine,
        "campaign_create",
        name="fixed",
        subject_id=SUBJECT,
        behavior="coin",
        question="does b land right more often than a?",
        aspects=[{"label": "coin", "run_ids": [a, b], "design": design(None, grader)}],
        campaign_arms=[{"label": "a", "role": "control", "from_run_ids": [a]}],
    )
    family = family_of(engine, str(filed["id"]))
    assert family.look_alpha is None and family.alpha == SIGNIFICANCE_ALPHA
    cuts.lost = engine.landings.drawn - engine.landings.arrived
    p, level = _grader_p(family, grader)
    return Outcome(reading_separates(family, grader), 1, cuts, ((FIXED_CASES, p, level),), dict(engine.landings.draws))


async def watched_over_planned_looks(tag: str, law: Law, grader: Grader, stop: Stop) -> Outcome:
    """One comparison under a plan of :data:`PLANNED` looks, declared before its first: each look launches
    :data:`LOOK_CASES` new cases on both arms, held to its end as :func:`_launched` holds it, joins the campaign and is
    read at α/L; the watcher stops at the first look that separates."""
    rng = random.Random(tag)
    engine = make_engine(rng, law, grader)
    cuts = Cuts()
    planned = await act(
        engine,
        "campaign_create",
        name="planned",
        subject_id=SUBJECT,
        behavior="coin",
        question="does b land right more often than a?",
        aspects=[{"label": "coin", "template_id": TEMPLATE, "run_ids": [], "design": design(PLANNED, grader)}],
    )
    campaign_id = str(planned["id"])
    reads: list[tuple[int, float | None, float]] = []
    for look in range(1, PLANNED + 1):
        a, b = await _launched(engine, rng, engine_cases((look - 1) * LOOK_CASES, LOOK_CASES), cuts, stop)
        await act(engine, "campaign_runs_add", campaign_id=campaign_id, run_ids=[a, b])
        if look == 1:
            await act(engine, "campaign_update", campaign_id=campaign_id, control_from_run_id=a)
        family = family_of(engine, campaign_id)
        assert family.look_alpha is not None and math.isclose(family.look_alpha, SIGNIFICANCE_ALPHA / PLANNED)
        reads.append((look * LOOK_CASES, *_grader_p(family, grader)))
        if reading_separates(family, grader):
            break
    cuts.lost = engine.landings.drawn - engine.landings.arrived
    return Outcome(reading_separates(family, grader), look, cuts, tuple(reads), dict(engine.landings.draws))


Reading = Literal["fixed", "looks"]


def comparison_tag(tag: str, reading: Reading, law: Law, grader: Grader, stop: Stop, index: int) -> str:
    """The seed of the ``index``-th comparison a simulation tagged ``tag`` reads."""
    return f"{tag}-{reading}-{law}-{grader}-{stop}-{index}"


async def one_comparison(tag: str, reading: Reading, law: Law, grader: Grader, stop: Stop) -> Outcome:
    """One comparison, read at a fixed count or over the plan of looks."""
    run = at_a_fixed_count if reading == "fixed" else watched_over_planned_looks
    return await run(tag, law, grader, stop)


async def simulate(
    reading: Reading, law: Law, grader: Grader, stop: Stop, count: int, tag: str, *, start: int = 0
) -> tuple[int, Cuts]:
    """How many of ``count`` comparisons (from the ``start``-th) separated, and their take-ups' cut rows by arm,
    summed."""
    separated, total = 0, Cuts()
    for index in range(start, start + count):
        outcome = await one_comparison(
            comparison_tag(tag, reading, law, grader, stop, index), reading, law, grader, stop
        )
        separated += outcome.separated
        total.holdings += outcome.cuts.holdings
        for arm, again in outcome.cuts.again.items():
            total.again[arm] += again
        total.lost += outcome.cuts.lost
    return separated, total


def said(reading: Reading, law: Law, grader: Grader, stop: Stop, separated: int, count: int, cuts: Cuts) -> str:
    """The share, its 95% Clopper-Pearson interval, and what the take-ups ran again, for a report or a failure."""
    return (
        f"{reading} {law} {grader} {stop}: {separated}/{count} separated, {exact_rate_interval(separated, count)}; "
        f"holdings {cuts.holdings}; rows run again a={cuts.again['a']} b={cuts.again['b']}; answers the cuts lost, "
        f"right/wrong: a {cuts.lost['a', True]}/{cuts.lost['a', False]}, b {cuts.lost['b', True]}/{cuts.lost['b', False]}"
    )


# --- the interrupt rule, held cell by cell -------------------------------------------------------------------------


async def held_launch(engine: Engine, cases: int) -> tuple[str, str]:
    """Launch both arms over ``cases`` cases with every coin held once drawn, and wait until the run's width of cells is
    in flight. Returns the launch's job id and its id."""
    engine.landings.held = asyncio.Event()
    batch = engine_cases(0, cases)
    for case in batch:
        off_loop(engine.ops.launch.eval_host.storage.save_test_case, case)
    engine.batch[:] = batch
    started = await run_launch(
        engine.ops, LaunchArguments(template_id=TEMPLATE, subject_id=SUBJECT, models=["a", "b"], k_runs=1), SCOPE
    )
    (handle,) = started.jobs
    async with asyncio.timeout(10):
        while engine.landings.holding < SETTINGS.max_concurrent_cells:
            await asyncio.sleep(0.005)
    return handle.job_id, parse_job_id(handle.job_id)[1]


def rows_of(engine: Engine, launch_id: str) -> list[Any]:
    storage = engine.ops.launch.eval_host.storage
    return [row for run_id in _run_ids(engine, launch_id) for row in list_results(storage, run_id, SCOPE)]


async def taken_up_and_read(engine: Engine, job_id: str, launch_id: str) -> ComparisonFamily:
    """Take the launch up to its end and read its runs as an engine campaign, control ``a``."""
    engine.landings.held = None
    await job_resume(engine.ops, job_id, SCOPE)
    await engine.ops.launch.job_manager.wait_for([launch_id])
    assert (await job_poll(engine.ops, job_id, SCOPE)).state == "completed"
    a, b = sorted(
        off_loop(engine.ops.launch.eval_host.storage.load_eval_runs, off_loop(_run_ids, engine, launch_id), SCOPE),
        key=lambda run: run.candidate_model,
    )
    filed = await act(
        engine,
        "campaign_create",
        name="held",
        subject_id=SUBJECT,
        behavior="coin",
        question="does b land right more often than a?",
        aspects=[{"label": "coin", "run_ids": [a.id, b.id], "design": design(None, "grade")}],
        campaign_arms=[{"label": "a", "role": "control", "from_run_ids": [a.id]}],
    )
    return family_of(engine, str(filed["id"]))


# --- the simulations -------------------------------------------------------------------------------------------------


#: The bounded test's mixture stake on coin pairs (``stats._mixture_log_path``), restated: ten bets, the j-th staking j/10
#: of the cap 0.9/m on every pair with weight j/55. A pair is a difference of -1, 0 or 1 on the range ±1, rescaled to 0,
#: 1/2 or 1 against the null mean m = 1/2, so the j-th bet's capital moves by 1 ± 0.09 j on a pair one arm alone got
#: right and not at all on a tie: after ``plus`` pairs to ``b`` and ``minus`` to ``a`` it is
#: ``(1 + 0.09 j)^plus (1 - 0.09 j)^minus``, whatever their order, and the downward bet's is that with the two swapped.
_STAKES = tuple(0.09 * j for j in range(1, 11))
_WEIGHTS = tuple(j / 55 for j in range(1, 11))


def hedged_log_capital(plus: int, minus: int) -> float:
    """The log of the even hedge's capital (half the upward mixture's, half the downward's) after ``plus`` coin pairs
    went to ``b`` alone and ``minus`` to ``a`` alone, ties moving nothing: the capital the engine's bounded test reads
    coin pairs by, which ``test_engine_fixed_count_bounded.py`` holds to the product's p."""
    up = sum(w * (1 + c) ** plus * (1 - c) ** minus for w, c in zip(_WEIGHTS, _STAKES, strict=True))
    down = sum(w * (1 + c) ** minus * (1 - c) ** plus for w, c in zip(_WEIGHTS, _STAKES, strict=True))
    return math.log(0.5 * up + 0.5 * down)


def exact_rates(level: float, cases: int, *, plus: Fraction = Fraction(1, 4)) -> list[Fraction]:
    """The chance the bounded test separates pairs of equal mean at ``level``, read once at each count from 1 to
    ``cases``, computed exactly: each pair goes to ``b`` alone with chance ``plus``, to ``a`` alone with the same chance,
    and ties otherwise (``plus`` = 1/4 for two fair coins). Its p is one over the most its capital reached
    (:func:`hedged_log_capital`), a function of the counts each way, so the chance it is below ``level`` by ``n`` pairs
    is the chance the walk of those counts has entered the region where the capital exceeds 1/level: a sum over the
    walk's states, kept as exact fractions. The order the pairs are bet in, fixed by their count, moves nothing under
    the null, every pair drawn alike.

    :return: The rate at each count from 1 to ``cases``.
    :raises AssertionError: A state's capital lies within 1e-9 of the line in log terms, where float rounding could
        put it on either side.
    """
    threshold = -math.log(level)
    alive: dict[tuple[int, int], Fraction] = {(0, 0): Fraction(1)}
    separated, rates = Fraction(0), []
    for _ in range(cases):
        grown: dict[tuple[int, int], Fraction] = {}
        for (up, down), weight in alive.items():
            for (more_up, more_down), chance in (((1, 0), plus), ((0, 1), plus), ((0, 0), 1 - 2 * plus)):
                key = (up + more_up, down + more_down)
                grown[key] = grown.get(key, Fraction(0)) + weight * chance
        alive = {}
        for (up, down), weight in grown.items():
            log_capital = hedged_log_capital(up, down)
            assert abs(log_capital - threshold) > 1e-9, (up, down)
            if log_capital > threshold:
                separated += weight
            else:
                alive[up, down] = weight
        rates.append(separated)
    return rates


@functools.cache
def exact_rate(reading: Reading) -> float:
    """The rate the engine's reading separates arms of equal accuracy (0.5) at, uninterrupted, by its p: at the fixed
    count (α), exactly (:func:`exact_rates`); over the plan of looks (α/L), each look's own exact rate at α/L summed, a
    bound on the watcher who stops at the first, whose looks are each read in the order fixed by their own count. A
    separation needs the interval to exclude zero too, which only removes some, so the product separates at most this.
    """
    if reading == "fixed":
        return float(exact_rates(SIGNIFICANCE_ALPHA, FIXED_CASES)[-1])
    rates = exact_rates(SIGNIFICANCE_ALPHA / PLANNED, LOOK_CASES * PLANNED)
    return float(sum(rates[LOOK_CASES * look - 1] for look in range(1, PLANNED + 1)))


@functools.cache
def t_test_rate(reading: Reading) -> float:
    """The rate the paired t test the engine read 0 to 1 by before separated such pairs at, computed exactly as the old
    reading was, over the counts each way, which are all a t statistic reads: 5.28% at a fixed count of 40, 4.49% over
    three looks with the watcher stopping at the first separation. What the bounded test replaced."""
    sizes, level = (
        ([FIXED_CASES], SIGNIFICANCE_ALPHA)
        if reading == "fixed"
        else ([LOOK_CASES] * PLANNED, SIGNIFICANCE_ALPHA / PLANNED)
    )
    alive: dict[tuple[int, int], float] = {(0, 0): 1.0}
    cases, separated = 0, 0.0
    for size in sizes:
        batch = {
            (up, down): math.comb(size, up)
            * math.comb(size - up, down)
            * 0.25 ** (up + down)
            * 0.5 ** (size - up - down)
            for up in range(size + 1)
            for down in range(size + 1 - up)
        }
        grown: dict[tuple[int, int], float] = {}
        for (up, down), weight in alive.items():
            for (more_up, more_down), chance in batch.items():
                key = (up + more_up, down + more_down)
                grown[key] = grown.get(key, 0.0) + weight * chance
        cases += size
        alive = {}
        for (up, down), weight in grown.items():
            a = [0.0] * up + [1.0] * down + [0.0] * (cases - up - down)
            b = [1.0] * up + [0.0] * down + [0.0] * (cases - up - down)
            p = composite_significance(a, b, paired=True).p_value
            if p is not None and p < level:
                separated += weight
            else:
                alive[up, down] = weight
    return separated


#: What a simulation's count shows, held to α by its Clopper-Pearson 95% bounds (``stats_certificate.judge``):
#:
#: - ``at_most``: the upper bound is at most α: the reading holds α, interrupted as the row says;
#: - ``above``: the lower bound is above α: the redrawn cells lean the reading past α.
Shown = Literal["at_most", "above"]

#: Each simulation, as (reading, latency law, grader, interrupt), and what it shows.
SHOWN: dict[tuple[Reading, Law, Grader, Stop], Shown] = {
    # The engine as it is: a first interrupt drains, so no cell is cut and none drawn afresh, whatever the latency law;
    # each such comparison reads exactly as the uninterrupted one on its first draws (:func:`drained_as_uninterrupted`),
    # which the certificate holds comparison by comparison, and the count holds α.
    **{
        (reading, law, grader, "drain"): "at_most"
        for reading in ("fixed", "looks")
        for law in LAWS
        for grader in ("check", "grade")
    },
    # A second interrupt in three of ten interrupted holdings, 5 ms after the first, cuts what is still in flight: the
    # slow arm's wrong answers, and nothing else, where its wrong answers are its slow ones. A check's family is the
    # grade's reading on the same values, so the grade's alone are drawn.
    **{(reading, law, "grade", "second"): "at_most" for reading in ("fixed", "looks") for law in LAWS},
    # The engine before it drained, every interrupt cutting: the comparison that shows why it drains.
    ("looks", "b-slower-20x", "check", "cut"): "above",
    ("looks", "b-slower-20x", "grade", "cut"): "above",
}
