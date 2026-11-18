"""The package's own reading of an on-ramp run, with the app as the case.

A stored run (its ``trials.jsonl`` rows) is handed to ``threetears.evals`` through its front door, as any adopter
hands it a comparison: an :class:`~threetears.evals.Eval` whose cases are the run's apps, whose arms replay each arm's
stored trials of an app, and whose one grader is the check read, returning what the trial stored. ``k`` is the
trials each arm ran on each app. The package then reads it as it reads every comparison: each app's share of its
trials passed, per arm, compared paired across apps (``_per_case_values``: repeats of a case are averaged before any
test sees them). No trial is pooled with another app's, and no model is called: the reading spends nothing.

A trial is replayed as the run ended it: one the run lost is lost again (its grade refuses it), and one whose runner
raised fails again, as the package read it in the run.

One check is read per comparison. The primary is the comparison's question, which the package tests alone and
one-sided, toward the candidate; the other checks are leads, read with no question, never decided on. Tiers are never
pooled: rows of more than one tier are read tier by tier.

The error rate this reading holds under the on-ramp's null (apps as cases, three trials, an arm-by-app spread) is the
package's certified claim (``small-n.once.onramp`` and ``small-n.continued.onramp.*`` in
``packages/evals/tests/fixtures/stats_certificate.json``); ``tests/test_onramp_reading.py`` guards that this replay
hands the package apps as cases, not trials. :func:`simulated_rows` draws stored-shaped rows under that model, for
the guard and for ``--power``, which says how often a plan of a given size separates an effect.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter, defaultdict, deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals import Eval, Result
from threetears.evals.storage import InMemoryDocumentStore

from .checks import GRADERS

#: Every check a trial is graded on, in the order the harness grades them.
CHECKS = tuple(grader.__name__ for grader in GRADERS)
#: The check a plan compares unless it names another: whether the agent's eval tells the app from its fixed twin.
PRIMARY = "measures_the_app"
#: The outcomes the package reads as a separation between arms.
SEPARATED = ("improved", "regressed")


class Unreadable(ValueError):
    """A stored run the package cannot read with apps as cases, and why."""


class LostTrial(RuntimeError):
    """A stored trial that holds no grade for the check read, replayed as the run ended it (:func:`_replay`)."""


def app_of(row: Mapping[str, Any]) -> str | None:  # Any: a trial's stored row
    """The app a stored trial ran on: the row's own ``app``, else its record's; ``None`` when neither holds one."""
    app = row.get("app")
    if isinstance(app, str):
        return app
    record = row.get("record")
    app = record.get("app") if isinstance(record, Mapping) else None
    return app if isinstance(app, str) else None


def tiers_of(rows: Sequence[Mapping[str, Any]]) -> dict[str | None, list[Mapping[str, Any]]]:  # Any: stored rows
    """The rows by tier, in the order each tier first appears; a row of a run that recorded no tier is under ``None``."""
    by_tier: dict[str | None, list[Mapping[str, Any]]] = {}
    for row in rows:
        tier = row.get("tier")
        by_tier.setdefault(tier if isinstance(tier, str) else None, []).append(row)
    return by_tier


@dataclass(frozen=True)
class Design:
    """What a stored run holds, as the package will read it: its arms (the control first), its apps, and ``k``."""

    arms: tuple[str, ...]
    apps: tuple[str, ...]
    k: int
    trials: Mapping[str, Mapping[str, tuple[Mapping[str, Any], ...]]]  # Any: arm -> app -> stored rows


def design_of(
    rows: Sequence[Mapping[str, Any]], arms: Sequence[str] = (), apps: Sequence[str] = ()
) -> Design:  # Any: stored rows
    """The arms, apps and repeats of one tier's stored rows.

    ``arms`` is the run's own order of its arms, the control first, and ``apps`` the order its cases ran in, as
    ``run.json`` records them: given, they decide which arm is the control and the case list the package reads (whose
    order its bet is committed by), so a file whose rows were reordered reads the same. Not given, each is in the order
    the rows first name it, which the harness writes control first and in case order.

    :raises Unreadable: Rows of more than one tier; a row with no app; arms other than ``arms``; arms that ran different
        apps; or an arm and app with a different count of trials from another, since ``k`` is one number for every case
        and arm.
    """
    if len(tiers_of(rows)) > 1:
        raise Unreadable(
            f"rows of tiers {', '.join(map(str, tiers_of(rows)))}: read each tier on its own, never pooled"
        )
    trials: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        app = app_of(row)
        if app is None:
            raise Unreadable(f"a trial of arm {row.get('arm')} (repeat {row.get('repeat')}) names no app")
        trials[str(row["arm"])][app].append(row)
    if not trials:
        raise Unreadable("the run holds no trial")
    if arms and sorted(arms) != sorted(trials):
        raise Unreadable(f"the rows' arms ({', '.join(trials)}) are not the run's ({', '.join(arms)})")
    arms = tuple(arms) or tuple(trials)
    seen = tuple(dict.fromkeys(app for arm in arms for app in trials[arm]))
    if apps and sorted(apps) != sorted(seen):
        raise Unreadable(f"the rows' apps ({', '.join(seen)}) are not the run's ({', '.join(apps)})")
    apps = tuple(apps) or seen
    if missing := [f"{arm} ran no trial on {app}" for arm in arms for app in apps if app not in trials[arm]]:
        raise Unreadable("; ".join(missing))
    counts = Counter(len(trials[arm][app]) for arm in arms for app in apps)
    if len(counts) > 1:
        each = ", ".join(f"{arm} {app} {len(trials[arm][app])}" for arm in arms for app in apps)
        raise Unreadable(f"the arms ran different counts of trials on the apps ({each}): k is one number")
    (k,) = counts
    stored = {arm: {app: tuple(trials[arm][app]) for app in apps} for arm in arms}
    return Design(arms, apps, k, stored)


def _replay(
    trials: Mapping[str, Sequence[Mapping[str, Any]]], check: str
) -> Callable[[dict[str, Any]], Awaitable[Any]]:
    """One arm's runner: each call on an app answers that app's next stored trial, as whether it passed ``check``.

    A trial as the run ended it: one the run counted lost answers ``None``, which the grade refuses, so the package
    loses it too (a grader that raises is the rig's fault, counted for nothing); one the run counted failed with no
    grade (its runner raised) raises here, so the package reads it as the run did: failed on the pass rate, with no
    value on the check.
    """
    left = {app: deque(rows) for app, rows in trials.items()}

    async def replay(case: dict[str, Any]) -> dict[str, bool | None]:  # Any: a case
        """The arm's next stored trial on this app, as it was graded."""
        row = left[case["app"]].popleft()
        score = row.get("scores", {}).get(check)
        if score is None and row.get("state") != "lost":
            raise LostTrial(f"the run's runner gave no answer to grade {check} on ({row.get('state')})")
        return {check: None if score is None else score == 1.0}

    return replay


def _stored_grade(check: str) -> Callable[[dict[str, Any], dict[str, bool | None]], bool]:
    """A check named ``check`` that passes a replayed trial when the run graded it passed, and refuses one the run
    lost."""

    def grade(case: dict[str, Any], out: dict[str, bool | None]) -> bool:  # Any: a case
        passed = out[check]
        if passed is None:
            raise LostTrial(f"the run lost this trial: it holds no {check} grade")
        return passed

    grade.__name__ = grade.__qualname__ = check
    grade.__doc__ = f"The trial passed {check} when the run graded it."
    return grade


def comparison(
    rows: Sequence[Mapping[str, Any]],
    check: str = PRIMARY,
    *,
    question: str | None = None,
    lead: bool = False,
    arms: Sequence[str] = (),
    apps: Sequence[str] = (),
) -> Eval:  # Any: stored rows
    """One tier's stored run as a comparison on one check, with the app as the case (see the module).

    With two arms or more it asks ``question`` (one naming the check when none is given), which the package tests
    alone and one-sided, toward the candidate passing more often. A ``lead`` asks none: a secondary check, which the
    package reads without deciding.

    :raises Unreadable: As :func:`design_of`; or a check no trial is graded on.
    """
    if check not in CHECKS:
        raise Unreadable(f"{check} is not a check the harness grades ({', '.join(CHECKS)})")
    design = design_of(rows, arms, apps)
    asked = None if lead else question or f"Do agents given {design.arms[-1]} pass {check} more often, app by app?"
    return Eval(
        [{"id": app, "app": app} for app in design.apps],
        runner={arm: _replay(design.trials[arm], check) for arm in design.arms},
        grader=[_stored_grade(check)],
        k=design.k,
        name="onramp",
        question=asked if len(design.arms) > 1 else None,
    )


async def read(
    rows: Sequence[Mapping[str, Any]],
    check: str = PRIMARY,
    *,
    question: str | None = None,
    lead: bool = False,
    arms: Sequence[str] = (),
    apps: Sequence[str] = (),
) -> Result:  # Any: stored rows
    """The package's reading of one tier's stored run on ``check``, apps as cases, held in memory (:func:`comparison`)."""
    return await comparison(rows, check, question=question, lead=lead, arms=arms, apps=apps).run(store=":memory:")


@dataclass(frozen=True)
class Stage:
    """One run of a continued comparison: its stored rows (one tier's), its arms (the control first) and its apps in
    the order its cases ran, as its ``run.json`` records them."""

    rows: tuple[Mapping[str, Any], ...]  # Any: stored rows
    arms: tuple[str, ...]
    apps: tuple[str, ...]


async def read_continued(
    stages: Sequence[Stage], check: str = PRIMARY, *, question: str | None = None, lead: bool = False
) -> Result:
    """The package's reading of runs continued as ONE comparison, apps as cases: the first run's apps, then each later
    run's appended after them, read by the package's own continuation (``quick/continuation.py``). Each stage is run
    into one store in turn, so every later run is the earlier comparison with cases appended: the cases already read
    keep the values they ran with, only the appended ones run, and the bet reads one committed order, its capital
    carried forward. The first stage alone reads as :func:`read` reads that run.

    :raises Unreadable: No stage, stages whose arms differ, or an app in two stages (a case is read once); or as
        :func:`comparison` on the rows read so far.
    """
    if not stages:
        raise Unreadable("a continued comparison reads at least one run")
    if any(stage.arms != stages[0].arms for stage in stages):
        raise Unreadable(f"the runs' arms differ ({'; '.join(', '.join(stage.arms) for stage in stages)})")
    apps = [app for stage in stages for app in stage.apps]
    if repeated := sorted({app for app in apps if apps.count(app) > 1}):
        raise Unreadable(f"an app is read once in a comparison, and {', '.join(repeated)} is in two runs")
    store = InMemoryDocumentStore()
    result: Result | None = None
    for upto in range(1, len(stages) + 1):
        rows = [row for stage in stages[:upto] for row in stage.rows]
        held = [app for stage in stages[:upto] for app in stage.apps]
        ev = comparison(rows, check, question=question, lead=lead, arms=stages[0].arms, apps=held)
        result = await ev.run(store=store)
    assert result is not None  # at least one stage, so at least one run
    return result


def _rate(rows: Sequence[Mapping[str, Any]], check: str) -> str:  # Any: stored rows
    graded = [row for row in rows if row.get("scores", {}).get(check) is not None]
    return f"{sum(1 for row in graded if row['scores'][check] == 1.0)} of {len(graded)}"


def _first_screen(result: Result, *, advised: bool) -> str:
    """The package's first screen of ``result`` on one line, as its page's lead reads it: the advice the print opens
    on, then the proof under it. A secondary check is a lead only, never advice on what to run, so it reads the proof
    alone."""
    printed = result.render().splitlines()
    # The print opens on the advice only where its verdict line leads on a reading (never under the case floor, where
    # the rows may still carry one), so the printed line itself says whether it did.
    if not printed[0].startswith(f"{result.name}: Advice: "):
        return printed[0]
    proof = printed[1].removeprefix(f"{result.name}: ")
    return f"{printed[0]} {proof}" if advised else printed[1]


async def read_lines(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary: str = PRIMARY,
    question: str | None = None,
    arms: Sequence[str] = (),
    apps: Sequence[str] = (),
    earlier: Sequence[Stage] = (),
) -> list[str]:  # Any: stored rows
    """What the package reads of a stored run, tier by tier: each check's verdict, the primary first and opening on the
    package's advice, with apps as cases, and each app's tally per arm, the units the verdict rests on. :func:`reading_lines` outside an event loop.

    With ``earlier`` runs, the run continues them as one comparison (:func:`read_continued`): every check is read over
    the earlier runs' apps and then this run's, and the tallies cover them all. A continued run is one tier's.
    """
    lines: list[str] = []
    if earlier and len(tiers_of(rows)) > 1:
        raise Unreadable("a continued comparison is one tier's: read each tier's run on its own")
    for tier, own in tiers_of(rows).items():
        here = design_of(own, arms, apps)
        stages = [*earlier, Stage(tuple(own), here.arms, here.apps)]
        everything = [row for stage in stages for row in stage.rows]
        design = design_of(everything, here.arms, [app for stage in stages for app in stage.apps])
        head = f"tier {tier}: " if tier is not None else ""
        continued = (
            f", continued: {' then '.join(str(len(stage.apps)) for stage in stages)} app(s) as one comparison"
            if earlier
            else ""
        )
        lines.append(
            f"{head}the package's reading, apps as cases: {len(design.apps)} app(s), {design.k} trial(s) per arm on "
            f"each{continued}"
        )
        if len(design.arms) < 2:
            lines.append("  one arm: there is no comparison to read")
            continue
        for check in (primary, *[name for name in CHECKS if name != primary]):
            role = "primary" if check == primary else "secondary, a lead only"
            result = await read_continued(stages, check, question=question, lead=check != primary)
            lines.append(f"  {check} ({role}): {_first_screen(result, advised=check == primary)}")
            for row in result.contrasts(check):
                lines.append(
                    f"    {row['arm']}: {row['outcome']}; {row['cases']} apps; {row['verdict']}"
                    + (f"; change {row['change']}" if row.get("change") else "")
                )
            tallies = [
                f"{app} " + ", ".join(_rate(design.trials[arm][app], check) for arm in design.arms)
                for app in design.apps
            ]
            lines.append(f"    by app ({', '.join(design.arms)}): {'; '.join(tallies)}")
    return lines


def reading_lines(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary: str = PRIMARY,
    question: str | None = None,
    arms: Sequence[str] = (),
    apps: Sequence[str] = (),
    earlier: Sequence[Stage] = (),
) -> list[str]:  # Any: stored rows
    """:func:`read_lines`, from code with no event loop running (the command line)."""
    return asyncio.run(read_lines(rows, primary=primary, question=question, arms=arms, apps=apps, earlier=earlier))


def simulated_rows(
    rng: random.Random,
    *,
    apps: int,
    k: int,
    effect: float,
    spread: float,
    arms: Sequence[str] = ("control", "candidate"),
) -> list[dict[str, Any]]:  # Any: stored-shaped rows
    """Stored-shaped rows of a run nobody ran: each app has a base pass share drawn from 0.1 to 0.6, each arm's share on
    each app is that base (plus ``effect`` for every arm after the first) moved by a normal draw of sd ``spread`` and
    held to 0 to 1, and each trial passes the primary check with its arm's share on its app.

    ``spread`` is the arm-by-app noise a binomial over trials has no term for: two runs of one design on one app that
    disagree. With ``effect`` 0 the arms differ only by noise.
    """
    rows: list[dict[str, Any]] = []
    for index in range(apps):
        app = f"app-{index + 1}"
        base = rng.uniform(0.1, 0.6)
        for place, arm in enumerate(arms):
            share = min(1.0, max(0.0, base + (effect if place else 0.0) + rng.gauss(0.0, spread)))
            for repeat in range(1, k + 1):
                passed = 1.0 if rng.random() < share else 0.0
                scores = {name: passed if name == PRIMARY else 1.0 for name in CHECKS}
                state = "passed" if passed else "failed"
                rows.append({"arm": arm, "app": app, "repeat": repeat, "state": state, "scores": scores})
    return rows


def separation_share(
    *, apps: int, k: int, effect: float, spread: float, reps: int, seed: int
) -> tuple[int, int]:  # how many of ``reps`` simulated runs the package read as separated on the primary check
    """How many of ``reps`` simulated runs (:func:`simulated_rows`, seeded) the package's reading separates on the
    primary check: the power of a plan of ``apps`` apps and ``k`` trials at ``effect``, or its error rate at 0."""
    rng = random.Random(seed)

    async def every() -> int:
        separated = 0
        for _ in range(reps):
            rows = simulated_rows(rng, apps=apps, k=k, effect=effect, spread=spread)
            result = await read(rows)
            separated += any(row["outcome"] in SEPARATED for row in result.contrasts(PRIMARY))
        return separated

    return asyncio.run(every()), reps


def power_lines(*, apps: int, k: int, effect: float, spread: float, reps: int, seed: int) -> list[str]:
    """``--power``: how often the package's reading of a plan this size separates ``effect``, with its exact range."""
    from .campaign import clopper_pearson  # campaign imports this module; the range is its one implementation

    separated, n = separation_share(apps=apps, k=k, effect=effect, spread=spread, reps=reps, seed=seed)
    low, high = clopper_pearson(separated, n)
    return [
        f"{apps} apps, k={k}, effect {effect:+.2f} on the primary check's share, arm-by-app spread {spread:.2f}, "
        f"seed {seed}: separated in {separated} of {n} simulated runs ({separated / n:.2f}, 95% range {low:.2f} to "
        f"{high:.2f})",
        "  simulated under packages/evals/onramp/reading.py simulated_rows, read by the package's own comparison: "
        "a plan's power, not a measurement",
    ]


def continued_power(
    earlier: Stage, *, add: int, scale: float, reps: int, seed: int, question: str | None = None
) -> tuple[int, int]:
    """How many of ``reps`` simulated continuations of the ``earlier`` run separate on the primary: each appends
    ``add`` apps drawn with replacement from the earlier run's apps, each drawn app's control share kept and its
    candidate share moved to the control's plus ``scale`` times the earlier difference (1: the effect the run observed
    holds; a third: a third of it), its k trials per arm drawn at those shares, and the two read as one comparison
    (:func:`read_continued`). The earlier run's own trials are its stored ones, read as they ran."""
    design = design_of(earlier.rows, earlier.arms, earlier.apps)
    control, candidate = design.arms[0], design.arms[-1]

    def share(rows: Sequence[Mapping[str, Any]]) -> float:  # Any: stored rows
        graded = [row["scores"][PRIMARY] for row in rows if row.get("scores", {}).get(PRIMARY) is not None]
        return sum(graded) / len(graded) if graded else 0.0

    shares = [(share(design.trials[control][app]), share(design.trials[candidate][app])) for app in design.apps]
    (tier,) = tiers_of(earlier.rows)  # one tier's run, which the continuation's rows share
    rng = random.Random(seed)

    async def every() -> int:
        separated = 0
        for _ in range(reps):
            rows: list[dict[str, Any]] = []
            apps = [f"continued-{index + 1}" for index in range(add)]
            for app in apps:
                low, high = rng.choice(shares)
                drawn = {control: low, candidate: min(1.0, max(0.0, low + scale * (high - low)))}
                for arm in design.arms:
                    for repeat in range(1, design.k + 1):
                        passed = 1.0 if rng.random() < drawn.get(arm, low) else 0.0
                        scores = {name: passed if name == PRIMARY else 1.0 for name in CHECKS}
                        state = "passed" if passed else "failed"
                        row = {"arm": arm, "app": app, "repeat": repeat, "state": state, "scores": scores}
                        rows.append({**row, "tier": tier} if tier is not None else row)
            later = Stage(tuple(rows), design.arms, tuple(apps))
            result = await read_continued([earlier, later], PRIMARY, question=question)
            separated += any(row["outcome"] in SEPARATED for row in result.contrasts(PRIMARY))
        return separated

    return asyncio.run(every()), reps
