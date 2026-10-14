"""``Eval``: the three-slot contract — what to test, the thing under test, and what counts as right — and its host.

``Eval(cases, runner, grader)`` is one value saying **what** is evaluated; :meth:`Eval.run` says **where** it runs.
The Python file, the TOML file (:mod:`threetears.evals.quick.toml_file`), the command line and a host's launch all
build the same value.

- **cases** are a list of JSON objects, or a file of them (``.jsonl``, ``.json``, ``.csv``, ``.toml``). Three keys
  are reserved and withheld from the runner and every grader: ``id`` names the case, ``stratum`` is the kind of case
  it is (its results are read again per stratum), and ``world`` is its starting state on a world with state.
- **runner** is a function (``async (case) -> output``, or a plain one, run in a worker thread), a world's runner
  (``World(...)(agent)``), a :class:`~threetears.evals.quick.logged.Log`, or a mapping of them. **A mapping is a
  comparison**: its first key is the control and every other arm is tested against it. Arms named as ``k=v`` pairs
  (``"prompt=v2, effort=low"``) are the cells of a factorial design. A mapping of one is one arm, named.
- **grader** is one grader or a list (:mod:`threetears.evals.quick.graders`): a function, a field name, an
  expression over a world, a :class:`~threetears.evals.quick.judged.Judge`, any of the first two wrapped by
  :func:`~threetears.evals.quick.graders.grader`.

Everything else is defaulted: the runs are kept in ``.evals/evals.sqlite`` under the working directory, in a scope
named after the eval, and the result says where.

**A host of your own** (:func:`as_host`) is what an eval runs in when its runs belong to a product's store and
vocabulary: ``Eval(...).run(host=as_host([...], store=...), scope=...)``.
"""

from __future__ import annotations

import csv
import json
import os
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel import GuardrailMargin
from threetears.evals.kernel.host import CompletionClients, EvalHost
from threetears.evals.schema import DEFAULT_LAUNCH_K_RUNS, DocumentStore
from threetears.evals.quick.graders import GraderPlan, expected_label, plan_graders
from threetears.evals.quick.levers import refuse_unusable_lever_names
from threetears.evals.quick.logged import Log
from threetears.evals.quick.one_call import (
    ARM_LEVER,
    CALLABLE_KIND,
    JUDGED_CALLABLE_KIND,
    SHARED_ARM_MODEL,
    CallableArm,
    callable_host,
    refuse_an_unusable_cap,
    run_arms,
)
from threetears.evals.quick.result import CREATED_BY, Result, calibrate, coordinates, declare, judged_rows
from threetears.evals.quick.tools import RECORD_ENV
from threetears.evals.quick.world import World, WorldRunner
from threetears.evals.storage import InMemoryDocumentStore, SqliteDocumentStore

#: Where an eval keeps its runs when it is told nothing, under the working directory.
DEFAULT_STORE = Path(".evals") / "evals.sqlite"

#: What ``store=`` names to keep the runs in memory, for this process alone.
MEMORY_STORE = ":memory:"

#: The case files an eval reads, by suffix.
CASE_FILES: tuple[str, ...] = (".jsonl", ".json", ".csv", ".toml")

#: What a store is named by: a file, :data:`MEMORY_STORE`, or a document store of your own.
StoreSpec = DocumentStore | str | os.PathLike[str] | None


def load_cases(cases: Sequence[Mapping[str, Any]] | str | os.PathLike[str]) -> list[dict[str, Any]]:
    """The cases as a list of JSON objects, read from a file when one is named.

    Args:
        cases: A list of mappings, or a file: ``.jsonl`` (one object a line), ``.json`` (a list of objects, or an
            object whose ``cases`` is one), ``.csv`` (a header row, then one case a row, every value a string) or
            ``.toml`` (an array of tables, ``[[cases]]``).

    Returns:
        The cases, each a ``dict``.

    Raises:
        ValueError: No cases, a file that is not there or of no format named, or a case that is not a JSON object.
    """
    if isinstance(cases, str | os.PathLike):
        path = Path(cases)
        if path.suffix not in CASE_FILES:
            raise ValueError(f"an eval reads its cases from a {', '.join(CASE_FILES)} file, not {path}")
        if not path.is_file():
            raise ValueError(f"the cases file {path} is not there (relative to {Path.cwd()})")
        loaded = _read_cases(path)
    elif isinstance(cases, Mapping) or not isinstance(cases, Sequence):
        raise ValueError(f"an eval's cases are a list of JSON objects or a file of them, not {type(cases).__name__}")
    else:
        loaded = list(cases)
    if not loaded:
        raise ValueError("an eval needs at least one case")
    if bad := [index for index, case in enumerate(loaded) if not isinstance(case, Mapping)]:
        raise ValueError(f"case(s) {', '.join(map(str, bad))} are not JSON objects; each case maps a field to a value")
    return [dict(case) for case in loaded]


def _read_cases(path: Path) -> list[Any]:
    """Every case a file holds, by its format.

    Raises:
        ValueError: A file its format cannot read, named with the line or the key at fault.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        cases: list[Any] = []
        for number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    cases.append(json.loads(line))
                except ValueError as unreadable:
                    raise ValueError(f"{path} line {number} is not JSON: {unreadable}") from unreadable
        return cases
    if path.suffix == ".json":
        try:
            held = json.loads(text)
        except ValueError as unreadable:
            raise ValueError(f"{path} is not JSON: {unreadable}") from unreadable
        held = held.get("cases") if isinstance(held, dict) else held
        if not isinstance(held, list):
            raise ValueError(f"{path} holds no list of cases: a JSON list of objects, or an object whose cases is one")
        return held
    if path.suffix == ".csv":
        return list(csv.DictReader(text.splitlines()))
    try:
        table = tomllib.loads(text)
    except tomllib.TOMLDecodeError as unreadable:
        raise ValueError(f"{path} is not TOML: {unreadable}") from unreadable
    held = table.get("cases")
    if not isinstance(held, list):
        raise ValueError(f"{path} holds no [[cases]] array of tables")
    return held


def open_store(store: StoreSpec) -> tuple[DocumentStore, str]:
    """The document store a ``store=`` names, and where it is, in words.

    Args:
        store: ``None`` for :data:`DEFAULT_STORE`; :data:`MEMORY_STORE` for memory; a file; or a store of your own.

    Returns:
        The store, and what a result's footer calls it.
    """
    if store is None:
        DEFAULT_STORE.parent.mkdir(parents=True, exist_ok=True)
        return SqliteDocumentStore(DEFAULT_STORE), str(DEFAULT_STORE)
    if isinstance(store, str) and store == MEMORY_STORE:
        return InMemoryDocumentStore(), "memory (this process only)"
    if isinstance(store, str | os.PathLike):
        path = Path(store)
        path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteDocumentStore(path), str(path)
    return store, type(store).__name__


def _runner_name(runner: Any) -> str | None:
    """What a runner is called when nothing names it: its function's name, its agent's, or its log's file."""
    if isinstance(runner, WorldRunner):
        runner = runner.agent
    if isinstance(runner, Log):
        return runner.path.stem
    name = getattr(runner, "__name__", None)
    return name if isinstance(name, str) and name.isidentifier() else None


def _refuse_an_unusable_runner(runner: Any, arm: str | None) -> None:
    said = "the runner" if arm is None else f"the arm {arm!r}"
    if not isinstance(runner, WorldRunner | Log) and not callable(runner):
        raise ValueError(
            f"{said} is a function (case) -> output, a world's runner (World(...)(agent)), a Log, or a mapping of "
            f"them; not {runner!r}"
        )


class Eval:
    """What is evaluated: the cases, the runner under test, and what counts as right.

    ``await ev.run()`` runs it and returns its :class:`~threetears.evals.quick.result.Result`;
    ``python -m threetears.evals FILE`` runs every ``Eval`` a file defines.

    Attributes:
        cases: The cases, each a JSON object, reserved keys and all.
        arms: The runners, by arm name; one unnamed arm is keyed by ``None``.
        control: The arm every other is tested against: the runner mapping's first key.
        plan: The graders, sorted by what reads each one.
        world: The world every arm's agent acts on, or ``None``.
        name: What titles the eval's campaign and report, and names its scope by default.
        k: Repeats per case, per arm.
        max_cost_usd: The most the whole eval may spend, in US dollars, shared equally between its arms; ``None``
            runs uncapped, which every summary states.
        measure_latency: Declare latency under test: the cases run one at a time, the arms one after another.
    """

    def __init__(
        self,
        cases: Sequence[Mapping[str, Any]] | str | os.PathLike[str],
        runner: Any,
        grader: Any,
        *,
        k: int = DEFAULT_LAUNCH_K_RUNS,
        name: str | None = None,
        max_cost_usd: float | None = None,
        measure_latency: bool = False,
    ) -> None:
        """Declare the eval; nothing runs until :meth:`run`.

        Args:
            cases: The cases, or a file of them (:func:`load_cases`), read now.
            runner: The thing under test: a function, ``World(...)(agent)``, a ``Log``, or a mapping of them, whose
                first key is the control.
            grader: One grader or a list: functions, field names, expressions over a world, a ``Judge``, each of the
                first two optionally wrapped by ``grader(...)``.
            k: Repeats per case, per arm.
            name: What titles its report and names its scope; ``None`` takes the cases file's stem, else the runner's
                name, else the arms joined by ``vs``.
            max_cost_usd: The most the eval may spend, shared equally between its arms.
            measure_latency: Declare latency under test.

        Raises:
            ValueError: No cases, an unreadable case file, a runner of no known form, a mapping with no arm or a blank
                arm name, arms naming ``k=v`` factors unlike one another, arms on two worlds, no grader or one of no
                known form, or an expression grader with no world with state to read.
        """
        self.cases = load_cases(cases)
        source = Path(cases) if isinstance(cases, str | os.PathLike) else None
        if isinstance(runner, Mapping):
            if not runner:
                raise ValueError("a mapping of runners is a comparison, and this one names no arm")
            if blank := [repr(arm) for arm in runner if not isinstance(arm, str) or not arm.strip()]:
                raise ValueError(f"an arm's name labels its run and its variant, and {', '.join(blank)} is blank")
            arms: dict[str | None, Any] = dict(runner)
        else:
            arms = {None: runner}
        for arm, given in arms.items():
            _refuse_an_unusable_runner(given, arm)
        worlds = {id(given.world): given.world for given in arms.values() if isinstance(given, WorldRunner)}
        if len(worlds) > 1:
            raise ValueError("every arm of an eval acts on one world: build the World once and run each agent on it")
        if worlds and len(worlds) == 1 and any(not isinstance(given, WorldRunner) for given in arms.values()):
            raise ValueError("an eval's arms all act on its world, or none does: run each agent on the World")
        self.arms = arms
        self.control = next(iter(arms))
        self.world: World | None = next(iter(worlds.values()), None)
        self.plan: GraderPlan = plan_graders(grader)
        if self.plan.goal_checks and (self.world is None or not self.world.stateful):
            raise ValueError(
                f"the expression graders {', '.join(map(repr, self.plan.goal_checks))} read a world's state, and the "
                "runner acts on none: run the agent on a world with state (World(State, tools=[...])(agent))"
            )
        self.factors = self._factors()
        named = [arm for arm in arms if arm is not None]
        self.name = (
            name
            or (source.stem if source is not None else None)
            or (_runner_name(arms[None]) if None in arms else None)
            or " vs ".join(named)
        )
        if not self.name or not self.name.strip():
            raise ValueError("an eval is named by name=, its cases file or its runner, and this one has none of them")
        refuse_an_unusable_cap(max_cost_usd)
        self.k = k
        self.max_cost_usd = max_cost_usd
        self.measure_latency = measure_latency

    def __repr__(self) -> str:
        """The eval, by name, cases and arms."""
        arms = [arm for arm in self.arms if arm is not None]
        return f"Eval({self.name!r}, {len(self.cases)} case(s){', arms ' + repr(arms) if arms else ''})"

    @property
    def comparison(self) -> bool:
        """Whether the eval compares arms: two or more, the first the control."""
        return len(self.arms) > 1

    def _factors(self) -> tuple[str, ...] | None:
        """The factors the arms name as ``k=v`` pairs, in order; ``None`` for arms named plainly, or one unnamed arm.

        Raises:
            ValueError: Arms naming ``k=v`` factors unlike one another, or mixing pairs and plain names.
        """
        named = {arm: coordinates(arm) for arm in self.arms if arm is not None}
        if not named:
            return None
        if all(pairs is None for pairs in named.values()):
            return None
        if any(pairs is None for pairs in named.values()):
            raise ValueError(
                'an eval\'s arms are named plainly ("baseline") or as k=v factors ("prompt=v2, effort=low"), not both'
            )
        orders = {tuple(pairs) for pairs in named.values() if pairs is not None}
        if len(orders) > 1:
            raise ValueError(
                f"every arm of a factorial names the same factors in the same order; these name {sorted(orders)}"
            )
        (order,) = orders
        refuse_unusable_lever_names([factor for factor in order if factor != "model"])
        return order

    @property
    def levers(self) -> tuple[str, ...]:
        """The levers every run states a level of beside its model: a factorial's factors other than ``model``."""
        return tuple(factor for factor in self.factors or () if factor != "model")

    def _arms(self, host: EvalHost | None) -> tuple[list[CallableArm], tuple[str, ...], dict[str, dict[str, str]]]:
        """Each arm as the launch runs it, the factors the campaign declares, and each arm's level of each."""
        launched: list[CallableArm] = []
        levels: dict[str, dict[str, str]] = {}
        on_arm_lever = (
            self.comparison
            and self.factors is None
            and (host is None or host.profile.host_sweepables.get(ARM_LEVER) is not None)
        )
        factors: tuple[str, ...] = (
            (ARM_LEVER,) if on_arm_lever else ("model", *self.levers) if self.factors else ("model",)
        )
        for arm, given in self.arms.items():
            candidate = given.agent if isinstance(given, WorldRunner) else given
            default = given.path.stem if isinstance(given, Log) else None
            if arm is None:
                launched.append(CallableArm(candidate, default_model=default))
                continue
            if on_arm_lever:
                launched.append(CallableArm(candidate, default_model=SHARED_ARM_MODEL, arm=arm))
                levels[arm] = {ARM_LEVER: arm}
                continue
            pairs = coordinates(arm) if self.factors else None
            if pairs is None:
                launched.append(CallableArm(candidate, model=arm))
                levels[arm] = {"model": arm}
                continue
            model = pairs.get("model")
            launched.append(
                CallableArm(
                    candidate,
                    model=model,
                    default_model=None if model is not None else SHARED_ARM_MODEL,
                    levers={lever: pairs[lever] for lever in self.levers} or None,
                )
            )
            levels[arm] = {"model": model or SHARED_ARM_MODEL, **{lever: pairs[lever] for lever in self.levers}}
        return launched, factors, levels

    def host(
        self, store: DocumentStore | str | os.PathLike[str] | None = None, *, clients: CompletionClients | None = None
    ) -> EvalHost:
        """The host this eval runs in when it is handed none: its graders' measures, its levers and its world.

        Args:
            store: Where the runs are kept (:func:`open_store`).
            clients: The completion clients the engine's own model calls use after the run; ``None`` for none.

        Returns:
            The host.
        """
        return as_host([self], store=store, clients=clients)

    # scope-default: an eval's scope is its own name unless the run names one, and every result states which
    async def run(
        self,
        *,
        store: DocumentStore | str | os.PathLike[str] | None = None,
        host: EvalHost | None = None,
        scope: str | None = None,
    ) -> Result:
        """Run every arm over every case ``k`` times, grade every answer, and read the campaign's report.

        Args:
            store: Where the runs are kept: ``None`` for ``.evals/evals.sqlite`` under the working directory,
                ``":memory:"`` for this process alone, a file, or a document store of your own. Never with ``host``.
            host: A host of your own (:func:`as_host`), whose storage and vocabulary the runs use.
            scope: The scope the runs and the campaign are stored in; ``None`` is the eval's name.

        Returns:
            The result: every arm's summary, the campaign and its report.

        Raises:
            ValueError: ``store`` beside ``host``; a host that does not declare what the eval's graders read; or
                anything the cases, the runner or the graders refuse, before anything runs.
            ValidationFailedError: The launch refused, or the host refuses the campaign's declaration.
        """
        if host is not None and store is not None:
            raise ValueError(
                f"store= is where the host this eval builds keeps the runs, and host {host.profile.host_id!r} brings "
                "its own storage; pass one or the other (as_host([...], store=...) builds a host over a store)"
            )
        scope_id = self.name if scope is None else scope
        if host is None:
            opened, where = open_store(store)
            host = as_host([self], store=opened)
        else:
            where = f"host {host.profile.host_id!r}"
            _refuse_undeclared_readings(host, self.plan)
        arms, factors, levels = self._arms(host)
        judge = self.plan.judge
        # Said before the run, which may make the recording it names.
        disclosures = tuple(self._disclosures())
        summaries = await run_arms(
            self.cases,
            arms,
            self.plan.scorers,
            scope_id=scope_id,
            host=host,
            # Named by its cases alone, as its id is: evals of other names over these cases share it unchanged.
            template_name=f"eval over {len(self.cases)} case(s)",
            expected=None if self.plan.classifier is None else partial(expected_label, name=self.plan.classifier),
            label_field=self.plan.classifier,
            judge=judge,
            world=self.world,
            goal_checks=self.plan.goal_checks,
            k=self.k,
            max_cost_usd=None if self.max_cost_usd is None else self.max_cost_usd / len(arms),
            # A comparison's arms are variants of one subject, never a subject each: the frontier ranks the variants of
            # a subject, so a model is a lever it ranks across rather than a subject of its own.
            subject=self.name if self.comparison else None,
            margins=self.plan.run_margins,
            measure_latency=self.measure_latency,
        )
        keyed = {
            arm if arm is not None else summary.candidate_model: summary
            for arm, summary in zip(self.arms, summaries, strict=True)
        }
        control = self.control if self.control is not None else summaries[0].candidate_model
        if not levels:
            levels = {control: {"model": summaries[0].candidate_model}}
        names = _case_names(self.cases)
        template = summaries[0].template_id or ""
        case_names = {f"{template}-{index}": case for index, case in enumerate(names)}

        def declared() -> Result:
            result = declare(
                host,
                keyed,
                factors,
                levels=levels,
                kind=CALLABLE_KIND if judge is None else JUDGED_CALLABLE_KIND,
                control=control,
                scope_id=scope_id,
                name=self.name,
                behavior="classify" if self.plan.classifier is not None else "score",
                repetitions=self.k,
                created_by=CREATED_BY,
                guardrail_margins=[
                    GuardrailMargin(dimension=dimension, margin=margin)
                    for dimension, margin in (judge.guard_margins if judge is not None else {}).items()
                ],
                measure_latency=self.measure_latency,
            )
            return replace(
                result,
                store=where,
                fields=tuple(self.plan.fields),
                judge=judge,
                disclosures=disclosures,
                calibrations=tuple(calibrate(judge, host, keyed, scope_id)),
                to_label=tuple(judged_rows(host, keyed, scope_id, case_names)) if judge is not None else (),
            )

        # Filing the campaign and reading its report and labels are store reads and writes, so they run off the loop.
        return await run_blocking(host.blocking_executor, declared)

    def _disclosures(self) -> list[str]:
        """What the result must say about how the runner's answers were come by, beside its readings."""
        said: list[str] = []
        if self.world is not None and self.world.recording is not None:
            recorded = os.environ.get(RECORD_ENV) == "1" or not self.world.recording.is_file()
            said.append(
                f"tool answers {'recorded to' if recorded else 'replayed from'} {self.world.recording}: every arm "
                "faced the same answers"
            )
        said += [f"outputs graded as logged in {given.path}" for given in self.arms.values() if isinstance(given, Log)]
        return said


def _case_names(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    """What each case is called: its ``id``, else its position."""
    return [str(case["id"]) if "id" in case else str(index) for index, case in enumerate(cases)]


def _refuse_undeclared_readings(host: EvalHost, plan: GraderPlan) -> None:
    """Refuse a host of the caller's own whose measures do not carry what the eval's graders declare.

    A margin, a range or a guardrail declared on a grader is read off the host's measure, so a host that declares
    the measure without it would run the eval as though nothing had been declared.

    Raises:
        ValueError: A grader's margin, range or guardrail the host's measure of it does not carry.
    """
    for scorer in plan.scorers:
        name = scorer.__name__
        descriptor = host.profile.measures.get(name)
        if descriptor is None:
            continue
        guard = plan.guardrails.get(name)
        wanted = {
            "materiality_threshold": plan.margins.get(name) if guard is None else guard.margin,
            "value_range": plan.ranges.get(name),
            "guardrail": True if guard is not None else None,
        }
        if mismatched := [
            key for key, value in wanted.items() if value is not None and getattr(descriptor, key) != value
        ]:
            raise ValueError(
                f"host {host.profile.host_id!r} declares the measure {name!r} without the {', '.join(mismatched)} its "
                "grader declares; build the host with as_host over this eval, so the two are one declaration"
            )


def as_host(
    evals: Sequence[Eval],
    *,
    store: DocumentStore | str | os.PathLike[str] | None,
    clients: CompletionClients | None = None,
) -> EvalHost:
    """A host for these evals: the vocabulary their graders, levers and world need, over one store.

    The bridge for a product whose runs belong to its own store, or that serves its evals to a team: every eval run
    with ``run(host=...)`` in it stores there, and the engine's other surfaces (the command line's ``--host``, the
    action catalogue) read them through the same host.

    Args:
        evals: The evals it hosts. Every function grader among them is one measure, declared with its margin, range
            and guardrail; they share their levers and their world.
        store: Where the runs are kept: a file, ``":memory:"``, a document store of your own, or ``None`` for
            ``.evals/evals.sqlite`` under the working directory.
        clients: The completion clients the engine's own model calls use after a run (a re-judge, an analysis), as
            :class:`~threetears.evals.kernel.host.EvalHost` takes them; ``None`` supplies none, and a call needing
            them is refused. A judge brings its own client for the run.

    Returns:
        The host.

    Raises:
        ValueError: No evals, two graders of one name declared differently, evals stating different levers, or evals on
            two worlds.
    """
    if not evals:
        raise ValueError("as_host builds a host for the evals given, and none was")
    scorers: dict[str, Callable[..., Any]] = {}
    margins: dict[str, float] = {}
    ranges: dict[str, tuple[float, float]] = {}
    guardrails: dict[str, Any] = {}
    for ev in evals:
        for scorer in ev.plan.scorers:
            known = scorers.get(scorer.__name__)
            if known is not None and known is not scorer and _declared(ev.plan, scorer) != _declared_by(evals, known):
                raise ValueError(
                    f"two graders named {scorer.__name__!r} declare different readings; one host reads one measure of "
                    "a name"
                )
            scorers.setdefault(scorer.__name__, scorer)
        margins.update(ev.plan.margins)
        ranges.update(ev.plan.ranges)
        guardrails.update(ev.plan.guardrails)
    levers = {ev.levers for ev in evals}
    if len(levers) > 1:
        raise ValueError(
            f"every run in one host states a level of every lever it declares, and these evals state {sorted(levers)}; "
            "host them apart"
        )
    worlds = {id(ev.world): ev.world for ev in evals if ev.world is not None}
    if len(worlds) > 1:
        raise ValueError("one host declares one world; host the evals on different worlds apart")
    opened, _where = open_store(store) if not _is_store(store) else (store, "")
    return callable_host(
        list(scorers.values()),
        levers=next(iter(levers)),
        world=next(iter(worlds.values()), None),
        arms=any(ev.comparison and ev.factors is None for ev in evals),
        margins=margins,
        ranges=ranges,
        store=opened,
        clients=clients,
        guardrails=guardrails,
    )


def _is_store(store: StoreSpec) -> bool:
    """Whether ``store`` is a document store already, rather than what names one."""
    return store is not None and not isinstance(store, str | os.PathLike)


def _declared(plan: GraderPlan, scorer: Callable[..., Any]) -> tuple[Any, ...]:
    """What a plan declares on one grader's reading: its margin, range and guardrail."""
    name = scorer.__name__
    return plan.margins.get(name), plan.ranges.get(name), plan.guardrails.get(name)


def _declared_by(evals: Sequence[Eval], scorer: Callable[..., Any]) -> tuple[Any, ...]:
    """What the eval holding ``scorer`` declares on its reading."""
    holder = next(ev for ev in evals if any(known is scorer for known in ev.plan.scorers))
    return _declared(holder.plan, scorer)


__all__ = [
    "CASE_FILES",
    "DEFAULT_STORE",
    "MEMORY_STORE",
    "Eval",
    "StoreSpec",
    "as_host",
    "load_cases",
    "open_store",
]
