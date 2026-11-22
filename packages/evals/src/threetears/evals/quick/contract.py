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
  expression over a world, a :class:`~threetears.evals.quick.judged.Judge`, a
  :class:`~threetears.evals.quick.claims.Claims`, any of the first two wrapped by
  :func:`~threetears.evals.quick.graders.grader`.

Everything else is defaulted: the runs are kept in ``.evals/evals.sqlite`` under the working directory, in a scope
named after the eval, and the result says where.

**A host of your own** (:func:`as_host`) is what an eval runs in when its runs belong to a product's store and
vocabulary: ``Eval(...).run(host=as_host([...], store=...), scope=...)``.
"""

from __future__ import annotations

import annotationlib
import ast
import csv
import inspect
import json
import math
import os
import textwrap
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeIs

from threetears.evals.analysis.summary import CaseResult, EvalSummary
from threetears.evals.kernel.eval_file import beside_eval_file
from threetears.evals.kernel.eval_scope import eval_running
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel import CONFUSION_CELL_MEASURE, GuardrailMargin
from threetears.evals.kernel.metrics import confusion_of
from threetears.evals.kernel.host import CompletionClients, EvalHost
from threetears.evals.schema import DEFAULT_LAUNCH_K_RUNS, DocumentStore
from threetears.evals.schema.logs import contain, get_logger
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.quick.graders import UNUSABLE_ANSWER, GraderPlan, expected_label, plan_graders
from threetears.evals.quick.levers import refuse_unusable_lever_names
from threetears.evals.quick.logged import Log
from threetears.evals.quick.one_call import (
    ARM_LEVER,
    CALLABLE_KIND,
    JUDGED_CALLABLE_KIND,
    SHARED_ARM_MODEL,
    CallableArm,
    Launched,
    callable_host,
    grade_judge_examples,
    launch_arms,
    launched_as_run,
    refuse_an_unusable_cap,
)
from threetears.evals.quick.claims import unchecked_truth
from threetears.evals.quick.continuation import (
    MEMORY_WHERE,
    design_facts,
    earlier_comparison,
    new_comparison_sentence,
    same_margin,
)
from threetears.evals.quick.result import (
    CREATED_BY,
    Result,
    blind_reading,
    model_slug,
    coordinates,
    declare,
    read_blind,
    refuse_an_unusable_margin,
    refuse_an_unusable_undo,
    with_judge_examples,
    with_said,
)
from threetears.evals.quick.tools import RECORD_ENV
from threetears.evals.quick.world import World, WorldRunner
from threetears.evals.run.curation import set_run_archived
from threetears.evals.storage import InMemoryDocumentStore, SqliteDocumentStore

log = get_logger(__name__)

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

    :param cases: A list of mappings, or a file: ``.jsonl`` (one object a line), ``.json`` (a list of objects, or an
        object whose ``cases`` is one), ``.csv`` (a header row, then one case a row, every value a string) or
        ``.toml`` (an array of tables, ``[[cases]]``).
    :return: The cases, each a ``dict``.
    :raises ValueError: No cases, a file that is not there or of no format named, or a case that is not a JSON object.


    A relative file is read beside the eval file being loaded (``python -m threetears.evals FILE``), else in the
    working directory.
    """
    if isinstance(cases, str | os.PathLike):
        path = beside_eval_file(cases)
        if path.suffix not in CASE_FILES:
            raise ValueError(f"an eval reads its cases from a {', '.join(CASE_FILES)} file, not {path}")
        if not path.is_file():
            raise ValueError(f"the cases file {path} is not there (in {Path.cwd()})")
        loaded = _read_cases(path)
    elif isinstance(cases, Mapping) or not isinstance(cases, Sequence):
        raise ValueError(f"an eval's cases are a list of JSON objects or a file of them, not {type(cases).__name__}")
    else:
        loaded = list(cases)
    if not loaded:
        raise ValueError("an eval needs at least one case")
    # Each by its place in the list from 1, as every line names a case with no id.
    if bad := [index + 1 for index, case in enumerate(loaded) if not isinstance(case, Mapping)]:
        raise ValueError(f"case(s) {', '.join(map(str, bad))} are not JSON objects; each case maps a field to a value")
    return [dict(case) for case in loaded]


def _read_cases(path: Path) -> list[Any]:
    """Every case a file holds, by its format.

    :raises ValueError: A file its format cannot read, named with the line or the key at fault.
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

    :param store: ``None`` for :data:`DEFAULT_STORE`; :data:`MEMORY_STORE` for memory; a file; or a store of your own.
    :return: The store, and what a result's footer calls it.
    """
    if store is None:
        DEFAULT_STORE.parent.mkdir(parents=True, exist_ok=True)
        return SqliteDocumentStore(DEFAULT_STORE), str(DEFAULT_STORE)
    if isinstance(store, str) and store == MEMORY_STORE:
        return InMemoryDocumentStore(), MEMORY_WHERE
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


def _bare(signature: inspect.Signature) -> inspect.Signature:
    """A signature as its parameters' names and defaults, the annotations left out: what the author wrote to call it."""
    return signature.replace(
        parameters=[
            parameter.replace(annotation=inspect.Parameter.empty) for parameter in signature.parameters.values()
        ],
        return_annotation=inspect.Signature.empty,
    )


class _CannotTell(Enum):
    """What reading an author's function finds when it cannot be read. Cannot tell means do not refuse: it is run."""

    NO_SIGNATURE = "no signature to read"
    NO_SOURCE = "no source to read"


def _refuse_an_unusable_runner(runner: Any, arm: str | None) -> None:
    said = "the runner" if arm is None else f"the arm {arm!r}"
    if not isinstance(runner, WorldRunner | Log) and not callable(runner):
        raise ValueError(
            f"{said} is a function (case) -> output, a world's runner (World(...)(agent)), a Log, or a mapping of "
            f"them; not {runner!r}"
        )
    if isinstance(runner, Log):
        return
    # A runner that cannot take what it is called with raises on every case, which would read as the product failing.
    called, handed, written = (
        (runner.agent, "case, tools", "async def") if isinstance(runner, WorldRunner) else (runner, "case", "def")
    )
    signature = _signature_as_called(called)
    if isinstance(signature, _CannotTell):
        return
    try:
        signature.bind(*handed.split(", "))
    except TypeError:
        name, fix = _declare_as(called, handed, written)
        raise ValueError(
            f"{said} {name}{_bare(signature)} cannot be called as {name}({handed}), which is how each case is handed to it; "
            f"{fix}"
        ) from None


def _signature_as_called(called: Any) -> inspect.Signature | _CannotTell:
    """The signature ``called`` is called through, or :attr:`_CannotTell.NO_SIGNATURE` when it has none to read.

    The outermost one: a decorator's wrapper is what is called, even one that copies the function's name with
    ``functools.wraps`` and takes other arguments than it. Annotations are read as forward references, never
    evaluated, so a name imported only for type checking is no error.
    """
    try:
        return inspect.signature(called, follow_wrapped=False, annotation_format=annotationlib.Format.FORWARDREF)
    except TypeError, ValueError:
        # inspect.signature's own refusal: a builtin or a C wrapper (functools.cache) shows no signature to read.
        return _CannotTell.NO_SIGNATURE


def _declare_as(called: Any, handed: str, written: str) -> tuple[str, str]:
    """What a runner is called in a refusal, and the declaration that would take ``handed``."""
    kind: type = type(called)
    if isinstance(called, partial):
        name = f"partial({getattr(called.func, '__name__', type(called.func).__name__)}, ...)"
        return name, f"declare its function so that what the partial leaves free is exactly ({handed})"
    if not inspect.isroutine(called) and not isinstance(called, type):
        return (
            kind.__name__,
            f"declare its __call__ to take exactly that: {written} __call__(self, {handed}): ...",
        )
    named = getattr(called, "__name__", None)
    name = named if isinstance(named, str) else kind.__name__
    return name, f"declare it to take exactly that: {written} {name}({handed}): ..."


def _a_field_some_case_expects(plan: GraderPlan, cases: Sequence[Mapping[str, Any]]) -> str | None:
    """A field the graders read off an answer that some case expects a value in, or ``None`` when there is none."""
    return next((name for name in plan.fields if any(case.get(name) is not None for case in cases)), None)


def _refuse_a_runner_that_returns_nothing(runner: Any, arm: str | None, field_name: str) -> None:
    """Refuse a runner whose own code returns no value anywhere: every answer would be ``None``, so a field graded off
    it reads none, and the run would score 0% for the author's wiring.

    Read from its source, and only where that settles it: a function, a method, a partial of one or an object's
    ``__call__``, not wrapped by a decorator (whose wrapper may return something the function does not), whose source
    can be read. A runner that has a ``return`` and returns ``None`` on every case is the product answering nothing,
    and is scored.

    :raises ValueError: Its code holds no ``return`` of a value, no ``yield`` and no ``raise``, naming it and what to
        return.
    """
    if isinstance(runner, Log | WorldRunner):
        return
    target = runner.func if isinstance(runner, partial) else runner
    if not inspect.isroutine(target) and not isinstance(target, type):
        target = type(target).__call__
    target = getattr(target, "__func__", target)
    if not inspect.isfunction(target) or target.__name__ == "<lambda>" or hasattr(target, "__wrapped__"):
        return  # cannot be read, or its value is its expression's: it cannot be told, so it is run
    tree = _source_of(target)
    if isinstance(tree, _CannotTell):
        return
    defined = tree.body[0] if tree.body else None
    if not isinstance(defined, ast.FunctionDef | ast.AsyncFunctionDef) or defined.name != target.__name__:
        return
    if _returns_a_value(defined.body):
        return
    said = "the runner" if arm is None else f"the arm {arm!r}"
    raise ValueError(
        f"{said} {target.__name__} returns nothing (it has no return of a value), so every answer would be None and "
        f"its grader could read {field_name!r} off none of them: return the answer from it"
    )


def _source_of(function: Callable[..., Any]) -> ast.Module | _CannotTell:
    """The parsed source of ``function``, or :attr:`_CannotTell.NO_SOURCE` when it has none to read."""
    try:
        return ast.parse(textwrap.dedent(inspect.getsource(function)))
    except OSError, SyntaxError:
        # Defined where no file holds it (exec, a REPL): OSError; the file changed since it was loaded: SyntaxError.
        return _CannotTell.NO_SOURCE


def _returns_a_value(body: Sequence[ast.stmt]) -> bool:
    """Whether a function body returns or yields a value, or raises, anywhere outside the functions and classes it
    defines: a runner that raises on purpose (one standing in for a refusing model) is the product failing, scored."""
    pending: list[ast.AST] = list(body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Yield | ast.YieldFrom | ast.Raise):
            return True
        if isinstance(node, ast.Return) and not (
            node.value is None or (isinstance(node.value, ast.Constant) and node.value.value is None)
        ):
            return True
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        pending.extend(ast.iter_child_nodes(node))
    return False


class Eval:
    """What is evaluated: the cases, the runner under test, and what counts as right.

    ``await ev.run()`` runs it and returns its :class:`~threetears.evals.quick.result.Result`;
    ``python -m threetears.evals FILE`` runs every ``Eval`` a file defines.

    :ivar cases: The cases, each a JSON object, reserved keys and all.
    :ivar arms: The runners, by arm name; one unnamed arm is keyed by ``None``.
    :ivar control: The arm every other is tested against: the runner mapping's first key.
    :ivar plan: The graders, sorted by what reads each one.
    :ivar world: The world every arm's agent acts on, or ``None``.
    :ivar name: What titles the eval's campaign and report, and names its scope by default.
    :ivar k: Repeats per case, per arm.
    :ivar max_cost_usd: The most the whole eval may spend, in US dollars: each arm's run is capped at an equal share of
        it, and a judge's examples, graded once the arms have run, draw on what the arms left of the whole. It is read
        as a run's cap is, against the spend already landed before each cell and each example's call, so what is in
        flight when it is reached finishes and the eval can end over it by that, which the result states; spend nobody
        priced stops what has not started, since a cap cannot bound it. ``None`` runs uncapped, which every summary
        states.
    :ivar measure_latency: Declare latency under test: the cases run one at a time, the arms one after another.
    :ivar question: The one sentence the eval answers, filed as its campaign's question; ``None`` asks none.
    :ivar margin: The question's margin in the measure's units, a non-inferiority question's; ``None`` declares none.
    :ivar undo: How hard switching to a candidate is to undo, which sets the evidence the advice switches at.
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
        question: str | None = None,
        margin: float | None = None,
        undo: Literal["easy", "hard"] = "easy",
    ) -> None:
        """Declare the eval; nothing runs until :meth:`run`.

        :param cases: The cases, or a file of them (:func:`load_cases`), read now.
        :param runner: The thing under test: a function, ``World(...)(agent)``, a ``Log``, or a mapping of them, whose
            first key is the control.
        :param grader: One grader or a list: functions, field names, expressions over a world, a ``Judge``,
            ``Claims``, each of the first two optionally wrapped by ``grader(...)``. A function annotated ``-> bool``, or not annotated, grades
            pass or fail; one annotated ``-> float`` or ``-> int``, or given a range, grades a number
            (:func:`~threetears.evals.quick.graders.declares_a_check`).
        :param k: Repeats per case, per arm.
        :param name: What titles its report and names its scope, where its runs are kept; ``None`` takes the cases
            file's stem, else the runner's name (a single arm's name, for a mapping of one). Several arms with neither
            are refused: a name built from the arms would move the runs every time an arm is added, renamed or dropped.
        :param max_cost_usd: The most the eval may spend: each arm's run capped at an equal share, and a judge's examples
            held to what the arms left (:attr:`max_cost_usd`).
        :param measure_latency: Declare latency under test.
        :param question: The one sentence a comparison answers (``"Does v2 read the currency right more often than
            v1?"``), as a writer eval's ``question`` is: filed as the campaign's question, so its page quotes it and
            reads its readings as asked rather than as leads. The question's measure is the first grader: the printed
            first line and the page's lead read it. Alone it asks whether a candidate does better (a superiority
            question): the pass rate reads shown better, shown worse or not separated, against zero, and no margin is
            filed. ``None`` asks none, and every reading is a lead.
        :param margin: How far a candidate's pass rate may fall below the control's before it is shown worse, in the
            measure's units as ``grader(margin=)``'s are (``0.05`` is 5 points): the author's own, which makes the
            question a non-inferiority one, its pass rate reading no worse than the margin, shown worse or neither.
            Needs ``question``; ``None`` declares none, and the package never assumes one.
        :param undo: How hard switching to a candidate would be to undo, declared with the question before anything
            runs. The first line advises what to do on the first grader's reading, beside what is proven (the glossary's "advice"): ``"easy"`` (the default: a prompt, a model, a config a revert undoes) switches at
            moderate evidence, ``"hard"`` (a migration, a published claim) at strong.
        :raises ValueError: A blank question, a margin that is no share strictly between 0 and 1 or one with no
            question, no cases, an unreadable case file, a runner of no known form, a mapping with no arm or a
            blank arm name, arms naming ``k=v`` factors unlike one another, arms on two worlds, no grader or one of no
            known form, an expression grader with no world with state to read, several arms with no name and no
            cases file to name the eval by, a runner that cannot be called with a case (a world's agent with a case
            and its tools), a runner whose code returns no value while a field grader reads its answer, a function
            grader that cannot take ``(case, out)``, a function grader written ``async def`` or whose return
            annotation is read as both kinds, neither, or names ``None``, or a case a field grader other than the
            classifier cannot read a label from.
        """
        if question is not None and not question.strip():
            raise ValueError("an eval's question is filed on its campaign, and this one is blank")
        refuse_an_unusable_margin(margin, question)
        refuse_an_unusable_undo(undo)
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
        _refuse_a_case_a_field_grader_cannot_read(self.plan, self.cases)
        if self.world is None and (read := _a_field_some_case_expects(self.plan, self.cases)) is not None:
            for arm, given in arms.items():
                _refuse_a_runner_that_returns_nothing(given, arm, read)
        if self.plan.goal_checks and (self.world is None or not self.world.stateful):
            raise ValueError(
                f"the expression graders {', '.join(map(repr, self.plan.goal_checks))} read a world's state, and the "
                "runner acts on none: run the agent on a world with state (World(State, tools=[...])(agent))"
            )
        self.factors = self._factors()
        named = [arm for arm in arms if arm is not None]
        if not name and source is None and len(named) > 1:
            raise ValueError(
                f'name this eval: pass name="..." (one that stays as the arms change), or give its cases as a file, '
                "whose stem names it; its name is where its runs are kept, and a name built from its arms "
                f"({', '.join(map(repr, named))}) would move them every time an arm changes"
            )
        self.name = (
            name
            or (source.stem if source is not None else None)
            or (_runner_name(arms[None]) if None in arms else None)
            or (named[0] if named else "")
        )
        if not self.name or not self.name.strip():
            raise ValueError("an eval is named by name=, its cases file or its runner, and this one has none of them")
        refuse_an_unusable_cap(max_cost_usd)
        self.k = k
        self.max_cost_usd = max_cost_usd
        self.measure_latency = measure_latency
        self.question = question
        self.margin = margin
        self.undo: Literal["easy", "hard"] = undo

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

        :raises ValueError: Arms naming ``k=v`` factors unlike one another, or mixing pairs and plain names.
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

        :param store: Where the runs are kept (:func:`open_store`).
        :param clients: The completion clients the engine's own model calls use after the run; ``None`` for none.
        :return: The host.
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

        :param store: Where the runs are kept: ``None`` for ``.evals/evals.sqlite`` under the working directory,
            ``":memory:"`` for this process alone, a file, or a document store of your own. Never with ``host``.
        :param host: A host of your own (:func:`as_host`), whose storage and vocabulary the runs use.
        :param scope: The scope the runs and the campaign are stored in; ``None`` is the eval's name.
        :return: The result: every arm's summary, the campaign and its report.
        :raises ValueError: ``store`` beside ``host``; a host that does not declare what the eval's graders read; or
            anything the cases, the runner or the graders refuse, before anything runs. Once the arms have run, a field
            grader whose field no answer of any arm carried (its runs are kept archived, out of every listing and
            reading; no campaign is filed).
        :raises ValidationFailedError: The launch refused, or the host refuses the campaign's declaration.
        """
        # Its arms, its judge and every task they start read as this one run, so a replay serving one of its calls
        # can tell this eval's calls from another's in the process (eval_running).
        with eval_running(self.name):
            return await self._run(store=store, host=host, scope=scope)

    async def _run(
        self,
        *,
        store: DocumentStore | str | os.PathLike[str] | None,
        host: EvalHost | None,
        scope: str | None,
    ) -> Result:
        """:meth:`run`'s body, inside its eval's run."""
        if host is not None and store is not None:
            raise ValueError(
                f"store= is where the host this eval builds keeps the runs, and host {host.profile.host_id!r} brings "
                "its own storage; pass one or the other (as_host([...], store=...) builds a host over a store)"
            )
        scope_id = self.name if scope is None else scope
        store_file: Path | None = None
        if host is None:
            opened, where = open_store(store)
            if isinstance(opened, SqliteDocumentStore) and (store is None or isinstance(store, str | os.PathLike)):
                store_file = DEFAULT_STORE if store is None else Path(store)
            host = as_host([self], store=opened)
        else:
            where = f"host {host.profile.host_id!r}"
            _refuse_undeclared_readings(host, self.plan)
        arms, factors, levels = self._arms(host)
        judge = self.plan.judge
        # Said before the run, which may make the recording it names.
        disclosures = tuple(self._disclosures())
        # The design and the control are fixed here, in the call, before any arm runs: the campaign, filed once they
        # have, is dated now.
        declared_at = utc_now_iso()
        # A comparison run again under its name continues the one it filed before, where nothing but its cases' tail
        # changed: the cases already run are read as they ran and only the cases appended run (one bet, read later).
        earlier = None
        said: str | None = None
        control_index = list(self.arms).index(self.control) if self.control in self.arms else 0
        if self.comparison:
            earlier = await run_blocking(
                host.blocking_executor, earlier_comparison, host, scope_id, self.name, launched_as_run
            )
            if earlier is not None and (
                earlier.design
                != design_facts(
                    self.plan.readings,
                    judge.guard_margins if judge is not None else {},
                    self.measure_latency,
                    self.question,
                )
                or not same_margin(earlier.margin, self.margin)
            ):
                earlier, said = (
                    None,
                    new_comparison_sentence("the graders, the guardrails, the question or its margin changed"),
                )
            elif earlier is not None and earlier.undo != self.undo:
                earlier, said = None, new_comparison_sentence("how hard the switch is to undo changed")

        def campaign(
            launched: Launched,
        ) -> tuple[dict[str, EvalSummary], str, dict[str, tuple[str, ...]], Callable[[], Result]]:
            """What ``launched`` files: each arm's summary by name, the control, each arm's runs, and the filing."""
            summaries = launched.summaries
            keyed = {
                arm if arm is not None else summary.candidate_model: summary
                for arm, summary in zip(self.arms, summaries, strict=True)
            }
            control = self.control if self.control is not None else summaries[0].candidate_model
            filed_levels = levels if levels else {control: {"model": summaries[0].candidate_model}}
            runs_of = {
                arm if arm is not None else summary.candidate_model: run_ids
                for arm, summary, run_ids in zip(self.arms, summaries, launched.run_ids, strict=True)
            }

            def filed() -> Result:
                return declare(
                    host,
                    keyed,
                    factors,
                    levels=filed_levels,
                    kind=CALLABLE_KIND if judge is None and self.plan.claims is None else JUDGED_CALLABLE_KIND,
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
                    # A continued comparison's design and control were fixed when it first ran, and nothing a
                    # continuation may change is part of them.
                    declared_at=(
                        earlier.declared_at
                        if launched.continued and earlier is not None and earlier.declared_at is not None
                        else declared_at
                    ),
                    question=self.question,
                    margin=self.margin,
                    undo=self.undo,
                    measure_order=self.plan.readings,
                    runs=runs_of,
                )

            return keyed, control, runs_of, filed

        async def file_cancelled(launched: Launched) -> None:
            # Filed as a stopped comparison is, and nothing more is read or asked of a model: the cancel goes on.
            try:
                await run_blocking(host.blocking_executor, campaign(launched)[3])
            except Exception as exc:
                contain(
                    exc,
                    log=log,
                    what="a cancelled comparison's runs could not be filed; a run again starts a new comparison",
                    level="error",
                )

        launched = await launch_arms(
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
            claims=self.plan.claims,
            world=self.world,
            goal_checks=self.plan.goal_checks,
            k=self.k,
            max_cost_usd=None if self.max_cost_usd is None else self.max_cost_usd / len(arms),
            # A comparison's arms are variants of one subject, never a subject each: the frontier ranks the variants of
            # a subject, so a model is a lever it ranks across rather than a subject of its own.
            subject=self.name if self.comparison else None,
            margins=self.plan.run_margins,
            measure_latency=self.measure_latency,
            earlier=earlier,
            control_index=control_index,
            said=said,
            # A comparison cancelled while it runs is filed as a stopped one is, before the cancel goes on: every result
            # it recorded is kept, and a run again runs only the cells it never recorded.
            on_cancel=file_cancelled if self.comparison else None,
        )
        summaries = launched.summaries
        if (
            refused := _a_field_no_answer_carries(self.plan, self.cases, summaries, world=self.world is not None)
        ) is not None:
            # The runs measured the author's wiring, not the product: archived, as an operator retires a contaminated
            # run, so no listing, history or frontier reads them, and no campaign is filed. A continued comparison's
            # earlier runs stay as they were: only the runs this call launched are archived.
            await run_blocking(host.blocking_executor, _archive_runs, host, launched.launched, scope_id)
            raise ValueError(refused)
        examples = None
        if judge is not None and judge.examples and summaries[0].template_id is not None:
            # Graded once for the eval, after its arms, through the judge every arm's answers were graded by, each
            # example judged as a cell's answer is (the classifier's label hidden alike), and held to what the arms
            # left of the eval's cap. A grading that fails is said on the reading, so the arms' results still land.
            examples = await grade_judge_examples(
                host,
                judge,
                summaries[0].template_id,
                scope_id,
                k=self.k,
                expected=None if self.plan.classifier is None else partial(expected_label, name=self.plan.classifier),
                max_cost_usd=self.max_cost_usd,
                landed_usd=_landed_usd(summaries),
            )
        keyed, control, runs_of, filed = campaign(launched)

        def declared() -> Result:
            said = (
                ((launched.said,) if launched.said is not None else ())
                + disclosures
                + tuple(_silent_served_model(host, keyed, scope_id, self.arms))
            )
            result = filed()
            closed, _loose = read_blind(judge, host, keyed, control, scope_id, self.name, runs=runs_of)
            drawn.extend([closed.read.id] if closed is not None else [])
            said_result = with_judge_examples(
                replace(
                    result,
                    store=where,
                    fields=tuple(self.plan.fields),
                    judge=judge,
                    disclosures=said,
                    **blind_reading(judge, closed, keyed),
                ),
                examples,
            )
            claims = self.plan.claims
            if claims is not None and not claims.checks_truth:
                # Nothing decided a claim's truth, so its false claims read 0 whatever the answers say: said beside them.
                said_result = with_said(said_result, unchecked_truth(claims.name), anchor="claims-unchecked")
            return said_result

        # Filing the campaign and reading its report and labels are store reads and writes, so they run off the loop.
        drawn: list[str] = []
        result = await run_blocking(host.blocking_executor, declared)
        if judge is not None and judge.reviewer is not None and drawn:
            # A model answers the sheet a person would, once the read is drawn: held to what the eval's cap has left,
            # ledgered, and kept beside a file store so the same draw is not paid for twice.
            kept = None
            if store_file is not None:
                kept = store_file.parent / "blind_reads" / f"{drawn[0]}.{model_slug(judge.reviewer)}.csv"
            result = await result.reviewed_by(judge.reviewer, max_cost_usd=self.max_cost_usd, kept=kept)
        return result

    def _disclosures(self) -> list[str]:
        """What the result must say about how the runner's answers were come by, beside its readings."""
        said: list[str] = []
        if self.world is not None and self.world.recording is not None:
            recorded = os.environ.get(RECORD_ENV) == "1" or not self.world.recording.is_file()
            said.append(
                f"tool answers {'recorded to' if recorded else 'replayed from'} {self.world.recording}: every arm "
                "faced the same answers"
            )
        for given in self.arms.values():
            if not isinstance(given, Log):
                continue
            said.append(f"outputs graded as logged in {given.path}")
            if any(answer.served_model is None for answers in given.outputs().values() for answer in answers):
                # The one coordinate a run launched now can lack, said as it launches and not when a campaign over it
                # will not compare: a log holds what its app pushed, and nothing else knows what answered.
                unnamed = (
                    f"outputs logged in {given.path} name no served model (identity coordinate served_model), so "
                    "nothing says which model answered and this run pools with no other run; log served_model= with "
                    "each output, or state it afterwards (runs_attest, coordinate served_model)"
                )
                log.warning("%s", unnamed)
                said.append(unnamed)
        return said


def _silent_served_model(
    host: EvalHost, keyed: Mapping[str, EvalSummary], scope_id: str, given: Mapping[str | None, Any]
) -> list[str]:
    """What the result says, once its arms ran, of each arm whose candidate rows name no served model.

    The other side of the launch notice a :class:`Log` gets: a runner that returns an :class:`Answer` with a model and
    no ``served_model`` stores a candidate row that names none, and nothing else knows what answered. Said on the
    result and as a WARNING, naming the coordinate and the remedy, never left to a campaign that will not compare.
    An arm that is a log was told at launch, and is not told twice.

    :param host: The host whose store holds the runs.
    :param keyed: Each arm's summary, by the arm's name.
    :param scope_id: The scope the runs are in.
    :param given: The eval's arms as given, by name.
    :return: One line per such arm.
    """
    logged = {name for name, runner in given.items() if isinstance(runner, Log)}
    lines: list[str] = []
    for arm, summary in keyed.items():
        if arm in logged or (None in logged and len(keyed) == 1):
            continue
        results = host.storage.query_eval_results_by_run(summary.run_id, scope_id)
        silent = sum(
            1 for result in results if any(row.role == "candidate" and not row.served_model for row in result.usage)
        )
        if not silent:
            continue
        line = (
            f"{silent} of {len(results)} result(s) of arm {arm} (run {summary.run_id}) hold a candidate call that names "
            "no served model (identity coordinate served_model), so nothing says which model answered and the run "
            "pools with no other run; return Answer(served_model=...) naming what answered, or state it afterwards "
            "(runs_attest, coordinate served_model)"
        )
        log.warning("%s", line)
        lines.append(line)
    return lines


def _landed_usd(summaries: Sequence[EvalSummary]) -> float | None:
    """What the arms' results landed, as a run's cost cap reads it: each result's cost summed, ``None`` where any
    result's is unknown (it spent what nobody priced), which a cap cannot bound."""
    landed: list[float] = []
    for summary in summaries:
        if summary.case_results is None:
            if summary.total_cost_usd is None:
                return None
            landed.append(summary.total_cost_usd)
            continue
        for result in summary.case_results:
            if result.cost_usd is None:
                return None
            landed.append(result.cost_usd)
    return math.fsum(landed)


def _refuse_a_case_a_field_grader_cannot_read(plan: GraderPlan, cases: Sequence[Mapping[str, Any]]) -> None:
    """Refuse, before anything runs, a case a field grader would raise on: one lacking the field, or holding no label.

    The classifier's cases are refused as the run starts (each by its expected label); a field graded beside it would
    otherwise raise on each such case only after the runner had answered it, excluding the cell as the rig's fault.

    :raises ValueError: The first case a field other than the classifier cannot be graded on, naming it.
    """
    for name in plan.fields:
        if name == plan.classifier:
            continue
        for index, case in enumerate(cases):
            if name in case and case[name] is None:
                continue  # expects the field absent
            try:
                expected_label(case, name)
            except ValueError as unreadable:
                fix = (
                    ""
                    if name in case
                    else f"; it carries {', '.join(sorted(case))}: grade one of those, or give every case its {name!r} "
                    "(null where the answer should carry none)"
                )
                # A case is called by its id, else by its place in the list from 1, as every line names it.
                raise ValueError(f"case {case.get('id', index + 1)}: {unreadable}{fix}") from None


def _archive_runs(host: EvalHost, run_ids: Sequence[str], scope_id: str) -> None:
    """Archive the runs of a refused eval: kept, and restorable, but out of every listing and reading."""
    for run_id in run_ids:
        set_run_archived(host.storage, run_id, scope_id, archived=True, profile=host.profile)


def _a_field_no_answer_carries(
    plan: GraderPlan, cases: Sequence[Mapping[str, Any]], summaries: Sequence[EvalSummary], *, world: bool
) -> str | None:
    """The refusal of a field grader no answer of any arm carried its field for: the author's wiring, not the
    product's failing; ``None`` when every graded field was carried somewhere, or cannot be told to be wiring.

    Every case would score 0 on it and the run exit 0: a measured 0% that reads as the app failing. Only answers the
    runner returned count, and only where some case expects a value in the field (a case's ``null`` expects it
    absent). An arm whose answers lack a field another arm's carry is the product's, scored as it is. So are answers
    that are all ``None`` from a runner that returns a value somewhere (one with no return is refused before it runs):
    a runner returning ``None`` where the model's output never parsed is the product answering nothing.
    """
    answered = [
        result
        for summary in summaries
        for result in summary.case_results or ()
        if result.outcome in ("scored", "not_judged")
    ]
    answers = [
        result.answer.get("answer") if world and isinstance(result.answer, dict) else result.answer
        for result in answered
    ]
    for name in plan.fields:
        if not answers or all(case.get(name) is None for case in cases):
            continue
        if any(
            _carries(answer, result, name, classifier=name == plan.classifier)
            for answer, result in zip(answers, answered, strict=True)
        ):
            continue
        if carried := sorted({key for answer in answers if isinstance(answer, dict) for key in answer}):
            shown = ", ".join(carried[:8]) + (f" and {len(carried) - 8} more" if len(carried) > 8 else "")
            fix = f"the answers carry {shown}: grade the field they carry, or have the runner return {name!r}"
        elif all(answer is None for answer in answers):
            continue  # the product answered nothing: scored, each case's answer stored as None
        else:
            first = next(answer for answer in answers if answer is not None)
            shown = (
                f"objects without it, such as {first[:80]}"
                if isinstance(first, str)
                else f"{type(first).__name__}s, which hold no fields"
            )
            fix = f"the answers are {shown}: return a dict, a pydantic model or a dataclass with {name!r}"
        return f"no answer has a field {name!r}, so its grader could score none of them; {fix}"
    return None


def _carries(answer: Any, result: CaseResult, name: str, *, classifier: bool) -> bool:
    """Whether one stored answer carries field ``name``: an object holding it, or a plain value, its own label.

    The stored answer is not always the answer graded: one JSON could not hold is stored as its ``repr``, a string like
    a plain value, and a named tuple as a list. So for the classifier, the label its grading read decides (a plain
    string that is not blank is always a label); any other field's grading leaves no such record, so an answer stored
    in a form that may have hidden the field is given the benefit.
    """
    if isinstance(answer, dict):
        return name in answer
    if answer is None:
        return False
    if classifier and not (isinstance(answer, str) and not answer.strip()):
        cell = result.scores.get(CONFUSION_CELL_MEASURE)
        read = confusion_of(cell) if isinstance(cell, str) else None
        return read is None or read[1] != UNUSABLE_ANSWER
    return True


def _refuse_undeclared_readings(host: EvalHost, plan: GraderPlan) -> None:
    """Refuse a host of the caller's own whose measures do not carry what the eval's graders declare.

    A margin, a range or a guardrail declared on a grader is read off the host's measure, so a host that declares
    the measure without it would run the eval as though nothing had been declared.

    :raises ValueError: A grader's margin, range or guardrail the host's measure of it does not carry.
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

    :param evals: The evals it hosts. Every function grader among them is one measure, declared with its margin, range
        and guardrail; they share their levers and their world.
    :param store: Where the runs are kept: a file, ``":memory:"``, a document store of your own, or ``None`` for
        ``.evals/evals.sqlite`` under the working directory.
    :param clients: The completion clients the engine's own model calls use after a run (a re-judge, an analysis), as
        :class:`~threetears.evals.kernel.host.EvalHost` takes them; ``None`` supplies none, and a call needing
        them is refused. A judge brings its own client for the run.
    :return: The host.
    :raises ValueError: No evals, two graders of one name declared differently, evals stating different levers, or evals
        on two worlds.
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


def _is_store(store: StoreSpec) -> TypeIs[DocumentStore]:
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
