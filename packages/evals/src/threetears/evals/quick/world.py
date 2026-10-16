"""A world for an eval's runner: state the agent acts on through tools, graded by the state it leaves.

A runner usually answers, and a grader grades the answer. Some agents do not answer — they *act*: switch a light,
file a ticket, move money. What matters then is the state they leave behind, and the engine already measures that: a
host declares its world (:class:`~threetears.evals.kernel.host.WorldRegistry`), each cell seeds it through a
:class:`~threetears.evals.kernel.WorldSession`, the runner reads it back once the agent is done, and goal-state checks
(``state.<field>``, ``calls("<world>.<tool>")``) grade what it holds. This module declares that world in a few lines
and drives the same machinery:

- :class:`World` takes the state as a pydantic model (each field one dimension, its ``description`` why it matters)
  and the tools that change it, each a plain function of the state and its typed parameters: a parameter's type is
  the JSON Schema the agent is shown and every call is held to. ``World(Room, tools=[switch])`` declares it, and
  ``room(agent)`` is the runner (:class:`WorldRunner`): ``agent(case, tools)`` is handed the tools bound to its cell's
  world.
- **Each case seeds its own world**, from its reserved ``world`` key: the cell binds a fresh state, seeds it through
  the session before the agent's first turn, and hands the agent a :class:`WorldTools`.
- **Each tool call that succeeds is recorded** on the cell's call ledger as ``<world>.<tool>`` with its parameters,
  which is what ``calls(...)`` reads. A call its schema refuses changes nothing and is not recorded; the agent is told
  why, as a ``ValueError`` (:class:`ToolRefused`).
- **What a grader sees** is a :class:`WorldOutcome`: the agent's ``answer``, the ``state`` it left (an instance of the
  state model) and the ``calls`` it made. An expression grader is a goal-state check over the same, graded by the
  engine's own :func:`~threetears.evals.run.evaluate_goal_state`. A tool that raises is the rig's fault and excludes
  the cell; a check that cannot be evaluated does too.
- **A world with no state** (``World(tools=[search])``) is a set of lookups: plain functions of their keyword
  arguments returning JSON. Handed ``recording=``, its answers are recorded once and replayed after
  (:class:`~threetears.evals.quick.tools.ToolRecording`), so every arm faces the same answers.

**Where the starting state lives.** The engine's template carries one seed per scenario; here every case carries its
own, because the starting state is what varies. So the template's seed stays empty, each case's seed rides on its
stored test case beside the case itself, and the template's id digests the seeds with the cases, so a different
starting state is a different case set.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import os
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, NoReturn, get_type_hints

from pydantic import BaseModel, ConfigDict, PydanticSchemaGenerationError, TypeAdapter, ValidationError

from threetears.evals.quick.answer import Answer
from threetears.evals.schema import (
    CallLedger,
    RecordedCall,
    EvalTestCase,
    JudgedArtifact,
    WorldSeed,
    extract_paths,
    referenced_actions,
)
from threetears.evals.kernel import (
    CandidateOutput,
    CandidatePreparationFailed,
    CellCassettes,
    CellSink,
    CellSpanWindow,
    VariantConfig,
    WorldSession,
)
from threetears.evals.kernel.eval_file import beside_eval_file
from threetears.evals.kernel.host import SeedRefused, WorldDimension, WorldRegistry, check_seed, schema_violations
from threetears.evals.schema import SubjectSnapshot
from threetears.evals.run import CellContext, GoalCheckUnevaluable, evaluate_goal_state, grade_goal_checks
from threetears.evals.run.check_controls import idle_end_state
from threetears.evals.schema.logs import get_logger


log = get_logger(__name__)


@dataclass(frozen=True)
class Dimension:
    """One piece of the world's state, as the engine declares it: one field of the state model.

    :ivar name: What a case seeds and a goal check reads it as (``state.<name>``).
    :ivar schema: The JSON Schema every value of it satisfies; a seed outside it is refused.
    :ivar matters: Why the state is worth seeding — the field's ``description``.
    """

    name: str
    schema: Mapping[str, Any]
    matters: str


#: JSON Schema keywords pydantic writes that say nothing about a value's shape, left out of what a call is held to.
_UNSHAPED = frozenset({"title", "default"})


def _shape(schema: Mapping[str, Any]) -> dict[str, Any]:
    """A pydantic-generated schema without the keywords that describe rather than constrain."""
    return {key: value for key, value in schema.items() if key not in _UNSHAPED}


def _snake(name: str) -> str:
    """``BookingCalendar`` as ``booking_calendar``: a state model's name as the world's."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class WorldTool:
    """One action the agent can take on the world: a function of the state and its typed parameters.

    ``act(state, **params)`` changes ``state`` and returns what the agent is told; it may be ``async``. Its
    ``__name__`` names the tool and the first line of its docstring describes it to the agent. Every parameter after
    the state is required and annotated: its type's JSON Schema is the schema the agent is shown and every call is
    held to.
    """

    def __init__(self, act: Callable[..., Any], *, state: type[BaseModel] | None) -> None:
        """Declare ``act`` as a tool, its parameters read from its type hints.

        :param act: The function. With a ``state`` model, its first parameter is the state.
        :ptype act: Callable[..., Any]
        :param state: The world's state model, or ``None`` for a world of lookups.
        :ptype state: type[BaseModel] | None
        :raises ValueError: ``act`` has no name a tool can carry, takes no state where the world has one, or has a
            parameter with no type annotation, a default, or a type no JSON Schema describes.
        """
        name = getattr(act, "__name__", "")
        if not name.isidentifier():
            raise ValueError(f"a tool is named by its function's __name__, and {act!r} has none; write it as a def")
        parameters = list(inspect.signature(act).parameters.values())
        if state is not None:
            if not parameters:
                raise ValueError(f"the tool {name} changes the world's state, so its first parameter is the state")
            parameters = parameters[1:]
        try:
            # The state model by its own name too, so a tool written beside a state model defined in a function, under
            # postponed annotations, still resolves the annotation naming it; Annotated metadata (a Field's
            # description) is kept, so it reaches the schema the agent is shown.
            local = None if state is None else {state.__name__: state}
            hints = get_type_hints(act, localns=local, include_extras=True)
        except NameError as unresolved:
            raise ValueError(f"the tool {name}'s annotations do not resolve: {unresolved}") from unresolved
        params: dict[str, dict[str, Any]] = {}
        for parameter in parameters:
            if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                raise ValueError(f"the tool {name}'s parameter {parameter.name} is not one a call can name")
            hint = hints.get(parameter.name)
            if hint is None:
                raise ValueError(
                    f"the tool {name}'s parameter {parameter.name} has no type: annotate it, since its type is the "
                    "schema the agent is shown and every call is held to"
                )
            try:
                params[parameter.name] = _shape(TypeAdapter(hint).json_schema())
            except PydanticSchemaGenerationError as unschemable:
                raise ValueError(f"the tool {name}'s parameter {parameter.name}: {unschemable}") from unschemable
        doc = inspect.getdoc(act)
        self.act = act
        self.name: str = name
        self.description: str = doc.splitlines()[0] if doc else f"The {name} tool."
        self.params: dict[str, dict[str, Any]] = params

    @property
    def input_schema(self) -> dict[str, Any]:
        """The tool's parameters as one JSON Schema object: every one required, nothing else admitted."""
        return {
            "type": "object",
            "properties": copy.deepcopy(self.params),
            "required": list(self.params),
            "additionalProperties": False,
        }


class ToolRefused(ValueError):
    """A tool call the world did not make: no such tool, or parameters its schema refuses. Nothing changed."""


class WorldOutcome(BaseModel):
    """What a world runner's graders see: the agent's answer, the state it left, and the calls it made.

    :ivar answer: What the agent returned.
    :ivar state: The world's state when the agent was done, as the state model; ``None`` for a world of lookups.
    :ivar calls: Every call that succeeded, in order (``tool`` is the world's name, ``action`` the tool's).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    answer: Any
    state: Any = None
    calls: list[RecordedCall] = []


@dataclass(frozen=True)
class WorldRunner:
    """A runner that hosts an agent on a world: ``World(...)(agent)``.

    :ivar world: The world every cell binds afresh.
    :ivar agent: ``agent(case, tools)``, called with each case and the tools on its cell's world; async.
    """

    world: World
    agent: Callable[[Mapping[str, Any], Any], Awaitable[Any]]


def _dimensions(state: type[BaseModel]) -> list[Dimension]:
    """Each field of the state model as a dimension: its schema from the model's, its reason its description.

    :raises ValueError: A field with no description, which is what the engine asks of every dimension.
    """
    schema = state.model_json_schema()
    defs = schema.get("$defs")
    dimensions: list[Dimension] = []
    for name, declared in state.model_fields.items():
        if not declared.description or not declared.description.strip():
            raise ValueError(
                f"the state field {state.__name__}.{name} has no description: say why it matters "
                f'({name}: ... = Field(description="..."))'
            )
        shape = _shape(schema["properties"][name])
        shape.pop("description", None)
        if defs:
            shape["$defs"] = defs
        dimensions.append(Dimension(name=name, schema=shape, matters=declared.description))
    return dimensions


class World:
    """A world: state each case seeds and tools the agent changes it with, or a set of lookups the agent calls.

    ``World(Room, tools=[switch])`` declares a world whose state is a ``Room``; ``World(tools=[search])`` one of
    lookups, with no state. Calling it on an agent gives the runner (:class:`WorldRunner`). :attr:`registry` is the
    engine's declaration of a world with state, which the run's host carries on its profile.

    :ivar state: The state model, or ``None``.
    :ivar name: The carrier its state reaches the agent through and the prefix its tools are recorded under
        (``calls("room.switch")``): the state model's name in snake case, ``tools`` for a world of lookups.
    :ivar tools: Its tools, by name.
    :ivar recording: Where a world of lookups records its answers and replays them from, or ``None``.
    """

    def __init__(
        self,
        state: type[BaseModel] | None = None,
        /,
        *,
        tools: Sequence[Callable[..., Any]] = (),
        recording: str | os.PathLike[str] | None = None,
    ) -> None:
        """Declare the world.

        :param state: The state, a pydantic model: each field one dimension a case seeds and a goal check reads, its
            ``Field(description=...)`` why it matters. ``None`` for a world of lookups.
        :ptype state: type[BaseModel] | None
        :param tools: What the agent can do: functions of the state and their typed parameters, or, with no state,
            functions of their keyword arguments returning JSON.
        :ptype tools: Sequence[Callable[..., Any]]
        :param recording: For a world of lookups: a file their answers are recorded to on the first run and replayed
            from after (``RECORD=1`` records it again). A world with state takes none: its tools move the state
            they are graded on, so they run in every cell. A relative path is read beside the eval file being
            loaded, else in the working directory.
        :ptype recording: str | os.PathLike[str] | None
        :raises ValueError: A state that is not a pydantic model, a field with no description, no tools for a world of
            lookups, two tools of one name, a recording for a world with state, or a declaration the engine's
            registry refuses.
        """
        if state is not None and not (isinstance(state, type) and issubclass(state, BaseModel)):
            raise ValueError(f"a world's state is a pydantic model class, not {state!r}")
        if state is None and not tools:
            raise ValueError("a world with no state is its tools; pass tools=[...]")
        if state is not None and recording is not None:
            raise ValueError(
                "a world with state takes no recording=: its tools move the state it is graded on, so a replayed "
                "answer would leave that state unmoved; they run in every cell"
            )
        self.state = state
        self.name = "tools" if state is None else _snake(state.__name__)
        self.recording = None if recording is None else beside_eval_file(recording)
        declared = [WorldTool(tool, state=state) for tool in tools]
        names = [tool.name for tool in declared]
        if repeated := sorted({tool for tool in names if names.count(tool) > 1}):
            raise ValueError(f"tools named {', '.join(repeated)} more than once")
        self.tools: dict[str, WorldTool] = {tool.name: tool for tool in declared}
        self.functions: dict[str, Callable[..., Any]] = {tool.name: tool.act for tool in declared}
        self.dimensions: tuple[Dimension, ...] = () if state is None else tuple(_dimensions(state))
        # The profile's table is the declaration's, over state no cell uses: every cell binds its own.
        self.registry: WorldRegistry | None = (
            None
            if state is None
            else WorldRegistry(
                (
                    WorldDimension(
                        name=dim.name,
                        schema=dim.schema,
                        matters=dim.matters,
                        carrier=self.name,
                        seed=f"{self.name}.seed.{dim.name}",
                        read=f"{self.name}.read.{dim.name}",
                        perceived_by=(_VIEW,),
                    )
                    for dim in self.dimensions
                ),
                bindings=self.bindings({}),
                subject_view=f"{self.name}.{_VIEW}",
                binds_per_cell=True,
            )
        )

    def __repr__(self) -> str:
        """The world, by state and tools."""
        state = "" if self.state is None else f"{self.state.__name__}, "
        return f"World({state}tools={list(self.tools)!r})"

    def __call__(self, agent: Callable[[Mapping[str, Any], Any], Awaitable[Any]]) -> WorldRunner:
        """The runner hosting ``agent`` on this world.

        :param agent: ``async def agent(case, tools)``, handed each case and the tools on its cell's world.
        :ptype agent: Callable[[Mapping[str, Any], Any], Awaitable[Any]]
        :return: The runner.
        :rtype: WorldRunner
        :raises ValueError: ``agent`` is not callable.
        """
        if not callable(agent):
            raise ValueError(f"a world hosts an agent, a callable (case, tools) -> answer, not {agent!r}")
        return WorldRunner(self, agent)

    @property
    def stateful(self) -> bool:
        """Whether the world has state, which each case seeds and goal checks read."""
        return self.state is not None

    def outcome(self, answer: Any, state: Mapping[str, Any] | None, calls: Sequence[RecordedCall]) -> WorldOutcome:
        """What graders see of one cell: the answer, the state as the state model, and the calls.

        :param answer: The agent's answer.
        :ptype answer: Any
        :param state: The cell's end state, by dimension; ``None`` for a world of lookups.
        :ptype state: Mapping[str, Any] | None
        :param calls: The calls that succeeded.
        :ptype calls: Sequence[RecordedCall]
        :return: The outcome.
        :rtype: WorldOutcome
        """
        model = None if self.state is None or state is None else self.state.model_construct(**dict(state))
        return WorldOutcome(answer=answer, state=model, calls=list(calls))

    @property
    def dimension_names(self) -> tuple[str, ...]:
        """Every dimension's name, in declaration order."""
        return tuple(dim.name for dim in self.dimensions)

    def bindings(self, state: dict[str, Any]) -> dict[str, Callable[..., Any]]:
        """The handles of a world whose state is ``state``: one cell's, when the kind binds it."""

        def seeder(dim: str) -> Callable[[Any], None]:
            return lambda value: state.__setitem__(dim, copy.deepcopy(value))

        def reader(dim: str) -> Callable[[], Any]:
            return lambda: copy.deepcopy(state.get(dim))

        def view(*, surfaces: Sequence[str]) -> dict[str, Any]:
            # One entry per surface asked for, the registry's convention: the conformance kit reads each
            # surface's rendering by its name, and found nothing when this returned the state itself.
            return {_VIEW: copy.deepcopy(state)} if _VIEW in surfaces else {}

        table: dict[str, Callable[..., Any]] = {f"{self.name}.{_VIEW}": view}
        for dim in self.dimension_names:
            table[f"{self.name}.seed.{dim}"] = seeder(dim)
            table[f"{self.name}.read.{dim}"] = reader(dim)
        return table

    def seed_of(self, values: Mapping[str, Any]) -> WorldSeed:
        """A case's starting state as the engine's seed: every value under this world's carrier."""
        return WorldSeed(namespaces={self.name: dict(values)})

    def refuse_unseedable(self, values: Any, *, case: int | str) -> dict[str, Any]:
        """The case's starting state, every field set, refused unless the state model and the seed walk admit it.

        A field the case leaves out takes its default; one with no default must be given, since a dimension left
        unset would read back as ``None`` — a value nobody seeded, which a goal check would grade as though one had.

        :param values: The case's ``world``.
        :ptype values: Any
        :param case: What the refusal calls the case: its name, or its position.
        :ptype case: int | str
        :return: The state, every field as JSON.
        :rtype: dict[str, Any]
        :raises ValueError: The state is not a mapping, the state model refuses it, or the seed walk refuses it.
        """
        assert self.state is not None, "only a world with state is seeded"
        if not isinstance(values, Mapping):
            raise ValueError(f"case {case}'s world is {values!r}; a starting state maps each field to its value")
        try:
            seeded = self.state.model_validate(dict(values)).model_dump(mode="json")
        except ValidationError as refused:
            raise ValueError(f"case {case}'s world is no {self.state.__name__}: {refused}") from refused
        try:
            assert self.registry is not None
            check_seed(self.registry, self.seed_of(seeded).namespaces, attached=(self.name,))
        except SeedRefused as refused:
            raise ValueError(f"case {case}'s world is a starting state the world refuses: {refused}") from refused
        return seeded

    def did_nothing_passes(
        self, checks: Sequence[str], cases: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ) -> dict[str, int]:
        """For each goal check, in how many cases a candidate that did nothing would pass it.

        The authoring gate's do-nothing control (:func:`~threetears.evals.run.check_controls.idle_end_state`),
        laid over each case's own starting state, since here the starting state is the case's: the seed untouched,
        no call made, nothing fired. A check this passes in every case does not beat doing nothing, whatever the
        candidate scores on it.

        Each check is graded on its own, so one that cannot be evaluated against a starting state costs only its own
        baseline. A check with no baseline is left out, never counted as passed or failed: its summary line then
        states no do-nothing figure, and the warning logged here names the check and why.

        :param checks: The goal checks.
        :ptype checks: Sequence[str]
        :param cases: Per case, its starting state and its variation parameters (``variation.*``).
        :ptype cases: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
        :return: Check -> cases passed, for every check evaluable against every case's starting state.
        :rtype: dict[str, int]
        """
        assert self.registry is not None, "only a world with state has goal checks"
        passes = dict.fromkeys(checks, 0)
        idles = [(idle_end_state(self.seed_of(seed), world=self.registry), variation) for seed, variation in cases]
        for check in list(passes):
            for idle, variation in idles:
                try:
                    outcomes = grade_goal_checks(
                        [check],
                        ledger=idle.ledger,
                        end_state=idle.end_state,
                        fired=idle.fired,
                        variation=variation,
                        world=self.registry,
                    )
                except GoalCheckUnevaluable as unevaluable:
                    log.warning(
                        "eval.quick goal check %r has no do-nothing baseline: %s; its pass rate is shown without one",
                        check,
                        unevaluable,
                    )
                    del passes[check]
                    break
                passes[check] += sum(outcome.passed for outcome in outcomes)
        return passes

    def action_parameters(self, tool: str, action: str) -> Mapping[str, Any] | None:
        """The parameter schema the candidate is shown for ``<tool>.<action>``, or None for a call this world lacks.

        The host's :attr:`~threetears.evals.kernel.host.HostProfile.action_parameters` reader for a quick world,
        so the goal-check gate reads a tool's parameters from the very schema each call is held to: a comparison
        over an ``enum``-, ``const``- or ``pattern``-closed parameter is a check of structure, and one over a free
        string is a reading of what the model wrote, refused as authoring refuses it.

        :param tool: The carrier a check names, this world's name for one of its tools.
        :ptype tool: str
        :param action: The tool's name.
        :ptype action: str
        :return: The tool's ``input_schema``, or None.
        :rtype: Mapping[str, Any] | None
        """
        declared = self.tools.get(action) if tool == self.name else None
        return None if declared is None else declared.input_schema

    def tool_actions(self, tool: str) -> frozenset[str]:
        """The actions ``tool`` offers: this world's tools under its own name, none under any other.

        The host's :attr:`~threetears.evals.kernel.host.HostProfile.tool_actions` reader for a quick world.

        :param tool: The carrier a check names.
        :ptype tool: str
        :return: The tools' names; empty for a carrier that is not this world.
        :rtype: frozenset[str]
        """
        return frozenset(self.tools) if tool == self.name else frozenset()

    def refuse_unreadable(self, checks: Sequence[str]) -> None:
        """Refuse a goal check reading state this world does not declare or calling a tool it does not have.

        Left to the run, either would grade *not established* or *never called* on every cell — a typo
        measured as the candidate's failure.

        :raises ValueError: A check reads an undeclared dimension or names an unknown tool.
        :raises DSLError: A check is not an expression of the goal language.
        """
        if self.registry is None:
            raise ValueError(
                f"the goal checks {', '.join(map(repr, checks))} read a world's state, and this world has none: "
                "declare its state (World(State, tools=[...]))"
            )
        for check in checks:
            if unknown := [path for path in extract_paths(check).world if self.registry.resolve_path(path) is None]:
                raise ValueError(f"goal check {check!r} reads state.{unknown[0]}, which world {self.name!r} lacks")
            for tool, action in referenced_actions(check):
                if tool != self.name or action not in self.tools:
                    raise ValueError(
                        f"goal check {check!r} names {tool}.{action}; this world's tools are "
                        + (", ".join(f"{self.name}.{known}" for known in self.tools) or "none")
                    )


#: The one surface the candidate sees the world through: :meth:`WorldTools.view`.
_VIEW = "view"


class WorldTools(Mapping[str, Callable[..., Awaitable[Any]]]):
    """What the candidate holds of its cell's world: the tools, bound to it, and a view of it.

    A world candidate is called ``candidate(case, tools)`` with one of these. It is the mapping a
    tool-using candidate is handed — ``await tools["switch"](to="on")`` — so one candidate shape serves
    both; :meth:`call` is the same call by name, and :attr:`declared` describes each tool (name,
    description, ``input_schema``) for a model's tool list.
    """

    def __init__(self, world: World, session: WorldSession, state: dict[str, Any]) -> None:
        """Bind the world's tools to one cell's session and state."""
        self.world = world
        self.session = session
        self.state = state
        self.ledger = CallLedger(refused=[])
        self.fault: str | None = None

    def __getitem__(self, tool: str) -> Callable[..., Awaitable[Any]]:
        """The tool, as an async function of its parameters."""
        if tool not in self.world.tools:
            raise KeyError(tool)
        return partial(self.call, tool)

    def __iter__(self) -> Iterator[str]:
        """The tools' names, in declaration order."""
        return iter(self.world.tools)

    def __len__(self) -> int:
        """How many tools there are."""
        return len(self.world.tools)

    @property
    def declared(self) -> tuple[WorldTool, ...]:
        """Every tool the candidate may call, in declaration order."""
        return tuple(self.world.tools.values())

    async def view(self) -> dict[str, Any]:
        """The world as the candidate sees it now, through the registry's subject view."""
        handle = self.session.registry.subject_view
        assert handle is not None
        rendered: dict[str, Any] = await self.session.registry.call(handle, surfaces=(_VIEW,))
        seen: dict[str, Any] = rendered[_VIEW]
        return seen

    async def call(self, tool: str, /, **params: Any) -> Any:
        """Call one tool on this cell's world and record the call; returns what the tool says.

        :raises ToolRefused: No such tool, or parameters its schema refuses. Nothing changed; the refusal is recorded
            on the ledger's refusal record, never among its calls, before it is raised.
        """
        declared = self.world.tools.get(tool)
        if declared is None:
            known = ", ".join(self.world.tools) or "none"
            self._refuse(tool, params, f"there is no tool {tool!r}; the tools are {known}")
        if violations := schema_violations(declared.input_schema, params, at=tool):
            self._refuse(tool, params, "; ".join(violations))
        assert self.world.state is not None, "a world with state hands its agent these tools"
        try:
            # The tool acts on the state model; what it leaves is written back as the cell's state, every field.
            model = self.world.state.model_validate(self.state)
            said = declared.act(model, **params)
            if inspect.isawaitable(said):
                said = await said
            self.state.clear()
            self.state.update(model.model_dump(mode="json"))
        # prawduct:ok-broad-except — the tool is the rig: what it raises excludes the cell, and the candidate is told it failed
        except Exception as raised:
            self.fault = f"the tool {tool} raised {type(raised).__name__}: {raised}"
            raise
        self.ledger.record(self.world.name, tool, params)
        return said

    def _refuse(self, tool: str, params: dict[str, Any], reason: str) -> NoReturn:
        """Record a refused call on the ledger, then refuse it: the record is written before the candidate hears no.

        :raises ToolRefused: Always, carrying ``reason``.
        """
        self.ledger.record_refusal(self.world.name, tool, params, reason=reason)
        raise ToolRefused(reason)


#: An agent on a world: an async callable taking one case and the tools on its cell's world.
WorldAgent = Callable[[Mapping[str, Any], Any], Awaitable[Any]]

#: Where a world case carries its starting state on its stored test case: this module's key, written and read here.
SEED_KEY = "seed"


def world_case_payload(seed: dict[str, Any]) -> dict[str, Any]:
    """What a world case adds to its stored test case's ``host_payload``: its starting state, under :data:`SEED_KEY`."""
    return {SEED_KEY: seed}


class WorldCellKind:
    """One cell of a world with state: seeds the case's world, runs the callable kind over it, grades the end state.

    Built per cell (it holds that cell's tools), from the :class:`~threetears.evals.run.CellContext` the
    runner hands the kind factory, so ``prepare`` knows the case whose starting state it seeds.
    """

    judged_artifact: JudgedArtifact
    #: The inner callable kind's: this kind is built per cell and binds the cell a fresh world of its own
    #: (``binds_per_cell``), so it adds nothing two cells could share.
    max_concurrent_cells: int | None

    def __init__(
        self,
        world: World,
        candidate: WorldAgent,
        inner: Callable[[Callable[[Mapping[str, Any]], Awaitable[Any]]], Any],
        cell: CellContext,
    ) -> None:
        """Bind the world, the candidate, and the cell.

        :param world: The world every cell seeds.
        :ptype world: World
        :param candidate: The world candidate.
        :ptype candidate: WorldAgent
        :param inner: Builds the callable kind over a one-argument candidate — the scorers, the expected
            labels and the judge, exactly as a world-less run grades.
        :ptype inner: Callable[[Callable[[Mapping[str, Any]], Awaitable[Any]]], Any]
        :param cell: The cell this kind drives.
        :ptype cell: CellContext
        """
        self._world = world
        self._cell = cell
        self._tools: WorldTools | None = None

        async def acting(case: Mapping[str, Any]) -> Any:
            tools = self._tools
            assert tools is not None, "prepare seeds the world before invoke runs the candidate"
            returned = await candidate(case, tools)
            # What a grader sees is the answer beside the state the agent left and the calls it made.
            answer = returned.value if isinstance(returned, Answer) else returned
            outcome = world.outcome(answer, tools.state, tools.ledger.calls)
            return dataclasses.replace(returned, value=outcome) if isinstance(returned, Answer) else outcome

        acting.__name__ = getattr(candidate, "__name__", "candidate")
        self._inner = inner(acting)
        self.judged_artifact = self._inner.judged_artifact
        self.max_concurrent_cells = self._inner.max_concurrent_cells

    async def prepare(
        self,
        *,
        subject_snapshot: SubjectSnapshot | None,
        variant_config: VariantConfig,
        world_seed: WorldSeed,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
        world: WorldSession | None,
    ) -> Any:
        """Bind a fresh world for the cell, seed it with the case's starting state, then prepare the candidate."""
        assert world is not None, "a world run's host declares the world, so every cell is handed its session"
        state: dict[str, Any] = {}
        world.bind(self._world.bindings(state))
        try:
            await world.seed(
                self._world.seed_of(self._cell.test_case.host_payload[SEED_KEY]), attached=(self._world.name,)
            )
        except SeedRefused as refused:
            raise CandidatePreparationFailed(f"apparatus: {refused}", termination="seed_failed") from refused
        self._tools = WorldTools(self._world, world, state)
        return await self._inner.prepare(
            subject_snapshot=subject_snapshot,
            variant_config=variant_config,
            world_seed=world_seed,
            span_window=span_window,
            cassettes=cassettes,
            world=None,
        )

    async def invoke(self, instance: Any, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Run the candidate on its world, then read the world back and grade the template's goal checks."""
        tools = self._tools
        assert tools is not None, "prepare seeds the world before invoke runs the candidate"
        output: CandidateOutput = await self._inner.invoke(instance, test_case, sink)
        if tools.fault is not None:
            return CandidateOutput(output=output.output, infra_errors=[tools.fault], call_ledger=tools.ledger)
        end_state = await tools.session.end_state()
        try:
            checks = evaluate_goal_state(
                template=self._cell.template,
                test_case=test_case,
                ledger=tools.ledger,
                end_state=end_state,
                fired=tools.session.fired,
                world=tools.session.registry,
            )
        except GoalCheckUnevaluable as unevaluable:
            return output.model_copy(
                update={"infra_errors": [*output.infra_errors, str(unevaluable)], "call_ledger": tools.ledger}
            )
        return output.model_copy(
            update={"mechanical_facts": [*output.mechanical_facts, *checks], "call_ledger": tools.ledger}
        )


__all__ = [
    "SEED_KEY",
    "Dimension",
    "ToolRefused",
    "World",
    "WorldAgent",
    "WorldCellKind",
    "WorldOutcome",
    "WorldRunner",
    "WorldTool",
    "WorldTools",
    "world_case_payload",
]
