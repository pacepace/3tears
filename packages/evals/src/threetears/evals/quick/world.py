"""A small world for :func:`~threetears.evals.quick.run_eval`: state the candidate acts on through tools, graded by its end.

The one-call path's candidate answers, and code grades the answer. Some candidates do not answer —
they *act*: switch a light, file a ticket, move money. What matters then is the state they leave
behind, and the engine already measures that: a host declares its world
(:class:`~threetears.evals.contracts.host.WorldRegistry`), each cell seeds it through a
:class:`~threetears.evals.contracts.WorldSession`, the runner reads it back once the candidate is done,
and goal-state checks (``state.<dimension>``, ``calls("<world>.<tool>")``) grade what it holds. This
module declares that world in a few lines and drives the same machinery:

- :class:`World` names the state (:class:`Dimension`, each a JSON Schema and the reason it matters)
  and the tools that change it (:class:`WorldTool`, a plain function of the state and its parameters). It
  builds the engine's registry: every dimension seeded and read back through its own handles, and seen
  by the candidate through the registry's subject view.
- **Each case seeds its own world.** ``run_eval(..., world=, seed=)`` reads the starting state off each
  case. The cell's kind binds a fresh state for its cell, seeds it through the session before the
  candidate's first turn, and hands the candidate a :class:`WorldTools`: the tools, bound to that state,
  and the view. The run's world placements record every dimension as seeded, since every case seeds it.
- **Each tool call that succeeds is recorded** on the cell's call ledger as ``<world>.<tool>`` with its
  parameters, which is what ``calls(...)`` and ``call_count(...)`` read. A call whose parameters its
  schema refuses changes nothing and is not recorded; the candidate is told why.
- **The end state is read back after the last turn** through the session — the same single reading the
  runner stores on the cell — and the template's goal-state checks are graded against it by the engine's
  own :func:`~threetears.evals.run.evaluate_goal_state`. A tool that raises is the rig's fault and
  excludes the cell; a check that cannot be evaluated does too.

**Where the starting state lives.** The engine's template carries one seed per scenario; here every case
carries its own, because the starting state is what varies. So the template's seed stays empty, each
case's seed rides on its stored test case beside the case itself (and its flat fields stay readable as
``variation.<field>``), and the template's id digests the seeds with the cases, so a different starting
state is a different case set.

**A world tool is an action, not a lookup.** :class:`WorldTools` is a plain mapping of tool name to async
function, the shape any tool-using candidate is handed, so one candidate shape serves both. What a world
tool adds is that it moves the graded state: it is validated against the schema the model is shown,
recorded on the call ledger, and must run in every cell — a recording replayed in its place would leave
the world unmoved and the end state graded wrong.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from threetears.evals.contracts import (
    CallLedger,
    CandidateOutput,
    CandidatePreparationFailed,
    CellCassettes,
    CellSink,
    CellSpanWindow,
    EvalTestCase,
    JudgedArtifact,
    VariantConfig,
    WorldSeed,
    WorldSession,
    extract_paths,
    referenced_actions,
)
from threetears.evals.contracts.host import (
    SeedRefused,
    SubjectSnapshot,
    WorldDimension,
    WorldRegistry,
    check_seed,
    schema_violations,
)
from threetears.evals.run import CellContext, GoalCheckUnevaluable, evaluate_goal_state, grade_goal_checks
from threetears.evals.run.check_controls import idle_end_state


@dataclass(frozen=True)
class Dimension:
    """One piece of the world's state.

    Attributes:
        name: What a case seeds and a goal check reads it as (``state.<name>``).
        schema: The JSON Schema every value of it satisfies; a seed outside it is refused.
        matters: Why the state is worth seeding — the engine asks every dimension for its reason.
    """

    name: str
    schema: Mapping[str, Any]
    matters: str


class WorldTool:
    """One action the candidate can take on the world: a function of the state and its parameters.

    ``act(state, **params)`` changes ``state`` — the cell's world, a ``dict`` of dimension name to
    value — and returns what the candidate is told. It may be ``async``. Its ``__name__`` names the tool
    and the first line of its docstring describes it to the candidate. Every parameter is required and
    declared with the JSON Schema its value must satisfy, which is also the schema the candidate is shown.
    """

    def __init__(self, act: Callable[..., Any], /, **params: Mapping[str, Any]) -> None:
        """Declare ``act`` as a tool taking ``params``, each named with its JSON Schema.

        Raises:
            ValueError: ``act`` has no name a tool can carry.
        """
        name = getattr(act, "__name__", "")
        if not name.isidentifier():
            raise ValueError(f"a tool is named by its function's __name__, and {act!r} has none; write it as a def")
        doc = inspect.getdoc(act)
        self.act = act
        self.name: str = name
        self.description: str = doc.splitlines()[0] if doc else f"The {name} tool."
        self.params: dict[str, dict[str, Any]] = {key: dict(schema) for key, schema in params.items()}

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


class World:
    """A small world: named state each case seeds, and tools the candidate changes it with.

    Built once and handed to :func:`~threetears.evals.quick.run_eval` (``world=``). :attr:`registry` is the
    engine's declaration of it, which the run's host carries on its profile.
    """

    def __init__(self, name: str, dimensions: Sequence[Dimension], *, tools: Sequence[WorldTool] = ()) -> None:
        """Declare the world.

        Args:
            name: The world's name: the carrier its state reaches the candidate through, and the prefix its
                tools are recorded under (``calls("<name>.<tool>")``).
            dimensions: Its state.
            tools: What the candidate can do to it.

        Raises:
            ValueError: No dimension, a blank or dotted name, two tools of one name, or a declaration the
                engine's registry refuses (a blank reason, a self-contradicting schema).
        """
        if not name.isidentifier():
            raise ValueError(f"a world's name prefixes its tools' calls, so it must be an identifier: {name!r}")
        if not dimensions:
            raise ValueError("a world needs at least one dimension of state")
        names = [tool.name for tool in tools]
        if repeated := sorted({tool for tool in names if names.count(tool) > 1}):
            raise ValueError(f"tools named {', '.join(repeated)} more than once")
        self.name = name
        self.dimensions = tuple(dimensions)
        self.tools: dict[str, WorldTool] = {tool.name: tool for tool in tools}
        # The profile's table is the declaration's, over state no cell uses: every cell binds its own.
        self.registry = WorldRegistry(
            (
                WorldDimension(
                    name=dim.name,
                    schema=dim.schema,
                    matters=dim.matters,
                    carrier=name,
                    seed=f"{name}.seed.{dim.name}",
                    read=f"{name}.read.{dim.name}",
                    perceived_by=(_VIEW,),
                )
                for dim in self.dimensions
            ),
            bindings=self.bindings({}),
            subject_view=f"{name}.{_VIEW}",
            binds_per_cell=True,
        )

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
        """The case's starting state, refused unless it sets every dimension to a value its schema admits.

        Every dimension, because a dimension a case leaves unset would read back as ``None`` — a value
        nobody seeded, which a goal check would grade as though one had.

        Args:
            values: The state ``seed=`` gave the case.
            case: What the refusal calls the case: its name, or its position.

        Raises:
            ValueError: The state is not a mapping, leaves a dimension unset, or the seed walk refuses it.
        """
        if not isinstance(values, Mapping):
            raise ValueError(f"seed= gave case {case} {values!r}; a starting state maps dimension to value")
        if missing := [dim for dim in self.dimension_names if dim not in values]:
            raise ValueError(f"seed= gave case {case} no {', '.join(missing)}; every case sets every dimension")
        try:
            check_seed(self.registry, self.seed_of(values).namespaces, attached=(self.name,))
        except SeedRefused as refused:
            raise ValueError(f"seed= gave case {case} a starting state the world refuses: {refused}") from refused
        return dict(values)

    def did_nothing_passes(
        self, checks: Sequence[str], cases: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ) -> dict[str, int] | None:
        """For each goal check, in how many cases a candidate that did nothing would pass it.

        The authoring gate's do-nothing control (:func:`~threetears.evals.run.check_controls.idle_end_state`),
        laid over each case's own starting state, since here the starting state is the case's: the seed untouched,
        no call made, nothing fired. A check this passes in every case does not beat doing nothing, whatever the
        candidate scores on it.

        Args:
            checks: The goal checks.
            cases: Per case, its starting state and its variation parameters (``variation.*``).

        Returns:
            Check -> cases passed; None when a check cannot be evaluated against a starting state, which the run
            itself reports.
        """
        passes = dict.fromkeys(checks, 0)
        for seed, variation in cases:
            idle = idle_end_state(self.seed_of(seed), world=self.registry)
            try:
                outcomes = grade_goal_checks(
                    list(checks),
                    ledger=idle.ledger,
                    end_state=idle.end_state,
                    fired=idle.fired,
                    variation=variation,
                    world=self.registry,
                )
            except GoalCheckUnevaluable:
                return None
            for outcome in outcomes:
                passes[outcome.expression] += outcome.passed
        return passes

    def action_parameters(self, tool: str, action: str) -> Mapping[str, Any] | None:
        """The parameter schema the candidate is shown for ``<tool>.<action>``, or None for a call this world lacks.

        The host's :attr:`~threetears.evals.contracts.host.HostProfile.action_parameters` reader for a quick world,
        so the goal-check gate reads a tool's parameters from the very schema each call is held to: a comparison
        over an ``enum``-, ``const``- or ``pattern``-closed parameter is a check of structure, and one over a free
        string is a reading of what the model wrote, refused as authoring refuses it.

        Args:
            tool: The carrier a check names, this world's name for one of its tools.
            action: The tool's name.

        Returns:
            The tool's ``input_schema``, or None.
        """
        declared = self.tools.get(action) if tool == self.name else None
        return None if declared is None else declared.input_schema

    def tool_actions(self, tool: str) -> frozenset[str]:
        """The actions ``tool`` offers: this world's tools under its own name, none under any other.

        The host's :attr:`~threetears.evals.contracts.host.HostProfile.tool_actions` reader for a quick world.

        Args:
            tool: The carrier a check names.

        Returns:
            The tools' names; empty for a carrier that is not this world.
        """
        return frozenset(self.tools) if tool == self.name else frozenset()

    def refuse_unreadable(self, checks: Sequence[str]) -> None:
        """Refuse a goal check reading state this world does not declare or calling a tool it does not have.

        Left to the run, either would grade *not established* or *never called* on every cell — a typo
        measured as the candidate's failure.

        Raises:
            ValueError: A check reads an undeclared dimension or names an unknown tool.
            DSLError: A check is not an expression of the goal language.
        """
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
        self.ledger = CallLedger()
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

        Raises:
            ToolRefused: No such tool, or parameters its schema refuses. Nothing changed, nothing recorded.
        """
        declared = self.world.tools.get(tool)
        if declared is None:
            raise ToolRefused(f"there is no tool {tool!r}; the tools are {', '.join(self.world.tools) or 'none'}")
        if violations := schema_violations(declared.input_schema, params, at=tool):
            raise ToolRefused("; ".join(violations))
        try:
            said = declared.act(self.state, **params)
            if inspect.isawaitable(said):
                said = await said
        # prawduct:ok-broad-except — the tool is the rig: what it raises excludes the cell, and the candidate is told it failed
        except Exception as raised:
            self.fault = f"the tool {tool} raised {type(raised).__name__}: {raised}"
            raise
        self.ledger.record(self.world.name, tool, params)
        return said


#: A world candidate: an async callable taking one case and the tools on its cell's world.
WorldCandidate = Callable[[Mapping[str, Any], WorldTools], Awaitable[Any]]

#: A case's starting state: takes the case, returns dimension name to value.
CaseSeed = Callable[[Mapping[str, Any]], Mapping[str, Any]]

#: Where a world case carries its starting state on its stored test case: this module's key, written and read here.
SEED_KEY = "seed"


def world_case_payload(seed: dict[str, Any]) -> dict[str, Any]:
    """What a world case adds to its stored test case's ``host_payload``: its starting state, under :data:`SEED_KEY`."""
    return {SEED_KEY: seed}


class WorldCellKind:
    """One cell of a world run: seeds the case's world, runs the callable kind over it, grades the end state.

    Built per cell (it holds that cell's tools), from the :class:`~threetears.evals.run.CellContext` the
    runner hands the kind factory, so ``prepare`` knows the case whose starting state it seeds.
    """

    judged_artifact: JudgedArtifact

    def __init__(
        self,
        world: World,
        candidate: WorldCandidate,
        inner: Callable[[Callable[[Mapping[str, Any]], Awaitable[Any]]], Any],
        cell: CellContext,
    ) -> None:
        """Bind the world, the candidate, and the cell.

        Args:
            world: The world every cell seeds.
            candidate: The world candidate.
            inner: Builds the callable kind over a one-argument candidate — the scorers, the expected
                labels and the judge, exactly as a world-less run grades.
            cell: The cell this kind drives.
        """
        self._world = world
        self._cell = cell
        self._tools: WorldTools | None = None

        async def acting(case: Mapping[str, Any]) -> Any:
            assert self._tools is not None, "prepare seeds the world before invoke runs the candidate"
            return await candidate(case, self._tools)

        acting.__name__ = getattr(candidate, "__name__", "candidate")
        self._inner = inner(acting)
        self.judged_artifact = self._inner.judged_artifact

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
    "CaseSeed",
    "Dimension",
    "WorldTool",
    "ToolRefused",
    "World",
    "WorldCandidate",
    "WorldCellKind",
    "WorldTools",
    "world_case_payload",
]
