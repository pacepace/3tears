"""The tools a ``run_eval`` candidate calls, as the engine's cassette seams record and replay them.

A candidate whose answer depends on what a tool said — a search, a price, a dice roll — is compared
fairly only when every arm faced the same tool answers. The engine's cassettes do that (``cassette_mode``
``'capture'`` records what the tools answered, ``'replay'`` serves the recording in place of running
them), through the seams a kind supplies (:mod:`threetears.evals.kernel.cassettes`). This module is
those seams for a candidate that is a plain function:

- **A tool is a plain function** (:data:`Tool`) of keyword arguments that returns a JSON value, sync or
  async. ``run_eval(..., tools={"search": search})`` declares it, and the candidate is then called with
  the case and its tools (:data:`ToolUsingCandidate`): ``await tools["search"](query="...")``.
- **Each declared tool is one recorded action.** :class:`CellTools` adapts every function to the
  engine's :class:`~threetears.evals.kernel.cassettes.ToolLike` and is the cell's
  :class:`~threetears.evals.kernel.cassettes.ActionSeam`, so a cassette run wraps every one of them
  and a replay calls none. A recording is keyed by the tool, its keyword arguments and which time the
  cell asked it, as every cassette is.
- **The candidate meets the same tools in every mode.** With cassettes off they run live through the
  same adapter, so what a candidate sees does not change when a run starts recording or replaying.
- **A miss is the rig's, even when the candidate swallows it.** A replay asked something its capture
  never recorded raises :class:`~threetears.evals.kernel.CassetteMiss` (an
  :class:`~threetears.evals.kernel.host.ApparatusError`) at the candidate's ``await``. A candidate
  catching every exception would absorb it into an ordinary answer, so the cell remembers the fault and
  :meth:`CellTools.raise_any_fault` re-raises it once the candidate is done, and the engine excludes the
  cell rather than scoring the candidate's manner toward a broken corpus.

**A tool that raises records nothing.** The cassette layer records what a tool answered; an exception is
the candidate's to handle in the live world, and a replay of that ask is a miss. A tool whose failure
the candidate should face under replay too returns it as a value (``{"error": "..."}``).
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from functools import partial
from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from threetears.evals.kernel.cassettes import Recordable, ToolLike, ToolWrap
from threetears.evals.kernel.host import ApparatusError

#: A tool a candidate calls: a function of keyword arguments returning a JSON value, sync or async. The
#: keyword arguments are what a recording is keyed by, so a call is made with keywords only.
Tool = Callable[..., Any]

#: What a tool-using candidate is handed beside its case: each declared tool by name, as an async function
#: of keyword arguments returning the tool's JSON answer.
CandidateTools = Mapping[str, Callable[..., Awaitable[Any]]]

#: A candidate that calls tools: an async callable taking one case and its tools, returning its answer.
ToolUsingCandidate = Callable[[Mapping[str, Any], CandidateTools], Awaitable[Any]]


class ToolAnswer(BaseModel):
    """One tool answer, as a cassette records it and a replay rebuilds it: the JSON value the tool returned."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: JsonValue


class FunctionTool:
    """A plain function as the engine's :class:`~threetears.evals.kernel.cassettes.ToolLike`: one action, its own name."""

    def __init__(self, name: str, function: Tool) -> None:
        """Bind the function to the name the candidate calls it by.

        Args:
            name: The tool's name, which is also its one action and the ``tool`` of its recordings.
            function: The function, called with the candidate's keyword arguments.
        """
        self._name = name
        self._function = function

    @property
    def name(self) -> str:
        """The tool's name."""
        return self._name

    def can_dispatch(self, action: str) -> bool:
        """Whether ``action`` is this tool's one action, its name."""
        return action == self._name

    async def act(self, action: str, parameters: dict[str, Any]) -> ToolAnswer:
        """Call the function with ``parameters`` and return its answer.

        Raises:
            ApparatusError: The function answered something JSON cannot hold, which no cassette could
                record or replay: the rig's fault, never the candidate's.
        """
        answered = self._function(**parameters)
        if inspect.isawaitable(answered):
            answered = await answered
        try:
            return ToolAnswer(value=answered)
        except ValidationError as unrecordable:
            raise ApparatusError(
                f"the tool {self._name!r} answered a {type(answered).__name__}, which a cassette cannot record; "
                "a run_eval tool returns a JSON value"
            ) from unrecordable


class CellTools:
    """One cell's tools: the cassette seams a kind wires, and the functions its candidate calls.

    Built by the callable kind's ``prepare`` for every cell of a run whose candidate declares tools, and
    handed to the cell's ``cassettes.wire`` when the run records or replays. It is its own
    :class:`~threetears.evals.kernel.cassettes.ActionSeam`: every tool is a recorded one, and none
    answers in the background, so it declares no delivery seam.
    """

    def __init__(self, tools: Mapping[str, Tool]) -> None:
        """Adapt each declared function to a tool.

        Args:
            tools: The declared tools, by name.
        """
        self._tools: dict[str, ToolLike] = {name: FunctionTool(name, function) for name, function in tools.items()}
        self._fault: ApparatusError | None = None

    @property
    def action_seam(self) -> CellTools:
        """The cell's tools, all recorded."""
        return self

    @property
    def delivery_seams(self) -> Mapping[str, Any]:
        """None: a function tool answers when it is called, never in the background."""
        return {}

    @property
    def recorded_tools(self) -> Mapping[str, type[Recordable]]:
        """Every declared tool, each answering a :class:`ToolAnswer`."""
        return dict.fromkeys(self._tools, ToolAnswer)

    def arm_tools(self, wrap: ToolWrap) -> None:
        """Swap the tools for the cassette run's wrapped ones, for the rest of the cell."""
        self._tools = wrap(self._tools)

    def for_candidate(self) -> dict[str, Callable[..., Awaitable[Any]]]:
        """The tools as the candidate calls them: by name, keyword arguments in, the JSON answer out."""
        return {name: partial(self._call, name) for name in self._tools}

    async def _call(self, tool: str, /, **parameters: Any) -> Any:
        """One call of one tool, through whatever the cell wired in its place; a rig fault is remembered."""
        try:
            answer = await self._tools[tool].act(tool, parameters)
        except ApparatusError as fault:
            self._fault = self._fault or fault
            raise
        if not isinstance(answer, ToolAnswer):
            raise ApparatusError(f"the tool {tool!r} was served a {type(answer).__name__}, not a ToolAnswer")
        return answer.value

    def raise_any_fault(self) -> None:
        """Re-raise the first rig fault a tool call met, whatever the candidate did with it.

        Raises:
            ApparatusError: A call this cell made met one — a replay miss, an exhausted or corrupt
                recording, an answer no cassette could record.
        """
        if self._fault is not None:
            raise self._fault


def refuse_unusable_tools(tools: Mapping[str, Tool] | None, cassette_mode: str | None) -> None:
    """Refuse tools a candidate could not be handed, and a cassette run of a candidate that declares none.

    Args:
        tools: The tools ``run_eval`` was given, or ``None``.
        cassette_mode: The cassette mode it was given.

    Raises:
        ValueError: ``tools`` is not a mapping, is empty, names a tool by something other than an
            identifier, or holds something that is not callable; or the run records or replays and the
            candidate declares no tools, so a replay would have nothing to serve and would run live.
    """
    if tools is None:
        if cassette_mode in ("capture", "replay"):
            raise ValueError(
                f"cassette_mode={cassette_mode!r} records or replays a candidate's tools, and this candidate "
                "declares none (tools=): nothing would be recorded, and a replay would run live. Declare the tools "
                "the candidate calls, or leave cassette_mode off"
            )
        return
    if isinstance(tools, str) or not isinstance(tools, Mapping) or not tools:
        raise ValueError("tools= is a non-empty mapping of tool name to function; omit it for a candidate with none")
    if unnamed := [repr(name) for name in tools if not isinstance(name, str) or not name.isidentifier()]:
        raise ValueError(f"a tool's name is how the candidate calls it, and {', '.join(unnamed)} is no identifier")
    if uncallable := [name for name, function in tools.items() if not callable(function)]:
        raise ValueError(f"the tool(s) {', '.join(uncallable)} are not callable")


__all__ = [
    "CandidateTools",
    "CellTools",
    "FunctionTool",
    "Tool",
    "ToolAnswer",
    "ToolUsingCandidate",
    "refuse_unusable_tools",
]
