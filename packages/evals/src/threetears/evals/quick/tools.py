"""The tools a world of lookups hands its agent, recorded once to a file and replayed from it after.

A runner whose answer depends on what a tool said — a search, a price, a dice roll — is compared fairly only when
every arm faced the same tool answers. A world with no state (``World(tools=[search], recording=...)``) does that:

- **A tool is a plain function** (:data:`Tool`) of keyword arguments that returns a JSON value, sync or async. The
  agent is called with the case and its tools: ``await tools["search"](query="...")``.
- **Each declared tool is one action.** :class:`CellTools` adapts every function to the engine's
  :class:`~threetears.evals.kernel.cassettes.ToolLike` and records each call that succeeded, which a grader reads on
  the :class:`~threetears.evals.quick.world.WorldOutcome`.
- **One replay rule.** Handed a recording (:class:`ToolRecording`), a file that is there is replayed and one that is
  not is recorded; ``RECORD=1`` records it again. An answer is keyed by the case, the tool, its keyword arguments and
  which time the cell asked it, so every repeat of a case and every arm is served the same answers in the order they
  were first given.
- **A miss is the rig's, even when the agent swallows it.** A replay asked something the recording does not hold
  raises :class:`ToolRecordingMiss`, an :class:`~threetears.evals.kernel.host.ApparatusError`, at the agent's
  ``await``. An agent catching every exception would absorb it into an ordinary answer, so the cell remembers the
  fault and :meth:`CellTools.raise_any_fault` re-raises it once the agent is done, and the engine excludes the cell
  rather than scoring the agent's manner toward a broken recording.

**A tool that raises records nothing.** A recording keeps what a tool answered; an exception is the agent's to
handle in the live world, and a replay of that ask is a miss. A tool whose failure the agent should face under
replay too returns it as a value (``{"error": "..."}``).
"""

from __future__ import annotations

import inspect
import json
import os
import threading
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from threetears.evals.kernel.cassettes import Recordable, ToolLike, ToolWrap
from threetears.evals.kernel.host import ApparatusError
from threetears.evals.schema import RecordedCall, canonical_digest

#: The environment variable that asks for a recording to be made again rather than replayed.
RECORD_ENV = "RECORD"

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

        :param name: The tool's name, which is also its one action and the ``tool`` of its recordings.
        :ptype name: str
        :param function: The function, called with the candidate's keyword arguments.
        :ptype function: Tool
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

        :raises ApparatusError: The function answered something JSON cannot hold, which no cassette could
            record or replay: the rig's fault, never the candidate's.
        """
        answered = self._function(**parameters)
        if inspect.isawaitable(answered):
            answered = await answered
        try:
            return ToolAnswer(value=answered)
        except ValidationError as unrecordable:
            raise ApparatusError(
                f"the tool {self._name!r} answered a {type(answered).__name__}, which a recording cannot keep; "
                "a world's lookup returns a JSON value"
            ) from unrecordable


class ToolRecordingMiss(ApparatusError, LookupError):
    """A replay asked a tool something its recording does not hold: the rig's fault, never the agent's."""


class ToolRecording:
    """A world of lookups' recording: what each tool answered, by case, tool, arguments and occurrence, in a JSONL file.

    One rule, as a model's recording follows it: a file that is there is replayed; a file that is not is recorded,
    every call made live and its first answer kept; ``RECORD=1`` records it afresh. Which it does is decided when a
    run starts (:meth:`begin`), and the run's result says which.

    :ivar path: The file.
    :ivar mode: ``"record"`` or ``"replay"``, as the last :meth:`begin` decided.
    """

    def __init__(self, path: Path) -> None:
        """Name the file; nothing is read before :meth:`begin`."""
        self.path = path
        self.mode: Literal["record", "replay"] = "replay"
        self._answers: dict[tuple[str, int], Any] = {}
        self._lock = threading.Lock()

    def begin(self) -> None:
        """Decide to record or replay, and read what the file holds.

        :raises ToolRecordingMiss: A line of the file is not one this recorder writes.
        """
        rerecord = os.environ.get(RECORD_ENV) == "1"
        self.mode = "record" if rerecord or not self.path.is_file() else "replay"
        if rerecord:
            self.path.unlink(missing_ok=True)
        self._answers = {}
        if not self.path.is_file():
            return
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                held = json.loads(line)
                self._answers[(held["key"], held["occurrence"])] = held["answer"]
            except (ValueError, KeyError, TypeError) as unreadable:
                raise ToolRecordingMiss(
                    f"{self.path} line {number} is not a line a tool recording writes ({unreadable}); record it again "
                    "with RECORD=1"
                ) from unreadable

    def served(self, key: str, occurrence: int, tool: str, params: Mapping[str, Any]) -> Any:
        """The answer recorded for one ask.

        :raises ToolRecordingMiss: The recording holds none: the ask changed since it was recorded.
        """
        try:
            return self._answers[(key, occurrence)]
        except KeyError:
            raise ToolRecordingMiss(
                f"{self.path} holds no answer for {tool}({json.dumps(dict(params), sort_keys=True)}) asked "
                f"{occurrence + 1} time(s) on this case; it changed since it was recorded: record it again with "
                "RECORD=1"
            ) from None

    def keep(self, key: str, occurrence: int, tool: str, params: Mapping[str, Any], answer: Any) -> None:
        """Append one answer, unless that ask's answer is already kept: the first answer is the recorded one."""
        with self._lock:
            if (key, occurrence) in self._answers:
                return
            self._answers[(key, occurrence)] = answer
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = {
                "key": key,
                "occurrence": occurrence,
                "tool": tool,
                "params": dict(params),
                "answer": answer,
                "recorded_at": datetime.now(UTC).isoformat(),
            }
            with self.path.open("a", encoding="utf-8") as recording:
                recording.write(json.dumps(line, sort_keys=True) + "\n")


class CellTools:
    """One cell's tools: the cassette seams a kind wires, and the functions its candidate calls.

    Built by the callable kind's ``prepare`` for every cell of a run whose candidate declares tools, and
    handed to the cell's ``cassettes.wire`` when the run records or replays. It is its own
    :class:`~threetears.evals.kernel.cassettes.ActionSeam`: every tool is a recorded one, and none
    answers in the background, so it declares no delivery seam.
    """

    def __init__(self, tools: Mapping[str, Tool], *, recording: ToolRecording | None = None) -> None:
        """Adapt each declared function to a tool.

        :param tools: The declared tools, by name.
        :ptype tools: Mapping[str, Tool]
        :param recording: Where their answers are recorded or replayed from; ``None`` calls them live.
        :ptype recording: ToolRecording | None
        """
        self._tools: dict[str, ToolLike] = {name: FunctionTool(name, function) for name, function in tools.items()}
        self._fault: ApparatusError | None = None
        self._recording = recording
        self._case: str = ""
        self._asked: Counter[str] = Counter()
        #: Every call that succeeded, in order: what a grader reads as the outcome's ``calls``.
        self.calls: list[RecordedCall] = []

    def bind_case(self, case: Mapping[str, Any]) -> None:
        """Name the case the cell asks about: what a recorded answer is keyed by, beside the tool and its arguments."""
        self._case = canonical_digest({"case": dict(case)})

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
        """One call of one tool: replayed, or made and recorded, or made; a rig fault is remembered."""
        key = canonical_digest({"case": self._case, "tool": tool, "params": parameters})
        occurrence = self._asked[key]
        self._asked[key] += 1
        try:
            if self._recording is not None and self._recording.mode == "replay":
                value = self._recording.served(key, occurrence, tool, parameters)
            else:
                answer = await self._tools[tool].act(tool, parameters)
                if not isinstance(answer, ToolAnswer):
                    raise ApparatusError(f"the tool {tool!r} was served a {type(answer).__name__}, not a ToolAnswer")
                value = answer.value
                if self._recording is not None:
                    self._recording.keep(key, occurrence, tool, parameters, value)
        except ApparatusError as fault:
            self._fault = self._fault or fault
            raise
        self.calls.append(RecordedCall(tool="tools", action=tool, params=dict(parameters)))
        return value

    def raise_any_fault(self) -> None:
        """Re-raise the first rig fault a tool call met, whatever the candidate did with it.

        :raises ApparatusError: A call this cell made met one — a replay miss, an exhausted or corrupt
            recording, an answer no cassette could record.
        """
        if self._fault is not None:
            raise self._fault


__all__ = [
    "RECORD_ENV",
    "CandidateTools",
    "CellTools",
    "FunctionTool",
    "Tool",
    "ToolAnswer",
    "ToolRecording",
    "ToolRecordingMiss",
    "ToolUsingCandidate",
]
