"""The adopting agent: one sentence, the installed package and an app's directory, and a loop of tool calls.

A :class:`Brain` picks each turn's :class:`Act`. :class:`ModelBrain` is a model (the live measurement);
:class:`ScriptedBrain` follows a fixed transcript (the harness proven offline, for free). :func:`attempt` drives one
run under its caps and keeps everything the run did: the :class:`Step` list, from which :func:`stuck_point` and
:func:`errors_hit` are read.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from threetears.evals import Model
from threetears.evals.providers import ProviderError
from threetears.evals.kernel import strict_response_format

from .sandbox import READ_LINES, Observation, Sandbox, is_door

#: The sentence the agent is given, and the report it is asked for.
TASK = """\
Add an eval to this app with 3tears-evals, run it, and report what it found as the JSON below.

{"eval_file": "<the eval file you wrote, relative to the app's directory>",
 "first_line": "<the first line your last run of the eval printed, verbatim>",
 "pass_rate": <your eval's first measure on its first runner, a share from 0 to 1>,
 "finished": <true when that run finished, false when it did not>}"""

SYSTEM = """\
You work in an app's directory, which you have not seen before, with one tool call per turn, answered as one JSON \
object: {{"tool": ..., "path": ..., "start": ..., "text": ..., "command": ...}}.
- read: the file at `path` (in the app's directory, or in the installed package), {lines} lines from line `start`.
- write: `text` to the file at `path`, relative to the app's directory.
- run: `command`, one of `python ...`, `python -m threetears.evals ...`, `ls [path]`, `cat path`; no shell, no pipes.
- done: your final report as `text`, which ends the task.
Leave unused fields empty. 3tears-evals is installed at {package}; its guide for agents is {llms}. Your commands \
run in a container with no network and no API key; the app runs offline."""

#: How many of the latest steps the prompt shows whole; older ones are cut to their first lines.
RECENT_STEPS = 6
OLD_STEP_CHARS = 400


class Act(BaseModel):
    """One turn: which tool, and its arguments (a tool reads only its own)."""

    model_config = ConfigDict(extra="forbid")

    tool: Literal["read", "write", "run", "done"]
    path: str
    start: int
    text: str
    command: str


_BLANK: dict[str, Any] = {"path": "", "start": 1, "text": "", "command": ""}  # Any: a field's empty value


#: How many times one call may be billed, at most: its send, and one resend after a send that broke before any
#: response began (the package's client resends such a send once, on a key with no fallbacks). A call is sent only when
#: the run's charge plus this many of its ceiling stays within the cap, so the cap bounds what the run can be billed.
BILLED_SENDS_AT_MOST = 2


@dataclass(frozen=True)
class Usage:
    """What one turn spent. ``None`` is unreported, never zero.

    :ivar ceiling_usd: The most the call could cost, bounded before it was sent; ``None`` when no cap asked.
    :ivar maybe_billed_sends: Sends of the call that broke before any response began and were sent again: each may
        have been billed, and ``cost_usd`` holds none of them.
    """

    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    ceiling_usd: float | None = None
    maybe_billed_sends: int = 0

    def charged(self) -> float | None:
        """The most this turn can have been billed: its cost and each maybe-billed send at the ceiling; a turn whose
        cost is unknown (it failed, or was cut off) at :data:`BILLED_SENDS_AT_MOST` ceilings. ``None`` when unbounded."""
        if self.cost_usd is None:
            return None if self.ceiling_usd is None else BILLED_SENDS_AT_MOST * self.ceiling_usd
        if self.maybe_billed_sends == 0:
            return self.cost_usd
        return None if self.ceiling_usd is None else self.cost_usd + self.maybe_billed_sends * self.ceiling_usd


@dataclass(frozen=True)
class Turn:
    """A brain's answer: the act, or ``None`` with why the reply was no act, and what asking cost."""

    act: Act | None
    usage: Usage
    problem: str = ""


@dataclass(frozen=True)
class Step:
    """One turn as it happened: the act (``None`` for a reply that was none) and what the tool did."""

    turn: int
    act: Act | None
    observation: Observation


class Brain(Protocol):
    """What picks each turn's act."""

    model: str

    async def ceiling(self, system: str, user: str) -> float | None:
        """The most the next turn can cost, in dollars; ``None`` when that cannot be bounded."""
        ...

    async def next(self, system: str, user: str, steps: Sequence[Step]) -> Turn:
        """The next turn, given the prompt and the steps so far."""
        ...


class ModelBrain:
    """A model picks each act. Every call goes through :class:`~threetears.evals.Model`, so the package prices it,
    and inside the harness's runner its spend lands on the trial's own ledger."""

    def __init__(self, model: Model) -> None:
        self._model = model
        self.model = model.model_name
        self._format = strict_response_format("Act", Act.model_json_schema())

    async def ceiling(self, system: str, user: str) -> float | None:
        # As the engine does before it prices a call: the read this task makes is the one the call is priced from.
        await self._model.refresh_prices()
        return self._model.price_ceiling(system=system, user=user, response_format=self._format)

    async def next(self, system: str, user: str, steps: Sequence[Step]) -> Turn:
        completion = await self._model.generate(system=system, user=user, response_format=self._format)
        usage = Usage(
            completion.input_tokens,
            completion.output_tokens,
            completion.cost_usd,
            maybe_billed_sends=completion.maybe_billed_sends,
        )
        if completion.stop_reason != "end_turn":
            return Turn(None, usage, f"the reply stopped short ({completion.stop_reason})")
        try:
            return Turn(Act.model_validate_json(completion.content), usage)
        except ValidationError as malformed:
            return Turn(None, usage, f"the reply was not one tool call: {malformed.errors()[0]['msg']}")


class ScriptedBrain:
    """A fixed transcript, one act per line, spending nothing.

    ``{package}`` and ``{llms}`` in a path or command are the build's; ``{first_line}`` in a report is the first line
    the last run printed, the one step a transcript cannot know ahead. Past its last line it says done.
    """

    model = "scripted"

    def __init__(self, transcript: Path, sandbox: Sandbox) -> None:
        self._acts = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
        self._names = {"package": str(sandbox.build.shown_package_dir), "llms": str(sandbox.build.shown_llms_txt)}

    async def ceiling(self, system: str, user: str) -> float | None:
        return 0.0

    async def next(self, system: str, user: str, steps: Sequence[Step]) -> Turn:
        index = len(steps)
        row = dict(self._acts[index]) if index < len(self._acts) else {"tool": "done"}
        runs = [step for step in steps if step.act is not None and step.act.tool == "run"]
        printed = runs[-1].observation.text.splitlines()[1:2] if runs else []
        names = {**self._names, "first_line": printed[0] if printed else ""}
        for key in ("path", "command", "text"):
            if key in row:
                row[key] = re.sub(r"\{(package|llms|first_line)\}", lambda m: names[m.group(1)], row[key])
        return Turn(Act.model_validate({**_BLANK, **row}), Usage(0, 0, 0.0))


@dataclass(frozen=True)
class Caps:
    """What bounds one run: turns, wall time, and dollars (``None``: a brain that spends nothing).

    ``wall_s`` is a deadline, held on every model call and command in flight, not only checked between turns: with
    the regrade's own budget (:data:`~.checks.REGRADE_BUDGET_S`) it fits inside the engine's cell budget.
    """

    turns: int = 40
    wall_s: float = 360.0
    cost_usd: float | None = None


#: How a run ended: the agent said done, or a cap stopped it, or a model call failed (``error``).
Stopped = Literal["done", "turns", "wall", "cap", "error"]


@dataclass
class Attempt:
    """One run of the agent, as it went."""

    steps: list[Step] = field(default_factory=list)
    usages: list[Usage] = field(default_factory=list)
    stopped: Stopped = "turns"
    report: str | None = None
    wall_s: float = 0.0

    def spent(self) -> float | None:
        """Dollars spent so far, as the provider reported them; ``None`` once any turn went unpriced."""
        costs = [usage.cost_usd for usage in self.usages]
        return None if any(cost is None for cost in costs) else sum(cost for cost in costs if cost is not None)

    def charged(self) -> float | None:
        """The most the run can have been billed (:meth:`Usage.charged`); ``None`` once any turn is unbounded."""
        charges = [usage.charged() for usage in self.usages]
        return None if any(c is None for c in charges) else sum(c for c in charges if c is not None)

    def maybe_billed_sends(self) -> int:
        """Sends that broke before any response began and were sent again, over the run."""
        return sum(usage.maybe_billed_sends for usage in self.usages)

    def tokens(self, which: Literal["input_tokens", "output_tokens"]) -> int | None:
        counts = [getattr(usage, which) for usage in self.usages]
        return None if any(count is None for count in counts) else sum(counts)


def prompt(steps: Sequence[Step]) -> str:
    """The user message for the next turn: the task, then every step, the latest :data:`RECENT_STEPS` whole."""
    parts = [TASK]
    for step in steps:
        act = "(no tool call)" if step.act is None else step.act.model_dump_json()
        said = step.observation.text
        if len(steps) - step.turn >= RECENT_STEPS:
            act, said = (_cut(text) for text in (act, said))
        parts.append(f"[turn {step.turn}] {act}\n{said}")
    return "\n\n".join(parts)


def _cut(text: str) -> str:
    """An old step's text, cut to its head."""
    return text if len(text) <= OLD_STEP_CHARS else text[:OLD_STEP_CHARS] + " [... cut; read it again if you need it]"


def _logged(log: Path | None, turn: int, usage: Usage | None, step: Step | None, stopped: str | None = None) -> None:
    """One line of the run's progress, written as it happens, so a run cut off keeps what it did."""
    if log is None:
        return
    row: dict[str, Any] = {"turn": turn}  # Any: JSON
    if usage is not None:
        row["usage"] = asdict(usage)
    if step is not None:
        row["act"] = None if step.act is None else step.act.model_dump()
        row["observation"] = asdict(step.observation)
    if stopped is not None:
        row["stopped"] = stopped
    with log.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row) + "\n")


async def attempt(sandbox: Sandbox, brain: Brain, caps: Caps, *, log: Path | None = None) -> Attempt:
    """Run the agent on ``sandbox`` until it says done, a cap stops it, or a model call fails.

    Each turn's usage and step are appended to ``log`` as they happen. A failed model call ends the run as ``error``
    with the failure as its last step; a run past ``caps.wall_s`` ends as ``wall``, whatever was in flight cut off.

    :raises ValueError: ``sandbox`` runs its commands on this machine and ``brain`` is not the scripted one: only a
        fixed transcript may run there.
    """
    if not sandbox.runner.isolated and not isinstance(brain, ScriptedBrain):
        raise ValueError(
            f"the sandbox runs commands on this machine, which only the scripted agent may do; {brain.model} runs in "
            "a trial container (DocsBuild.in_image)"
        )
    system = SYSTEM.format(lines=READ_LINES, package=sandbox.build.shown_package_dir, llms=sandbox.build.shown_llms_txt)
    run = Attempt()
    started = time.monotonic()
    try:
        async with asyncio.timeout(caps.wall_s):
            await _turns(sandbox, brain, caps, system, run, log)
    except TimeoutError:
        run.stopped = "wall"
    run.wall_s = round(time.monotonic() - started, 3)
    _logged(log, len(run.usages), None, None, run.stopped)
    return run


async def _turns(sandbox: Sandbox, brain: Brain, caps: Caps, system: str, run: Attempt, log: Path | None) -> None:
    """The run's turns, into ``run``; a model call cut off by the wall is charged as an unknown cost first."""
    for turn in range(1, caps.turns + 1):
        user = prompt(run.steps)
        bound: float | None = None
        if caps.cost_usd is not None:
            bound, charged = await brain.ceiling(system, user), run.charged()
            if bound is None or charged is None or charged + BILLED_SENDS_AT_MOST * bound > caps.cost_usd:
                run.stopped = "cap"
                return
        try:
            answer = await brain.next(system, user, run.steps)
        except (ProviderError, httpx.HTTPError) as broke:  # a decline, or a send that broke and was not resent
            why = f"the model call failed: {broke}"
            usage = Usage(None, None, None, ceiling_usd=bound)
            step = Step(turn, None, Observation(ok=False, text=why, first_error=why))
            run.usages.append(usage)
            run.steps.append(step)
            _logged(log, turn, usage, step)
            run.stopped = "error"
            return
        except asyncio.CancelledError:
            # Cut off by the wall (or the cell) mid-call: it may have been billed, and no cost came back.
            run.usages.append(Usage(None, None, None, ceiling_usd=bound))
            _logged(log, turn, run.usages[-1], None)
            raise
        usage = replace(answer.usage, ceiling_usd=bound)
        run.usages.append(usage)
        act = answer.act
        if act is None:
            step = Step(turn, None, Observation(ok=False, text=answer.problem, first_error=answer.problem))
            run.steps.append(step)
            _logged(log, turn, usage, step)
            continue
        if act.tool == "done":
            run.stopped, run.report = "done", act.text
            _logged(log, turn, usage, Step(turn, act, Observation(ok=True, text="")))
            return
        sandbox.turn = turn
        if act.tool == "read":
            observation = sandbox.read(act.path, act.start)
        elif act.tool == "write":
            observation = sandbox.write(act.path, act.text)
        else:
            observation = await sandbox.run(act.command)
        run.steps.append(Step(turn, act, observation))
        _logged(log, turn, usage, run.steps[-1])
    run.stopped = "turns"


#: The newcomer's errors the docs and messages are measured on, each by an id and what its message says, read off what
#: each run printed: a run met an error when the package said it, whatever caused it.
ERRORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("no_key", re.compile(r"no key for \w+")),
    ("cases_missing", re.compile(r"the cases file .+ is not there")),
    ("eval_not_at_module_level", re.compile(r"defines no Eval at module level")),
    ("grader_unnamed", re.compile(r"a (?:grader|scorer)'s __name__ names its measure")),
    ("field_not_in_answer", re.compile(r"answered no \S+ it can be graded on")),
    ("runner_raised", re.compile(r"the candidate raised \w+")),
    ("grader_raised", re.compile(r"the scorer \w+ raised \w+")),
    ("eval_unnamed", re.compile(r"name this eval \(name=")),
    ("recording_refused", re.compile(r"RecordingRefused|recording \S+: .+; ")),
)


def errors_hit(steps: Sequence[Step]) -> dict[str, int]:
    """Each newcomer error a run met, by id, with the first turn it appeared on. ``help_first`` is the door's
    ``--help``, which today points away from the file door."""
    hit: dict[str, int] = {}
    for step in steps:
        if step.act is None or step.act.tool != "run":
            continue
        argv = step.act.command.split()
        if argv[:3] == ["python", "-m", "threetears.evals"] and (len(argv) == 3 or argv[3] in ("-h", "--help")):
            hit.setdefault("help_first", step.turn)
        for name, pattern in ERRORS:
            if pattern.search(step.observation.text):
                hit.setdefault(name, step.turn)
    return hit


def failed(step: Step) -> bool:
    """Whether a turn failed: a reply that was no tool call, a call refused, or a run that ended badly."""
    return not step.observation.ok


def stuck_point(steps: Sequence[Step]) -> dict[str, Any] | None:  # Any: a span's turns, count and message
    """The longest span of consecutive failing turns, and the message that opened it; ``None`` when none failed.

    Only a run of the eval door that worked ends a span. Anything else that worked (a read, a write, ``ls``, ``cat``, a
    probe script) neither extends nor ends it: it is the agent looking for help, or trying a fix. Ties go to the
    earliest span.
    """
    spans: list[list[Step]] = []
    current: list[Step] = []
    for step in steps:
        if failed(step):
            current.append(step)
        elif step.act is not None and step.act.tool == "run" and is_door(step.act.command.split()) and current:
            spans.append(current)
            current = []
    if current:
        spans.append(current)
    if not spans:
        return None
    worst = max(spans, key=len)
    opened = worst[0].observation
    return {
        "from_turn": worst[0].turn,
        "to_turn": worst[-1].turn,
        "failed_turns": len(worst),
        "opened_by": opened.first_error or opened.text.splitlines()[0],
    }


def commands(steps: Sequence[Step]) -> list[dict[str, Any]]:  # Any: a command's fields
    """Every command run, with its exit code and its first stderr line (or why it was refused)."""
    return [
        {
            "turn": step.turn,
            "command": step.act.command,
            "exit_code": step.observation.exit_code,
            "first_error": step.observation.first_error,
            "door": is_door(step.act.command.split()),
        }
        for step in steps
        if step.act is not None and step.act.tool == "run"
    ]


def parse_report(text: str | None) -> tuple[dict[str, Any] | None, str]:  # Any: the agent's JSON
    """The agent's final report as a mapping, or ``None`` and why it is none."""
    if text is None:
        return None, "the run ended before the agent reported"
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        report = json.loads(match.group(0) if match else text)
    except json.JSONDecodeError as unreadable:
        return None, f"the report is not JSON: {unreadable.msg}"
    if not isinstance(report, Mapping):
        return None, "the report is not a JSON object"
    return dict(report), ""
