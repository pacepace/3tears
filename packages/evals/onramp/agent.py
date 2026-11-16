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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from threetears.evals import Model
from threetears.evals.kernel import strict_response_format
from threetears.evals.negotiation import MAX_RESENDS
from threetears.evals.providers import ProviderError

from .sandbox import READ_LINES, Observation, Sandbox, is_door

#: The sentence the agent is given, and the report it is asked for.
TASK = """\
Add an eval to this app with 3tears-evals, run it, and report what it found as the JSON below.

{"eval_file": "<the eval file you wrote, relative to the app's directory>",
 "first_line": "<the first line your last run of the eval printed, verbatim and whole>",
 "measure": "<the measure pass_rate reads, named as the run printed it>",
 "pass_rate": <that measure on your eval's first runner, a share from 0 to 1>,
 "finished": <true when that run finished, false when it did not>}"""

#: The most tokens one reply may hold, its reasoning included.
REPLY_TOKENS = 16_384

SYSTEM = """\
You work in an app's directory, which you have not seen before, with one tool call per turn, answered as one JSON \
object: {{"tool": ..., "path": ..., "start": ..., "text": ..., "command": ...}}, every field present and the ones \
the tool does not read left empty ("" or 1).
- read: the file at `path` (in the app's directory, or in the installed package), {lines} lines from line `start`.
- write: `text` to the file at `path`, relative to the app's directory.
- run: `command`, one of `python ...`, `python -m threetears.evals ...`, `ls [path]`, `cat path`; no shell, no pipes.
- done: your final report, the JSON the task asks for, as `text`; it ends the task.
Your working directory is the app's directory, shown as `.`: give paths relative to it. Nothing outside it and the \
installed package can be read. 3tears-evals is installed at {package}; its guide for agents is {llms}. Your \
commands run in a container with no network and no API key; the app runs offline. A reply holds at most {tokens} \
tokens, your reasoning included. The task stops after {wall:.0f} seconds or {turns} turns, whichever comes first, \
with no report unless you have sent done; each turn takes 10 to 40 seconds."""

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


#: The fields each tool reads: a reply may leave out any other, which is then empty.
READS: Mapping[str, tuple[str, ...]] = {"read": ("path",), "write": ("path", "text"), "run": ("command",), "done": ()}
#: The keys of the report the task asks for.
REPORT_KEYS = frozenset({"eval_file", "first_line", "measure", "pass_rate", "finished"})
#: How a reply that was no tool call is answered: what the shape is.
SHAPE = (
    'send one JSON object a turn, {"tool": "read"|"write"|"run"|"done", "path": ..., "start": ..., "text": ..., '
    '"command": ...}, with the report as the text of done'
)


def _objects(content: str) -> list[Any]:  # Any: JSON values
    """Every JSON value ``content`` holds one after another, or ``[]`` when it holds none whole."""
    decoder, found, at = json.JSONDecoder(), [], 0
    while at < len(content):
        if content[at].isspace():
            at += 1
            continue
        try:
            value, at = decoder.raw_decode(content, at)
        except json.JSONDecodeError:
            return []
        found.append(value)
    return found


def read_reply(content: str) -> tuple[Act | None, str, str]:
    """A model's reply read as one tool call: the act, or ``None`` and why it is none, and how it was read.

    Where a reply's intent is unambiguous it is taken: a field the tool does not read may be left out, and a bare
    report (a JSON object holding the report's keys and no other tool) is the ``done`` it means (read as ``"bare
    report"``). Anything else that is no tool call (two objects, a field no tool has, a tool that is none, a field its
    tool reads left out) is refused, naming what was wrong and the shape: never guessed.
    """
    values = _objects(content)
    if len(values) != 1:
        held = "no JSON object" if not values else f"{len(values)} JSON values"
        return None, f"the reply held {held}; {SHAPE}", ""
    (value,) = values
    if not isinstance(value, dict):
        return None, f"the reply is not a JSON object; {SHAPE}", ""
    fields = set(Act.model_fields)
    extra = sorted(set(value) - fields)
    if extra and set(extra) <= REPORT_KEYS and value.get("tool", "done") == "done":
        report = {key: value[key] for key in value if key in REPORT_KEYS}
        return Act.model_validate({**_BLANK, "tool": "done", "text": json.dumps(report)}), "", "bare report"
    if extra:
        return None, f"{', '.join(extra)} is no field of a tool call; {SHAPE}", ""
    tool = value.get("tool")
    if tool not in READS:
        return None, f"tool {tool!r} is none of read, write, run, done; {SHAPE}", ""
    missing = [name for name in READS[tool] if name not in value]
    if missing:
        return None, f"{tool} reads {', '.join(missing)}, which the reply left out; {SHAPE}", ""
    try:
        return Act.model_validate({**_BLANK, **value}), "", ""
    except ValidationError as malformed:
        error = malformed.errors()[0]
        where = ".".join(str(part) for part in error["loc"])
        return None, f"{where}: {error['msg']}; {SHAPE}", ""


def _holds_object(text: str) -> bool:
    """Whether ``text`` holds a JSON object, as the report is read from it."""
    report, _ = parse_report(text)
    return report is not None


def done_report(act: Act) -> tuple[str, str]:
    """The report a ``done`` gives, and the field it was read from: ``text``, as asked; or ``command``, when ``text``
    holds no JSON object and ``command`` does (the report in the wrong field, its intent plain)."""
    if not _holds_object(act.text) and _holds_object(act.command):
        return act.command, "command"
    return act.text, "text"


_BLANK: dict[str, Any] = {"path": "", "start": 1, "text": "", "command": ""}  # Any: a field's empty value


#: How many sends of one call may be billed, at most. A call is up to ``MAX_RESENDS + 1`` requests: the negotiation
#: sends a request again for each setting a provider refused, ``MAX_RESENDS`` times at most. The client sends a request
#: again once when its first send broke before any response began, and that send may have been served. A refusal or a
#: decline is not billed, so each refused request may have been billed once (its first send, broken), and the last
#: twice (its first send, broken, and the send that was served or lost): ``MAX_RESENDS + 2``. A call is sent only when
#: the run's charge plus this many of its ceiling stays within the cap, and :class:`SendMeter` refuses any send past it,
#: so the cap bounds what the run can be billed.
BILLED_SENDS_AT_MOST = MAX_RESENDS + 2
#: The statuses of an answer the provider may have billed: a served call, and a gateway's timeout (the provider may
#: still have served it). Every other status is a refusal or a decline, which is not billed.
BILLED_STATUSES = frozenset({200, 504, 524})


class SendsSpent(httpx.RequestError):
    """A send refused before it went: the call it belongs to has made every send its reservation covers."""


@dataclass
class CallSends:
    """One call's sends, as :class:`SendMeter` counts them: every send made, and those that may have been billed."""

    made: int = 0
    maybe_billed: int = 0


#: The call the current task is making, whose sends the meter counts; ``None`` outside a call.
_CALL: ContextVar[CallSends | None] = ContextVar("onramp_call", default=None)


@contextmanager
def metered_call() -> Iterator[CallSends]:
    """One call: every send made inside it, in this task, is counted on the :class:`CallSends` it yields."""
    sends = CallSends()
    counting = _CALL.set(sends)
    try:
        yield sends
    finally:
        _CALL.reset(counting)


class SendMeter(httpx.AsyncBaseTransport):
    """The live model's transport: it counts each call's sends that may have been billed, here, as they go, and
    refuses a send past :data:`BILLED_SENDS_AT_MOST` before it goes.

    A send may have been billed when it brought an answer the provider may have billed (:data:`BILLED_STATUSES`) or
    no answer at all (it broke, or was cut off). The count is the harness's own, never what a reply reports: what a run
    is charged rests on it (:class:`ModelBrain`). A send outside a call (a catalogue read) passes uncounted.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._inner = inner if inner is not None else httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        call = _CALL.get()
        if call is None or request.method != "POST":
            return await self._inner.handle_async_request(request)
        if call.maybe_billed >= BILLED_SENDS_AT_MOST:
            raise SendsSpent(
                f"the call has made {call.maybe_billed} sends that may have been billed, every one its reservation "
                "covers; it is not sent again",
                request=request,
            )
        call.made += 1
        try:
            response = await self._inner.handle_async_request(request)
        except BaseException:
            call.maybe_billed += 1  # sent, and no answer came: it may have been served
            raise
        if response.status_code in BILLED_STATUSES:
            call.maybe_billed += 1
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


@dataclass(frozen=True)
class Usage:
    """What one turn spent. ``None`` is unreported, never zero.

    :ivar ceiling_usd: The most the call could cost, bounded before it was sent; ``None`` when no cap asked.
    :ivar maybe_billed_sends: Sends of the call that may have been billed besides the one ``cost_usd`` prices: each
        broke, or was cut off, before its answer came.
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
    """A brain's answer: the act, or ``None`` with why the reply was no act, and what asking cost.

    :ivar reply: The reply as the model gave it, when it was read as no act, or read as one it was not written as:
        kept so the run can be read again.
    :ivar read_as: How the act was read when not as written (``"bare report"``); empty otherwise.
    """

    act: Act | None
    usage: Usage
    problem: str = ""
    reply: str = ""
    read_as: str = ""


@dataclass(frozen=True)
class Step:
    """One turn as it happened: the act (``None`` for a reply that was none) and what the tool did.

    :ivar reply: The model's reply, when it was read as no act or not as written (:attr:`Turn.reply`).
    """

    turn: int
    act: Act | None
    observation: Observation
    reply: str = ""


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
    and inside the harness's runner its spend lands on the trial's own ledger.

    The model is built by ``model_on`` on the transport it must send through, a :class:`SendMeter` over ``inner``
    (``httpx``'s own when ``None``; a test's mock): the meter counts each call's sends that may have been billed, which
    is what the call is charged on, and refuses one past the reservation.
    """

    def __init__(
        self, model_on: Callable[[SendMeter], Model], *, inner: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._model = model_on(SendMeter(inner))
        self.model = self._model.model_name
        self._format = strict_response_format("Act", Act.model_json_schema())

    async def aclose(self) -> None:
        """Close the model, and with it the client it opened on the meter."""
        await self._model.aclose()

    async def ceiling(self, system: str, user: str) -> float | None:
        # As the engine does before it prices a call: the read this task makes is the one the call is priced from.
        await self._model.refresh_prices()
        return self._model.price_ceiling(system=system, user=user, response_format=self._format)

    async def next(self, system: str, user: str, steps: Sequence[Step]) -> Turn:
        with metered_call() as sends:
            completion = await self._model.generate(system=system, user=user, response_format=self._format)
        # Every send the meter counted may have been billed; the one served is priced in cost_usd. A replayed call was
        # never sent, and counts none.
        priced = 1 if completion.cost_usd is not None and sends.maybe_billed else 0
        usage = Usage(
            completion.input_tokens,
            completion.output_tokens,
            completion.cost_usd,
            maybe_billed_sends=sends.maybe_billed - priced,
        )
        if completion.stop_reason != "end_turn":
            why = f"the reply stopped short ({completion.stop_reason}) at {REPLY_TOKENS} tokens, reasoning included"
            return Turn(None, usage, why, completion.content)
        act, problem, read_as = read_reply(completion.content)
        if act is None:
            return Turn(None, usage, f"the reply was not one tool call: {problem}", completion.content)
        return Turn(act, usage, reply=completion.content if read_as else "", read_as=read_as)


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
    #: As long as the engine's cell budget leaves after the regrade and the copies' stop (600 - 90 - 2 x 20, less a
    #: margin): ``Eval`` takes no longer cell, so the design's 15 minutes cannot be had here.
    wall_s: float = 450.0
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
    #: Where the report was read from: ``text`` as asked, ``command``, or a ``bare report`` (:func:`read_reply`).
    report_from: str = ""
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


#: The share of the wall cap past which each prompt says how much time is left.
WARN_AT = 0.8


def prompt(steps: Sequence[Step], left_s: float | None = None) -> str:
    """The user message for the next turn: the task, then every step, the latest :data:`RECENT_STEPS` whole, and, when
    ``left_s`` is given, how many seconds are left before the wall cap stops the run."""
    parts = [TASK]
    for step in steps:
        act = "(no tool call)" if step.act is None else step.act.model_dump_json()
        said = step.observation.text
        if len(steps) - step.turn >= RECENT_STEPS:
            act, said = (_cut(text) for text in (act, said))
        parts.append(f"[turn {step.turn}] {act}\n{said}")
    if left_s is not None:
        parts.append(
            f"[time] {max(left_s, 0.0):.0f} seconds are left before the task stops with no report: if you have run "
            "your eval, send done with its report now."
        )
    return "\n\n".join(parts)


def _cut(text: str) -> str:
    """An old step's text, cut to its head."""
    return text if len(text) <= OLD_STEP_CHARS else text[:OLD_STEP_CHARS] + " [... cut; read it again if you need it]"


def _logged(
    log: Path | None,
    turn: int,
    usage: Usage | None,
    step: Step | None,
    stopped: str | None = None,
    files: Mapping[str, str] | None = None,
) -> None:
    """One line of the run's progress, written as it happens, so a run cut off keeps what it did; after a turn that
    may change the directory, its state (:meth:`~.sandbox.Sandbox.manifest`), so the run can be graded again as it
    stood at any turn."""
    if log is None:
        return
    row: dict[str, Any] = {"turn": turn}  # Any: JSON
    if usage is not None:
        row["usage"] = asdict(usage)
    if step is not None:
        row["act"] = None if step.act is None else step.act.model_dump()
        row["observation"] = asdict(step.observation)
        if step.reply:
            row["reply"] = step.reply
    if stopped is not None:
        row["stopped"] = stopped
    if files is not None:
        row["files"] = dict(files)
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
    system = SYSTEM.format(
        lines=READ_LINES,
        package=sandbox.build.shown_package_dir,
        llms=sandbox.build.shown_llms_txt,
        tokens=REPLY_TOKENS,
        wall=caps.wall_s,
        turns=caps.turns,
    )
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
    """The run's turns, into ``run``; a model call cut off by the wall is charged as an unknown cost first. Past
    :data:`WARN_AT` of the wall cap, each prompt says how many seconds are left (a recording of a run that got there
    replays up to that turn)."""
    started = time.monotonic()
    for turn in range(1, caps.turns + 1):
        elapsed = time.monotonic() - started
        user = prompt(run.steps, caps.wall_s - elapsed if elapsed >= WARN_AT * caps.wall_s else None)
        bound: float | None = None
        if caps.cost_usd is not None:
            bound, charged = await brain.ceiling(system, user), run.charged()
            # A turn the provider reported no price for stops the run too: what the rest would cost is unknowable.
            unpriced = run.spent() is None
            if bound is None or charged is None or unpriced or charged + BILLED_SENDS_AT_MOST * bound > caps.cost_usd:
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
            failure = Observation(ok=False, text=answer.problem, first_error=answer.problem)
            step = Step(turn, None, failure, answer.reply)
            run.steps.append(step)
            _logged(log, turn, usage, step)
            continue
        if act.tool == "done":
            run.stopped = "done"
            run.report, run.report_from = done_report(act)
            run.report_from = answer.read_as or run.report_from
            _logged(log, turn, usage, Step(turn, act, Observation(ok=True, text=""), answer.reply))
            return
        sandbox.turn = turn
        if act.tool == "read":
            observation = sandbox.read(act.path, act.start)
        elif act.tool == "write":
            observation = sandbox.write(act.path, act.text)
        else:
            observation = await sandbox.run(act.command)
        run.steps.append(Step(turn, act, observation))
        changes = act.tool == "write" or (act.tool == "run" and act.command.split()[:1] in (["python"], ["python3"]))
        _logged(log, turn, usage, run.steps[-1], files=sandbox.manifest() if changes else None)
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


#: What a door run prints when it scored no result: every case failed by the candidate.
NOTHING_SCORED = re.compile(r"\d+ result\(s\): 0 scored\b")


def scored_nothing(step: Step) -> bool:
    """Whether a turn ran the eval door and it scored no result (exit 0, every case failed by the candidate)."""
    return (
        step.act is not None
        and step.act.tool == "run"
        and is_door(step.act.command.split())
        and bool(NOTHING_SCORED.search(step.observation.text))
    )


def failed(step: Step) -> bool:
    """Whether a turn failed: a reply that was no tool call, a call refused, a run that ended badly, or a door run that
    scored nothing (it exits 0 with every case failed)."""
    return not step.observation.ok or scored_nothing(step)


def scored_door_turns(steps: Sequence[Step]) -> list[int]:
    """The turns whose door run worked and scored a result: where the agent had an eval that measured something."""
    return [
        step.turn
        for step in steps
        if step.act is not None and step.act.tool == "run" and is_door(step.act.command.split())
        and step.observation.ok and not scored_nothing(step)
    ]  # fmt: skip


#: What caused a failure, as far as the harness can tell it mechanically: the harness's own rules and limits (a reply
#: that was no tool call, a sandbox path, a refused command, a token or time limit, a model call that failed), the docs
#: (a file the guide names that is not there), the agent (a report sent with no eval written), or :data:`CANNOT_TELL`.
#: The package's messages and most of the agent's own mistakes are told apart only by reading the run.
HARNESS, DOCS, AGENT, CANNOT_TELL = "harness", "docs", "agent", "cannot tell"
CAUSES = (HARNESS, DOCS, AGENT, CANNOT_TELL)
#: What a failed turn's first error says when the harness's own rules or limits failed it.
_HARNESS_SAID = (
    "the reply was not one tool call",
    "the reply stopped short",
    "the model call failed",
    "is not in the app's directory or the installed package",
    "is outside the app's directory and the installed package",
    "is not a file in the app's directory",
    "run takes python",
    "the command does not parse",
    "cat takes one file",
    "timed out after",
    "could not be copied",
    "a read takes",
)
_MISSING = re.compile(r"^(?P<path>\S+): no such file$")


def cause(step: Step, guide: str = "") -> str:
    """What caused a failed turn, when the harness can tell it mechanically (:data:`CAUSES`): its own rules and limits,
    or the docs for a read of a file the build's guide (``guide``, its text) names and the build does not ship."""
    said = step.observation.first_error or step.observation.text
    if step.act is None or any(phrase in said for phrase in _HARNESS_SAID):
        return HARNESS
    missing = _MISSING.match(said)
    if missing is not None and len(named := Path(missing["path"]).parts[-2:]) == 2 and "/".join(named) in guide:
        return DOCS  # the guide names the file, by its directory and name, and the build does not have it
    return CANNOT_TELL


def stuck_point(steps: Sequence[Step], guide: str = "") -> dict[str, Any] | None:  # Any: a span's fields
    """The longest span of consecutive failing turns, the message that opened it and its :func:`cause`; ``None`` when
    none failed.

    Only a run of the eval door that worked and scored ends a span (one that scored nothing is a failed turn). Anything else that worked (a read, a write, ``ls``, ``cat``, a
    probe script) neither extends nor ends it: it is the agent looking for help, or trying a fix. Ties go to the
    earliest span.
    """
    spans: list[list[Step]] = []
    current: list[Step] = []
    for step in steps:
        if failed(step):
            current.append(step)
        elif step.act is not None and step.act.tool == "run" and is_door(step.act.command.split()) and current:
            # failed() was false: a door run that worked and scored, which ends the span.
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
        "cause": cause(worst[0], guide),
    }


def format_error_turns(steps: Sequence[Step]) -> list[int]:
    """The turns whose reply was refused as no tool call, or cut off at the token limit: the harness's format rules."""
    return [step.turn for step in steps if step.act is None and not step.observation.text.startswith("the model call")]


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
