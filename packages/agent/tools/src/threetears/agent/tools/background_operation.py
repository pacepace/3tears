"""A long operation a pod runs in the background: started (or requested) by one tool, reported by another.

**Why two tools, not one.** Some operations a tool pod runs on request take minutes (a
load from a warehouse, a rebuild of every map layer), longer than the hub waits on a tool
call. A tool that awaited one would answer every successful run with a timeout. So the
start tool begins the run and answers at once, and the status tool says whether it is
running and how the last one ended.

**One run at a time, and one way in.** :meth:`BackgroundOperation.start` is the only way
to begin a run, whether a pod starts one itself (a first load when its tables are empty,
:meth:`BackgroundOperation.start_when_needed`) or a caller asks through a tool, so
:attr:`~BackgroundOperation.running` and :attr:`~BackgroundOperation.last` describe every
run. Two tools start one, for two kinds of operation:

- :class:`StartOperationTool`, for an operation a second run of would repeat: a start while one
  runs is refused, answered :data:`~threetears.agent.tools.base_tool.CONFLICT`.
- :class:`RequestOperationTool`, for an operation that drains requests (its body a
  ``CoalescedRun.drain``, built with ``request=``): :meth:`BackgroundOperation.request` records the
  request and starts a run, and a request while one runs is never refused nor lost: the run in
  progress is followed by one more, which takes it (the drain may already have made its last look
  for requests).

:class:`OperationStatusTool` reports either kind.

**A run that raises is kept as failed, with its reason.** The run is a background task
with no caller to raise to, so its exception is recorded on the outcome and logged; the
status tool reports it until the next run ends.

**The outcome lives in this process only.** A restarted pod reports ``idle`` until its
next run ends. An operation whose result must outlive the process records it in its own
tables; this keeps only what the status tool reports.

Usage::

    load = BackgroundOperation("enr-load", loader.load)
    server.register(StartOperationTool(name="enr.reload", description="...",
                                       operation=lambda: load, status_tool="enr.load_status"))
    server.register(OperationStatusTool(name="enr.load_status", description="...",
                                        operation=lambda: load))
    load.start_when_needed(loader.needs_load, retry_on=(DataLayerUnavailableError,), retry_seconds=30)
    ...
    await load.stop()  # on shutdown

The tools take the operation through a callable, so a pod can register them before the
operation exists (its storage is reached only after it connects); until then both refuse
the call, saying so.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Generic, Literal, TypeVar

from threetears.observe import get_logger

from threetears.agent.tools.base_tool import CONFLICT, TOOL_NOT_READY, MCPToolDefinition, TearsTool, ToolResult

__all__ = [
    "BackgroundOperation",
    "OperationOutcome",
    "OperationState",
    "OperationStatusTool",
    "RequestOperationTool",
    "StartOperationTool",
]

log = get_logger(__name__)

ResultT = TypeVar("ResultT")

#: where an operation stands: never run since the process started, running, or how the last run ended
OperationState = Literal["idle", "running", "succeeded", "failed"]

#: the version both tools carry unless the pod names another
_DEFAULT_TOOL_VERSION: Final = "1.0"

#: what both tools answer before the pod has built the operation
_NOT_READY: Final = "the pod is not ready to run this yet (its storage is still being reached); try again shortly"


@dataclass(frozen=True)
class OperationOutcome(Generic[ResultT]):
    """how one run ended.

    :ivar started_at: when the run began
    :ivar finished_at: when it ended
    :ivar result: what the run returned; None when it failed
    :ivar error: why it failed, as ``<ExceptionType>: <message>``; None when it succeeded
    """

    started_at: datetime
    finished_at: datetime
    result: ResultT | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """whether the run returned rather than raised.

        :return: True for a run that returned
        :rtype: bool
        """
        return self.error is None


class BackgroundOperation(Generic[ResultT]):
    """one operation, run in the background at most once at a time, its last outcome kept.

    :param name: the operation's name, for task names and logs
    :ptype name: str
    :param run: the operation's body; each start awaits a fresh call of it
    :ptype run: Callable[[], Awaitable[ResultT]]
    :param request: records that a run is wanted (``CoalescedRun.request``), for an operation whose
        body drains requests; :meth:`request` needs it
    :ptype request: Callable[[], Awaitable[None]] | None
    """

    def __init__(
        self,
        name: str,
        run: Callable[[], Awaitable[ResultT]],
        *,
        request: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._name = name
        self._run = run
        self._request = request
        self._again = False
        self._task: asyncio.Task[None] | None = None
        self._watcher: asyncio.Task[None] | None = None
        self._started_at: datetime | None = None
        self._last: OperationOutcome[ResultT] | None = None

    @property
    def name(self) -> str:
        """the operation's name.

        :return: the name
        :rtype: str
        """
        return self._name

    @property
    def running(self) -> bool:
        """whether a run is in progress.

        :return: True while one runs
        :rtype: bool
        """
        return self._task is not None and not self._task.done()

    @property
    def started_at(self) -> datetime | None:
        """when the run in progress began; None when none runs.

        :return: the start time
        :rtype: datetime | None
        """
        return self._started_at if self.running else None

    @property
    def last(self) -> OperationOutcome[ResultT] | None:
        """how the last finished run ended; None before one has finished in this process.

        :return: the outcome
        :rtype: OperationOutcome[ResultT] | None
        """
        return self._last

    @property
    def state(self) -> OperationState:
        """running, or how the last run ended, or idle before any has.

        :return: the state
        :rtype: OperationState
        """
        state: OperationState
        if self.running:
            state = "running"
        elif self._last is None:
            state = "idle"
        elif self._last.succeeded:
            state = "succeeded"
        else:
            state = "failed"
        return state

    def start(self) -> bool:
        """begin a run in the background, unless one is running.

        :return: True when a run was started; False when one was already running
        :rtype: bool
        """
        started = not self.running
        if started:
            # a run that starts now sees every request already recorded
            self._again = False
            self._started_at = datetime.now(UTC)
            self._task = asyncio.create_task(self._run_and_keep_outcome(), name=self._name)
            log.info("background operation started", extra={"extra_data": {"operation": self._name}})
        return started

    async def request(self) -> bool:
        """record that a run is wanted, then start one, or have the run in progress run once more.

        The request is recorded first, so a run on another replica may take it too; a run in progress
        here is followed by one more, since it may already have made its last look for requests. Its
        end checks for that in the same step it ends in, so no request falls between.

        :return: True when a run was started; False when the run in progress will run once more
        :rtype: bool
        :raises TypeError: when the operation was built without ``request=``
        """
        if self._request is None:
            raise TypeError(f"{self._name}: built without request=, so it cannot take a request")
        await self._request()
        started = self.start()
        if not started:
            self._again = True
        return started

    async def wait(self) -> None:
        """wait for the run in progress, if any, to end.

        :return: nothing
        :rtype: None
        """
        if self._task is not None:
            await self._task

    def start_when_needed(
        self,
        needed: Callable[[], Awaitable[bool]],
        *,
        retry_on: tuple[type[BaseException], ...],
        retry_seconds: float,
    ) -> bool:
        """in the background: ask ``needed`` until it answers, and start a run if it says yes.

        For a pod's first run, when it cannot know yet whether one is needed (its tables are
        out of reach until the hub grants it storage). Only the errors in ``retry_on`` are
        retried; any other ends the wait, is kept as the last outcome (the status reports it failed, with
        its reason) and is raised from :meth:`wait_until_settled`. If a
        run was started some other way meanwhile, that run is the one; none is added. While
        one such wait is still deciding, another is refused, so :meth:`stop` and
        :meth:`wait_until_settled` always reach the wait that is running.

        :param needed: whether a run is needed now
        :ptype needed: Callable[[], Awaitable[bool]]
        :param retry_on: the errors that mean "not reachable yet", retried after ``retry_seconds``
        :ptype retry_on: tuple[type[BaseException], ...]
        :param retry_seconds: the wait between attempts
        :ptype retry_seconds: float
        :return: True when the wait started; False when one was already deciding
        :rtype: bool
        """
        started = self._watcher is None or self._watcher.done()
        if started:
            self._watcher = asyncio.create_task(
                self._start_when_needed(needed, retry_on=retry_on, retry_seconds=retry_seconds),
                name=f"{self._name}-when-needed",
            )
        else:
            log.warning(
                "a wait to start the operation is already deciding; this one is refused",
                extra={"extra_data": {"operation": self._name}},
            )
        return started

    async def wait_until_settled(self) -> None:
        """wait for :meth:`start_when_needed` to decide, and for any run it started to end.

        :return: nothing
        :rtype: None
        :raises Exception: what ``needed`` raised that ``retry_on`` did not name
        """
        if self._watcher is not None:
            await self._watcher
        await self.wait()

    async def stop(self) -> None:
        """cancel the wait of :meth:`start_when_needed` and the run in progress, if any.

        :return: nothing
        :rtype: None
        """
        for task in (self._watcher, self._task):
            if task is not None and not task.done():
                task.cancel()
                # NOSILENT: the cancellation requested on the line above, awaited so the task ends first
                with suppress(asyncio.CancelledError):
                    await task

    def status(self, render_result: Callable[[ResultT], Any] | None = None) -> dict[str, Any]:
        """where the operation stands, as plain JSON values.

        :param render_result: turns a result into JSON values; the result as it is when None
        :ptype render_result: Callable[[ResultT], Any] | None
        :return: ``state``, ``running``, ``started_at`` (of the run in progress) and ``last``
            (``started_at``, ``finished_at``, ``result``, ``error``), times in ISO 8601
        :rtype: dict[str, Any]
        """
        last = self._last
        started = self.started_at
        rendered: dict[str, Any] | None = None
        if last is not None:
            result: Any = last.result
            if result is not None and render_result is not None:
                result = render_result(result)
            rendered = {
                "started_at": last.started_at.isoformat(),
                "finished_at": last.finished_at.isoformat(),
                "result": result,
                "error": last.error,
            }
        return {
            "state": self.state,
            "running": self.running,
            "started_at": None if started is None else started.isoformat(),
            "last": rendered,
        }

    async def _start_when_needed(
        self,
        needed: Callable[[], Awaitable[bool]],
        *,
        retry_on: tuple[type[BaseException], ...],
        retry_seconds: float,
    ) -> None:
        """the body of :meth:`start_when_needed`.

        :param needed: whether a run is needed now
        :ptype needed: Callable[[], Awaitable[bool]]
        :param retry_on: the errors retried
        :ptype retry_on: tuple[type[BaseException], ...]
        :param retry_seconds: the wait between attempts
        :ptype retry_seconds: float
        :return: nothing
        :rtype: None
        """
        answer: bool | None = None
        waiting_since = datetime.now(UTC)
        while answer is None:
            try:
                answer = await needed()
            except retry_on as exc:
                log.warning(
                    "cannot tell yet whether the operation is needed; retrying",
                    extra={
                        "extra_data": {
                            "operation": self._name,
                            "error": f"{type(exc).__name__}: {exc}",
                            "retry_seconds": retry_seconds,
                        }
                    },
                )
                await asyncio.sleep(retry_seconds)
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- kept, logged and re-raised: the wait may never be awaited, so the status and the log are where this error is sure to be seen
                error = f"{type(exc).__name__}: {exc}"
                # kept as the last outcome, so the status says failed with the reason rather than idle;
                # unless a run ended after the wait began, which is newer news than this failure
                if self._last is None or self._last.finished_at < waiting_since:
                    self._last = OperationOutcome(started_at=waiting_since, finished_at=datetime.now(UTC), error=error)
                log.error(
                    "cannot tell whether the operation is needed; no run will start",
                    extra={"extra_data": {"operation": self._name, "error": error}},
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                raise
        if answer:
            self.start()
        else:
            log.info("background operation not needed", extra={"extra_data": {"operation": self._name}})

    async def _run_and_keep_outcome(self) -> None:
        """run, once more for each request made meanwhile, and keep how the last run ended.

        :return: nothing
        :rtype: None
        """
        again = True
        while again:
            await self._run_once()
            # read and cleared in the step the task ends in: a request after this finds the task done
            # and starts a run of its own
            again, self._again = self._again, False
            if again:
                self._started_at = datetime.now(UTC)
                log.info(
                    "background operation runs once more for a request", extra={"extra_data": {"operation": self._name}}
                )

    async def _run_once(self) -> None:
        """run once and keep how it ended, a run that raised included.

        :return: nothing
        :rtype: None
        """
        started = self._started_at or datetime.now(UTC)
        try:
            result = await self._run()
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- a background run has no caller to raise to; the failure is logged and kept for the status tool
            error = f"{type(exc).__name__}: {exc}"
            self._last = OperationOutcome(started_at=started, finished_at=datetime.now(UTC), error=error)
            log.error(
                "background operation failed",
                extra={"extra_data": {"operation": self._name, "error": error}},
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        else:
            self._last = OperationOutcome(started_at=started, finished_at=datetime.now(UTC), result=result)
            log.info("background operation finished", extra={"extra_data": {"operation": self._name}})


class _OperationTool(TearsTool):
    """what both tools share: a name, a version, a description, and the operation once built.

    :param name: the tool's namespaced name (``<provider>.<verb>``)
    :ptype name: str
    :param description: the sentence a caller routes on
    :ptype description: str
    :param operation: the operation, or None until the pod has built it
    :ptype operation: Callable[[], BackgroundOperation[Any] | None]
    :param version: the tool's version
    :ptype version: str
    """

    # an operator's action as much as an agent's: reachable over the tools API too. the face only
    # makes the call addressable; the hub still requires the caller's grant before it runs
    face_api = True

    def __init__(
        self,
        *,
        name: str,
        description: str,
        operation: Callable[[], BackgroundOperation[Any] | None],
        version: str = _DEFAULT_TOOL_VERSION,
    ) -> None:
        super().__init__()
        self._name = name
        self._description = description
        self._operation = operation
        self._version = version

    async def execute(self, **kwargs: Any) -> ToolResult:
        """refuse until the operation exists, then answer for it.

        :param kwargs: none are taken
        :ptype kwargs: Any
        :return: the tool's answer
        :rtype: ToolResult
        """
        operation = self._operation()
        result: ToolResult
        if operation is None:
            result = ToolResult(success=False, content="", error=_NOT_READY, error_code=TOOL_NOT_READY)
        else:
            result = self.answer(operation)
        return result

    def answer(self, operation: BackgroundOperation[Any]) -> ToolResult:
        """the tool's answer for a built operation.

        :param operation: the operation
        :ptype operation: BackgroundOperation[Any]
        :return: the answer
        :rtype: ToolResult
        :raises NotImplementedError: in this base
        """
        raise NotImplementedError

    def mcp_schema(self) -> MCPToolDefinition:
        """the tool's definition; it takes no arguments.

        :return: the definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self._name,
            version=self._version,
            description=self._description,
            input_schema={"type": "object", "properties": {}},
        )

    def mcp_name(self) -> str:
        """the tool's namespaced name.

        :return: the name given
        :rtype: str
        """
        return self._name

    def mcp_version(self) -> str:
        """the tool's version.

        :return: the version given
        :rtype: str
        """
        return self._version


class StartOperationTool(_OperationTool):
    """starts the operation and answers at once; a start while one runs is a ``CONFLICT``.

    :param name: the tool's namespaced name (``<provider>.<verb>``)
    :ptype name: str
    :param description: the sentence a caller routes on
    :ptype description: str
    :param operation: the operation, or None until the pod has built it
    :ptype operation: Callable[[], BackgroundOperation[Any] | None]
    :param status_tool: the status tool's name, which the answer points the caller to
    :ptype status_tool: str
    :param version: the tool's version
    :ptype version: str
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        operation: Callable[[], BackgroundOperation[Any] | None],
        status_tool: str,
        version: str = _DEFAULT_TOOL_VERSION,
    ) -> None:
        super().__init__(name=name, description=description, operation=operation, version=version)
        self._status_tool = status_tool

    def answer(self, operation: BackgroundOperation[Any]) -> ToolResult:
        """start a run, or refuse because one is running.

        :param operation: the operation
        :ptype operation: BackgroundOperation[Any]
        :return: that it started, or a ``CONFLICT`` naming the run in progress
        :rtype: ToolResult
        """
        result: ToolResult
        if operation.start():
            result = ToolResult(
                success=True,
                content=f"started; {self._status_tool} says when it ends and how",
                metadata={"started": True},
            )
        else:
            started = operation.started_at
            since = "" if started is None else f" since {started.isoformat()}"
            result = ToolResult(
                success=False,
                content="",
                error=f"a run is already in progress{since}; {self._status_tool} says when it ends",
                error_code=CONFLICT,
            )
        return result


class RequestOperationTool(_OperationTool):
    """asks the operation to run (:meth:`BackgroundOperation.request`); a request while one runs is not refused.

    For an operation that drains requests, built with ``request=``, so the request and the run are
    one object's and cannot name different runs. Where :class:`StartOperationTool` answers a second
    start ``CONFLICT``, this answers that the run in progress is followed by one that takes it.

    :param name: the tool's namespaced name (``<provider>.<verb>``)
    :ptype name: str
    :param description: the sentence a caller routes on
    :ptype description: str
    :param operation: the operation, or None until the pod has built it
    :ptype operation: Callable[[], BackgroundOperation[Any] | None]
    :param status_tool: the status tool's name, which the answer points the caller to
    :ptype status_tool: str
    :param version: the tool's version
    :ptype version: str
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        operation: Callable[[], BackgroundOperation[Any] | None],
        status_tool: str,
        version: str = _DEFAULT_TOOL_VERSION,
    ) -> None:
        super().__init__(name=name, description=description, operation=operation, version=version)
        self._status_tool = status_tool

    async def execute(self, **kwargs: Any) -> ToolResult:
        """refuse until the operation exists; then ask it to run.

        :param kwargs: none are taken
        :ptype kwargs: Any
        :return: that it started, or that the run in progress is followed by one that takes the request
        :rtype: ToolResult
        """
        operation = self._operation()
        result: ToolResult
        if operation is None:
            result = ToolResult(success=False, content="", error=_NOT_READY, error_code=TOOL_NOT_READY)
        else:
            started = await operation.request()
            content = (
                f"started; {self._status_tool} says when it ends and how"
                if started
                else f"requested; it runs after the run in progress, and {self._status_tool} says when it ends"
            )
            result = ToolResult(success=True, content=content, metadata={"started": started})
        return result


class OperationStatusTool(_OperationTool):
    """says whether the operation is running and how the last run ended.

    :param name: the tool's namespaced name (``<provider>.<noun>``)
    :ptype name: str
    :param description: the sentence a caller routes on
    :ptype description: str
    :param operation: the operation, or None until the pod has built it
    :ptype operation: Callable[[], BackgroundOperation[Any] | None]
    :param render_result: turns a result into JSON values for the answer; the result as it is when None
    :ptype render_result: Callable[[Any], Any] | None
    :param progress: what the operation is doing now, as JSON values with a short ``summary`` line
        (phase, counts, timings): answered as ``progress`` whether or not a run is in progress, so a
        caller sees a long run move and what the operation's other work is doing between runs
    :ptype progress: Callable[[], dict[str, Any]] | None
    :param version: the tool's version
    :ptype version: str
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        operation: Callable[[], BackgroundOperation[Any] | None],
        render_result: Callable[[Any], Any] | None = None,
        progress: Callable[[], dict[str, Any]] | None = None,
        version: str = _DEFAULT_TOOL_VERSION,
    ) -> None:
        super().__init__(name=name, description=description, operation=operation, version=version)
        self._render_result = render_result
        self._progress = progress

    def answer(self, operation: BackgroundOperation[Any]) -> ToolResult:
        """the operation's status, as a sentence and as metadata.

        :param operation: the operation
        :ptype operation: BackgroundOperation[Any]
        :return: the status (:meth:`BackgroundOperation.status`)
        :rtype: ToolResult
        """
        status = operation.status(self._render_result)
        last = status["last"]
        sentence = status["state"]
        if status["running"]:
            sentence = f"running since {status['started_at']}"
        if self._progress is not None:
            try:
                progress = self._progress()
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- the status (the last run's error above all) must answer even when what reports progress fails; logged and said
                log.exception("operation %s: its progress could not be read", operation.name)
                progress = {"summary": f"progress unavailable ({type(exc).__name__})", "error": str(exc)}
            status["progress"] = progress
            if progress.get("summary"):
                sentence += f"; now: {progress['summary']}"
        if last is not None:
            ended = "failed" if last["error"] is not None else "succeeded"
            detail = last["error"] if last["error"] is not None else last["result"]
            sentence += f"; the last run {ended} at {last['finished_at']}: {detail}"
        return ToolResult(success=True, content=sentence, metadata=status)
