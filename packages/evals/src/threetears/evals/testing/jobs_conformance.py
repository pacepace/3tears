"""The jobs conformance kit: no action's call makes a completion request, and every spend action's job makes some.

A transport serves calls under one fixed timeout, so no call may wait on a model: an action whose work calls one
starts that work as a job and returns :class:`~threetears.evals.ops.JobsStarted`. Nothing in an action's
declaration can show that it keeps the rule — the request is made, or not, deep inside its handler — so this kit
runs every action and watches the completion clients.

**The watch.** A host builds its :class:`~threetears.evals.ops.OpsHost` for the kit with its completion-client
factory wrapped by the kit's :class:`CompletionGuard` (``EvalHost(..., clients=guard.wrap(my_clients))``). A
request made through a wrapped client where :func:`~threetears.evals.kernel.running_job` is ``None`` — in a call,
not in a job's work — is counted and refused (:class:`CompletionOutsideJob`); one made in a job is counted and sent.
Clients a host builds outside that factory are not watched, so a host routes every client its actions use through it.

**The cases** (:func:`jobs_conformance_cases`), over a catalogue the host's actions included:

- **every action's call makes no completion request** — its handler run on its own ``example``, on a fresh host;
  what else the call does (a not-found refusal of the example's ids, most often) is not this kit's question;
- **every spend action's happy path reaches its work as a job** — the action's :data:`~threetears.evals.actions.HappyPath`
  arranges what its work needs and names the parameters; its call must make no request, must return
  ``JobsStarted``, and each job, polled through ``job_poll`` to its end, must have made at least one request. The
  last is the positive control: a happy path that never reaches the work passes the first two vacuously, and is
  refused here as vacuous.

A spend action without a happy path is refused when it is declared (:class:`~threetears.evals.actions.Action`), so a
host's action cannot be forgotten. Under pytest::

    from threetears.evals.testing import jobs_conformance_cases

    @pytest.mark.parametrize("case", jobs_conformance_cases(my_catalogue()), ids=lambda case: case.name)
    async def test_my_actions_keep_the_job_rule(case) -> None:
        await case.check(build_my_conformance_host)

Plain Python: it imports no test runner.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from threetears.evals.actions import Action, ActionCatalogue, Caller, ConformanceSample, ConformanceStage
from threetears.evals.kernel import EvalServiceError, running_job
from threetears.evals.kernel.host import CompletionClients
from threetears.evals.ops import JobsStarted, OpsHost, job_poll

__all__ = [
    "DEFAULT_JOB_CONFORMANCE_TIMEOUT_S",
    "BuildConformanceHost",
    "CompletionGuard",
    "CompletionOutsideJob",
    "JobsConformanceCase",
    "JobsConformanceFailure",
    "JobsConformanceHost",
    "jobs_conformance_cases",
]

#: How long a happy path's jobs are polled for before the case gives up on them, in seconds.
DEFAULT_JOB_CONFORMANCE_TIMEOUT_S = 60.0


class JobsConformanceFailure(AssertionError):
    """An action broke the job rule, or its happy path proved nothing. The message names the action and the rule.

    An :class:`AssertionError`, so a test runner reports it as a failed assertion rather than an error in the test.
    """


class CompletionOutsideJob(AssertionError):
    """A completion request made outside any job's work, refused by the kit's guarded client before it was sent."""


class _GuardedClient:
    """A host's bound client, its requests counted — and refused outside a job — by a :class:`CompletionGuard`."""

    def __init__(self, inner: Any, guard: CompletionGuard, role: str) -> None:
        self._inner = inner
        self._guard = guard
        self._role = role

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def generate(self, **request: Any) -> Any:
        """Count the request; refuse it outside a job, send it inside one."""
        if running_job() is None:
            where = f"a {self._role} request to {getattr(self._inner, 'model_name', '?')}"
            self._guard.outside.append(where)
            raise CompletionOutsideJob(f"{where} was made outside any job's work")
        self._guard.inside += 1
        return await self._inner.generate(**request)

    async def aclose(self) -> None:
        """Release the host's client."""
        await self._inner.aclose()

    async def __aenter__(self) -> _GuardedClient:
        """Enter the host's client, when it is a context manager of its own."""
        if hasattr(self._inner, "__aenter__"):
            await self._inner.__aenter__()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Leave the host's client as it would be left: through its own exit, or released."""
        if hasattr(self._inner, "__aexit__"):
            await self._inner.__aexit__(exc_type, exc, tb)
        else:
            await self._inner.aclose()


@dataclass
class CompletionGuard:
    """The kit's watch on a host's completion clients: what was asked outside a job, and how often inside one.

    Attributes:
        outside: Each request made outside a job's work, described; every one was refused.
        inside: How many requests a job's work made.
    """

    outside: list[str] = field(default_factory=list)
    inside: int = 0

    def wrap(self, clients: CompletionClients) -> CompletionClients:
        """The host's client factory, every client it builds watched.

        Args:
            clients: The host's own factory.

        Returns:
            A factory the host builds its :class:`~threetears.evals.kernel.host.EvalHost` with for the kit.
        """

        def guarded(role: Any, model: str | None, *, temperature: float | None = None) -> Any:
            return _GuardedClient(clients(role, model, temperature=temperature), self, str(role))

        return guarded

    def reset(self) -> None:
        """Forget every request counted so far."""
        self.outside.clear()
        self.inside = 0


@dataclass(frozen=True, kw_only=True)
class JobsConformanceHost:
    """What a host hands the kit for one case: its operations host, who calls, and its sample launch.

    Attributes:
        host: The host the action runs in, built over a fresh store with its client factory wrapped by the
            case's :class:`CompletionGuard`.
        caller: Who the kit calls as.
        sample: A launch the host runs end to end (:class:`~threetears.evals.actions.ConformanceSample`).
    """

    host: OpsHost
    caller: Caller
    sample: ConformanceSample


#: Builds a fresh host for one case, its completion clients wrapped by the guard it is handed.
BuildConformanceHost = Callable[[CompletionGuard], JobsConformanceHost]


@dataclass(frozen=True)
class JobsConformanceCase:
    """One action held to one rule.

    Attributes:
        name: ``<action>.<rule>``, for a test id.
        rule: The rule, in a sentence.
        check: Runs the case against a host the builder makes; raises :class:`JobsConformanceFailure` when broken.
    """

    name: str
    rule: str
    check: Callable[[BuildConformanceHost], Awaitable[None]]


def _fail(action: Action, rule: str, observed: str) -> JobsConformanceFailure:
    return JobsConformanceFailure(f"action {action.name!r} breaks '{rule}': {observed}")


async def _settled(host: OpsHost, job_id: str, scope_id: str, timeout_s: float) -> None:
    """Poll a job through ``job_poll`` until it reads done, as an agent would."""
    async with asyncio.timeout(timeout_s):
        while not (await job_poll(host, job_id, scope_id)).done:
            await asyncio.sleep(0.01)


def _call_case(action: Action) -> JobsConformanceCase:
    rule = "its call makes no completion request"

    async def check(build: BuildConformanceHost) -> None:
        guard = CompletionGuard()
        built = build(guard)
        try:
            params = action.params.model_validate(dict(action.example))
            try:
                await action.handler(built.host, built.caller, params)
            # prawduct:ok-broad-except — the example's ids name nothing; how the call ends is not this case's question
            except Exception:
                # NOSILENT: the example's ids name nothing in the conformance host, so the call may refuse; how it
                # ends is not this case's question -- only whether it made a completion request, read just below.
                pass
            if guard.outside:
                raise _fail(action, rule, f"its example call made {', '.join(guard.outside)}")
        finally:
            await built.host.launch.job_manager.shutdown()

    return JobsConformanceCase(f"{action.name}.call", rule, check)


def _happy_path_case(action: Action, timeout_s: float) -> JobsConformanceCase:
    rule = "its happy path makes no request in its call, returns JobsStarted, and its job makes one"

    async def check(build: BuildConformanceHost) -> None:
        assert action.happy_path is not None  # a spend action declares one, or is refused when declared
        guard = CompletionGuard()
        built = build(guard)
        host, caller = built.host, built.caller
        try:
            arranged = await action.happy_path(ConformanceStage(host=host, caller=caller, sample=built.sample))
            try:
                params = action.params.model_validate(dict(arranged))
            except ValueError as invalid:  # a pydantic ValidationError is one
                raise _fail(action, rule, f"its happy path names parameters that do not validate: {invalid}") from None
            guard.reset()
            try:
                started = await action.handler(host, caller, params)
            except CompletionOutsideJob as refused:
                raise _fail(action, rule, f"its call made a completion request: {refused}") from None
            except EvalServiceError as refused:
                raise _fail(action, rule, f"its call refused its happy path: {refused.message}") from None
            if guard.outside:
                raise _fail(action, rule, f"its call made {', '.join(guard.outside)}")
            if not isinstance(started, JobsStarted):
                raise _fail(action, rule, f"its call returned {type(started).__name__}, not JobsStarted")
            for job in started.jobs:
                try:
                    await _settled(host, job.job_id, caller.scope_id, timeout_s)
                except TimeoutError:
                    raise _fail(action, rule, f"its job {job.job_id} did not end within {timeout_s:g}s") from None
            if guard.outside:
                raise _fail(action, rule, f"a request was made outside its job: {', '.join(guard.outside)}")
            if guard.inside == 0:
                raise _fail(
                    action,
                    rule,
                    "its job ran to its end and made no completion request, so its happy path never reached the work "
                    "— the case is vacuous; make the happy path arrange what the work needs",
                )
        finally:
            await host.launch.job_manager.shutdown()

    return JobsConformanceCase(f"{action.name}.happy_path", rule, check)


def jobs_conformance_cases(
    catalogue: ActionCatalogue, *, job_timeout_s: float = DEFAULT_JOB_CONFORMANCE_TIMEOUT_S
) -> tuple[JobsConformanceCase, ...]:
    """Every case the catalogue's actions are held to: each action's call, and each spend action's happy path.

    Args:
        catalogue: The actions — the engine's and the host's (:func:`~threetears.evals.actions.eval_catalogue`).
        job_timeout_s: How long a happy path's jobs may run before the case fails.

    Returns:
        The cases, in catalogue order: an action's call case, then, for a spend action, its happy-path case.
    """
    cases: list[JobsConformanceCase] = []
    for action in catalogue.actions:
        cases.append(_call_case(action))
        if action.permission == "spend":
            cases.append(_happy_path_case(action, job_timeout_s))
    return tuple(cases)
