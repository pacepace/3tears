"""Which eval's run the code running now belongs to: what tells a replay that a recorded call went to another eval.

:meth:`~threetears.evals.quick.contract.Eval.run` marks its whole extent with :func:`eval_running`, so its arms'
cells, its judge's calls and every task they start read one :class:`RunningEval` through :func:`running_eval`. A
replaying client (:class:`~threetears.evals.testing.RecordedCompletions`) notes it beside each call it serves, so a
call its recording has already served to another eval in the process is refused for that cause, by name, rather than
as a prompt asked more often than it was recorded.

A context variable, so it is per task: two evals run at once in one process each see their own. An eval run twice is
two runs, told apart by identity, never by name.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class RunningEval:
    """One run of one eval, equal only to itself.

    :ivar name: The eval's name, as a refusal names the run that took a call.
    """

    name: str


#: The eval run this context's work belongs to; ``None`` outside every one.
_RUNNING_EVAL: ContextVar[RunningEval | None] = ContextVar("threetears_evals_running_eval", default=None)


def running_eval() -> RunningEval | None:
    """The eval run the calling task's work belongs to, or ``None`` outside every one.

    :return: The run, or ``None``.
    :rtype: RunningEval | None
    """
    return _RUNNING_EVAL.get()


@contextmanager
def eval_running(name: str) -> Iterator[RunningEval]:
    """Mark the enclosed code, and every task it creates, as one run of eval ``name``.

    :param name: The eval's name.
    :ptype name: str
    :yield: The run.
    :rtype: Iterator[RunningEval]
    """
    run = RunningEval(name)
    token = _RUNNING_EVAL.set(run)
    try:
        yield run
    finally:
        _RUNNING_EVAL.reset(token)


__all__ = ["RunningEval", "eval_running", "running_eval"]
