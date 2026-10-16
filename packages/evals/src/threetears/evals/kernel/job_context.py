"""Whether the code running now is a job's work: the one fact that tells a transport call from the work it started.

No transport call waits on a model: an action whose work calls one starts that work as a job and returns
:class:`~threetears.evals.ops.JobsStarted`, and the job makes the calls. That is a property of where a call is
made, and nothing in a completion client's arguments says it, so the job runner says it: the job manager
(:class:`~threetears.evals.run.EvalJobManager`) runs each job's work inside :func:`job_running`, and
:func:`running_job` reads it from anywhere below — a judge, a generator, a host's own client.

The value is a :class:`~contextvars.ContextVar`, so it is per task and never per process: a job's work and every
task it creates see it, the caller that started the job does not, and two hosts in one process never read each
other's. Its one reader in the engine is the jobs conformance kit
(:func:`~threetears.evals.testing.jobs_conformance_cases`), whose completion client refuses a request made where
:func:`running_job` is ``None``; a host may read it the same way, to refuse or to label its own calls.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

#: The job whose work this context is running, by its record's id; ``None`` outside every job.
_RUNNING_JOB: ContextVar[str | None] = ContextVar("threetears_evals_running_job", default=None)


def running_job() -> str | None:
    """The id of the job whose work is running in this context, or ``None`` outside every job.

    :return: The job's record id as the job manager runs it (a run's id, a task's id), or ``None`` — a transport
        call, a script, anything that is not a job's work.
    :rtype: str | None
    """
    return _RUNNING_JOB.get()


@contextmanager
def job_running(job_id: str) -> Iterator[None]:
    """Mark the enclosed code, and every task it creates, as job ``job_id``'s work.

    The job manager's alone to enter: it wraps each job's work in it, so a completion request made inside reads
    as made by a job. Reset on leaving, however the work ends.

    :param job_id: The job's record id.
    :ptype job_id: str
    :yield: Nothing.
    :rtype: Iterator[None]
    """
    token = _RUNNING_JOB.set(job_id)
    try:
        yield
    finally:
        _RUNNING_JOB.reset(token)


__all__ = ["job_running", "running_job"]
