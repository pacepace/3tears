"""A run taken up again, as the ``job_resume`` action takes one up, read back as the run it started."""

from __future__ import annotations

from threetears.evals.run.launch import LaunchHost
from threetears.evals.run.resume import take_up_run
from threetears.evals.schema.models import EvalRun


async def resumed_run(host: LaunchHost, run_id: str, scope_id: str, *, max_cost_usd: float | None = None) -> EvalRun:
    """Take ``run_id`` up again through :func:`~threetears.evals.run.resume.take_up_run` and return the run it
    started; what its admission told the caller is left out."""
    return (await take_up_run(host, run_id, scope_id, max_cost_usd=max_cost_usd)).record
