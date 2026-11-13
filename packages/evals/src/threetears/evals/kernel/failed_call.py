"""A cell's output for a model call that failed: one routing for every candidate kind.

A kind's cell that made a model call which raised asks the host's describer whose failure it was
(:func:`~threetears.evals.kernel.provider.describe_failure`), and the answer decides whether the cell counts against
the arm or is excluded as the harness's, and whether the run goes on. That decision is the engine's, not a kind's, so
it lives here once, beside the output it builds, and each kind words only which of its calls failed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, assert_never

from threetears.evals.kernel.candidate_kind import CandidateOutput, CandidateTelemetry
from threetears.evals.schema.base import VerbatimJsonObject
from threetears.evals.schema.completion import FailureOwner


def failed_call_output(
    owner: FailureOwner,
    description: str,
    *,
    refused: str,
    declined: str,
    failed: str,
    log: Callable[[str, int], None],
    unserved: str | None = None,
    telemetry: CandidateTelemetry | None = None,
    host_measures: dict[str, bool | float | str] | None = None,
    kind_payload: VerbatimJsonObject | None = None,
) -> CandidateOutput:
    """A cell's output for a model call that failed, routed by whose failure it was.

    The rule every candidate kind's cell holds a failed call to, in one place:

    - the calling ``account``'s refusal excludes the cell as the harness's (``infra_errors``) and sets
      ``account_refused``, which stops the run, since every model behind the key gets the same answer;
    - the ``provider``'s decline (a rate limit, an overload) excludes the cell the same way and the run goes on, since
      the next call may be served;
    - a model the provider does not serve (``unserved``) excludes the cell the same way and sets ``model_unserved``,
      so the cell is never judged and a run whose cells keep meeting it stops, since every call naming that model
      gets the same answer;
    - anything else is the ``candidate``'s, charged to the arm (``candidate_errors``).

    Each case is worded by the caller, so its record says which call failed; the description follows it, and an
    excluded cell's record starts ``apparatus:`` as every harness fault's does.

    :param owner: Whose failure the call was, as the host's describer read it.
    :ptype owner: FailureOwner
    :param description: The failure as described, safe to store.
    :ptype description: str
    :param refused: What was refused for the account, e.g. ``"reporter: the generator call was refused for the
        calling account"``.
    :ptype refused: str
    :param declined: What the provider declined.
    :ptype declined: str
    :param failed: What failed, charged to the candidate.
    :ptype failed: str
    :param unserved: Which call named a model the provider does not serve, e.g. ``"reporter: the generator's model is
        not served"``; ``None`` words it from ``failed``. Optional, so a caller written before ``unserved`` existed keeps
        working: its describer never returns it.
    :ptype unserved: str | None
    :param log: Logs the chosen wording at a level: ``ERROR`` for an account's refusal, ``WARNING`` otherwise. A
        caller holding the raised exception logs through
        :func:`~threetears.evals.kernel.provider.log_provider_failure`, which appends the
        description and decides the traceback.
    :ptype log: Callable[[str, int], None]
    :param telemetry: What the cell spent before it failed.
    :ptype telemetry: CandidateTelemetry | None
    :param host_measures: The kind's measures of the cell.
    :ptype host_measures: dict[str, bool | float | str] | None
    :param kind_payload: The kind's payload for the cell.
    :ptype kind_payload: VerbatimJsonObject | None
    :return: The routed output.
    :rtype: CandidateOutput
    """
    kept: dict[str, Any] = {  # Any: CandidateOutput's fields
        "telemetry": telemetry if telemetry is not None else CandidateTelemetry(),
        "host_measures": host_measures if host_measures is not None else {},
        "kind_payload": kind_payload,
    }
    if owner == "account":
        log(refused, logging.ERROR)
        return CandidateOutput(infra_errors=[f"apparatus: {refused}: {description}"], account_refused=True, **kept)
    if owner == "provider":
        log(declined, logging.WARNING)
        return CandidateOutput(infra_errors=[f"apparatus: {declined}: {description}"], **kept)
    if owner == "unserved":
        said = unserved if unserved is not None else f"{failed}: the provider does not serve the model it named"
        log(said, logging.WARNING)
        return CandidateOutput(infra_errors=[f"apparatus: {said}: {description}"], model_unserved=True, **kept)
    if owner == "candidate":
        log(failed, logging.WARNING)
        return CandidateOutput(candidate_errors=[f"{failed}: {description}"], **kept)
    assert_never(owner)


__all__ = ["failed_call_output"]
