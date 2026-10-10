"""The completion conformance check: whether a host's completion type carries every attribute eval reads.

Eval reads a completion by attribute name (:class:`~threetears.evals.schema.completion.CompletionResult`),
and the usage ledger reads its attributes defensively, through ``getattr`` with a default, because judge
and simulator test doubles legitimately supply only part of the set. The cost of that is silence: a host
whose completion type renames one of those attributes — ``served_model`` to ``response_model``, say —
gets every usage row from it degraded to "unreported", and every alias it was launched on recorded as
not recorded, with no error anywhere. Nothing in the engine can see the host's concrete type, so the
host's own suite has to. Under pytest::

    from threetears.evals.testing import check_completion_conformance

    def test_my_completion_conforms() -> None:
        check_completion_conformance(my_client_result_for_a_canned_response())

**Hand it an instance, not the class**: a dataclass field with no default and a Pydantic field are not
class attributes, so a class can lack an attribute every one of its instances has.

**What it cannot see.** That an attribute holds the RIGHT thing — a ``served_model`` copied from the
request passes here and makes two models served under one alias compare equal. And a value outside the
declared types beyond the one checked below: the stop reason, which a raw provider string (OpenAI's
``length``) would make read as a finished completion.
"""

from __future__ import annotations

from typing import get_args

from threetears.evals.schema.completion import COMPLETION_RESULT_ATTRIBUTES, USAGE_LEDGER_ATTRIBUTES, StopReason

__all__ = ["CompletionConformanceFailure", "check_completion_conformance"]


class CompletionConformanceFailure(AssertionError):
    """A host's completion lacks an attribute eval reads, or reports a stop reason outside eval's vocabulary.

    An :class:`AssertionError`, so a test runner reports it as a failed assertion rather than an error in
    the test. The message names every offending attribute.
    """


def check_completion_conformance(completion: object) -> None:
    """Fail, naming the attribute, when a host's completion does not carry what eval reads off it.

    Every :class:`~threetears.evals.schema.completion.CompletionResult` member is checked, the usage
    ledger's own (:data:`~threetears.evals.schema.completion.USAGE_LEDGER_ATTRIBUTES`) among them: those
    are the ones whose absence nothing else would report, so the message says which they are.

    Args:
        completion: An instance of the host's completion type, as its client returns one.

    Raises:
        CompletionConformanceFailure: An attribute is missing, or ``stop_reason`` is not a
            :data:`~threetears.evals.schema.completion.StopReason`.
    """
    missing = [name for name in COMPLETION_RESULT_ATTRIBUTES if not hasattr(completion, name)]
    problems: list[str] = []
    if silent := [name for name in missing if name in USAGE_LEDGER_ATTRIBUTES]:
        problems.append(
            f"missing {', '.join(silent)}, which the usage ledger reads with a default — every usage row from this "
            "completion would record it as unreported, without an error"
        )
    if loud := [name for name in missing if name not in USAGE_LEDGER_ATTRIBUTES]:
        problems.append(f"missing {', '.join(loud)}, which eval reads off every completion")
    if "stop_reason" not in missing and (reason := getattr(completion, "stop_reason")) not in get_args(StopReason):
        problems.append(
            f"stop_reason {reason!r} is not one of {', '.join(get_args(StopReason))}: a provider's own finish reason "
            "must be mapped first, or a truncated completion reads as finished"
        )
    if problems:
        raise CompletionConformanceFailure(f"{type(completion).__name__}: " + "; ".join(problems))
