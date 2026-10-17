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

**The optional member.** ``sent_settings`` (what the request actually carried, a
:class:`~threetears.evals.schema.ClientRequestSettings`) is not required: a completion without it conforms,
and the engine records its calls as unverified. When the completion carries it, its type is checked, since a
wrong one is refused on every call the engine reads. A host that opts in passes
``require_sent_settings=True``, so its suite fails the day the member is dropped or renamed, or holds
``None`` (a client that stopped saying what it sent), rather than every later run quietly reading as
unverified::

    check_completion_conformance(my_client_result_for_a_canned_response(), require_sent_settings=True)

**What it cannot see.** That an attribute holds the RIGHT thing — a ``served_model`` copied from the
request passes here and makes two models served under one alias compare equal, and a ``sent_settings``
that echoes what the client was asked rather than what its request carried passes here too and records a
setting nobody checked as sent. And a value outside the declared types beyond the ones checked below: the
stop reason, which a raw provider string (OpenAI's ``length``) would make read as a finished completion,
and ``sent_settings``.
"""

from __future__ import annotations

from typing import get_args

from threetears.evals.schema.completion import COMPLETION_RESULT_ATTRIBUTES, USAGE_LEDGER_ATTRIBUTES, StopReason
from threetears.evals.schema.models import ClientRequestSettings

__all__ = ["CompletionConformanceFailure", "check_completion_conformance"]


class CompletionConformanceFailure(AssertionError):
    """A host's completion lacks an attribute eval reads, or holds a stop reason or sent settings eval cannot read.

    An :class:`AssertionError`, so a test runner reports it as a failed assertion rather than an error in
    the test. The message names every offending attribute.
    """


def check_completion_conformance(completion: object, *, require_sent_settings: bool = False) -> None:
    """Fail, naming the attribute, when a host's completion does not carry what eval reads off it.

    Every :class:`~threetears.evals.schema.completion.CompletionResult` member is checked, the usage
    ledger's own (:data:`~threetears.evals.schema.completion.USAGE_LEDGER_ATTRIBUTES`) among them: those
    are the ones whose absence nothing else would report, so the message says which they are. The
    optional ``sent_settings`` (:data:`~threetears.evals.schema.completion.OPTIONAL_COMPLETION_ATTRIBUTES`)
    is checked for its type when present, and required only when asked.

    :param completion: An instance of the host's completion type, as its client returns one.
    :ptype completion: object
    :param require_sent_settings: Fail when the completion carries no ``sent_settings``, or carries ``None``: for a
        host whose client reports what each request carried, so dropping the member, or a client that stops
        filling it, is caught here. Hand it a completion of a call made with an output cap (a client built
        with none reports ``None``, since a settings value names a cap).
    :ptype require_sent_settings: bool
    :raises CompletionConformanceFailure: An attribute is missing, ``stop_reason`` is not a
        :data:`~threetears.evals.schema.completion.StopReason`, ``sent_settings`` is present and is neither
        ``None`` nor a :class:`~threetears.evals.schema.ClientRequestSettings`, or it is absent or ``None`` and
        ``require_sent_settings`` is set.
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
            "must be mapped first (an absent or unknown one to 'truncated'), or a completion cut short reads as finished"
        )
    if not hasattr(completion, "sent_settings"):
        if require_sent_settings:
            problems.append(
                "missing sent_settings, which this check was asked to require: the optional member saying what each "
                "request actually carried (a ClientRequestSettings, or None when the client cannot say)"
            )
    elif (sent := getattr(completion, "sent_settings")) is None:
        if require_sent_settings:
            problems.append(
                "sent_settings is None, and this check was asked to require it: the completion says nothing of what "
                "its request carried, so every call from it records as unverified (report Negotiated.sent_settings, "
                "or what your request builder put on the request)"
            )
    elif not isinstance(sent, ClientRequestSettings):
        problems.append(
            f"sent_settings holds a {type(sent).__name__}, not a ClientRequestSettings or None: the engine refuses "
            "every completion carrying it (CompletionProtocolDefect)"
        )
    if problems:
        raise CompletionConformanceFailure(f"{type(completion).__name__}: " + "; ".join(problems))
