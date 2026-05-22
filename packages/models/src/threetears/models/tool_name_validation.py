"""validate tool-call names emitted by chat models against a strict regex.

Production incident, metallm conv ``019e3e26-9870-7a03-8f04-8cc6a4f5f418``
(2026-05-19): a misbehaving provider response surfaced a tool-call name
that carried embedded XML-attribute fragments left over from a model
formatting error (``memory_recall" name="memory_recall``). The junk
name propagated through metallm's reverse-translation layer, reached
the tool-dispatch layer, and was persisted to the database as an
unrecoverable invocation record before the dispatcher's own error
handling could trip.

The dotted/underscored canonical tool name shape used across 3tears
consumers is constrained by every supported provider's tool-name
validator -- the strictest of which is Anthropic's
``^[a-zA-Z0-9_-]{1,128}$``. The 3tears canonical form additionally
allows ``.`` (the dotted form before wire translation) and clamps the
length to 64 (matching the underlying tools' registered names). This
module pins that contract as a regex + helper functions so every
wrapper between the chat model and the application can drop junk
names before they reach downstream dispatch / logging / persistence
layers.

The wrappers in :mod:`threetears.models.providers.openrouter` and
:mod:`threetears.models.providers.anthropic` invoke
:func:`filter_invalid_tool_calls` on every streamed / generated
``AIMessage``: each rejected name is logged once (truncated to 80
characters) and dropped from the ``invalid_tool_calls`` list so
downstream consumers never see it. Valid tool names pass through
unchanged.

This is intentionally separate from
:mod:`threetears.models.tool_name_translation`. Translation hides
provider-specific naming quirks (dot vs. underscore); validation
defends against malformed names regardless of which side of the
translation produced them.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "ToolNameValidationError",
    "filter_invalid_tool_calls",
    "is_valid_tool_name",
    "validate_tool_name",
]


# pattern is the union of every observed provider validator plus the
# dotted-canonical form 3tears uses internally:
#   - alnum
#   - underscore (wire form after translation)
#   - dot (canonical dotted form)
#   - hyphen (some legacy tool ids)
# length clamped at 64 to match the registered tool names; longer values
# are always wrong regardless of pattern.
_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")


class ToolNameValidationError(ValueError):
    """raised when a tool name fails the canonical-form regex.

    Subclasses :class:`ValueError` so callers that catch
    ``ValueError`` (e.g. wrappers using ``try / except ValueError``
    to recover from malformed input) keep working without an
    explicit exception import. Carries the rejected name on the
    instance for log-friendly contexts that need the bad value
    without re-parsing the message.

    :ivar bad_name: the rejected tool name (verbatim; may be empty
        string, may contain control characters, MUST be truncated by
        the caller before logging to prevent log-injection)
    :ptype bad_name: str
    """

    def __init__(self, bad_name: str) -> None:
        """build the error with the rejected name attached.

        :param bad_name: the value that failed validation
        :ptype bad_name: str
        """
        self.bad_name = bad_name
        super().__init__(f"tool name failed validation: {bad_name!r} does not match {_TOOL_NAME_PATTERN.pattern}")


def is_valid_tool_name(name: str) -> bool:
    """check whether ``name`` matches the canonical 3tears tool-name regex.

    The regex (``^[a-zA-Z0-9_.-]{1,64}$``) covers every observed
    provider validator (Anthropic's ``^[a-zA-Z0-9_-]{1,128}$`` is the
    strictest external one; 3tears additionally allows ``.`` for the
    dotted canonical form and clamps the length to 64).

    :param name: candidate tool name (any string; ``None`` or
        non-string callers must coerce first)
    :ptype name: str
    :return: ``True`` if ``name`` matches the regex, ``False``
        otherwise
    :rtype: bool
    """
    if not isinstance(name, str):
        return False
    return _TOOL_NAME_PATTERN.match(name) is not None


def validate_tool_name(name: str) -> None:
    """raise :class:`ToolNameValidationError` when ``name`` fails the regex.

    Use at hard boundaries where an invalid name is a programming
    error or an untrusted input that should never reach dispatch
    (e.g. before persisting a tool invocation). For the streaming
    wrapper path where rejected names should be dropped silently
    with a single log entry, prefer
    :func:`filter_invalid_tool_calls`.

    :param name: candidate tool name
    :ptype name: str
    :raises ToolNameValidationError: when the name fails the regex
    """
    if not is_valid_tool_name(name):
        raise ToolNameValidationError(name)


def filter_invalid_tool_calls(
    invalid_tool_calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """split an ``invalid_tool_calls`` list by name validity.

    Streaming chat-model responses surface malformed tool calls in
    ``invalid_tool_calls``: each entry carries the partial name +
    args + an error string. Some of those entries have names that
    might still be recoverable (e.g. a JSON-parse error on otherwise
    valid args); others carry junk names that cannot be dispatched
    under any circumstances and must be dropped before they reach
    downstream code. This function performs that split.

    The ``rejected`` list is suitable for log emission by the
    caller; the caller is expected to truncate ``name`` before
    logging to bound output size and prevent log injection. The
    ``kept`` list replaces the consumer's view of
    ``invalid_tool_calls`` so the recovery path sees only entries
    that could plausibly be repaired.

    :param invalid_tool_calls: candidate ``invalid_tool_calls`` list
        (typically ``message.invalid_tool_calls`` from a streaming
        chat-model response)
    :ptype invalid_tool_calls: list[dict[str, Any]]
    :return: tuple of ``(kept, rejected)`` where ``kept`` is the
        sub-list whose names match the canonical regex and
        ``rejected`` is the sub-list whose names do not
    :rtype: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    """
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for call in invalid_tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if isinstance(name, str) and is_valid_tool_name(name):
            kept.append(call)
        else:
            rejected.append(call)
    return kept, rejected
