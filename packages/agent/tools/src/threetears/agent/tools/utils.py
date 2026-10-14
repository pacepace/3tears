"""Shared utilities for agent tools: the three ways a tool's result reads.

A tool hands its caller text. Whether the call did its job is in that text's
first characters, so a caller reads it without parsing a sentence:

- ``[TOOL ERROR]``: the tool could not do what it was asked.
- ``[NO CHANGE]``: the tool ran and there was nothing for it to do: the thing
  was already there, the thing named does not exist, or a cap was reached.
  Nothing changed and nothing broke, so a caller that refuses a failed tool
  for the rest of a turn must not refuse this one.
- anything else: the call worked.

:func:`tool_result_status` is the one reader of both prefixes.
"""

from __future__ import annotations

from typing import Final, Literal

__all__ = [
    "NO_CHANGE_PREFIX",
    "TOOL_ERROR_PREFIX",
    "ToolResultStatus",
    "tool_error",
    "tool_no_change",
    "tool_result_status",
]

#: How a result opens when the tool could not do what it was asked.
TOOL_ERROR_PREFIX: Final[str] = "[TOOL ERROR]"

#: How a result opens when the tool ran and changed nothing because nothing was there to change.
NO_CHANGE_PREFIX: Final[str] = "[NO CHANGE]"

#: What a tool's result says of its call.
ToolResultStatus = Literal["ok", "no_change", "error"]


def tool_error(tool_name: str, action: str, error: str) -> str:
    """Format a standardized tool error message."""
    return f"{TOOL_ERROR_PREFIX} {tool_name}: {action} failed — {error}"


def tool_no_change(tool_name: str, what: str) -> str:
    """The result of a call that ran and changed nothing: already there, not found, at a cap.

    :param tool_name: the tool answering
    :ptype tool_name: str
    :param what: why nothing changed, and what is there instead, in a sentence
    :ptype what: str
    :return: the result text, opening with :data:`NO_CHANGE_PREFIX`
    :rtype: str
    """
    return f"{NO_CHANGE_PREFIX} {tool_name}: {what}"


def tool_result_status(result: object) -> ToolResultStatus:
    """What a tool's result says of its call, read from how the text opens.

    Anything that is not text is ``ok``: a tool that returns structure reports
    failure its own way.

    :param result: what the tool returned
    :ptype result: object
    :return: ``error``, ``no_change`` or ``ok``
    :rtype: ToolResultStatus
    """
    if not isinstance(result, str):
        return "ok"
    opening = result.lstrip()
    if opening.startswith(TOOL_ERROR_PREFIX):
        return "error"
    if opening.startswith(NO_CHANGE_PREFIX):
        return "no_change"
    return "ok"
