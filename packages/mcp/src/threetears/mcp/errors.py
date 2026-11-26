"""the error model: a closed, documented set of codes a failed call carries on its protocol result.

every failed ``tools/call`` answered by :class:`~threetears.mcp.server.McpServer` carries one error, under
:data:`ERROR_META_KEY` in the result's ``_meta``::

    {"code": "INVALID_ARGUMENTS", "message": "'name' is a required property", "retryable": False, "argument": "name"}

``code`` is from the server's :class:`ErrorCodes`: the framework's own refusals, the codes a ``ToolResult`` of
3tears-agent-tools names (with the same meaning), and the codes the server registered. ``retryable`` says whether
the same call can succeed later unchanged; ``argument`` names the argument at fault, when one is. The result's
content stays what it was (the framework's refusal text, or the tool's own content), so a client that reads only
the text is unaffected, and one that branches on the code reads it from ``_meta``.

a handler fails with a code by raising :class:`McpToolError` or returning
:func:`~threetears.mcp.server.tool_error` with ``code=``; an exception type the server mapped with
:meth:`ErrorCodes.map_exception` answers its code; anything else a handler raises is ``HANDLER_ERROR``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

__all__ = [
    "AUTHZ_ERROR",
    "BUILTIN_ERROR_CODES",
    "CONFLICT",
    "ERROR_META_KEY",
    "HANDLER_ERROR",
    "IDENTITY_UNAVAILABLE",
    "INVALID_ARGUMENTS",
    "MALFORMED_REQUEST",
    "OUTPUT_VALIDATION_ERROR",
    "PERMISSION_DENIED",
    "TOOLS_UNAVAILABLE",
    "TOOL_AUTHORIZATION_UNAVAILABLE",
    "TOOL_ERROR",
    "TOOL_NOT_READY",
    "UNKNOWN_RESOURCE",
    "UNKNOWN_TOOL",
    "ErrorCode",
    "ErrorCodes",
    "McpToolError",
    "error_detail",
]

#: the ``_meta`` key of a failed result that holds its error: a valid ``_meta`` prefix (each label starts with a
#: letter, so not ``3tears``) followed by the name
ERROR_META_KEY: Final = "threetears.mcp/error"

#: what a code may be: a letter, then letters, digits and underscores. lower case is allowed, since a server's own
#: codes may already be lower case on its wire
_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class ErrorCode:
    """one code of the set: its name on the wire, what it means, and whether the same call can succeed later.

    :ivar code: the name a client branches on
    :ivar description: what the code means, for the server's documentation and its operators
    :ivar retryable: whether the same call, unchanged, can succeed later (an outage, a lost race) rather than
        meeting the same refusal
    """

    code: str
    description: str
    retryable: bool = False

    def __post_init__(self) -> None:
        """refuse a code no client could match.

        :return: nothing
        :rtype: None
        :raises ValueError: when ``code`` is not a letter followed by letters, digits and underscores, or
            ``description`` is empty
        """
        if not _CODE.fullmatch(self.code):
            raise ValueError(
                f"error code {self.code!r} must be a letter followed by letters, digits and underscores",
            )
        if not self.description:
            raise ValueError(f"error code {self.code!r} needs a description")


UNKNOWN_TOOL = ErrorCode("UNKNOWN_TOOL", "no tool of that name is served")
TOOLS_UNAVAILABLE = ErrorCode("TOOLS_UNAVAILABLE", "the server's tool set could not be read", retryable=True)
INVALID_ARGUMENTS = ErrorCode(
    "INVALID_ARGUMENTS",
    "the arguments do not match the tool's input schema; `argument` names the one at fault when there is one",
)
IDENTITY_UNAVAILABLE = ErrorCode(
    "IDENTITY_UNAVAILABLE",
    "the caller's identity could not be resolved (no credential, or one that no longer resolves)",
)
PERMISSION_DENIED = ErrorCode(
    "PERMISSION_DENIED",
    "the caller may not do this; the same call meets the same refusal until its grants change",
)
AUTHZ_ERROR = ErrorCode(
    "AUTHZ_ERROR",
    "the caller's permission could not be decided (the authorizer failed or is reloading)",
    retryable=True,
)
HANDLER_ERROR = ErrorCode("HANDLER_ERROR", "the tool failed in a way it did not name")
OUTPUT_VALIDATION_ERROR = ErrorCode(
    "OUTPUT_VALIDATION_ERROR",
    "the tool's result does not match its declared output schema (a bug in the tool)",
)
TOOL_ERROR = ErrorCode("TOOL_ERROR", "the tool reported a failure and named no code")
UNKNOWN_RESOURCE = ErrorCode("UNKNOWN_RESOURCE", "no resource or resource template matches that URI")
# the codes a 3tears-agent-tools ToolResult names, with the same meaning, so a TearsTool's refusal keeps its code
CONFLICT = ErrorCode(
    "CONFLICT",
    "what the call changes was changed by someone else at the same moment; nothing was written",
    retryable=True,
)
MALFORMED_REQUEST = ErrorCode(
    "MALFORMED_REQUEST",
    "the call cannot be served as asked (an argument names something the tool does not have)",
)
TOOL_AUTHORIZATION_UNAVAILABLE = ErrorCode(
    "TOOL_AUTHORIZATION_UNAVAILABLE",
    "the tool could not confirm what the caller may have, so it answered nothing",
    retryable=True,
)
TOOL_NOT_READY = ErrorCode(
    "TOOL_NOT_READY", "the tool cannot answer yet; the same call can succeed shortly", retryable=True
)

#: every code the framework itself answers or recognises, by name
BUILTIN_ERROR_CODES: Mapping[str, ErrorCode] = MappingProxyType(
    {
        code.code: code
        for code in (
            UNKNOWN_TOOL,
            TOOLS_UNAVAILABLE,
            INVALID_ARGUMENTS,
            IDENTITY_UNAVAILABLE,
            PERMISSION_DENIED,
            AUTHZ_ERROR,
            HANDLER_ERROR,
            OUTPUT_VALIDATION_ERROR,
            TOOL_ERROR,
            UNKNOWN_RESOURCE,
            CONFLICT,
            MALFORMED_REQUEST,
            TOOL_AUTHORIZATION_UNAVAILABLE,
            TOOL_NOT_READY,
        )
    }
)


class McpToolError(Exception):
    """raised by a handler to fail its call with a code of the server's set.

    :param code: the code, as an :class:`ErrorCode` or its name
    :ptype code: ErrorCode | str
    :param message: what went wrong, for the caller
    :ptype message: str
    :param argument: the argument at fault, when one is
    :ptype argument: str | None
    """

    def __init__(self, code: ErrorCode | str, message: str, *, argument: str | None = None) -> None:
        """capture the code, the message and the argument at fault.

        :param code: the code
        :ptype code: ErrorCode | str
        :param message: what went wrong
        :ptype message: str
        :param argument: the argument at fault
        :ptype argument: str | None
        :return: nothing
        :rtype: None
        """
        super().__init__(message)
        self.code = code.code if isinstance(code, ErrorCode) else code
        self.message = message
        self.argument = argument


class ErrorCodes:
    """a server's closed set of error codes, and the exceptions it maps to them.

    holds :data:`BUILTIN_ERROR_CODES` and the codes registered on it. a code is registered once; one that differs
    from another only in case is refused, so one condition never travels under two spellings.

    :param codes: the server's own codes, registered at construction
    :ptype codes: Iterable[ErrorCode]
    """

    def __init__(self, codes: Iterable[ErrorCode] = ()) -> None:
        """start from the built-in codes and register ``codes``.

        :param codes: the server's own codes
        :ptype codes: Iterable[ErrorCode]
        :return: nothing
        :rtype: None
        :raises ValueError: as :meth:`register`
        """
        self._codes: Mapping[str, ErrorCode] = BUILTIN_ERROR_CODES
        self._exceptions: tuple[tuple[type[BaseException], ErrorCode], ...] = ()
        for code in codes:
            self.register(code)

    def register(self, code: ErrorCode) -> ErrorCode:
        """add ``code`` to the set.

        :param code: the code
        :ptype code: ErrorCode
        :return: the code, so a module can register and keep it in one line
        :rtype: ErrorCode
        :raises ValueError: when a code of that name, in any case, is already in the set
        """
        clash = next((held for held in self._codes if held.upper() == code.code.upper()), None)
        if clash is not None:
            raise ValueError(f"error code {code.code!r} is already in the set (as {clash!r})")
        self._codes = MappingProxyType({**self._codes, code.code: code})
        return code

    def map_exception(self, exception: type[BaseException], code: ErrorCode | str) -> None:
        """answer ``code`` when a handler raises ``exception`` (or a subclass of it).

        the most specific mapping wins: an exception is matched along its class's method resolution order.

        :param exception: the exception type
        :ptype exception: type[BaseException]
        :param code: a code of the set, or its name
        :ptype code: ErrorCode | str
        :return: nothing
        :rtype: None
        :raises ValueError: when ``code`` is not in the set, or ``exception`` is already mapped
        """
        known = self.get(code.code if isinstance(code, ErrorCode) else code)
        if known is None:
            raise ValueError(f"error code {code!r} is not in the set; register it first")
        if any(mapped is exception for mapped, _ in self._exceptions):
            raise ValueError(f"{exception.__name__} is already mapped to an error code")
        self._exceptions = (*self._exceptions, (exception, known))

    def get(self, code: str) -> ErrorCode | None:
        """the code of that name, or ``None`` when the set has none.

        :param code: the name
        :ptype code: str
        :return: the code
        :rtype: ErrorCode | None
        """
        return self._codes.get(code)

    def for_exception(self, exc: BaseException) -> ErrorCode | None:
        """the code ``exc``'s type is mapped to, the most specific mapping first; ``None`` when none is.

        :param exc: what a handler raised
        :ptype exc: BaseException
        :return: the code
        :rtype: ErrorCode | None
        """
        mapped = dict(self._exceptions)
        for cls in type(exc).__mro__:
            if cls in mapped:
                return mapped[cls]
        return None

    def all(self) -> list[ErrorCode]:
        """every code of the set, the built-in ones first, for a server's documentation.

        :return: the codes
        :rtype: list[ErrorCode]
        """
        return list(self._codes.values())


def error_detail(code: ErrorCode, message: str, argument: str | None = None) -> dict[str, Any]:
    """the error as a failed result carries it under :data:`ERROR_META_KEY`.

    :param code: the code
    :ptype code: ErrorCode
    :param message: what went wrong
    :ptype message: str
    :param argument: the argument at fault, when one is
    :ptype argument: str | None
    :return: ``{"code", "message", "retryable"}``, with ``"argument"`` when one is named
    :rtype: dict[str, Any]
    """
    detail: dict[str, Any] = {"code": code.code, "message": message, "retryable": code.retryable}
    if argument is not None:
        detail["argument"] = argument
    return detail
