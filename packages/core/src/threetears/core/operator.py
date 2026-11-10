"""An operator's own command: the marks that tell the no-whole-reads rule a whole write or read is the operator's.

The rule (:mod:`threetears.enforcement.no_whole_reads`) names every whole read and write in a repo. Two shapes are
an operator asking for exactly that, never a load path doing it on its own:

- :func:`operator_reset`: an explicit reset or truncate an operator orders. It may ``TRUNCATE`` (or call a
  backend's ``truncate``); every read inside it is still judged as anywhere else.
- :func:`operator_command`: an operator's explicit one-off command that must read whole to be complete (a backup's
  restore check over its scratch copy, a person's erasure across every agent). Never a load or cache path.

Each mark is checked, not trusted. The gate refuses a marked function that anything in the repo calls, so it is
reachable only from its own entry (a route, a CLI command), and it refuses one whose ``proof`` does not name a test
that names the function: that test proves the entry needs the operator, and what else the command's reason claims.

Both return the function unchanged, so a framework's own decorator above them sees the function it wrote.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, TypeVar

__all__ = [
    "operator_command",
    "operator_reset",
]

_F = TypeVar("_F", bound=Callable[..., object])

#: the attribute :func:`operator_reset` sets on the function, holding its reason
_OPERATOR_RESET_ATTRIBUTE: Final[str] = "__operator_reset__"

#: the attribute :func:`operator_command` sets on the function, holding its reason
_OPERATOR_COMMAND_ATTRIBUTE: Final[str] = "__operator_command__"


def _mark(attribute: str, reason: str, proof: str) -> Callable[[_F], _F]:
    """a decorator setting ``attribute`` to ``reason`` on the function and returning it unchanged.

    :param attribute: the attribute
    :ptype attribute: str
    :param reason: why the operator asks for it
    :ptype reason: str
    :param proof: ``<test file>::<test function>`` proving the entry is the operator's
    :ptype proof: str
    :return: the decorator
    :rtype: Callable[[_F], _F]
    :raises ValueError: when the reason or the proof is empty
    """
    if not reason.strip() or "::" not in proof:
        raise ValueError("an operator mark needs a reason and a proof of the form <test file>::<test function>")

    def decorate(function: _F) -> _F:
        setattr(function, attribute, reason)
        return function

    return decorate


def operator_reset(reason: str, *, proof: str) -> Callable[[_F], _F]:
    """mark a function as an operator's explicit reset: it may truncate, and nothing in the repo may call it.

    :param reason: which reset the operator orders
    :ptype reason: str
    :param proof: ``<test file>::<test function>``: a test naming the function, proving its entry needs the operator
    :ptype proof: str
    :return: the decorator
    :rtype: Callable[[_F], _F]
    """
    return _mark(_OPERATOR_RESET_ATTRIBUTE, reason, proof)


def operator_command(reason: str, *, proof: str) -> Callable[[_F], _F]:
    """mark a function as an operator's explicit one-off command: it may read whole, as its completeness needs.

    :param reason: which command, and why it reads whole
    :ptype reason: str
    :param proof: ``<test file>::<test function>``: a test naming the function, proving its entry needs the operator
    :ptype proof: str
    :return: the decorator
    :rtype: Callable[[_F], _F]
    """
    return _mark(_OPERATOR_COMMAND_ATTRIBUTE, reason, proof)
