"""An operator's own command: the marks that tell the no-whole-reads rule a whole write or read is the operator's.

The rule (:mod:`threetears.enforcement.no_whole_reads`) names every whole read and write in a repo. Two shapes are
an operator asking for exactly that, never a load path doing it on its own:

- :func:`operator_reset`: an explicit reset or truncate an operator orders. It may ``TRUNCATE`` (or call a
  backend's ``truncate``); every read inside it is still judged as anywhere else.
- :func:`operator_command`: an operator's explicit one-off command that must read whole to be complete (a backup's
  restore check over its scratch copy, a person's erasure across every agent). Never a load or cache path.

Two more marks name a whole pass the rule accepts for its own reason, never an operator's:

- :func:`security_backstop`: a periodic pass that re-decides every record so a missed revocation cannot leave
  access open (a NATS connection fence). Pace decides each; the gate prints every one it accepts. It needs a reason
  naming the risk, a proof test that the pass catches a missed revocation, and a speed test held to a calibrated
  budget (:func:`threetears.core.testing.calibrate.budget_problem`) at a realistic row count.
- :func:`source_change_read`: a read of an external source's own change record past a saved mark
  (``WHERE upload_timestamp > $1 OR winnerdatetime > $2``), on a relation no declaration of the repo or 3tears
  names. Only a ``WHERE`` that is exactly the named stamps past bound marks passes; its proof test shows the mark
  advances and an unchanged load reads no row.

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
    "security_backstop",
    "source_change_read",
]

_F = TypeVar("_F", bound=Callable[..., object])

#: the attribute :func:`operator_reset` sets on the function, holding its reason
_OPERATOR_RESET_ATTRIBUTE: Final[str] = "__operator_reset__"

#: the attribute :func:`operator_command` sets on the function, holding its reason
_OPERATOR_COMMAND_ATTRIBUTE: Final[str] = "__operator_command__"

#: the attribute :func:`security_backstop` sets on the function, holding its reason
_SECURITY_BACKSTOP_ATTRIBUTE: Final[str] = "__security_backstop__"

#: the attribute :func:`source_change_read` sets on the function, holding the stamps it reads past
_SOURCE_CHANGE_READ_ATTRIBUTE: Final[str] = "__source_change_read__"


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


def security_backstop(reason: str, *, proof: str, speed: str) -> Callable[[_F], _F]:
    """mark a function as a security backstop: a periodic whole pass that re-decides every record, so a missed
    revocation cannot leave access open. Pace decides each one; the gate prints every backstop it accepts.

    :param reason: the risk the pass covers (30+ characters)
    :ptype reason: str
    :param proof: ``<test file>::<test function>``: a test naming the function, proving the pass catches a missed
        revocation
    :ptype proof: str
    :param speed: ``<test file>::<test function>``: a speed test of the pass at a realistic row count, held to a
        calibrated budget (``budget_problem``)
    :ptype speed: str
    :return: the decorator
    :rtype: Callable[[_F], _F]
    :raises ValueError: when the speed test names no test function
    """
    if "::" not in speed:
        raise ValueError("a security backstop needs a speed test of the form <test file>::<test function>")
    return _mark(_SECURITY_BACKSTOP_ATTRIBUTE, reason, proof)


def source_change_read(
    *, proof: str, stamps: tuple[str, ...] = (), ids: tuple[str, ...] = (), unit: str = ""
) -> Callable[[_F], _F]:
    """mark a function as reading an external source by its own change record. Its reads of a relation no
    declaration names (never an L3 or catalog table) pass when:

    - ``stamps``: their ``WHERE`` is exactly ``stamp > mark`` on these stamps (an ``OR`` of them);
    - ``ids``: their ``WHERE`` binds one of these id columns to a bound set (``IN`` / ``= ANY`` of parameters), the ids
      the source's runs record named;
    - ``unit="table"``: the relation is a small static input read whole, only when the source's inputs record says
      it moved (and that one-row-per-input record itself).

    :param proof: ``<test file>::<test function>``: a test naming the function, proving it reads only what the
        source's change record moved (the mark advances; an unchanged load reads no row)
    :ptype proof: str
    :param stamps: the change-record columns the read compares with its saved marks
    :ptype stamps: tuple[str, ...]
    :param ids: the id columns the read binds to the ids the source's runs record named
    :ptype ids: tuple[str, ...]
    :param unit: ``"table"`` for a static input read whole when its inputs record moved
    :ptype unit: str
    :return: the decorator
    :rtype: Callable[[_F], _F]
    :raises ValueError: when it names no stamp, no id and no unit, or a unit other than ``table``
    """
    if not (stamps or ids or unit) or unit not in {"", "table"}:
        raise ValueError("a source change read names its stamps, its ids, or unit='table'")
    return _mark(_SOURCE_CHANGE_READ_ATTRIBUTE, ", ".join((*stamps, *ids, unit)).strip(", "), proof)
