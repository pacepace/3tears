"""the asyncio send path the NATS send bound relies on, checked by name on any interpreter.

:func:`threetears.nats._nats_py_internals.bound_socket_sends` bounds each ``sendmsg`` of a large publish by replacing
``_get_sendmsg_buffer`` on asyncio's plain socket transport. That works only while asyncio sends through it: the
transport picks ``_write_sendmsg`` when the platform has ``sendmsg``, ``writelines`` sends through ``_write_ready``,
``_write_sendmsg`` hands ``sendmsg`` what ``_get_sendmsg_buffer`` returns and drops what the kernel took through
``_adjust_leftover_buffer``. Should a Python release change any of that, the replaced method would simply never be
called and every large publish would hold the loop again, with nothing failing. :func:`send_path_gaps` names each
such change; ``_nats_py_internals`` refuses to import while it names any, and a test runs it under every Python minor
the packages declare.

It reads two asyncio module names, ``_SelectorSocketTransport`` and ``_HAS_SENDMSG``, under the per-file SLF001
ignore with their reasons in ``tests/enforcement/_underscore_exemptions.txt``; the methods are read by the names in
``_READS``. The module imports nothing but the standard library, so that test can load it by path into a bare interpreter of
another Python version (``python -I``), without this workspace's packages.
"""

from __future__ import annotations

import json
from asyncio import selector_events
from types import ModuleType

__all__ = ["send_path_gaps"]

#: the plain socket transport class, as asyncio names it
TRANSPORT_NAME = "_SelectorSocketTransport"

#: (method, name its code must read): the steps from ``writelines`` to the bounded ``_get_sendmsg_buffer``
_READS: tuple[tuple[str, str], ...] = (
    ("__init__", "_write_sendmsg"),
    ("writelines", "_write_ready"),
    ("_write_sendmsg", "_get_sendmsg_buffer"),
    ("_write_sendmsg", "_adjust_leftover_buffer"),
    ("_get_sendmsg_buffer", "_buffer"),
)


def send_path_gaps(module: ModuleType = selector_events) -> list[str]:
    """every step of the send path the bound relies on that ``module`` no longer has.

    :param module: asyncio's ``selector_events`` (another module only to show the check can fail)
    :ptype module: ModuleType
    :return: one line per gap, empty when the bound is honoured
    :rtype: list[str]
    """
    try:
        owner = module._SelectorSocketTransport
    except AttributeError:
        return [f"asyncio.selector_events.{TRANSPORT_NAME}: absent"]
    try:
        has_sendmsg = module._HAS_SENDMSG
    except AttributeError:
        has_sendmsg = None
    gaps: list[str] = []
    if has_sendmsg is not True:
        gaps.append(
            "asyncio.selector_events._HAS_SENDMSG: not True, so the socket transport sends with send() and never "
            "reads _get_sendmsg_buffer"
        )
    for method, reads in _READS:
        code = getattr(getattr(owner, method, None), "__code__", None)
        if code is None:
            gaps.append(f"{TRANSPORT_NAME}.{method}: absent")
        elif reads not in code.co_names:
            gaps.append(f"{TRANSPORT_NAME}.{method}: no longer reads {reads}")
    return gaps


if __name__ == "__main__":
    print(json.dumps(send_path_gaps()))
