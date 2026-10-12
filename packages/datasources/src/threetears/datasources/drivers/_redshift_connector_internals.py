"""the one owner of every ``redshift_connector`` private attribute this package touches.

``redshift_connector`` has no public way to reach the socket a :class:`redshift_connector.Connection`
talks over, and :class:`~threetears.datasources.drivers.redshift_driver.RedshiftDriver` needs it
twice: to apply the granular TCP keepalive knobs ``connect()`` does not accept, and to lift the
login timeout ``connect()`` leaves on the socket for the connection's whole life. This module is
the only place in ``threetears.datasources`` that reads a ``redshift_connector`` name with a
leading underscore. The driver calls :func:`connection_socket`, never the attribute, so a release
that renames it breaks here, by name, and in the test that pins the surface
(``tests/unit/test_redshift_connector_internals.py``).

**Verified against redshift_connector 2.1.7** (``redshift_connector/core.py``,
``Connection.__init__``). Moving to a new release: read ``Connection.__init__`` in it, confirm
every entry below is still assigned there, and run the unit and live suites on it.

The surface, and why each is used:

``Connection._usock``
    :func:`connection_socket`. The plain or TLS-wrapped socket ``Connection.__init__`` opens and
    keeps. ``connect()`` takes only the ``tcp_keepalive`` bool, so the idle / interval / count
    knobs are set with ``setsockopt`` on this socket; and ``connect(timeout=...)`` sets a socket
    timeout that would otherwise fail every statement running longer than the login bound.
"""

from __future__ import annotations

from typing import Any

__all__ = ["connection_socket"]


def connection_socket(conn: Any) -> Any:
    """the socket a ``redshift_connector`` connection talks over, or ``None`` when it exposes none.

    Read as an attribute -- the spelling every check sees -- rather than through ``getattr`` with
    the name as a string, which hid the dependency from all of them. A release that renames it
    degrades to ``None``, and each caller says what that costs.

    :param conn: a live ``redshift_connector`` connection
    :ptype conn: Any
    :return: the connection's socket, or ``None``
    :rtype: Any
    """
    result: Any = None
    try:
        result = conn._usock
    except AttributeError:
        # NOSILENT: an absent socket is the answer this returns, and each caller logs or raises
        # what that means for it -- a keepalive left at the system default, or a refused login.
        result = None
    return result
