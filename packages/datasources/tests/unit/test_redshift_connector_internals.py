"""the ``redshift_connector`` private surface the Redshift driver depends on is still there.

The driver reaches one private attribute, through
:mod:`threetears.datasources.drivers._redshift_connector_internals`. A release that renames it
degrades the keepalive to the system default and refuses every login (the login timeout could not
be lifted), so it must fail here, by name, before anything ships against it.

The surface is read from the installed library's source rather than from a constructed connection:
``Connection.__init__`` opens a network socket, and a stand-in object carrying the attribute would
prove only that the stand-in carries it.
"""

from __future__ import annotations

import ast
import importlib.metadata
import inspect
import textwrap
from unittest.mock import MagicMock

import redshift_connector

from threetears.datasources.drivers._redshift_connector_internals import connection_socket


def _attributes_assigned_in_init() -> set[str]:
    """every ``self.<name>`` that ``redshift_connector.Connection.__init__`` assigns.

    :return: the assigned attribute names
    :rtype: set[str]
    """
    source = textwrap.dedent(inspect.getsource(redshift_connector.Connection.__init__))
    assigned: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                assigned.add(target.attr)
    return assigned


def test_the_installed_redshift_connector_still_keeps_its_socket_where_the_driver_reads_it() -> None:
    """``Connection.__init__`` assigns the socket attribute the internals module reads."""
    assigned = _attributes_assigned_in_init()
    assert "_usock" in assigned, (
        f"redshift_connector {importlib.metadata.version('redshift-connector')} no longer assigns the "
        "socket attribute threetears.datasources.drivers._redshift_connector_internals reads. Read that "
        "module's docstring before supporting this release."
    )


def test_the_surface_check_can_fail() -> None:
    """non-vacuity: the walk finds real assignments and not a name the library never had."""
    assigned = _attributes_assigned_in_init()
    assert "_sock" in assigned
    assert "_renamed_in_a_future_release" not in assigned


def test_a_connection_without_a_socket_answers_none() -> None:
    """a connection object that carries no socket at all yields ``None``, never an AttributeError."""
    assert connection_socket(object()) is None
    assert connection_socket(MagicMock(spec=["cursor", "commit", "close"])) is None
