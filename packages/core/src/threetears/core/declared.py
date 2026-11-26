"""A read that is whole by nature, declared where it is written.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) names every whole read in a repo. A few are
whole by what they answer, never by a load path's choice (Pace, 2026-11-12): the one cold start that fills an empty
copy, a panel answering for every state, a backup's own count and checksum, a warehouse relation that holds no key to
read by, a schema change, the drain of a process-local buffer, an operator's reset; and (coordinator, 2026-11-12) a
read serving writers older than a rolling upgrade, until the release it names (``until=``). Each is declared on the
exact statement, never on a whole function and never in a list::

    count = await whole_read(
        conn.fetchval(_COUNT_SQL),
        kind="backup_verification",
        reason="the manifest's counts are taken inside the dump's own snapshot",
        ruling="Pace 2026-11-12: backup count and checksum read whole",
        proof="packages/backup/tests/test_cluster.py::test_counts_only_inside_a_backup_snapshot",
        speed="packages/backup/tests/integration/test_speed.py::test_count_tables_budget",
    )

:func:`whole_read` returns what it is handed, unchanged (an awaitable stays one), so it costs nothing at run time. The
gate checks each declaration, not trusts it: its kind lets through only that kind's whole reads in the wrapped
expression; the reason is 30 characters or more; the ruling names Pace and a date; ``proof`` is a test naming the
enclosing function that proves the statement is reached only in that situation; ``speed`` (for the kinds that read
our own tables at size) is a speed test calling ``budget_problem``. A call into a function holding a declared read is
still a whole read unless that call is declared with the same kind, so a chain is declared at each level. The gate
prints every declaration it accepts.

A statement that returns nothing (a mark, ``whole.add(table)``) is declared by :func:`whole_read_block`, the statement
form: its ``with`` body is the declared read, its arguments the same as :func:`whole_read`'s::

    with whole_read_block(kind="schema_change", reason=..., ruling=..., proof=..., speed=...):
        whole.add(change.table)
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Final, Literal, TypeVar

__all__ = ["sentinel_templates", "whole_read", "whole_read_block"]

_T = TypeVar("_T")

#: the situations a read may be declared whole in, as Pace ruled them (2026-11-12); a new one is a rule change he
#: approves
WholeReadKind = Literal[
    "cold_start",
    "every_state_answer",
    "backup_verification",
    "unbounded_by_warehouse",
    "schema_change",
    "drain",
    "operator_reset",
    "rolling_upgrade",
    # emptying a process's own L1 copy of a table whole when the table is dropped (Pace 2026-11-15); the gate admits
    # one DELETE with no WHERE of a relation no declared L3 table is
    "l1_table_drop",
    # a recovery (Pace 2026-11-26: recovery reloads stay, but are never silent): a copy reloading a table from the
    # writer's full on the delta-files volume after a lost epoch log or a lost delta file, and the lost-write
    # backstop recording a lost window's whole change again; each counted (collection_whole_reloads_total), said at
    # WARNING, and a replay fails on any
    "recovery",
]

#: the kinds, for the run-time check
_KINDS: Final[frozenset[str]] = frozenset(
    {
        "cold_start",
        "every_state_answer",
        "backup_verification",
        "unbounded_by_warehouse",
        "schema_change",
        "drain",
        "operator_reset",
        "rolling_upgrade",
        "l1_table_drop",
        "recovery",
    }
)


def whole_read(
    value: _T,
    *,
    kind: WholeReadKind,
    reason: str,
    ruling: str,
    proof: str,
    speed: str = "",
    until: str = "",
) -> _T:
    """declare ``value`` (the call reading whole, usually not yet awaited) a read whole by nature; return it unchanged.

    :param value: the read: a statement's call (``conn.fetch(...)``), or a call into a function holding one
    :ptype value: _T
    :param kind: the situation it reads whole in
    :ptype kind: WholeReadKind
    :param reason: which situation, and why nothing narrower answers it (30+ characters)
    :ptype reason: str
    :param ruling: Pace's ruling allowing it, with its date (``"Pace 2026-11-12: ..."``)
    :ptype ruling: str
    :param proof: ``<test file>::<test function>``: a test naming the enclosing function, proving the statement is
        reached only in that situation
    :ptype proof: str
    :param speed: ``<test file>::<test function>``: a speed test at real size calling ``budget_problem``, for
        ``cold_start``, ``every_state_answer``, ``backup_verification``, ``schema_change`` and ``l1_table_drop`` (a
        ``backup_verification`` of a bare ``count(*)`` or row checksum over one table, which the gate checks, needs
        none: it is the bare operation)
    :ptype speed: str
    :param until: for ``rolling_upgrade`` only, the release (``"0.96.0"``) the read is gone by: it serves writers older
        than the roll, and the gate refuses it once the package reaches that release
    :ptype until: str
    :return: ``value``
    :rtype: _T
    :raises ValueError: when the kind is unknown, or the reason, ruling or proof is missing, or ``until`` is missing
        from a ``rolling_upgrade`` or given to any other kind
    """
    _check_declaration(kind, reason, ruling, proof, until)
    return value


def whole_read_block(
    *,
    kind: WholeReadKind,
    reason: str,
    ruling: str,
    proof: str,
    speed: str = "",
    until: str = "",
) -> AbstractContextManager[None]:
    """declare the statements of a ``with`` body a read whole by nature: :func:`whole_read`'s statement form, for a
    statement that returns nothing (``whole.add(table)``), so no ``None`` is handed through an expression. The gate
    covers the body's lines as it covers a wrapped expression's, and checks the same arguments.

    :param kind: the situation it reads whole in
    :ptype kind: WholeReadKind
    :param reason: which situation, and why nothing narrower answers it (30+ characters)
    :ptype reason: str
    :param ruling: Pace's ruling allowing it, with its date (``"Pace 2026-11-12: ..."``)
    :ptype ruling: str
    :param proof: ``<test file>::<test function>``, as for :func:`whole_read`
    :ptype proof: str
    :param speed: the speed test, as for :func:`whole_read`
    :ptype speed: str
    :param until: for ``rolling_upgrade`` only, the release the read is gone by
    :ptype until: str
    :return: a context that does nothing
    :rtype: AbstractContextManager[None]
    :raises ValueError: when the kind is unknown, or the reason, ruling or proof is missing, or ``until`` is missing
        from a ``rolling_upgrade`` or given to any other kind
    """
    _check_declaration(kind, reason, ruling, proof, until)
    return nullcontext()


def _check_declaration(kind: str, reason: str, ruling: str, proof: str, until: str) -> None:
    """refuse a declaration missing what :func:`whole_read` and :func:`whole_read_block` require.

    :param kind: the declared kind
    :ptype kind: str
    :param reason: the reason
    :ptype reason: str
    :param ruling: the ruling
    :ptype ruling: str
    :param proof: the proof test
    :ptype proof: str
    :param until: the release a ``rolling_upgrade`` read is gone by
    :ptype until: str
    :return: nothing
    :rtype: None
    :raises ValueError: when the kind is unknown, or the reason, ruling or proof is missing, or ``until`` is missing
        from a ``rolling_upgrade`` or given to any other kind
    """
    if (kind == "rolling_upgrade") != bool(until.strip()):
        raise ValueError("a rolling_upgrade read names until=<the release it is gone by>; no other kind takes until")
    if kind not in _KINDS or not reason.strip() or not ruling.strip() or "::" not in proof:
        raise ValueError(
            f"a whole read declares a kind of {sorted(_KINDS)}, a reason, Pace's ruling and a proof "
            "<test file>::<test function>"
        )


#: a sentinel: a name a template holds in place of a table or column it is filled with at run time (``__table__``)
_SENTINEL_SHAPE: Final = ("__", "__")


def sentinel_templates(
    *,
    templates: tuple[str, ...],
    table: str,
    tables: str,
    columns: Mapping[str, str],
    renderer: str | None = None,
) -> None:
    """declare, at module level, that SQL templates of this module read a table chosen from a closed map: each holds
    ``table`` (a sentinel, ``"__table__"``) where the table goes and each of ``columns``' sentinels where that
    entry's column goes, filled at run time only from the map named by ``tables``.

    The no-whole-reads rule judges each named template once per entry of the map, with the entry's table and
    columns in place of the sentinels, and only for the entries that fill every column sentinel the template holds
    (an entry whose column is None is written another template by the code). A template the declaration does not
    name is judged as written (its table unknown). Nothing happens at run time; the arguments are checked::

        sentinel_templates(
            templates=("_CUSTOMER_ONLY_SHADOW", "_ANCHOR_ONLY_SHADOW"),
            table="__table__",
            tables="aibots.hub.broker.acl.PLATFORM_CUSTOMER_SCOPED_TABLES",
            columns={"__customer_column__": "customer_column", "__anchor_column__": "anchor_column"},
        )

    :param templates: the module-level names the templates are bound to
    :ptype templates: tuple[str, ...]
    :param table: the table's sentinel
    :ptype table: str
    :param tables: the map's dotted path (``package.module.NAME``): a module-level dict literal of table name onto
        a call whose keyword arguments are the columns (a string or None each)
    :ptype tables: str
    :param columns: each column sentinel onto the keyword of the map's entries that fills it
    :ptype columns: Mapping[str, str]
    :param renderer: the module-level function of this module that renders one of these templates, filled from the
        map, into SQL text: a keyed helper reading ``Derived(<renderer>(...), params)`` written in this module reads
        one of the templates, so the rule judges that relation as each template's instances (and the helper's read
        over it) rather than as a run-time text it cannot see. Any other run-time ``Derived`` text stays unseen. None
        when the module hands no template to a helper
    :ptype renderer: str | None
    :return: nothing
    :rtype: None
    :raises ValueError: when no template is named, a sentinel is not ``__name__``-shaped, ``tables`` is not a
        dotted path to a name, or ``renderer`` is not a name
    """
    if not templates or not all(isinstance(name, str) and name for name in templates):
        raise ValueError("sentinel_templates names at least one template, each a module-level name")
    for sentinel in (table, *columns):
        if not (
            sentinel.startswith(_SENTINEL_SHAPE[0]) and sentinel.endswith(_SENTINEL_SHAPE[1]) and len(sentinel) > 4
        ):
            raise ValueError(f"a sentinel is spelled __name__, got {sentinel!r}")
    if tables.count(".") < 1 or not all(part.isidentifier() for part in tables.split(".")):
        raise ValueError(f"tables is the map's dotted path (package.module.NAME), got {tables!r}")
    if renderer is not None and not renderer.isidentifier():
        raise ValueError(f"renderer is a module-level function's name, got {renderer!r}")
