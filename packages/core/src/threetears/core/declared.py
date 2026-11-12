"""A read that is whole by nature, declared where it is written.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) names every whole read in a repo. A few are
whole by what they answer, never by a load path's choice (Pace, 2026-11-12): the one cold start that fills an empty
copy, a panel answering for every state, a backup's own count and checksum, a warehouse relation that holds no key to
read by, a schema change, the drain of a process-local buffer, an operator's reset. Each is declared on the exact
statement, never on a whole function and never in a list::

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
"""

from __future__ import annotations

from typing import Final, Literal, TypeVar

__all__ = ["whole_read"]

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
        ``cold_start``, ``every_state_answer``, ``backup_verification`` and ``schema_change`` (a
        ``backup_verification`` of a bare ``count(*)`` or row checksum over one table, which the gate checks, needs
        none: it is the bare operation)
    :ptype speed: str
    :return: ``value``
    :rtype: _T
    :raises ValueError: when the kind is unknown, or the reason, ruling or proof is missing
    """
    if kind not in _KINDS or not reason.strip() or not ruling.strip() or "::" not in proof:
        raise ValueError(
            f"a whole read declares a kind of {sorted(_KINDS)}, a reason, Pace's ruling and a proof "
            "<test file>::<test function>"
        )
    return value
