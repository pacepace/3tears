"""How a public root keeps a name a retired generation left importable for one minor, warning on every read.

A host-contract name, or one a host binds at import, does not vanish in the release that retires what it served: the
root keeps it in ``__all__`` and serves it from its module ``__getattr__`` (:func:`served_retired`), so an import still
succeeds and every read warns with a :class:`DeprecationWarning` naming the replacement. The next minor removes it,
recorded in ``MIGRATIONS.toml``. What a retired name holds is defined beside the root that serves it, in that root's
``_retired`` module: a value that registers nothing launchable, or a function that refuses with the replacement named.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass


#: Why the analysis reporter kind's names retired, and what replaces the kind, as each warning and refusal says it.
REPORTER_RETIRED = (
    "the analysis reporter kind is retired with the memo writer and report planner it measured; evaluate the report "
    "writer with the report_writer kind (writer_template_create, writer_case_mint)"
)


@dataclass(frozen=True)
class Retired:
    """One retired name: what reading it returns, and why it retired, naming its replacement.

    :ivar value: What the name holds for its last minor.
    :ivar why: One clause, naming what retired and what replaces it, as the warning says it.
    """

    value: object
    why: str


def served_retired(root: str, name: str, retired: Mapping[str, Retired]) -> object:
    """Serve ``name`` from ``root``'s retired names with a :class:`DeprecationWarning`, for a module ``__getattr__``.

    :param root: The public root asked, in full (``threetears.evals.analysis``).
    :ptype root: str
    :param name: The attribute asked for.
    :ptype name: str
    :param retired: The root's retired names.
    :ptype retired: Mapping[str, Retired]
    :return: What the retired name holds.
    :rtype: object
    :raises AttributeError: ``name`` is no retired name of the root, as a module normally would.
    """
    entry = retired.get(name)
    if entry is None:
        raise AttributeError(f"module {root!r} has no attribute {name!r}")
    # stacklevel 3: past this function and the root's __getattr__, to the line that read the name.
    warnings.warn(
        f"{root}.{name} is retired and leaves in the next minor: {entry.why}", DeprecationWarning, stacklevel=3
    )
    return entry.value


__all__ = ["REPORTER_RETIRED", "Retired", "served_retired"]
