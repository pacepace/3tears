"""How a public root keeps a name a retired generation left importable for one minor, warning on every read.

A host-contract name, or one a host binds at import, does not vanish in the release that retires what it served: the
root keeps it in ``__all__`` and serves it from its module ``__getattr__`` (:func:`served_retired`), so an import still
succeeds and every read warns with a :class:`DeprecationWarning` naming the replacement. The next minor removes it,
recorded in ``MIGRATIONS.toml``. What a retired name holds is defined beside the root that serves it, in that root's
``_retired`` module: a value that registers nothing launchable, or a function that refuses with the replacement named.
A retired member of a class a host still calls (a method, a property) warns the same way from its own body
(:func:`warn_retired`) and keeps answering what it did.
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


#: Why a host's style and caveat kinds retired, as each warning says it: their only readers were the retired writers'
#: prompts.
HOST_STYLE_RETIRED = (
    "the report writer reads no host style or caveat kinds (their only readers were the retired memo writer's and "
    "report planner's prompts); drop the argument"
)


#: Why the run's in-cell cost-cap names retired, as each warning says it: the cap is read between cells, so nothing
#: inside a cell is refused, reserved against or told the cap is reached.
COST_CAP_INSIDE_A_CELL_RETIRED = (
    "the run's cost cap is read between cells, against what the recorded cells landed, so nothing inside a cell is "
    "refused or told the cap is reached"
)


#: Why the generation ceiling retired, as its warning says it: the only generation it bounded was the report planner's.
GENERATION_CEILING_RETIRED = (
    "it bounded the retired report planner's generation; derive a report's budget_s from Ceilings().writer_ceiling_s"
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
    # stacklevel 4: past warn_retired, this function and the root's __getattr__, to the line that read the name.
    warn_retired(f"{root}.{name}", entry.why, stacklevel=4)
    return entry.value


def warn_retired(name: str, why: str, *, stacklevel: int = 3) -> None:
    """Warn that ``name`` is retired and leaves in the next minor, with a :class:`DeprecationWarning` naming why.

    For a retired member (a method or a property a host still calls), which no module ``__getattr__`` serves.

    :param name: The retired name in full (``threetears.evals.kernel.CellSink.reserve``).
    :ptype name: str
    :param why: One clause, naming what retired and what replaces it.
    :ptype why: str
    :param stacklevel: Frames above this function the warning is attributed to; the default, 3, names the line that
        called the member calling this.
    :ptype stacklevel: int
    """
    warnings.warn(f"{name} is retired and leaves in the next minor: {why}", DeprecationWarning, stacklevel=stacklevel)


__all__ = [
    "COST_CAP_INSIDE_A_CELL_RETIRED",
    "GENERATION_CEILING_RETIRED",
    "HOST_STYLE_RETIRED",
    "REPORTER_RETIRED",
    "Retired",
    "served_retired",
    "warn_retired",
]
