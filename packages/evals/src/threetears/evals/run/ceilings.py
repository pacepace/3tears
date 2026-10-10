"""The override-or-default ceiling cascade both of a run's per-run bounds resolve through.

A launch's override may only LOWER the host's configured ceiling, never raise it
(:func:`refuse_raised_ceiling`); that rule lives here, once, for both currencies and every
surface that launches.

An eval run is bounded in two currencies — LLM dollars
(:class:`~threetears.evals.run.budget.EvalRunCostCap`) and metered third-party calls
(:class:`~threetears.evals.run.metering.MeteredCallLedger`) — and each answers the same three
questions in the same order: take the launch's override or fall back to the configured
default, drop the ceiling entirely when the host has enforcement switched off, and name
which tier of that cascade answered so a stored number can still say where it came from.

None of that is currency-specific, and it existed twice, statement for statement, until this
module. The reason it had to stop being two copies is not tidiness: a run document carries
both origins side by side and an operator compares them, so a later change to the origin
vocabulary — or to what enforcement-off means — landing in one copy leaves a run recording two
origins computed under two different rules, with nothing going red.

The currency difference is carried by :data:`Ceiling` rather than by a second copy, so dollars
come back as dollars and calls as calls.

**What is deliberately NOT here is how each class then USES the answer.** The cost cap keeps
its resolved ceiling and switches an ``enabled`` flag off; the ledger takes the effective
ceiling and reads ``None`` as unbounded. Those constructors are mirror-opposite on purpose,
are asserted in both directions, and are the part that genuinely differs — which is why each
class keeps its own ``for_run`` and only the cascade under it is shared.

**Everything after the override is keyword-only, and that is load-bearing rather than style.**
``float`` and ``int`` are mutually assignable, so a caller that transposed the override and the
default would resolve a plausible-looking number with no type error anywhere — the failure mode
that removing the ``from_config`` factories was written up for, in a repo that runs no static
type gate. Keywords make the transposition a :class:`TypeError` at the call site instead of a
wrong ceiling in a stored run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from threetears.evals.schema.models import CostCapOrigin

#: The currency a ceiling is counted in: dollars for the cost cap, calls for the metered-call
#: ledger. Constrained to those two rather than left unbound, so a resolved ceiling comes back as
#: the type it went in as — a caller working in dollars never has to narrow an ``int | float`` it
#: can never receive.
#:
#: **A constraint list is matched EXACTLY, which is not how a bare annotation behaves**, and the
#: difference has already been misread once. ``int`` is assignable to a parameter annotated
#: ``float`` (the numeric tower), so ``float | int`` in an ordinary annotation is a redundant union
#: — which is why :func:`resolve_ceiling_origin` below takes a bare ``float | None``. A constrained
#: ``TypeVar`` does not promote: an ``int`` argument solves to ``int`` whichever order the
#: constraints are written in, so ``MeteredCallLedger``'s ``-> int | None`` holds. Verified against
#: these modules with mypy (a ``reveal_type`` probe) rather than reasoned about. Re-run it before
#: changing this line.
Ceiling = TypeVar("Ceiling", float, int)


class CeilingRaisedError(ValueError):
    """A launch named a per-run ceiling ABOVE the host's configured one.

    A launch's override may only lower the host's ceiling. The host's configured ceiling is the
    most the host's operator agreed a run may spend (or call), and whoever holds the launch tool —
    an agent among them — is not the operator, so an override that could raise it would make the
    host's ceiling a suggestion. Raised by :func:`refuse_raised_ceiling`; every launch surface
    translates it into its own refusal before anything is paid for.
    """


def refuse_raised_ceiling(override: Ceiling | None, *, configured: Ceiling, name: str, configured_name: str) -> None:
    """Refuse a per-run override above the host's configured ceiling — the one rule every surface applies.

    An override at or below the configured ceiling is a choice the launch may make (it lowers, or
    restates, the bound); one above it is refused rather than clamped, because a clamp would record
    a ceiling nobody named and leave the caller believing it had the room it asked for. The rule
    holds whether or not enforcement is switched on: the argument's contract is "lower only", and a
    launch accepted with enforcement off would be refused, unchanged, the moment it was switched on.

    Args:
        override: The per-run ceiling the launch named, or ``None`` (which inherits and is always
            allowed).
        configured: The host's configured ceiling, as a value.
        name: What the launch argument is called, for the refusal.
        configured_name: What the host calls its configured ceiling, for the refusal.

    Raises:
        CeilingRaisedError: ``override`` is above ``configured``.
    """
    if override is not None and override > configured:
        raise CeilingRaisedError(
            f"{name}={override} is above the host's ceiling ({configured_name}={configured}); a launch may only "
            f"lower the host's ceiling, never raise it — launch with {name} at or below {configured}, or ask the "
            f"host's operator to raise {configured_name}"
        )


def resolve_ceiling(override: Ceiling | None, *, configured: Ceiling) -> Ceiling:
    """Resolve the override-or-default ceiling, before enforcement is consulted.

    The bottom of the cascade, and the one both the enforcing bound and the figure a run
    persists read through — which is what stops the number a run records from drifting from
    the number that bounded it. It applies :func:`refuse_raised_ceiling` itself, so no path that
    resolves a ceiling can resolve one above the host's; the launch surfaces call that rule
    earlier, before anything is paid for, so a refusal here means a surface skipped it.

    Args:
        override: The per-run ceiling the launch named, validated ``> 0`` at the service
            boundary. ``None`` inherits the default below.
        configured: The default ceiling, supplied as a value. This module reads no
            configuration and holds none.

    Returns:
        The ceiling, regardless of whether enforcement is switched on.

    Raises:
        CeilingRaisedError: ``override`` is above ``configured``.
    """
    refuse_raised_ceiling(override, configured=configured, name="the per-run ceiling", configured_name="configured")
    return override if override is not None else configured


def resolve_effective_ceiling(
    override: Ceiling | None, *, configured: Ceiling, enforcement_enabled: bool
) -> Ceiling | None:
    """Return the ceiling a run is actually bounded by, or ``None`` when nothing bounds it.

    Differs from :func:`resolve_ceiling` by one thing: enforcement. A configured ceiling with
    enforcement off bounds nothing, and recording the number anyway would claim a limit the run
    never had.

    Args:
        override: The per-run ceiling the launch named, or ``None``.
        configured: The default ceiling, as a value.
        enforcement_enabled: Whether the caller enforces eval ceilings at all.

    Returns:
        The effective ceiling, or ``None`` when enforcement is off and the run is unbounded.

    Raises:
        CeilingRaisedError: ``override`` is above ``configured`` — refused with enforcement off too.
    """
    ceiling = resolve_ceiling(override, configured=configured)
    return ceiling if enforcement_enabled else None


def resolve_ceiling_origin(override: float | None, *, enforcement_enabled: bool) -> CostCapOrigin:
    """Return which tier of the cascade supplied the ceiling a run is bounded by.

    The companion of :func:`resolve_effective_ceiling`, deliberately answering from the same
    launch: the ceiling is stored resolved, so the number alone cannot say whether the launch
    named it or the default did — and the second is the one that moves under a stored run when
    nobody touched the launch.

    Needs no default ceiling of its own, and is deliberately NOT generic in :data:`Ceiling`,
    because the question is which tier answered rather than what it answered — only the override's
    PRESENCE is read, and a ``TypeVar`` appearing once in a signature constrains nothing. The bare
    ``float`` accepts the ledger's ``int`` ceilings by ordinary assignability, which is a different
    mechanism from the constraint matching above and is why ruff refuses ``float | int`` here while
    the ``TypeVar`` needs both members.

    Args:
        override: The per-run ceiling the launch named, or ``None``. Only its presence is read.
        enforcement_enabled: Whether the caller enforces eval ceilings at all.

    Returns:
        A :data:`~threetears.evals.schema.models.CostCapOrigin` value: ``"uncapped"`` when enforcement is
        off (no ceiling bound the run, whatever the cascade would have resolved), else
        ``"chosen"`` for a launch-supplied ceiling (at or below the configured one — the only
        kind a launch may supply, :func:`refuse_raised_ceiling`) and ``"inherited"`` for the configured default.
    """
    if not enforcement_enabled:
        return "uncapped"
    return "chosen" if override is not None else "inherited"


__all__ = [
    "Ceiling",
    "CeilingRaisedError",
    "refuse_raised_ceiling",
    "resolve_ceiling",
    "resolve_ceiling_origin",
    "resolve_effective_ceiling",
]
