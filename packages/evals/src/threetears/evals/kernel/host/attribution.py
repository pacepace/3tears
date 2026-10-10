"""Host attribution for registry refusals — one policy, shared by every registry.

A registry is constructed by the host's own declaration module, before any
:class:`~threetears.evals.kernel.host.profile.HostProfile` exists, so it cannot know whose it is
until a profile binds it. What it does with that name is a single policy, and this module
is where the policy lives rather than in each registry that applies it.
"""

from __future__ import annotations

from threetears.observe import get_logger


log = get_logger(__name__)


class HostAttributed:
    """Mixin giving a registry the ``host 'x': `` prefix its runtime refusals carry.

    Applied by every host registry, so the attribution rule is stated once. Before this
    was a mixin it was four verbatim copies bound by duck-typing, which is the shape a
    rule takes just before the copies stop agreeing.

    **A registry carried by two profiles names neither.** A refusal attributed to the
    wrong host is worse to act on than one attributed to none — and a registry is a
    module-level singleton in every host that has shipped one, so a second profile over it
    is an ordinary construction rather than a defect. Tests build one routinely.

    **That drop is announced exactly once**, because a silently empty prefix is
    indistinguishable from a registry no profile ever bound, and those two states want
    different responses: the first is a construction that gave up attribution, the second
    is wiring that never ran.
    """

    _host_ids: set[str]
    _host_drop_announced: bool

    def _init_attribution(self) -> None:
        """Initialise the attribution state. Call from the registry's ``__init__``."""
        self._host_ids = set()
        self._host_drop_announced = False

    def bind_host(self, host_id: str) -> None:
        """Name the host whose profile carries this registry, for error text only.

        Called by :meth:`~threetears.evals.kernel.host.profile.HostProfile.__post_init__`, which
        carries why a registry cannot know this at construction and why it needs to know it
        at all. Idempotent for a repeated bind of the same id, which is the common case:
        one profile rebuilt over one module-level registry.

        Args:
            host_id: The binding profile's opaque id.
        """
        # `bool(...)`, not the bare set. `set() and X` evaluates to the SET OBJECT, so a bare
        # capture holds a live reference that the very next line mutates — and an empty set
        # becomes truthy before the branch below reads it, announcing a dropped attribution on
        # the FIRST bind of every registry.
        first_other = bool(self._host_ids) and host_id not in self._host_ids
        self._host_ids.add(host_id)
        if first_other and not self._host_drop_announced:
            self._host_drop_announced = True
            log.warning(
                "%s is now bound by more than one host profile (%s); its refusals will carry no host "
                "attribution from here, because naming the wrong one is worse than naming none",
                type(self).__name__,
                ", ".join(sorted(self._host_ids)),
            )

    @property
    def _host(self) -> str:
        """The ``host 'x': `` prefix for a refusal, empty unless exactly one profile bound this."""
        return f"host {next(iter(self._host_ids))!r}: " if len(self._host_ids) == 1 else ""


__all__ = [
    "HostAttributed",
]
