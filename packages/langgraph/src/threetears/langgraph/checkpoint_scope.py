"""the tenancy decision a checkpoint saver cannot be built without.

:class:`ThreeTierCheckpointSaver` stores full serialized graph state -- the most
sensitive live data in the system -- keyed by ``thread_id``. The customer that
state belongs to is folded into that key rather than added as a column, for the
reasons recorded on
:meth:`~threetears.langgraph.checkpoint.ThreeTierCheckpointSaver.storage_thread_id`.

This module holds the DECISION rather than the mechanism. The mechanism shipped
first, as an optional ``customer_id=None`` parameter, and the default was the
defect: saying nothing meant "address every customer's keyspace", which is what
every existing caller said, so tenancy was a convention rather than a gate. The
type here removes the default and leaves exactly two answers, both of which a
caller has to write down:

- :meth:`CheckpointScope.for_customer` -- this saver belongs to one customer.
- :meth:`CheckpointScope.unscoped` -- this saver deliberately belongs to none,
  and here is why.

The second is a legitimate answer, not a failure state: a single-tenant
deployment has one keyspace and nothing to separate. What it is not is a
silence. It names a reason, it logs a warning when it is constructed, and it is
greppable across an estate by its own constructor name -- so "which of our
deployments still runs unscoped, and on what grounds" is a question with an
answer.

**This is defence in depth and a purge handle, not an authorization system.** A
host still decides which customer a request belongs to and builds the saver
accordingly. What a scope adds is that a saver built for the wrong customer
reads nothing rather than reading someone else's conversation.
"""

from __future__ import annotations

from typing import final
from uuid import UUID

from threetears.observe import get_logger

__all__ = [
    "CheckpointScope",
]

log = get_logger(__name__)


@final
class CheckpointScope:
    """which customer's checkpoint keyspace a saver addresses.

    An immutable value with exactly two legal shapes, reached only through
    :meth:`for_customer` and :meth:`unscoped`. There is no public constructor and
    no default: ``CheckpointScope()`` raises, so the unscoped answer cannot be
    arrived at by omission the way ``customer_id=None`` could.

    Read the answer back through :attr:`customer_id` (``None`` when unscoped) and
    :attr:`reason` (``None`` when scoped).

    :param customer_id: the customer this scope names, or None when unscoped
    :ptype customer_id: UUID | None
    :param reason: why this scope names no customer, or None when scoped
    :ptype reason: str | None
    """

    _customer_id: UUID | None
    _reason: str | None

    __slots__ = ("_customer_id", "_reason")

    def __init__(self, *args: object, **kwargs: object) -> None:
        """refuse direct construction.

        A bare constructor would hand the unsafe answer back its default --
        build one with nothing and get "sees everything", with no reason recorded
        and no warning logged. The two named constructors are the only doors, and
        this says so in the error rather than failing on a missing attribute
        later.

        :param args: ignored; present only so any call shape reaches the message
        :ptype args: object
        :param kwargs: ignored; present only so any call shape reaches the message
        :ptype kwargs: object
        :return: never returns
        :rtype: None
        :raises TypeError: always
        """
        raise TypeError(
            "CheckpointScope has no public constructor. Use "
            "CheckpointScope.for_customer(customer_id) for a tenant-scoped saver, or "
            "CheckpointScope.unscoped(reason='...') to deliberately address the un-tenanted keyspace.",
        )

    @classmethod
    def for_customer(cls, customer_id: UUID) -> CheckpointScope:
        """scope every key a saver addresses to one customer.

        The normal answer. The customer is folded into the stored ``thread_id``
        and therefore into the L3 bound parameter, the L2 bucket key, and the L1
        thread key, so a saver holding this scope cannot NAME another customer's
        row at any tier.

        :param customer_id: the customer whose keyspace the saver addresses
        :ptype customer_id: UUID
        :return: a scope naming that customer
        :rtype: CheckpointScope
        :raises TypeError: when customer_id is not a UUID
        """
        if not isinstance(customer_id, UUID):
            # not incidental type policing: the rendered customer becomes a LIKE
            # pattern in adelete_customer_threads, and that statement needs no
            # ESCAPE clause only because a UUID's text form contains no ``%`` and
            # no ``_``. an arbitrary string carries no such guarantee, so a
            # purge built from one could match wider than its own customer.
            raise TypeError(
                f"CheckpointScope.for_customer() needs a uuid.UUID, got {type(customer_id).__name__}. "
                "The customer is rendered into a storage key and into a LIKE pattern, both of which "
                "rely on a UUID's text form.",
            )
        return cls._create(customer_id=customer_id, reason=None)

    @classmethod
    def unscoped(cls, reason: str) -> CheckpointScope:
        """deliberately address the un-tenanted keyspace, and say why.

        Produces byte-identical keys and statements to a saver built before
        tenancy existed, so an existing deployment adopts the required-scope API
        by passing this and migrating no data.

        The reason is mandatory because the point of this constructor is that the
        answer was given rather than defaulted into. It is logged at WARNING on
        construction, so an unscoped deployment is visible to an operator reading
        a running system as well as to anyone grepping the source for
        ``CheckpointScope.unscoped``.

        :param reason: why this saver addresses no customer, non-empty
        :ptype reason: str
        :return: a scope naming no customer
        :rtype: CheckpointScope
        :raises ValueError: when reason is empty or only whitespace
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "CheckpointScope.unscoped() needs a non-empty reason. An unscoped saver addresses "
                "every customer's checkpoints, so the reason is the record that it was chosen.",
            )
        log.warning(
            "checkpoint saver scope is UNSCOPED: keys carry no customer, so one saver addresses "
            f"and can purge every customer's checkpoints. reason: {reason}",
            extra={"reason": reason},
        )
        return cls._create(customer_id=None, reason=reason)

    @classmethod
    def _create(cls, *, customer_id: UUID | None, reason: str | None) -> CheckpointScope:
        """build an instance, bypassing the refusing ``__init__``.

        :param customer_id: the customer this scope names, or None
        :ptype customer_id: UUID | None
        :param reason: why this scope names no customer, or None
        :ptype reason: str | None
        :return: the constructed scope
        :rtype: CheckpointScope
        """
        scope = object.__new__(cls)
        object.__setattr__(scope, "_customer_id", customer_id)
        object.__setattr__(scope, "_reason", reason)
        return scope

    def __setattr__(self, name: str, value: object) -> None:
        """refuse every attribute write.

        A saver holds its scope for its whole life and derives its key prefix
        once at construction, so a mutated scope would not re-scope the saver --
        it would only make the saver and its scope disagree about which customer
        it serves.

        :param name: attribute name
        :ptype name: str
        :param value: attempted value
        :ptype value: object
        :return: never returns
        :rtype: None
        :raises AttributeError: always
        """
        raise AttributeError(f"CheckpointScope is immutable; cannot set {name!r}")

    def __delattr__(self, name: str) -> None:
        """refuse every attribute delete, for the reason in :meth:`__setattr__`.

        :param name: attribute name
        :ptype name: str
        :return: never returns
        :rtype: None
        :raises AttributeError: always
        """
        raise AttributeError(f"CheckpointScope is immutable; cannot delete {name!r}")

    @property
    def customer_id(self) -> UUID | None:
        """the customer this scope names, or None when unscoped.

        :return: customer identifier, or None
        :rtype: UUID | None
        """
        return self._customer_id

    @property
    def reason(self) -> str | None:
        """why this scope names no customer, or None when it names one.

        :return: the recorded reason, or None
        :rtype: str | None
        """
        return self._reason

    def __repr__(self) -> str:
        """render as the call that would rebuild it.

        :return: source-shaped representation
        :rtype: str
        """
        if self._customer_id is None:
            return f"CheckpointScope.unscoped(reason={self._reason!r})"
        return f"CheckpointScope.for_customer({self._customer_id!r})"

    def __eq__(self, other: object) -> bool:
        """compare by value; two scopes naming one customer are one scope.

        :param other: value to compare against
        :ptype other: object
        :return: True when both scopes give the same answer
        :rtype: bool
        """
        if not isinstance(other, CheckpointScope):
            return NotImplemented
        # read the other instance through its public surface: a same-class
        # private read is still a private read, and these two properties are
        # exactly the fields.
        return self._customer_id == other.customer_id and self._reason == other.reason

    def __hash__(self) -> int:
        """hash by the same two fields equality uses.

        :return: hash of the scope's answer
        :rtype: int
        """
        return hash((self._customer_id, self._reason))
