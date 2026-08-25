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
type here removes the default and leaves exactly three answers, each of which a
caller has to write down:

- :meth:`CheckpointScope.for_customer` -- this saver belongs to one customer.
- :meth:`CheckpointScope.unscoped` -- this saver deliberately belongs to none,
  and here is why.
- :meth:`CheckpointScope.from_config` -- this saver serves many customers, and
  each call says which one.

The second is a legitimate answer, not a failure state: a single-tenant
deployment has one keyspace and nothing to separate. What it is not is a
silence. It names a reason, it logs a warning when it is constructed, and it is
greppable across an estate by its own constructor name -- so "which of our
deployments still runs unscoped, and on what grounds" is a question with an
answer.

The third exists because the first two both assume ONE customer per saver
INSTANCE, and a large class of host cannot honestly say either. A process that
serves every customer from one compiled graph -- metallm, where ``customer_id``
IS the ``user_id`` so every user is a customer, or the survey engine's admin
pod, which is that shape by design -- builds its saver once in lifespan startup
with no request in scope. Under the first two answers such a host had to pick
``unscoped`` and rely on thread-id unguessability for isolation.
:meth:`from_config` lets it keep its one saver and resolve the customer out of
each call's ``config["configurable"]`` instead, which is where LangGraph already
carries per-call identity.

**A config-resolved scope fails CLOSED, always.** A missing key, a ``None``, or
a value that is not a :class:`~uuid.UUID` raises. It never degrades to the
un-tenanted keyspace -- a host that forgot the key gets a loud error rather than
one shared keyspace it believes is isolated, which would be strictly worse than
having no tenancy at all.

**This is defence in depth and a purge handle, not an authorization system.** A
host still decides which customer a request belongs to and says so, by building
the saver or by populating the config. What a scope adds is that a call made for
the wrong customer reads nothing rather than reading someone else's
conversation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, final
from uuid import UUID

from threetears.observe import get_logger

__all__ = [
    "DEFAULT_CUSTOMER_CONFIG_KEY",
    "CheckpointScope",
]

log = get_logger(__name__)


#: the ``configurable`` key a :meth:`CheckpointScope.from_config` scope reads by
#: default. the name the estate already uses for this dimension everywhere else.
DEFAULT_CUSTOMER_CONFIG_KEY = "customer_id"


@final
class CheckpointScope:
    """which customer's checkpoint keyspace a saver addresses.

    An immutable value with exactly three legal shapes, reached only through
    :meth:`for_customer`, :meth:`unscoped` and :meth:`from_config`. There is no
    public constructor and no default: ``CheckpointScope()`` raises, so the
    unscoped answer cannot be arrived at by omission the way ``customer_id=None``
    could.

    Read the answer back through :attr:`customer_id` (``None`` unless the scope
    names one customer for its whole life), :attr:`reason` (``None`` unless the
    scope names no customer at all) and :attr:`config_key` (``None`` unless the
    scope resolves a customer per call). Exactly one of the three is non-``None``
    on any instance.

    Resolve a call's customer through :meth:`customer_for_config` when a config
    is in hand, and through :meth:`customer_for_operation` when one is not --
    which is the case for the two purge methods.

    :param customer_id: the customer this scope names, or None
    :ptype customer_id: UUID | None
    :param reason: why this scope names no customer, or None
    :ptype reason: str | None
    :param config_key: the configurable key each call's customer arrives under,
        or None
    :ptype config_key: str | None
    """

    _customer_id: UUID | None
    _reason: str | None
    _config_key: str | None

    __slots__ = ("_config_key", "_customer_id", "_reason")

    def __init__(self, *args: object, **kwargs: object) -> None:
        """refuse direct construction.

        A bare constructor would hand the unsafe answer back its default --
        build one with nothing and get "sees everything", with no reason recorded
        and no warning logged. The three named constructors are the only doors,
        and this says so in the error rather than failing on a missing attribute
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
            "CheckpointScope.for_customer(customer_id) for a saver that serves one customer, "
            "CheckpointScope.from_config() for a saver that serves many and reads the customer out of "
            "each call's configurable, or "
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
        return cls._create(customer_id=customer_id, reason=None, config_key=None)

    @classmethod
    def from_config(cls, key: str = DEFAULT_CUSTOMER_CONFIG_KEY) -> CheckpointScope:
        """resolve the customer per call, out of each call's ``configurable``.

        The multi-tenant answer, for a host that serves many customers from ONE
        process with ONE compiled graph and therefore ONE saver instance built
        before any request exists. :meth:`for_customer` cannot express that
        (there is no single customer at construction) and :meth:`unscoped` is
        untrue of it (there are many customers, and they must not share a
        keyspace), so the customer arrives with the work instead: LangGraph
        already threads a ``RunnableConfig`` into every checkpoint call, and this
        reads one key out of its ``configurable``.

        **Fails closed, without exception.** A call whose ``configurable`` lacks
        *key*, carries ``None`` under it, or carries anything that is not a
        :class:`~uuid.UUID` raises from :meth:`customer_for_config`. There is no
        fallback to the un-tenanted keyspace and no default customer, because a
        host that forgot the key would otherwise end up with every customer's
        conversations in one keyspace while believing itself isolated -- worse
        than never having tenanted at all.

        The two purge methods receive no config and so cannot use this path; they
        take the customer as a keyword-only argument and refuse without one. See
        :meth:`customer_for_operation`.

        :param key: the ``configurable`` key each call's customer arrives under
        :ptype key: str
        :return: a scope that resolves its customer per call
        :rtype: CheckpointScope
        :raises TypeError: when key is not a str
        :raises ValueError: when key is empty or only whitespace
        """
        if not isinstance(key, str):
            raise TypeError(
                f"CheckpointScope.from_config() needs a str key, got {type(key).__name__}. "
                "The key indexes a call's configurable mapping.",
            )
        if not key.strip():
            raise ValueError(
                "CheckpointScope.from_config() needs a non-empty key. An empty key names no entry, so "
                "every call would fail to resolve a customer.",
            )
        return cls._create(customer_id=None, reason=None, config_key=key)

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
        return cls._create(customer_id=None, reason=reason, config_key=None)

    @classmethod
    def _create(cls, *, customer_id: UUID | None, reason: str | None, config_key: str | None) -> CheckpointScope:
        """build an instance, bypassing the refusing ``__init__``.

        :param customer_id: the customer this scope names, or None
        :ptype customer_id: UUID | None
        :param reason: why this scope names no customer, or None
        :ptype reason: str | None
        :param config_key: the configurable key each call's customer arrives
            under, or None
        :ptype config_key: str | None
        :return: the constructed scope
        :rtype: CheckpointScope
        """
        scope = object.__new__(cls)
        object.__setattr__(scope, "_customer_id", customer_id)
        object.__setattr__(scope, "_reason", reason)
        object.__setattr__(scope, "_config_key", config_key)
        return scope

    def customer_for_config(self, config: Mapping[str, Any] | None) -> UUID | None:
        """the customer a config-bearing call addresses under this scope.

        The one resolver every read and write path goes through, so the three
        answers differ in exactly one place rather than at every call site:

        - :meth:`for_customer` -- returns the scope's own customer. The config is
          NOT consulted, so a ``customer_id`` that happens to sit in a
          ``configurable`` for some unrelated reason cannot re-scope a saver that
          already stated its answer.
        - :meth:`unscoped` -- returns ``None``, and the caller builds bare keys.
          Also does not consult the config, for the same reason.
        - :meth:`from_config` -- reads ``config["configurable"][key]`` and
          insists on a :class:`~uuid.UUID`.

        The last case fails CLOSED at every step: a config without a
        ``configurable`` mapping, a mapping without the key, a key holding
        ``None``, and a key holding a non-UUID all raise. None of them fall back
        to ``None``, which is what the un-tenanted keyspace looks like from
        here, so a host that forgot the key gets a loud error rather than one
        shared keyspace it believes is isolated. The UUID requirement is the same
        one :meth:`for_customer` imposes and for the same reason: the customer is
        rendered into a ``LIKE`` pattern by the per-customer purge, and that
        statement needs no ``ESCAPE`` clause only because a UUID's text form
        contains no ``%`` and no ``_``.

        :param config: the call's runnable config, or None
        :ptype config: Mapping[str, Any] | None
        :return: the customer this call addresses, or None when the scope names
            none
        :rtype: UUID | None
        :raises TypeError: when a config-resolved scope finds a non-UUID value
        :raises ValueError: when a config-resolved scope finds no usable value
        """
        key = self._config_key
        if key is None:
            return self._customer_id

        configurable: Any = None if config is None else config.get("configurable")
        if not isinstance(configurable, Mapping):
            raise ValueError(
                f"CheckpointScope.from_config(key={key!r}) needs config['configurable'] to be a mapping "
                f"carrying the customer, got {type(configurable).__name__}. Refusing rather than "
                "addressing the un-tenanted keyspace.",
            )
        if key not in configurable:
            raise ValueError(
                f"CheckpointScope.from_config(key={key!r}) found no {key!r} in the call's configurable. "
                "A config-resolved saver serves many customers, so it refuses a call that names none "
                "rather than addressing the un-tenanted keyspace, where every customer would share one "
                f"key. Add {key!r} to the configurable dict alongside 'thread_id'.",
            )
        value: Any = configurable[key]
        if value is None:
            raise ValueError(
                f"CheckpointScope.from_config(key={key!r}) found {key!r} set to None. A missing customer "
                "is refused rather than treated as the un-tenanted keyspace.",
            )
        if not isinstance(value, UUID):
            raise TypeError(
                f"CheckpointScope.from_config(key={key!r}) needs a uuid.UUID, got "
                f"{type(value).__name__}. The customer is rendered into a storage key and into a LIKE "
                "pattern, both of which rely on a UUID's text form.",
            )
        return value

    def customer_for_operation(self, customer: UUID | None, *, operation: str) -> UUID | None:
        """the customer an operation addresses when no config is available.

        Two saver methods receive no ``RunnableConfig`` at all -- the per-thread
        purge and the per-customer purge -- so :meth:`customer_for_config` cannot
        serve them. They take the customer as a keyword-only argument instead and
        pass it through here, which reconciles it against the scope:

        - :meth:`for_customer` -- ``None`` means "the one you already know".
          Restating that same customer is agreement and is accepted; naming a
          DIFFERENT one is refused, so the argument cannot become a way to reach
          outside the scope.
        - :meth:`unscoped` -- ``None`` is the only answer. A customer is refused,
          because an unscoped saver's keys carry none, and honouring it silently
          would purge the bare-keyed rows while the caller believed a customer's
          rows had gone.
        - :meth:`from_config` -- a customer is REQUIRED. There is no config to
          read one from and no instance customer to fall back on, so the
          operation refuses rather than guessing whose data to destroy or
          silently addressing the un-tenanted keyspace.

        Every refusal names the constructor the saver was built with, because
        that -- not this argument -- is what a caller has to reconcile.

        :param customer: the customer the caller named, or None
        :ptype customer: UUID | None
        :param operation: the method name, used in the refusal message
        :ptype operation: str
        :return: the customer the operation addresses, or None when unscoped
        :rtype: UUID | None
        :raises TypeError: when customer is neither None nor a UUID
        :raises ValueError: when the customer and the scope cannot be reconciled
        """
        if customer is not None and not isinstance(customer, UUID):
            raise TypeError(
                f"{operation}() needs a uuid.UUID customer, got {type(customer).__name__}. "
                "The customer is rendered into a storage key and into a LIKE pattern, both of which "
                "rely on a UUID's text form.",
            )

        if self._config_key is not None:
            if customer is None:
                raise ValueError(
                    f"{operation}() needs an explicit customer on a saver built with "
                    f"CheckpointScope.from_config(key={self._config_key!r}). That saver serves many "
                    "customers and this call carries no config to resolve one from, so it refuses "
                    f"rather than guess. Pass {operation}(..., customer=<uuid>).",
                )
            return customer

        if self._customer_id is None:
            if customer is not None:
                raise ValueError(
                    f"{operation}() was given a customer, but this saver was built with "
                    f"CheckpointScope.unscoped(reason={self._reason!r}). Its keys carry no customer, so "
                    "honouring one would address the bare-keyed rows while the caller believed a "
                    "customer's rows had been reached. Rebuild the saver with "
                    "CheckpointScope.for_customer() or CheckpointScope.from_config() to scope it.",
                )
            return None

        if customer is not None and customer != self._customer_id:
            raise ValueError(
                f"{operation}() was given customer {customer}, but this saver was built with "
                f"CheckpointScope.for_customer({self._customer_id}). A saver cannot address a customer "
                "it was not built for; build one for that customer, or use "
                "CheckpointScope.from_config() for a saver that serves many.",
            )
        return self._customer_id

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
        """the customer this scope names for its whole life, or None.

        ``None`` for both of the other two answers: an unscoped scope names no
        customer at all, and a config-resolved one names a different customer on
        every call, so neither has one to report here.

        :return: customer identifier, or None
        :rtype: UUID | None
        """
        return self._customer_id

    @property
    def reason(self) -> str | None:
        """why this scope names no customer, or None when it names one.

        Non-``None`` only for :meth:`unscoped`. A config-resolved scope names one
        customer per call, so it is not an opt-out and records no reason.

        :return: the recorded reason, or None
        :rtype: str | None
        """
        return self._reason

    @property
    def config_key(self) -> str | None:
        """the configurable key each call's customer arrives under, or None.

        Non-``None`` only for :meth:`from_config`.

        :return: the configurable key, or None
        :rtype: str | None
        """
        return self._config_key

    def __repr__(self) -> str:
        """render as the call that would rebuild it.

        :return: source-shaped representation
        :rtype: str
        """
        if self._config_key is not None:
            return f"CheckpointScope.from_config(key={self._config_key!r})"
        if self._customer_id is None:
            return f"CheckpointScope.unscoped(reason={self._reason!r})"
        return f"CheckpointScope.for_customer({self._customer_id!r})"

    def __eq__(self, other: object) -> bool:
        """compare by value; two scopes giving one answer are one scope.

        :param other: value to compare against
        :ptype other: object
        :return: True when both scopes give the same answer
        :rtype: bool
        """
        if not isinstance(other, CheckpointScope):
            return NotImplemented
        # read the other instance through its public surface: a same-class
        # private read is still a private read, and these three properties are
        # exactly the fields.
        return (
            self._customer_id == other.customer_id
            and self._reason == other.reason
            and self._config_key == other.config_key
        )

    def __hash__(self) -> int:
        """hash by the same three fields equality uses.

        :return: hash of the scope's answer
        :rtype: int
        """
        return hash((self._customer_id, self._reason, self._config_key))
