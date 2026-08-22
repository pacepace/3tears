"""Eager, bounded BIND of the shared L2 collections bucket (``coll-task-04a`` KVC-05).

Every process that wires an L2-live :class:`~threetears.core.collections.registry.CollectionRegistry`
must open ``{ns}-collections`` BEFORE it configures that registry, and the reasons are not
interchangeable with "it works either way":

- **a config mismatch must raise at WIRING time, not in a request path.**
  :meth:`~threetears.core.collections.base.BaseCollection._ensure_kv` resolves the bucket on the
  first read, so a bucket carrying a configuration this process refuses (``allow_direct`` unset,
  which puts every KV read back on the body-carried ``$JS.API.STREAM.MSG.GET`` form that no
  key-scoped ``$KV.`` grant can constrain) would otherwise surface as a
  :class:`~threetears.nats.errors.KvConfigMismatch` raised under load.
- **it pins the handle every later opener shares.** :meth:`~threetears.nats.NatsClient.kv_bucket`
  caches by full bucket name, so the first open in a process decides the configuration for the life
  of that process. Opening here is what makes every registry in the process see one bucket.

BINDS rather than declares. The hub owns this bucket's canonical configuration and every other
principal's grant carries ``$JS.API.STREAM.INFO.KV_{ns}-collections`` with no ``CREATE``, so the
declaring path can only ever issue a ``STREAM.CREATE`` the broker never answers -- a permissions
refusal arrives as a JetStream deadline, not as an error -- and then fall through to the bind that
was always going to succeed. The bind also RESTORES the config check: the declare path reconciles
only on err_code 10058 and otherwise falls through with no comparison at all.

**Two failures, told apart, and neither is handled the way a generic retry helper would.**
:func:`threetears.observe.resilience.retry_with_backoff` never raises, so wrapping this in it would
downgrade a ``KvConfigMismatch`` to one log line and carry the process on into exactly the
silently-broken state the raise exists to prevent.

- ``KvConfigMismatch`` -- the live bucket carries a configuration this process refuses. Retrying
  cannot clear it, so it propagates on the FIRST attempt and the process dies.
- ``KvError`` -- the bind itself failed, and the dominant cause on a cold cluster is that the
  declaring identity has not run yet: the hub declares this bucket in its own lifespan and nothing
  sequences any other process behind it. That IS transient, and supervisors run these services on
  bounded restart budgets a fast crash-loop burns through in seconds. So it is retried with bounded
  exponential backoff, and raised once the budget is spent.

The client parameter is typed :class:`~threetears.nats.kv.KvDeclaring`, the narrow "can declare
or bind a bucket" slice, rather than the whole ``NatsClient``. That is what lets an in-memory
double satisfy it by construction, and it is why every consumer shares THIS function instead of
keeping a copy whose only real difference was a looser annotation.

This lives in ``threetears.core`` rather than in ``threetears.nats`` because the bucket NAME is
:attr:`~threetears.core.collections.base.BaseCollection.L2_BUCKET_SUFFIX` -- a wire fact owned by
the collection base, shared by the hub's canonical declaration, every consumer's eager bind and the
minted grant. A second literal for it would be a second source of truth.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final

from threetears.core.collections.base import BaseCollection
from threetears.nats.errors import KvError
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.nats.kv import KvBucketLike, KvDeclaring

__all__ = [
    "COLLECTIONS_BIND_ATTEMPTS",
    "COLLECTIONS_BIND_BACKOFF_SECONDS",
    "COLLECTIONS_BIND_MAX_BACKOFF_SECONDS",
    "COLLECTIONS_BUCKET_SUFFIX",
    "bind_collections_bucket",
]

log = get_logger(__name__)

#: the shared L2 bucket every ``BaseCollection`` in every process opens. taken from the collection
#: base rather than spelled again here.
COLLECTIONS_BUCKET_SUFFIX: Final[str] = BaseCollection.L2_BUCKET_SUFFIX

#: how many times the eager BIND is retried before the process gives up. sized against the
#: cold-cluster race it exists for: the hub declares the bucket in its own lifespan and nothing
#: sequences a consumer behind it, so the first bind can precede the declaration by however long hub
#: startup takes. the schedule below tops out at 30s, so 20 attempts span several minutes -- long
#: enough for a hub doing migrations, short enough that a genuinely missing grant is reported rather
#: than hung on forever.
COLLECTIONS_BIND_ATTEMPTS: Final[int] = 20
COLLECTIONS_BIND_BACKOFF_SECONDS: Final[float] = 2.0
COLLECTIONS_BIND_MAX_BACKOFF_SECONDS: Final[float] = 30.0


async def bind_collections_bucket(
    nats_client: KvDeclaring,
    *,
    component: str | None = None,
    attempts: int = COLLECTIONS_BIND_ATTEMPTS,
    backoff_seconds: float = COLLECTIONS_BIND_BACKOFF_SECONDS,
    max_backoff_seconds: float = COLLECTIONS_BIND_MAX_BACKOFF_SECONDS,
) -> KvBucketLike:
    """bind the shared L2 collections bucket, retrying only the transient half.

    Call once at startup, immediately after connect and BEFORE any
    :meth:`~threetears.core.collections.registry.CollectionRegistry.configure` that wires an L2
    client. See the module docstring for why both properties are load-bearing.

    :param nats_client: connected canonical NATS wrapper client
    :ptype nats_client: KvDeclaring
    :param component: name of the binding process, carried on every log line. Several
        processes bind this same bucket and the interesting failure is an ORDERING one --
        which of them reached it before the hub declared it -- so a log that does not say
        who is speaking cannot answer the question it is there for
    :ptype component: str | None
    :param attempts: how many bind attempts the process spends before giving up
    :ptype attempts: int
    :param backoff_seconds: delay before the second attempt; doubles thereafter
    :ptype backoff_seconds: float
    :param max_backoff_seconds: ceiling the doubling delay is clamped to
    :ptype max_backoff_seconds: float
    :return: the bound bucket handle, also installed in the client's bucket cache
    :rtype: KvBucketLike
    :raises KvError: the bucket could not be bound within the attempt budget -- it does not exist
        (nothing has declared it) or this principal is not granted it
    :raises KvConfigMismatch: the live bucket carries a configuration this process refuses; raised
        on the first attempt, because config drift does not heal
    """
    backoff = backoff_seconds
    failure: KvError | None = None
    bucket: KvBucketLike | None = None
    for attempt in range(1, attempts + 1):
        try:
            bucket = await nats_client.ensure_kv_bucket(name=COLLECTIONS_BUCKET_SUFFIX, create_if_missing=False)
            failure = None
            break
        except KvError as exc:
            failure = exc
            if attempt == attempts:
                break
            log.warning(
                "collections KV bucket not bindable yet, retrying: component=%s bucket=%s "
                "attempt=%d/%d retry_in=%.1fs: %s",
                component or "unnamed",
                COLLECTIONS_BUCKET_SUFFIX,
                attempt,
                attempts,
                backoff,
                exc,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff_seconds)
    if failure is not None or bucket is None:
        raise KvError(
            f"collections KV bucket {COLLECTIONS_BUCKET_SUFFIX!r} could not be bound by "
            f"{component or 'unnamed'} after "
            f"{attempts} attempts. this process BINDS the bucket and never declares it, so either "
            f"the declaring identity (the hub, in its lifespan) has not run, or this principal's "
            f"NATS grant does not cover the bucket. last error: {failure}"
        ) from failure
    log.info(
        "collections KV bucket bound",
        extra={
            "extra_data": {
                "component": component,
                "bucket": COLLECTIONS_BUCKET_SUFFIX,
                "create_if_missing": False,
            }
        },
    )
    return bucket
