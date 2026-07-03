"""agent-side schema-priming integration (schema-priming-task-01b).

wires the materialized documented-schema digest the hub writes
(schema-priming-task-01a) into the agent's read path. the
:class:`SchemaPrimingIntegration` holds the agent-side
:class:`~threetears.datasources.DataSourceSchemaDigestCollection` (bound
over the ``system.platform.rbac`` proxy by the host) and resolves the
agent's declared datasource NAMES to their datasource ids ONCE per pod,
so the :class:`~threetears.langgraph.SchemaPrimingMiddleware` can read
each digest BY PRIMARY KEY -- a hot-L1 lookup -- rather than re-deriving
the documented schema per turn.

the name->id resolution is memoized on the integration. this is a
POD-LEVEL fact (the agent's datasources are customer-uniform and do not
vary per conversation), NOT per-conversation state, so the memo does not
violate the per-conversation isolation rule -- it is the agent's fixed
datasource binding, the same for every conversation the pod multiplexes.

reads ONLY; the hub owns every write to the digest. soft-fails to empty
everywhere so a missing collection / unresolved datasource / proxy fault
yields no injection rather than crashing the turn.

this module imports ONLY :mod:`threetears.observe`; it holds a
duck-typed digest collection so it stays free of any concrete
collection or proxy import.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from threetears.observe import get_logger

__all__ = ["SchemaPrimingIntegration"]

log = get_logger(__name__)


class SchemaPrimingIntegration:
    """holds the digest collection + resolves the agent's datasource ids.

    :ivar digest_collection: agent-side
        :class:`~threetears.datasources.DataSourceSchemaDigestCollection`
        over the rbac proxy, or ``None`` (priming disabled, soft-fail)
    :ivar customer_id: the agent's customer scope for name->id resolution
    :ivar datasource_names: the agent's declared datasource names
    """

    def __init__(
        self,
        digest_collection: Any = None,
        customer_id: UUID | None = None,
        datasource_names: list[str] | None = None,
    ) -> None:
        """capture the digest collection + the agent's datasource binding.

        :param digest_collection: agent-side digest collection over the
            proxy; ``None`` disables priming
        :ptype digest_collection: Any
        :param customer_id: the agent's customer scope
        :ptype customer_id: UUID | None
        :param datasource_names: declared datasource names to prime
        :ptype datasource_names: list[str] | None
        :return: nothing
        :rtype: None
        """
        self.digest_collection = digest_collection
        self.customer_id = customer_id
        self.datasource_names = list(datasource_names or [])
        # memoized name->id resolution (pod-level, customer-uniform; NOT
        # per-conversation state). ``None`` = not yet resolved (or last
        # attempt faulted -> retry next turn). KNOWN low-severity staleness:
        # once resolved, a datasource later DISABLED / renamed / reassigned
        # keeps being primed until the pod restarts -- the resolution is not
        # re-evaluated. acceptable: the digest is documented schema (not row
        # data), and the agent falls back to the live schema tool if the
        # stale table no longer matches. revisit with a TTL if it bites.
        self._resolved_ids: list[UUID] | None = None

    async def datasource_ids(self) -> list[UUID]:
        """resolve (once) the agent's declared datasource names to ids.

        reads ``platform.datasources`` over the proxy, scoped to the
        declared names + active status + the live-access visibility model:
        the agent's OWN customer rows, PUBLIC shared rows, and RESTRICTED
        shared rows explicitly granted to this customer via
        ``datasource_customers``. a bare ``customer_id IS NULL`` would leak
        restricted datasources (both public and restricted carry NULL
        customer_id), so it is NOT used. memoized only on a NON-EMPTY
        success: the first turn that resolves something caches it and later
        turns reuse; a transient fault OR a zero-row result returns ``[]``
        for that turn WITHOUT memoizing, so the next turn retries (priming
        recovers once the proxy is back / the datasource appears; it never
        bricks for the pod life).

        :return: the resolved active datasource ids, or ``[]``
        :rtype: list[UUID]
        """
        if self._resolved_ids is not None:
            return self._resolved_ids
        # nothing to resolve against -> a legitimate empty result; memoize
        # it (there is nothing to retry).
        if not (self.digest_collection is not None and self.customer_id is not None and self.datasource_names):
            self._resolved_ids = []
            return self._resolved_ids
        pool = getattr(self.digest_collection, "l3_pool", None)
        if pool is None:
            self._resolved_ids = []
            return self._resolved_ids
        try:
            # cache-bypass: one-time name->id resolution over the proxy; the
            # per-turn read is the by-pk digest get, not this scan. the
            # visibility scope MUST match the live datasource access model,
            # NOT a bare `customer_id IS NULL` (which would leak RESTRICTED
            # shared datasources: both 'public' AND 'restricted' carry
            # customer_id NULL, but a restricted one is shared only with
            # customers granted via platform.datasource_customers). admit:
            # the agent's OWN customer rows, PUBLIC shared rows, and
            # RESTRICTED shared rows explicitly granted to this customer.
            rows = await pool.fetch(
                "SELECT id FROM datasources "
                "WHERE name = ANY($2) AND status = 'active' AND ("
                "  customer_id = $1"
                "  OR visibility = 'public'"
                "  OR id IN ("
                "       SELECT datasource_id FROM datasource_customers"
                "        WHERE customer_id = $1"
                "  )"
                ")",
                self.customer_id,
                self.datasource_names,
            )
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- proxy fault must soft-fail this turn and retry next, never brick the pod
            # a TRANSIENT fault (proxy not ready, NATS hiccup) must NOT
            # memoize: leave _resolved_ids None so the next turn retries.
            # priming injects nothing THIS turn, but recovers -- never
            # silently bricks for the pod lifetime.
            log.warning(
                "schema-priming datasource resolution failed (soft-fail this turn, will retry): %s: %s",
                type(exc).__name__,
                exc,
            )
            return []
        resolved = [row["id"] for row in rows]
        # only memoize a NON-EMPTY result. a successful zero-row query means
        # the declared datasource does not resolve YET (not created/applied,
        # grant not present) -- treat it like the transient path and retry
        # next turn, so the pod primes once the datasource appears rather
        # than caching [] for its whole lifetime.
        if resolved:
            self._resolved_ids = resolved
        return resolved

    async def get_digest(self, datasource_id: UUID) -> Any:
        """read one datasource's digest BY PRIMARY KEY (hot-L1).

        :param datasource_id: datasource whose digest to read
        :ptype datasource_id: UUID
        :return: the digest entity, or ``None`` when absent / disabled
        :rtype: Any
        """
        if self.digest_collection is None:
            return None
        return await self.digest_collection.get(datasource_id)
