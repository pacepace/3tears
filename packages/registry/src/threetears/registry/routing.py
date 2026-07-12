"""routing strategies for load-balanced tool call distribution.

defines protocol for pluggable routing strategies and provides
least-connections implementation as default. strategies select
a single endpoint from a list of candidates for each tool call.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Protocol, runtime_checkable

__all__ = [
    "LeastConnectionsStrategy",
    "RoutingStrategy",
]

if TYPE_CHECKING:
    from threetears.registry.catalog import ToolEndpoint


@runtime_checkable
class RoutingStrategy(Protocol):
    """protocol for tool endpoint selection strategies.

    implementations filter and rank endpoints to select the
    best candidate for each incoming tool call. strategies
    receive the full endpoint list and are responsible for
    filtering to available-only candidates.
    """

    def select(self, endpoints: list[ToolEndpoint]) -> ToolEndpoint | None:
        """select single endpoint from candidates.

        :param endpoints: list of all endpoints for a tool
        :ptype endpoints: list[ToolEndpoint]
        :return: selected endpoint, or None if no viable endpoint
        :rtype: ToolEndpoint | None
        """
        ...


class LeastConnectionsStrategy:
    """selects endpoint with fewest in-flight calls.

    standard least-connections load balancing strategy.
    filters to available endpoints, then picks the one
    with the lowest in_flight count.

    ``in_flight`` is a PER-REPLICA, process-local signal: it counts only
    the calls THIS registry replica has forwarded (it is never persisted
    -- ``ToolEndpoint.to_dict`` omits it and ``from_dict`` resets it to
    zero). there is no shared, authoritative in-flight signal across
    replicas. a deterministic tie-break (e.g. always the first endpoint
    in list order) therefore makes every replica converge on the SAME
    pod whenever their local counts agree -- which is the steady state
    under uniform traffic, and always the case at cold start when every
    count is zero. that thundering-herd defeats the point of running
    multiple pods. so ties for the lowest count are broken UNIFORMLY AT
    RANDOM: each replica independently spreads its load across the
    equally-loaded pods, and the aggregate distribution stays even
    without any cross-replica coordination.
    """

    def select(self, endpoints: list[ToolEndpoint]) -> ToolEndpoint | None:
        """select available endpoint with lowest in_flight count.

        among the available endpoints tied for the lowest ``in_flight``
        count, one is chosen uniformly at random so concurrent registry
        replicas do not all converge on the same pod (see class docstring).

        :param endpoints: list of all endpoints for a tool
        :ptype endpoints: list[ToolEndpoint]
        :return: endpoint with fewest in-flight calls, or None if none available
        :rtype: ToolEndpoint | None
        """
        available = [ep for ep in endpoints if ep.status == "available"]
        if not available:
            return None
        lowest = min(ep.in_flight for ep in available)
        least_loaded = [ep for ep in available if ep.in_flight == lowest]
        result = least_loaded[secrets.randbelow(len(least_loaded))]
        return result
