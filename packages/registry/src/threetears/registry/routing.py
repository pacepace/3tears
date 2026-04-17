"""routing strategies for load-balanced tool call distribution.

defines protocol for pluggable routing strategies and provides
least-connections implementation as default. strategies select
a single endpoint from a list of candidates for each tool call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

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
    with the lowest in_flight count. ties broken by
    list order (first registered wins).
    """

    def select(self, endpoints: list[ToolEndpoint]) -> ToolEndpoint | None:
        """select available endpoint with lowest in_flight count.

        :param endpoints: list of all endpoints for a tool
        :ptype endpoints: list[ToolEndpoint]
        :return: endpoint with fewest in-flight calls, or None if none available
        :rtype: ToolEndpoint | None
        """
        available = [ep for ep in endpoints if ep.status == "available"]
        if not available:
            return None
        result = min(available, key=lambda ep: ep.in_flight)
        return result
