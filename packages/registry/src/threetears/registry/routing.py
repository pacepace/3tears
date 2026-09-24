"""routing strategies for load-balanced tool call distribution.

defines protocol for pluggable routing strategies and provides
least-connections implementation as default. strategies select
a single endpoint from a list of candidates for each tool call.

which endpoints a caller may be routed to at all is decided before any
strategy runs, by :func:`endpoints_callable_by`: an agent's in-process
endpoint serves only that agent, a Tool Pod's serves everyone.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

from threetears.nats import Subjects
from threetears.observe import get_logger

__all__ = [
    "LeastConnectionsStrategy",
    "RoutingStrategy",
    "endpoints_callable_by",
]

if TYPE_CHECKING:
    from threetears.registry.catalog import ToolEndpoint

log = get_logger(__name__)


def endpoints_callable_by(endpoints: Iterable[ToolEndpoint], caller_id: UUID | None) -> list[ToolEndpoint]:
    """the endpoints one caller may be routed to, before any strategy chooses among them.

    The catalog merges every pod serving ``name@version`` into one entry, and that is right for a
    Tool Pod: it is a shared service, and any replica answers any caller the same way. It is wrong
    for a tool an AGENT serves in-process -- a drafts tool, a conversation-recall tool -- because
    that tool answers from its own agent's state. When several agents serve the same name, a call
    from one agent that lands on another's process is answered from the wrong agent's scope, or
    finds nothing in the wrong agent's per-conversation store. So ownership is a precondition of
    routing, not a preference: an in-process endpoint is callable only by the agent that owns it,
    and replicas of that agent still share its calls through the strategy.

    The owner is read from the endpoint's pod-id by
    :meth:`~threetears.nats.Subjects.agent_inprocess_owner_id`, which is proof rather than a
    claim: only the owning agent's connection is granted the probe subject that promotes a dotted
    endpoint to available. ``caller_id`` must be VERIFIED wherever the answer is an authority (the
    call path); discovery passes the requester's claimed id, which only decides what it is shown.

    An endpoint whose pod-id names no agent is callable by no one. Registration refuses such an id,
    so one here was loaded from shared state written before that check; reading it as a Tool Pod
    would make it callable by everyone.

    :param endpoints: every endpoint of one catalog entry
    :ptype endpoints: Iterable[ToolEndpoint]
    :param caller_id: the calling principal -- an agent's id, a tool pod's id -- or ``None`` when
        the caller names no agent, which leaves it only the Tool Pod endpoints
    :ptype caller_id: UUID | None
    :return: the endpoints this caller may be routed to, in catalog order
    :rtype: list[ToolEndpoint]
    """
    result: list[ToolEndpoint] = []
    for endpoint in endpoints:
        try:
            owner = Subjects.agent_inprocess_owner_id(endpoint.pod_id)
        except ValueError as exc:
            log.warning(
                "tool endpoint's pod-id names no agent; it is routable by no caller",
                extra={"extra_data": {"pod_id": endpoint.pod_id, "detail": str(exc)}},
            )
            continue
        if owner is None or owner == caller_id:
            result.append(endpoint)
    return result


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
