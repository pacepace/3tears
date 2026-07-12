"""tests for routing strategies (LeastConnectionsStrategy)."""

from __future__ import annotations

from datetime import UTC, datetime

from threetears.registry.catalog import ToolEndpoint
from threetears.registry.routing import LeastConnectionsStrategy, RoutingStrategy


# -- helpers --


def _make_endpoint(
    pod_id: str = "pod-001",
    status: str = "available",
    in_flight: int = 0,
) -> ToolEndpoint:
    """create tool endpoint for testing.

    :param pod_id: identifier of pod serving this tool
    :ptype pod_id: str
    :param status: availability status
    :ptype status: str
    :param in_flight: number of currently in-flight calls
    :ptype in_flight: int
    :return: test tool endpoint
    :rtype: ToolEndpoint
    """
    result = ToolEndpoint(
        pod_id=pod_id,
        status=status,
        in_flight=in_flight,
        date_last_heartbeat=datetime.now(UTC),
    )
    return result


# -- selection tests --


class TestLeastConnectionsSelection:
    """tests for LeastConnectionsStrategy endpoint selection."""

    def test_selects_endpoint_with_lowest_in_flight(self) -> None:
        """strategy selects endpoint with lowest in_flight count among three candidates."""
        endpoints = [
            _make_endpoint(pod_id="pod-high", in_flight=5),
            _make_endpoint(pod_id="pod-low", in_flight=2),
            _make_endpoint(pod_id="pod-max", in_flight=8),
        ]
        strategy = LeastConnectionsStrategy()
        selected = strategy.select(endpoints)
        assert selected is not None
        assert selected.pod_id == "pod-low"
        assert selected.in_flight == 2

    def test_returns_none_for_empty_list(self) -> None:
        """strategy returns None when given empty endpoint list."""
        strategy = LeastConnectionsStrategy()
        result = strategy.select([])
        assert result is None

    def test_returns_none_for_all_unavailable(self) -> None:
        """strategy returns None when all endpoints are unavailable."""
        endpoints = [
            _make_endpoint(pod_id="pod-a", status="unavailable", in_flight=0),
            _make_endpoint(pod_id="pod-b", status="unavailable", in_flight=0),
            _make_endpoint(pod_id="pod-c", status="unavailable", in_flight=0),
        ]
        strategy = LeastConnectionsStrategy()
        result = strategy.select(endpoints)
        assert result is None

    def test_filters_unavailable_before_selecting(self) -> None:
        """strategy ignores unavailable endpoints and picks lowest in_flight among available."""
        endpoints = [
            _make_endpoint(pod_id="pod-down", status="unavailable", in_flight=0),
            _make_endpoint(pod_id="pod-busy", status="available", in_flight=5),
            _make_endpoint(pod_id="pod-idle", status="available", in_flight=1),
        ]
        strategy = LeastConnectionsStrategy()
        selected = strategy.select(endpoints)
        assert selected is not None
        assert selected.pod_id == "pod-idle"

    def test_single_endpoint_returns_it(self) -> None:
        """strategy returns single available endpoint without error."""
        endpoints = [
            _make_endpoint(pod_id="pod-only", in_flight=3),
        ]
        strategy = LeastConnectionsStrategy()
        selected = strategy.select(endpoints)
        assert selected is not None
        assert selected.pod_id == "pod-only"

    def test_tie_breaking_returns_a_tied_endpoint(self) -> None:
        """strategy returns one of the endpoints tied for lowest in_flight."""
        endpoints = [
            _make_endpoint(pod_id="pod-first", in_flight=2),
            _make_endpoint(pod_id="pod-second", in_flight=2),
        ]
        strategy = LeastConnectionsStrategy()
        selected = strategy.select(endpoints)
        assert selected is not None
        assert selected.pod_id in {"pod-first", "pod-second"}
        assert selected.in_flight == 2

    def test_tie_breaking_is_randomized_across_tied_endpoints(self) -> None:
        """ties for lowest in_flight are broken at random, not by list order.

        this is the multi-replica convergence guard: a deterministic
        list-order tie-break would make every registry replica route to
        the same pod whenever their local in_flight counts agree (the
        steady state, and always so at cold start when all counts are
        zero). repeated selection over equally-loaded endpoints must
        reach BOTH pods, proving the tie-break spreads load.
        """
        strategy = LeastConnectionsStrategy()
        seen: set[str] = set()
        for _ in range(200):
            endpoints = [
                _make_endpoint(pod_id="pod-first", in_flight=0),
                _make_endpoint(pod_id="pod-second", in_flight=0),
            ]
            selected = strategy.select(endpoints)
            assert selected is not None
            seen.add(selected.pod_id)
        assert seen == {"pod-first", "pod-second"}

    def test_zero_in_flight_preferred(self) -> None:
        """strategy prefers endpoint with zero in_flight over endpoints with positive counts."""
        endpoints = [
            _make_endpoint(pod_id="pod-busy-a", in_flight=3),
            _make_endpoint(pod_id="pod-idle", in_flight=0),
            _make_endpoint(pod_id="pod-busy-b", in_flight=7),
        ]
        strategy = LeastConnectionsStrategy()
        selected = strategy.select(endpoints)
        assert selected is not None
        assert selected.pod_id == "pod-idle"
        assert selected.in_flight == 0


# -- protocol compliance tests --


class TestRoutingStrategyProtocol:
    """tests for RoutingStrategy protocol compliance."""

    def test_protocol_compliance(self) -> None:
        """LeastConnectionsStrategy satisfies RoutingStrategy runtime_checkable protocol."""
        strategy = LeastConnectionsStrategy()
        assert isinstance(strategy, RoutingStrategy)
