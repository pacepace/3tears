"""A tool server's assertion replay guard is any ``ReplayLedger`` (hub spec 12e): the test double a consumer's test
hands it (``FakeReplayGuard``) is taken as the real guard is, so the seam type-checks without the concrete class."""

from __future__ import annotations

from threetears.agent.tools.server import ToolServer
from threetears.core.testing.replay_guard import FakeReplayGuard

from packages.agent.tools.tests.unit.tools.pod_auth import jwks_provider


# held by mypy (typecheck.sh checks tests): ToolServer's parameter takes the double, no ignore
def _guarded(guard: FakeReplayGuard) -> ToolServer:
    return ToolServer(
        namespace="3tears",
        nats_url="nats://127.0.0.1:4222",
        pod_id="ledger-seam-pod",
        jwks_provider=jwks_provider,
        assertion_replay_guard=guard,
    )


def test_the_tool_server_takes_the_double_as_its_assertion_replay_guard() -> None:
    server = _guarded(FakeReplayGuard(fresh=True))
    assert server.pod_id == "ledger-seam-pod"
