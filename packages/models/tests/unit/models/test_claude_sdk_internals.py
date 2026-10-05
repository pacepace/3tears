"""the Claude Agent SDK private surface the CLI pool depends on is present in the installed SDK.

Nothing else fails loudly when it is not: every use of it is behind a checkout that counts the
failure as a moved surface and, after three, turns pooling off for the process. An SDK release
that renames one of these members must fail here, by name, before anything ships against it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("claude_agent_sdk")

from threetears.models._claude_sdk_internals import (  # noqa: E402
    PRIVATE_SURFACE,
    PrivateAttribute,
    cli_process_pid,
    install_tool_server,
    missing_private_attributes,
    remove_tool_server,
    send_control_request,
)


def test_the_installed_sdk_has_every_private_member_the_pool_uses() -> None:
    """each declared member exists, with the declared kind, on a real, unconnected SDK object."""
    import claude_agent_sdk

    missing = missing_private_attributes()
    assert not missing, (
        f"the Claude Agent SDK {claude_agent_sdk.__version__} no longer provides what "
        f"threetears.models._claude_sdk_internals depends on: {missing}. Read the module docstring "
        "before moving the SDK's version cap."
    )


def test_the_check_reports_a_member_the_sdk_does_not_have() -> None:
    """non-vacuity: the check above can fail, for an absent member and for a changed kind."""
    surface = (
        PrivateAttribute("ClaudeSDKClient", "_renamed_in_a_future_release", "declared"),
        PrivateAttribute("Query", "sdk_mcp_servers", list),
        PrivateAttribute("Query", "sdk_mcp_servers", "method"),
    )
    assert missing_private_attributes(surface) == [
        "ClaudeSDKClient._renamed_in_a_future_release: absent",
        "Query.sdk_mcp_servers: expected list, found dict",
        "Query.sdk_mcp_servers: no longer a coroutine method",
    ]


def test_the_declared_surface_covers_every_class_the_pool_reaches_into() -> None:
    """the surface names the three SDK classes the pool reaches into, and is not empty."""
    assert {attribute.owner for attribute in PRIVATE_SURFACE} == {
        "ClaudeSDKClient",
        "Query",
        "SubprocessCLITransport",
    }


class TestTheCliPid:
    """the CLI pid is read off the SDK's own process object while its internals still hold it."""

    def test_the_pid_the_sdk_holds_is_returned(self) -> None:
        client = SimpleNamespace(_transport=SimpleNamespace(_process=SimpleNamespace(pid=4321)))

        assert cli_process_pid(client) == 4321

    def test_internals_that_moved_read_as_no_pid_rather_than_raising(self) -> None:
        """the pool's fallback to /proc depends on this answering None, not on it raising."""
        for client in (SimpleNamespace(), SimpleNamespace(_transport=None), SimpleNamespace(_transport=object())):
            assert cli_process_pid(client) is None


class TestTheControlProtocol:
    async def test_a_control_request_reaches_the_sdk_with_its_timeout_and_its_answer_comes_back(self) -> None:
        sent: list[tuple[dict[str, Any], float]] = []

        async def send(request: dict[str, Any], timeout: float) -> Any:
            sent.append((request, timeout))
            return {"rewound": True}

        client = SimpleNamespace(_query=SimpleNamespace(_send_control_request=send))

        answer = await send_control_request(client, {"subtype": "rewind_conversation"}, timeout=1.5)

        assert answer == {"rewound": True}
        assert sent == [({"subtype": "rewind_conversation"}, 1.5)]

    @pytest.mark.asyncio
    async def test_a_tool_server_is_installed_and_removed_where_the_sdk_routes_tool_calls(self) -> None:
        servers: dict[str, Any] = {}
        bridges: dict[str, Any] = {}
        client = SimpleNamespace(_query=SimpleNamespace(sdk_mcp_servers=servers, _sdk_mcp_bridges=bridges))
        server = object()

        await install_tool_server(client, "langchain-tools", server)
        assert servers == {"langchain-tools": server}
        # SDK 0.2.163 routes a tools/call through the bridge, not the table: one is built for it
        assert bridges["langchain-tools"].name == "langchain-tools"
        await remove_tool_server(client, "langchain-tools")
        await remove_tool_server(client, "langchain-tools")
        assert servers == {} and bridges == {}, "removing an absent server must be a no-op"
