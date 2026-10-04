"""the one owner of every Claude Agent SDK private member ``threetears.models`` touches.

The Claude Agent SDK (``claude-agent-sdk``) has no public way to do three things the CLI pool
(:mod:`threetears.models.claude_cli_pool`) needs. This module is the only place in
``threetears.models`` that reads or calls an SDK name with a leading underscore, or reaches the
SDK's private ``Query`` object at all. The pool calls the functions below, never the attributes,
so an SDK release that renames one breaks here, by name, and in the test that checks this
module's declared surface (``tests/unit/models/test_claude_sdk_internals.py``) -- rather than as
an ``AttributeError`` deep inside a checkout, which the pool counts as a moved surface and answers
by turning pooling off for the process.

**Verified against claude-agent-sdk 0.2.116** (``client.py``, ``_internal/query.py`` and
``_internal/transport/subprocess_cli.py`` read). ``packages/models/pyproject.toml`` caps the SDK
below the first release not verified here (``<0.3``). Moving the cap: read the same three files in
the new release, check every entry of :data:`PRIVATE_SURFACE` against them, run the models suite
on it, and add the version to this paragraph.

The surface, and why each is used:

``ClaudeSDKClient._query``
    the connected client's control-protocol object (``claude_agent_sdk._internal.query.Query``).
    The client wraps a handful of control requests publicly (``set_model``, ``interrupt``, ...);
    the three the pool needs it does not, so they are sent on the ``Query`` directly.
``Query._send_control_request``
    :func:`send_control_request`. Sends ``apply_flag_settings`` (switch the CLI to the agent that
    holds a call's system prompt), ``mcp_set_servers`` (swap the in-process tool server per
    checkout; ``reconnect_mcp_server`` refuses SDK servers) and ``rewind_conversation`` (empty the
    conversation between callers). The SDK exposes none of the three.
``Query.sdk_mcp_servers``, ``Query._sdk_mcp_bridges``
    :func:`install_tool_server` and :func:`remove_tool_server`. ``sdk_mcp_servers`` is the table of
    in-process servers; from SDK 0.2.163 a CLI's ``tools/call`` is routed through
    ``_sdk_mcp_bridges``, one ``SdkMcpBridge`` per server, built when the client connects. A server
    installed after that needs its own bridge, or its calls answer "server not found": replacing
    both entries is what makes the next ``mcp_set_servers`` serve the borrower's tools.
``ClaudeSDKClient._transport``, ``SubprocessCLITransport._process``
    :func:`cli_process_pid`. The CLI subprocess's pid, so the pool can kill the process and its
    descendants. The SDK exposes its process nowhere; the pool falls back to scanning ``/proc``
    for its own marker when this answers ``None``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "PRIVATE_SURFACE",
    "PrivateAttribute",
    "cli_process_pid",
    "install_tool_server",
    "missing_private_attributes",
    "remove_tool_server",
    "send_control_request",
]


@dataclass(frozen=True)
class PrivateAttribute:
    """one Claude Agent SDK attribute this module depends on.

    :ivar owner: the SDK class that carries it
    :ivar name: the attribute name
    :ivar kind: what it must be on a freshly constructed instance -- a type, ``"method"`` for a
        coroutine method, or ``"declared"`` for an instance attribute the SDK sets to ``None``
        until the client connects
    """

    owner: str
    name: str
    kind: type | str


#: every SDK attribute this module reads, writes or calls.
#: :func:`missing_private_attributes` checks each against the installed SDK.
PRIVATE_SURFACE: Final[tuple[PrivateAttribute, ...]] = (
    PrivateAttribute("ClaudeSDKClient", "_query", "declared"),
    PrivateAttribute("ClaudeSDKClient", "_transport", "declared"),
    PrivateAttribute("Query", "_send_control_request", "method"),
    PrivateAttribute("Query", "sdk_mcp_servers", dict),
    PrivateAttribute("Query", "_sdk_mcp_bridges", dict),
    PrivateAttribute("SubprocessCLITransport", "_process", "declared"),
)


async def send_control_request(client: Any, request: dict[str, Any], *, timeout: float) -> Any:
    """send one control request to a connected client's CLI and return its answer.

    :param client: a connected ``ClaudeSDKClient``
    :ptype client: Any
    :param request: the request body, ``subtype`` first
    :ptype request: dict[str, Any]
    :param timeout: seconds before the SDK gives up waiting for the answer
    :ptype timeout: float
    :return: the CLI's answer, as the SDK hands it back
    :rtype: Any
    :raises Exception: what the SDK raises -- a bare ``Exception`` for a refusal and for a timeout
        (told apart by ``__cause__``), its own error types for a broken transport
    :raises AttributeError: when the SDK no longer carries the members this function reads
    """
    return await client._query._send_control_request(request, timeout=timeout)


async def install_tool_server(client: Any, name: str, server: Any) -> None:
    """make ``server`` the in-process MCP server the SDK routes ``name`` 's tool calls to.

    The server goes in the SDK's table and gets a bridge of its own, the one the SDK routes a
    ``tools/call`` through; a bridge already under ``name`` is closed first.

    :param client: a connected ``ClaudeSDKClient``
    :ptype client: Any
    :param name: the server name the CLI addresses
    :ptype name: str
    :param server: the server instance
    :ptype server: Any
    :return: nothing
    :rtype: None
    :raises AttributeError: when the SDK no longer carries the members this function reads
    """
    from claude_agent_sdk._internal.sdk_mcp_bridge import SdkMcpBridge  # noqa: PLC0415

    query = client._query
    await _close_bridge(query, name)
    query.sdk_mcp_servers[name] = server
    query._sdk_mcp_bridges[name] = SdkMcpBridge(name, server)


async def _close_bridge(query: Any, name: str) -> None:
    """close and drop the bridge under ``name``, if there is one.

    :param query: the client's ``Query``
    :ptype query: Any
    :param name: the server name
    :ptype name: str
    :return: nothing
    :rtype: None
    """
    bridge = query._sdk_mcp_bridges.pop(name, None)
    if bridge is not None:
        await bridge.aclose()


async def remove_tool_server(client: Any, name: str) -> None:
    """stop the SDK routing tool calls for ``name``; a no-op when nothing is installed under it.

    :param client: a connected ``ClaudeSDKClient``
    :ptype client: Any
    :param name: the server name the CLI addresses
    :ptype name: str
    :return: nothing
    :rtype: None
    :raises AttributeError: when the SDK no longer carries the members this function reads
    """
    await _close_bridge(client._query, name)
    client._query.sdk_mcp_servers.pop(name, None)


def cli_process_pid(client: Any) -> Any:
    """the CLI subprocess's pid as the SDK holds it, or ``None`` when its internals have moved.

    :param client: the connected ``ClaudeSDKClient``
    :ptype client: Any
    :return: whatever the SDK's process object reports as its pid, or ``None``
    :rtype: Any
    """
    result: Any = None
    try:
        result = client._transport._process.pid
    except AttributeError:
        # NOSILENT: the SDK's internals moved; the pool falls back to scanning /proc for its
        # session's marker, and warns only if that finds nothing either.
        result = None
    return result


def _instances() -> dict[str, object]:
    """one freshly constructed, unconnected instance of each SDK class in :data:`PRIVATE_SURFACE`.

    :return: instances by class name
    :rtype: dict[str, object]
    """
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # noqa: PLC0415 -- the claude-cli extra
    from claude_agent_sdk._internal.query import Query  # noqa: PLC0415
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport  # noqa: PLC0415

    transport = SubprocessCLITransport(prompt="", options=ClaudeAgentOptions())
    return {
        "ClaudeSDKClient": ClaudeSDKClient(),
        "Query": Query(transport, is_streaming_mode=True),
        "SubprocessCLITransport": transport,
    }


def missing_private_attributes(surface: tuple[PrivateAttribute, ...] = PRIVATE_SURFACE) -> list[str]:
    """every entry of ``surface`` the installed SDK does not provide as declared.

    Constructs each class without connecting; no CLI starts.

    :param surface: the attributes to check; :data:`PRIVATE_SURFACE` unless a test proves the
        check can fail
    :ptype surface: tuple[PrivateAttribute, ...]
    :return: one line per missing or changed attribute, empty when the surface is intact
    :rtype: list[str]
    """
    instances = _instances()
    missing: list[str] = []
    for attribute in surface:
        instance = instances[attribute.owner]
        label = f"{attribute.owner}.{attribute.name}"
        if not hasattr(instance, attribute.name):
            missing.append(f"{label}: absent")
            continue
        value = getattr(instance, attribute.name)
        if attribute.kind == "method":
            if not inspect.iscoroutinefunction(value):
                missing.append(f"{label}: no longer a coroutine method")
        elif attribute.kind == "declared":
            continue
        elif isinstance(attribute.kind, type) and not isinstance(value, attribute.kind):
            missing.append(f"{label}: expected {attribute.kind.__name__}, found {type(value).__name__}")
    return missing
