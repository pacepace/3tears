"""The MCP server installs and runs without the grant store.

The grant store (``LocalGrantAuthorizer`` over ``mcp_tool_grants``, kept coherent across pods
by an epoch broadcast) is the ``grants`` extra. It is the only part of the package that needs
core, epoch, nats, sqlalchemy and asyncpg, so a host that brings its own ``Authorizer`` must be
able to serve MCP with none of them installed. Where the extra is installed they load with the
package, at its top (owner ruling, 2026-10-24: no lazy imports in serving code).

Each check runs in a fresh interpreter with the extra's modules refused at import, as an install
without the extra refuses them.
"""

from __future__ import annotations

import subprocess
import sys

import threetears.mcp
from threetears.mcp import rbac

#: refuses the grant store's distributions, as an install without the extra would
BLOCK_GRANTS = """
import importlib.abc, sys
_REFUSED = (
    "threetears.core", "threetears.epoch", "threetears.nats",
    "sqlalchemy", "asyncpg", "aiosqlite", "nats", "nkeys", "uuid_utils",
)
class _NoGrants(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == refused or name.startswith(refused + ".") for refused in _REFUSED):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _NoGrants())
"""

SERVE_ONE_CALL = """
import asyncio, json
from uuid import uuid4

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from threetears.mcp import EnvVarIdentityProvider, Identity, McpServer, McpTool, ToolRegistry


class AllowAll:
    async def allows(self, identity: Identity, permission: str) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


async def echo(**kwargs):
    return json.dumps(kwargs)


async def main() -> None:
    registry = ToolRegistry()
    registry.register(
        McpTool(
            name="echo",
            description="echo the arguments back as JSON",
            input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
            required_permission="t.echo.call",
            handler=echo,
        )
    )
    server = McpServer(
        name="lean",
        identity_provider=EnvVarIdentityProvider(principal_id=uuid4()),
        authorizer=AllowAll(),
        registry=registry,
    )
    await server.start()
    app = server.build_http_app()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mcp.test")
    try:
        async with app.router.lifespan_context(app), client:
            async with streamable_http_client("http://mcp.test/mcp", http_client=client) as streams:
                read, write = streams[0], streams[1]  # 1.x yields a third value (the session id), 2.x two
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("echo", {"value": "hi"})
                    assert result.model_dump(by_alias=True)["isError"] is False, result
                    assert json.loads(result.content[0].text) == {"value": "hi"}
    finally:
        await server.stop()


asyncio.run(main())
loaded = sorted(
    name for name in sys.modules
    if any(name == refused or name.startswith(refused + ".") for refused in _REFUSED)
)
assert not loaded, loaded
print("served")
"""


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """run *code* in a fresh interpreter.

    :param code: the program
    :ptype code: str
    :return: the finished process
    :rtype: subprocess.CompletedProcess[str]
    """
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=60)


def test_a_server_with_its_own_authorizer_serves_a_call_without_the_grant_store() -> None:
    result = _run(BLOCK_GRANTS + SERVE_ONE_CALL)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "served"


def test_reaching_a_grant_store_name_without_the_extra_names_the_extra() -> None:
    code = BLOCK_GRANTS + (
        "import threetears.mcp\n"
        "for name in ('McpToolGrantCollection', 'McpToolGrantEntity', 'mcp_tool_grants_table'):\n"
        "    try:\n"
        "        getattr(threetears.mcp, name)\n"
        "    except ImportError as exc:\n"
        "        assert '3tears-mcp[grants]' in str(exc), str(exc)\n"
        "    else:\n"
        "        raise AssertionError(f'{name} resolved without the extra')\n"
        "print('refused')"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused"


def test_building_the_grant_authorizer_without_the_extra_names_the_extra() -> None:
    code = BLOCK_GRANTS + (
        "from threetears.mcp import LocalGrantAuthorizer\n"
        "async def no_grants():\n"
        "    return []\n"
        "try:\n"
        "    LocalGrantAuthorizer(grant_loader=no_grants)\n"
        "except ImportError as exc:\n"
        "    assert '3tears-mcp[grants]' in str(exc), str(exc)\n"
        "else:\n"
        "    raise AssertionError('LocalGrantAuthorizer built without the extra')\n"
        "print('refused')"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused"


def test_the_grant_store_names_on_the_package_are_the_rbac_module_s_own() -> None:
    assert threetears.mcp.McpToolGrantCollection is rbac.McpToolGrantCollection
    assert threetears.mcp.McpToolGrantEntity is rbac.McpToolGrantEntity
    assert threetears.mcp.mcp_tool_grants_table is rbac.mcp_tool_grants_table
