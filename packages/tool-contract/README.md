# 3tears-tool-contract

The `TearsTool` contract on its own: `TearsTool`, `ToolResult`, `MCPToolDefinition`, the closed set of result
codes, and the input coercion `TearsTool.run` applies. A tool written against it is served on the platform mesh by
3tears-agent-tools and over MCP by 3tears-mcp, with one definition.

## Install

```bash
pip install 3tears-tool-contract          # the contract: 3tears-observe and 3tears-tool-schema, nothing else
pip install "3tears-tool-contract[mcp]"   # plus its MCP face, over 3tears-mcp
```

The base installs no NATS client, Postgres driver, LangGraph or LangChain:
`tests/enforcement/test_tool_contract_installs_light.py` checks what the lock resolves an install of it (and of
3tears-mcp's base) to. 3tears-agent-tools depends on it, and `threetears.agent.tools.base_tool` and
`threetears.agent.tools.coercion` re-export the same objects, so a tool imported from either path is the same class.

## A tool

```python
from threetears.tool_contract import MCPToolDefinition, TearsTool, ToolResult

class Echo(TearsTool):
    face_mcp = True

    def mcp_name(self) -> str:
        return "example.echo"

    def mcp_version(self) -> str:
        return "1.0.0"

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="echo the text back",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        )

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content=kwargs["text"])
```

`run` is the callable surface: it coerces loose model-supplied values (an empty or JSON-encoded string for an
object or array field) toward the declared types, then calls `execute`. A subclass overrides `execute`, never
`run`. A failed `ToolResult` may name one of `TOOL_RESULT_ERROR_CODES` (`CONFLICT`, `MALFORMED_REQUEST`,
`PERMISSION_DENIED`, `TOOL_AUTHORIZATION_UNAVAILABLE`, `TOOL_NOT_READY`) in `error_code`; any other code is refused
when the result is built.

`face_rest`'s type, `RestAffordance`, is defined in 3tears core (the registry reads it from the manifest too); the
contract imports it for type checking only, so it needs no core at run time. A tool that declares a REST face is a
platform tool, and installs core with agent-tools.

## The MCP face

`threetears.tool_contract.mcp_face.mcp_tool` (the `mcp` extra) turns a `TearsTool` into the `McpTool` that
3tears-mcp's `McpServer` serves:

```python
from threetears.mcp import ErrorCodes, McpServer, ToolRegistry
from threetears.tool_contract.mcp_face import mcp_tool

registry = ToolRegistry()
registry.register(mcp_tool(Echo(), required_permission="example.echo", name="echo"))
server = McpServer(..., registry=registry, error_codes=ErrorCodes([...]))
```

| From the tool | To the MCP call |
|---|---|
| `mcp_schema().description`, `.input_schema`, `.output_schema` | the listed tool's description, input schema and output schema |
| `mcp_name()` | its name, unless `name=` gives the one the MCP face lists (an MCP client already prefixes a tool with its server's name, and some LLM tool APIs reject a dot) |
| a successful `ToolResult` | its `content` as the text of the result; when the tool declares an output schema, the content parsed as a JSON object is the `structuredContent` too |
| a failed `ToolResult` | `isError`, its `content` (else its `error`) as the text, and its code on the protocol result (`_meta["threetears.mcp/error"]`): `error_code` when set, else `metadata["error_code"]` when it is a string, else `TOOL_ERROR`. A code the server never registered answers `HANDLER_ERROR` |
| an exception from `run` | `HANDLER_ERROR`, or the code the server mapped the exception to |

The call context reaches `execute` the way it reaches a plain 3tears-mcp handler: an `execute` that declares a
parameter annotated `CallContext` (from `threetears.mcp`) is passed the call's context in it, as a keyword that
`run` passes through unchanged. Give it a default of `None` so the same tool runs on the mesh, where there is none:

```python
async def execute(self, *, text: str, ctx: CallContext | None = None) -> ToolResult:
    if ctx is not None:
        await ctx.report_progress(1, 2, "echoing")
    ...
```

`mcp_tool` refuses a tool whose `face_mcp` is `False`: that flag is the tool's own declaration of whether it is
reachable over MCP. `MCPToolDefinition.timeout_seconds` is not applied by the face; the MCP client's own deadline
governs the call.
