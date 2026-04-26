# 3tears Agent Tools

Tool framework for LLM agents. Provides tool routing, execution, context management, MCP integration, and a set of builtin tools.

Part of the [3tears](https://github.com/pacepace/3tears) framework.

## ToolServer baseline audit (audit-task-01 Phase 3)

`ToolServer.handle_call` stamps every dispatch with a unified `AuditEvent` envelope (`event_type='tool.call'`) via `threetears.agent.audit.publish_audit`. The baseline emission fires in a `finally` block so success, failure (tool returned `success=False`), and error (tool raised) outcomes all produce a row. Identity axes carry from the active `ToolCallScope` (`actor_user_id`, `calling_agent_id`, `owner_agent_id`, `customer_id`, `correlation_id`); `resource_namespace_id` / `resource_namespace_type` stay `None` at the baseline layer since the tool resolves its target inside `execute`. Per-tool additive events (e.g. `workspace.fs_write`) still publish via `publish_audit` and ride alongside the baseline row under the same `correlation_id` — the `(correlation_id, event_type)` partial unique index on `platform_audit.audit_events` keeps them distinct. Emission is fire-and-forget: NATS publish failures log WARN and never taint the tool's response.

## Tool-as-namespace emission (namespace-task-01 Phase 2, three-tier-task-01 Phase F)

Tool namespace materialization is hub-owned. `ToolServer.publish_registration` writes the `RegistrationManifest` (carrying `pod_id` + `tools` + the new `owner_agent_id` / `customer_id` envelope fields), and the hub-side `ToolNamespaceEmitter` (in `aibots.hub.tools.namespace_emitter`) subscribes to `{ns}.tools.register` and upserts one `platform.namespaces` row of type `tool` per tool — this is the SOLE writer in the platform.

Agent-spun ToolServers stamp `agent_id` + `customer_id` on the `RegistrationManifest` so the hub emitter lands rows with the right owner scope; platform-built-in pods (admin tool server, datasource tool pod) leave both `None` and the row lands with NULL owner columns (admitted under v061's widened `namespaces_row_scope_customer_ck` carve-out for `tool` type alongside `system` / `model`).

The canonical `name` shape is `tools.<sanitized-mcp>.<sanitized-version>` (per `build_namespace_name`); `metadata` carries the pre-sanitized natural-identity fields `mcp_name` / `mcp_version` / `pod_id` so downstream pattern matching (hub access materializer agent.yaml `access.tools` patterns + registry authorizer canonical-name lookup) does not need to reverse the sanitization rules. Deterministic `uuid5` derived from `(mcp_name, version, owner_agent_id_hex)` keeps concurrent emitters race-safe via `ON CONFLICT (id) DO UPDATE`.

The legacy `ToolServer.register_tool` / `deregister_tool` helpers still emit through an injected `namespace_collection` for callers that wire one explicitly, but in production deployments the hub-side emitter is the source of truth and the in-process emission is redundant.

## Installation

```bash
pip install 3tears-agent-tools

# Optional extras for builtin tools
pip install "3tears-agent-tools[calculator]"   # simpleeval
pip install "3tears-agent-tools[units]"        # pint
pip install "3tears-agent-tools[fetch]"        # trafilatura
pip install "3tears-agent-tools[document]"     # PyMuPDF, python-docx, openpyxl
pip install "3tears-agent-tools[all]"          # everything
```

## Components

### ToolRouter

Routes user messages to the appropriate tool using a lightweight LLM call. Includes recall-intent detection to avoid re-invoking tools when users ask about previous results.

```python
from threetears.agent.tools import ToolRouter, is_recall_intent

# Quick check — no LLM call needed
if is_recall_intent("show me what the calculator said"):
    # User wants to recall, not invoke

# Full routing with LLM
router = ToolRouter(chat_model)
decision = await router.route(user_message, tool_descriptions)
# decision.tool_name, decision.reasoning
```

### ToolExecutor

Invokes a tool-LLM: sends the user message to a secondary model configured for a specific task.

```python
from threetears.agent.tools import ToolExecutor

executor = ToolExecutor()
result = await executor.invoke_with_tools(
    chat_model=tool_model,
    user_message="What is 42 * 17?",
    tools=[calculator_tool],
    tool_name="calculator",
)
# result.content, result.tool_calls
```

### ToolContextManager

Tracks tool invocations and results across a conversation for recall support.

```python
from threetears.agent.tools import ToolContextManager

ctx = ToolContextManager()
await ctx.record_invocation("calculator", "42 * 17", "714")
await ctx.get_recall_context("calculator")  # Returns formatted recall string
```

### McpClient

MCP (Model Context Protocol) integration for connecting to external tool servers.

```python
from threetears.agent.tools import McpClient

async with McpClient(server_config) as client:
    tools = await client.list_tools()
    result = await client.invoke_tool("tool_name", {"param": "value"})
```

### Builtin Tools

Register all builtin tools at once:

```python
from threetears.agent.tools import register_builtins, ToolRegistry

registry = ToolRegistry()
register_builtins(registry)
# Registers: calculator, unit_converter, dice_roller, date_time,
#            random_number, web_fetch, text_transform, parse_document
```

### Todo Tools

Todo list management behind a storage protocol:

```python
from threetears.agent.tools import TodoStorage, load_todo_tools_from_storage

class MyTodoStorage(TodoStorage):
    async def add(self, conv_id, user_id, title, list_name, msg_id) -> dict: ...
    async def list_all(self, conv_id) -> list[dict]: ...
    # ... other methods

tools = load_todo_tools_from_storage(my_storage, snapshot_callback=on_snapshot)
```

### Protocols

For media-related capabilities, implement these protocols:

```python
from threetears.agent.tools import (
    ImageGenerationBackend,
    MediaStorage,
    VisionProvider,
    TranscriptionProvider,
)
```

### Document Parsing

Parse PDF, DOCX, XLSX, and plain text with optional OCR:

```python
from threetears.agent.tools import parse_document, OcrConfig

result = await parse_document(
    file_bytes=data,
    filename="report.pdf",
    ocr_config=OcrConfig(enabled=True),
)
# result.sections — list of DocumentSection with title, content, page numbers
```
