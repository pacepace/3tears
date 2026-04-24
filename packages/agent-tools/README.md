# 3tears Agent Tools

Tool framework for LLM agents. Provides tool routing, execution, context management, MCP integration, and a set of builtin tools.

Part of the [3tears](https://github.com/pacepace/3tears) framework.

## ToolServer baseline audit (audit-task-01 Phase 3)

`ToolServer.handle_call` stamps every dispatch with a unified `AuditEvent` envelope (`event_type='tool.call'`) via `threetears.agent.audit.publish_audit`. The baseline emission fires in a `finally` block so success, failure (tool returned `success=False`), and error (tool raised) outcomes all produce a row. Identity axes carry from the active `ToolCallScope` (`actor_user_id`, `calling_agent_id`, `owner_agent_id`, `customer_id`, `correlation_id`); `resource_namespace_id` / `resource_namespace_type` stay `None` at the baseline layer since the tool resolves its target inside `execute`. Per-tool additive events (e.g. `workspace.fs_write`) still publish via `publish_audit` and ride alongside the baseline row under the same `correlation_id` — the `(correlation_id, event_type)` partial unique index on `platform_audit.audit_events` keeps them distinct. Emission is fire-and-forget: NATS publish failures log WARN and never taint the tool's response.

## Tool-as-namespace emission (namespace-task-01 Phase 2, three-tier-task-01 Phase F)

`ToolServer.register_tool` writes a `platform.namespaces` row of type `tool` for every registered tool (name shape `tool:<mcp_name>:<version>`); `deregister_tool` removes the paired row for each `name@version` key that was present in the in-memory registry. The `ToolServer` constructor takes a required keyword-only `namespace_collection: NamespaceCollection` alongside the existing `agent_id` and `customer_id` — typed `Any` at the package boundary per the Phase F layering decision because `threetears.agent.tools` sits below the concrete `NamespaceCollection` implementation in the import graph; the real Collection is wired by the bootstrap caller. Platform-built-in tool pods pass `namespace_collection=None` (suppresses emission) and leave `agent_id`/`customer_id` unset, landing rows with NULL owner columns (same shape as `shared`-type workspaces) when emission is enabled elsewhere; production agent-spun pods MUST supply a Collection or namespace materialization silently falls behind and rbac resolution fails open. `register_tool` persists the namespace via `NamespaceCollection.save_entity` using a deterministic `uuid5` derived from `(mcp_name, version, agent_id_hex)` so concurrent registrations converge through `ON CONFLICT (id) DO UPDATE`; `deregister_tool` pairs each removed key with a `NamespaceCollection.delete` on the same derived id. Emission failures raise unchanged so misconfigured deployments surface immediately rather than silently dropping namespaces. The resulting namespace id is what `threetears.registry.rbac_authorizer.RbacEvaluatorAuthorizer` resolves against on every tool dispatch.

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
