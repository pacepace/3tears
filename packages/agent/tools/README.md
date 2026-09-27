# 3tears Agent Tools

Tool framework for LLM agents. Provides tool routing, execution, context management, MCP integration, and a set of builtin tools.

Part of the [3tears](https://github.com/pacepace/3tears) framework.

## ToolServer baseline audit

`ToolServer.handle_call` stamps every dispatch with a unified `AuditEvent` envelope (`event_type='tool.call'`) via `threetears.agent.audit.publish_audit`. The baseline emission fires in a `finally` block so success, failure (tool returned `success=False`), and error (tool raised) outcomes all produce a row. Identity axes carry from the active `ToolCallScope` (`actor_user_id`, `calling_agent_id`, `owner_agent_id`, `customer_id`, `correlation_id`); `resource_namespace_id` / `resource_namespace_type` stay `None` at the baseline layer since the tool resolves its target inside `execute`. Per-tool additive events (e.g. `workspace.fs_write`) still publish via `publish_audit` and ride alongside the baseline row under the same `correlation_id`, which ties a request's events together and is not a deduplication key: every `tool.call` in a turn shares it, and each is its own row. Each envelope's `id` is its identity, so a JetStream redelivery (which repeats the `id`) collapses to one row. Emission is fire-and-forget: NATS publish failures log WARN and never taint the tool's response.

## Tool-as-namespace emission

Tool namespace materialization is platform-owned. `ToolServer.publish_registration` writes the `RegistrationManifest` (carrying `pod_id` + `tools` + the `owner_agent_id` / `customer_id` envelope fields), and a platform-side namespace emitter subscribes to `{ns}.tools.register` and upserts one `namespaces` row of type `tool` per tool. This is the sole writer in the platform.

Agent-spun ToolServers stamp `agent_id` + `customer_id` on the `RegistrationManifest` so the emitter lands rows with the right owner scope; platform-built-in pods (admin tool server, datasource tool pod) leave both `None` and the row lands with NULL owner columns (admitted under the widened `namespaces_row_scope_customer_ck` carve-out for `tool` type alongside `system` / `model`).

The canonical `name` shape is `tools.<sanitized-mcp>.<sanitized-version>` (per `build_namespace_name`); `metadata` carries the pre-sanitized natural-identity fields `mcp_name` / `mcp_version` / `pod_id` so downstream pattern matching (platform access materializer agent.yaml `access.tools` patterns + registry authorizer canonical-name lookup) does not need to reverse the sanitization rules. Deterministic `uuid5` derived from `(mcp_name, version, owner_agent_id_hex)` keeps concurrent emitters race-safe via `ON CONFLICT (id) DO UPDATE`.

`ToolServer` holds no `NamespaceCollection` and has no constructor parameter to take one: `register_tool` / `deregister_tool` publish the manifest and nothing else. The pod-side emitter that once wrote and deleted these rows is deleted -- its write could not land (the agent's L3 proxy resolves platform-scoped writes to the per-agent `agent_<hex>` schema, which has no `namespaces` table) and its delete raised on every call (it passed a bare `UUID` to a Collection keyed on the composite `(row_scope, namespace_id)`). `packages/agent/tools/tests/enforcement/test_no_agent_side_namespace_writes.py` fails a build that brings any of it back.

## Tool pod lifecycle: shutdown and owner process

`ToolServerBootstrap` runs every tool pod's lifecycle. Two guarantees about how a pod stops:

- **SIGTERM always ends the process.** SIGTERM and SIGINT run one shutdown path, `ToolServerBootstrap.shutdown_server`. If the server's own shutdown raises (a NATS drain against a reconnecting server raised `ConnectionResetError` in production) or overruns its bound, the failure is logged once at ERROR with its cause, the serve loop is ended anyway, and `run()` exits `EX_SOFTWARE` (70). A caller driving `run_async()` itself receives `ToolPodShutdownError`, chaining the cause. `ToolServer.shutdown` also releases `serve()` in a `finally`, so a pod not run by the bootstrap does not hang either.
- **A pod can follow the process that owns it.** Set `THREETEARS_TOOL_POD_OWNER_PID` to the pid of the process that spawned the pod, and the pod shuts itself down through the same path once that process no longer exists (a WARNING names the pid). Opt-in; unset watches nothing. The check is `kill(pid, 0)`, portable across macOS and Linux; when the owner is the pod's parent, being reparented counts as the owner gone too. The value must be a decimal integer greater than 1 and not the pod's own pid -- `0`, negatives, `1`, blank and non-integers are refused at startup as a `ToolPodConfigError`, and `run()` exits `EX_CONFIG` (78).

| Variable | Default | Meaning |
|----------|---------|---------|
| `THREETEARS_TOOL_POD_OWNER_PID` | unset | pid of the owning process; the pod exits when it is gone |
| `THREETEARS_TOOL_POD_OWNER_POLL_INTERVAL_SECONDS` | `1.0` | how often the owner is checked (positive) |
| `THREETEARS_TOOL_POD_SHUTDOWN_TIMEOUT_SECONDS` | `20.0` | bound on the server's shutdown, and separately on the teardown after it (positive) |

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

# Quick check -- no LLM call needed
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
from threetears.agent.tools import DocumentParseError, OcrConfig, parse_document

try:
    result = await parse_document(
        data,
        "application/pdf",
        "report.pdf",
        ocr_config=OcrConfig(enabled=True),
    )
except DocumentParseError as exc:
    ...  # exc.reason is "unsupported_type" or "parse_failed"; the parser's error is exc.__cause__
# result.sections -- list of DocumentSection with heading, content, page numbers
```

A document that cannot be read raises `DocumentParseError`; its failure is never returned as the document's text.
