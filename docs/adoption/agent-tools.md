# 3tears-agent-tools

`threetears.agent.tools` -- tool framework. `TearsTool` base, `ToolServer`
for NATS registration plus dispatch plus audit, context management, built-in
tools, and tool-group aliases.

## Problem

An LLM agent needs a consistent way to define tools, register them so
other pods can discover and call them, dispatch calls with proper context,
and audit what ran -- without every app rebuilding that plumbing around a
specific LLM framework's tool interface.

## What it does

- `TearsTool` -- the base class for defining a tool.
- `ToolServer` -- registers tools over NATS, dispatches calls, and audits
  them.
- Context management for tool execution.
- Built-in tools and tool-group aliases for common capability bundles.

## Design philosophy

The platform, not the package, is the sole writer of tool-namespace rows --
registration state stays consistent across pods because there's exactly one
writer, not one per consumer. Tool execution is audited through the shared
`agent-audit` envelope rather than a bespoke one, keeping tool-call history
queryable alongside every other domain's audit trail.

## When to adopt

Any LLM agent that needs to call tools, especially across more than one
pod, where tool discovery and routing need to be consistent rather than
hardcoded per caller.

## Composes with

- [`agent-audit`](agent-audit.md) -- tool calls are audited through the
  shared envelope.
- [`agent-memory`](agent-memory.md) -- built-in context wiring references
  memory collections directly.
- [`langgraph`](langgraph.md) -- tool calls integrate with LangGraph-built
  agents.
- [`conversations`](conversations.md) -- tool context items key off
  `conversation_id`.
- [`media-contracts`](media-contracts.md) -- tools that accept or produce
  media use this contract rather than a provider-specific type.

Authorization is deliberately kept out of this package's dependency graph:
it accepts an ACL cache typed as `Any` rather than importing `agent-acl`
directly, so a host wires in whatever authorizer it uses.
[`registry`](registry.md) is referenced only in docstring type hints, not a
real dependency -- add it yourself if you need multi-pod tool routing.

## Install

```bash
pip install 3tears-agent-tools
# built-in tool bundles are gated behind extras, e.g.:
pip install "3tears-agent-tools[document]"   # pdf/docx/xlsx tools
pip install "3tears-agent-tools[vision]"     # image tools
pip install "3tears-agent-tools[ocr]"        # OCR tools
pip install "3tears-agent-tools[all]"        # every built-in bundle
```
