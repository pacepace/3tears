# 3tears-agent-workspace

Workspace primitives for 3tears agents: workspace entities, sandbox, format handlers, and tools.

This package provides `WorkspaceConfig` and related Pydantic models consumed by the platform SDK to declare per-agent workspace configuration (templates, bind root, read/write allow-lists, validator hooks).

## Audit emission

Workspace tools publish unified `threetears.agent.audit.AuditEvent` envelopes via `threetears.agent.audit.publish_audit` on the dotted subject `{ns}.audit.workspace.{event_type_verb}` (e.g. `workspace.fs_write`, `workspace.materialize`, `workspace.rollback`). This package does NOT define its own audit envelope type and does NOT run its own consumer — `audit-task-01` Phase 3 retired the per-domain `WorkspaceAuditEnvelope` + hub-side `WorkspaceAuditConsumer`; the hub's `UnifiedAuditConsumer` subscribes to the whole `{ns}.audit.>` subtree and owns persistence. Every dispatch also produces a baseline `event_type='tool.call'` row stamped by `ToolServer` (see `3tears-agent-tools`); the workspace-specific event is additive under the same `correlation_id`.
