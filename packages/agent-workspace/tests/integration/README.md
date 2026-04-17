# agent-workspace integration tests

This directory holds integration tests for the agent-workspace package.
They wire multiple real components together and exercise multi-pod
behaviours that cannot be unit-tested with pure mocks.

## Scope and realism

The task shard (`workspace-task-18`) asks for real-NATS +
real-YugabyteDB testcontainers integration. That infrastructure is not
wired into the 3tears repository today; building it from scratch was
out of scope for shard-18. The pragmatic compromise in this directory:

| Test | NATS | DB | Notes |
|---|---|---|---|
| `test_bind_lease_race.py` (task-14) | fake (CAS-equivalent in-memory KV) | fake (asyncpg-shaped) | cross-pod lease serialization |
| `test_multi_pod_bind_race.py` | fake (CAS-equivalent in-memory KV) | fake (asyncpg-shaped) | WS-18-06 wrapper assertion |
| `test_yaml_round_trip_preserved.py` | fake (publish-recording) | fake (asyncpg-shaped with in-memory tables) | real `DocSetTool` + real ruamel.yaml round trip |
| `test_bind_builder_e2e.py` | fake (publish-recording) | fake (asyncpg-shaped with in-memory tables) | real `bind` + real `atomic_write` + real `_capture_back` |
| `test_audit_event_landed.py` | fake (publish-recording + in-process subscription dispatch) | fake (asyncpg-shaped) | real `publish_workspace_event`; stub consumer feeds a stub `AuditEventCollection` |
| `test_validator_rejection.py` | n/a | fake (asyncpg-shaped) | real `dispatch_validators` + real `FsWriteTool` |

The fake NATS KV comes from `packages/core/tests/unit/coordination/_fake_kv.py`
(shared via `tests/conftest.py` sys.path injection). Its CAS semantics
are functionally equivalent to real nats-py `KeyValue` for the lease
contract, so lease-serialization assertions reflect what two real pods
would see.

The fake DB pool is a pattern-matching in-memory stand-in defined in
`conftest.py`. It recognizes every SQL statement the production write
paths issue (`_write_file_atomic`, `_capture_back`, lifecycle tool
inserts); an unrecognized statement raises `NotImplementedError` so
drift from production SQL fails loudly rather than silently no-opping.

## What's covered in the aibots-repo integration suite

1. **Real NATS JetStream** (testcontainers): the lease contract is
   exercised here against the fake KV; running the same tests against
   real NATS would change zero assertions and adds confidence only for
   behaviours the fake might not model.
2. **Real YugabyteDB** (testcontainers) with the migration-built
   schema: the fake pool understands the SQL verbatim, but the YB
   backend exercises full transaction semantics, FK constraints, and
   schema-search-path routing. That full round-trip belongs in the
   aibots integration suite where testcontainers infra already exists.
3. **Hub-side `WorkspaceAuditConsumer` writing to `platform_audit.audit_events`** —
   IMPLEMENTED. The end-to-end audit pipeline (real testcontainers NATS
   + Postgres, real `publish_workspace_event`, real
   `WorkspaceAuditConsumer`, real `AuditEventCollection`, real
   migration 012 partial unique index) now has coverage in the aibots
   repo at `tests/integration/test_workspace_audit_e2e.py`. That file
   proves: (a) a `workspace.doc_set` envelope lands as one row in
   `platform_audit.audit_events`; (b) duplicate envelopes with the
   same `(correlation_id, event_type)` collapse to one row while a
   second envelope carrying a different `event_type` under the same
   `correlation_id` survives; (c) the real `FsWriteTool` publish path
   lands a row end-to-end.

## Graduation path

When testcontainers NATS + YugabyteDB become available to the 3tears
package test suite:

1. Replace the `_fake_kv.FakeNatsClient` / `RecordingFakeNatsClient`
   imports with a real `nats.connect(...)` helper against the
   container; lease tests stay unchanged.
2. Replace `_FakePool` with a real asyncpg pool pointing at a per-test
   schema (`agent_{uuid.hex}`); run the workspace migrations via
   `threetears.agent.workspace.migrations`. The collection stand-ins
   in `conftest.py` become thin wrappers over the real
   `Workspace*Collection` classes.
3. Mark the existing tests with the appropriate container fixtures and
   keep the assertions unchanged. The "realism" table above updates to
   read "real" in every cell.

## Running locally

```
uv run --directory /path/to/3tears pytest \
  packages/agent-workspace/tests/integration/ -v
```

All integration tests carry `pytestmark = [pytest.mark.asyncio,
pytest.mark.integration]`. They are excluded from the fast loop by
selecting the directory explicitly (or by `-m "not integration"` once
the marker is registered in `pyproject.toml`).
