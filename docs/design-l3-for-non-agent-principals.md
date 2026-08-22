# Design: L3 for a Non-Agent Principal

**Status: design, not a shard.** L3 for a tool pod is conditional on a pod
actually needing durable state. `coll-task-07c` gives tool pods L1+L2, which is a
complete pattern and covers most cases.

**`build-plan-principal-convergence.md` Chunk 13 is the further-developed
version of this design and governs where the two differ.** It settles what this
document left open:

- **Schema name: `ns_<hex>`**, derived from the owning namespace row's id.
  `agent_<hex>` stays and no live schema is renamed.
- **The isolation bar is a deliverable, not a test note:** a non-agent
  namespace's `schema_name` must never name an `agent_*` schema, enforced at the
  **write**, in the declaration handler where the value is derived -- not at the
  read.
- The `workspace` type deliberately points at the owner agent's schema, so the
  new shape is explicitly barred from that behaviour rather than inheriting it.
- `idx_namespaces_schema_name_non_workspace` already makes non-workspace
  `schema_name` unique; the new bar must compose with it.
- Only `namespace_provisioner.py` and `datasources/data_sync.py` carry the narrow
  `^agent_[0-9a-f]{32}$` pin. `broker/proxy.py`'s regex is already
  `^[a-z][a-z0-9_]*$` -- which is also why it is **not** the isolation boundary an
  earlier design doc claimed.

What this document still contributes, and Chunk 13 does not address, is the
sentinel-`agent_id`-vs-generalized-principal question below.

Confirm the requirement before turning either into shards.

---

## The gap

Agent identity is welded into the L3 path at the schema level.
`14-eng-ai-bot/src/aibots/hub/broker/namespace_provisioner.py:62-70`:

```python
:return: schema name in format agent_{full_uuid_hex} (38 chars total)
return f"agent_{extract_agent_id_hex(agent_id)}"
```

The broker sets `search_path` to that schema for the requesting agent. A tool pod
has no `agent_id`, so there is no schema for it and no grant reaching one.

This is the isolation model, not an oversight. The question is what the
equivalent unit of isolation is for a principal that is not an agent.

---

## Generalize the existing provisioner; do not write a parallel one

`namespace_provisioner.py` is agent-specific in **seven** places:
`build_schema_name`; `_SCHEMA_NAME_RE` (`^agent_[0-9a-f]{32}$`);
`extract_agent_id_hex` (public, in `__all__`); `build_agent_namespace_name`
(uses the agent plural prefix); the `provision_agent_namespace(agent_id: UUID, ...)`
entry point; its two derivation calls; and the `NamespaceType.AGENT` /
`owner_agent_id` write.

What *is* already generic, and is the reason to parameterize rather than
rewrite: the idempotent short-circuit, CREATE SCHEMA + migrate +
drop-on-failure, and concurrent-peer convergence. That last one is hard-won
behaviour nobody should reimplement.

**Parameterize `(schema_prefix, namespace_type, owner column)` on the existing
function.** `run_migrations_for_schema(db_url, schema_name)` in
`broker/migrations.py` has a schema-generic *signature* but an agent-scoped
*body* -- its docstring opens "apply agent-scope migrations" and it hard-codes
the agent runner. So migration support is a runner-registration change plus a
body change, not a new path.

The provisioner correctly stays in the **hub**, not 3tears: it writes
`platform.*` on the hub's own pool and is deploy policy.

---

## The unresolved question: sentinel id vs. generalized principal

There is an existing precedent that contradicts the obvious instinct, and it must
be addressed rather than silently overruled.

The instinct -- and an earlier draft's anti-pattern -- is "do not give a non-agent
an `agent_id` so it fits the existing path; that collapses two identities and
every downstream RBAC decision inherits the confusion."

But `packages/registry/.../rbac_stack.py` **already does exactly that**:
`REGISTRY_SERVICE_SENTINEL_AGENT_ID` is a uuid5 sentinel, defined with a written
rationale and passed further down as `NatsProxyL3Backend(agent_id=str(...))` for
a non-agent principal. And all seven broker request models in
`hub/broker/proxy.py` carry `agent_id: UUID`.

Two coherent answers:

1. **Generalize `NatsProxyL3Backend` and the broker envelope to a principal.**
   The right answer, and large -- it touches seven request models and the broker's
   resolution path.
2. **Keep the sentinel pattern** and state why it stands for the registry and for
   tool pods alike.

Pick one. Leaving both in the tree is the actual failure mode, and it is the
state today by accident.

---

## The naming decision, if a new schema kind is introduced

The platform already has the shape: a **namespace**. `namespace_type` includes
`tool` with plural prefix `tools`, and `platform.namespaces` carries
`schema_name`. Tools live under `tools.` and a tool namespace is
`tools.<mcp_name>.<version>`.

So the candidate is: the tool namespace owns the schema, as the agent owns
`agent_{hex}`. One schema per tool namespace, shared by every replica.

Derive from a **UUID**, never from the human-readable namespace string -- the
hex-of-UUID choice exists to eliminate collisions and stay inside PostgreSQL's
63-char limit, and a name-derived schema reintroduces both. Note the version
question here differs from the L2 one: an L2 cache may cold-start on a version
bump, but a durable schema must not be orphaned by one. If the schema keys on
the versioned namespace, a version bump strands the data.

---

## The security constraint

The user's condition, verbatim: *"as long as we can secure this so that some app
that is using us for its collections can't read and write platform objects."*
That is the whole gate.

`validate_platform_writes` in `hub/broker/proxy.py` is the existing enforcement -- 
sqlglot AST-based, and **write-only**. Its own docstring says SELECT targets are
"informational"; read isolation rests on the upstream ACL system-namespace
short-circuit, not on the gate.

`search_path` is also weaker than it looks: a schema-qualified
`SELECT * FROM platform.agents` does not need it to resolve. Confirm what
actually stops a qualified read from a broker client before assuming anything
does.

And: a `schema_name` reaching `SET search_path` from an untrusted field was a
live hole fixed earlier in this work, with the fix pinning `schema_name` as a
derived, agent-immutable column. Same discipline applies -- **derived by the hub,
never accepted from the caller.**

Non-negotiable: a non-agent principal must be **refused** on both a qualified
read and a qualified write against `platform.*`, proven by a live probe, not by
reading the gate.

---

## Requirements, when this becomes shards

| ID | Requirement | Priority |
|----|-------------|----------|
| L3T-01 | A non-agent principal can be provisioned an L3 schema owned by its namespace, via the **parameterized existing** provisioner | P0 |
| L3T-02 | The schema name is derived from a UUID by the hub; never from the caller, never from a display string | P0 |
| L3T-03 | The broker resolves the schema from the authenticated principal, never from the request envelope | P0 |
| L3T-04 | A non-agent principal is refused a qualified read of `platform.*`, proven live | P0 |
| L3T-05 | Same for a qualified write | P0 |
| L3T-06 | A tool pod cannot reach another tool namespace's schema | P0 |
| L3T-07 | Provisioning happens at registration, not on first use by the pod | P1 |
| L3T-08 | Migration runner registration for the new schema kind | P1 |
| L3T-09 | The sentinel-vs-principal question is resolved one way across the tree | P0 |

---

## Known adjacent hazards

- **The datasource tool path** reaches `platform.*` on the hub's own superuser
  pool, in process, with no platform-write gate, and its borrowed-pool branch
  never sets `search_path` (recorded in `14-eng-ai-bot/docs/design-one-write-path.md`).
  Not this work's job to fix; do not model the new path on it.
- **DDL and DML must be separate transactions on YugabyteDB.** Provisioning is
  DDL; the bookkeeping row is DML. Two `execute()` calls.
- **Agent-create DDL cannot join a YugabyteDB transaction**, which is why an
  operation-as-transaction design was rejected earlier. Do not assume atomicity
  across the DDL and the registry write.

---

## Verification, when built

Live probes from the principal's credential, each **refused**:

```sql
SELECT * FROM platform.agents;
UPDATE platform.namespaces SET name = 'x';
SELECT * FROM <other_namespace_schema>.anything;
```

and each succeeding against its own schema. A unit test over the sqlglot gate is
necessary and not sufficient: the gate has been reasoned about correctly and been
wrong repeatedly in this work, and every one of those was caught by running a
probe rather than by reading the code.
