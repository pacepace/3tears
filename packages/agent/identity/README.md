# 3tears-agent-identity

Versioned identity blocks for LLM agents — the substrate for **self-evolution**:
the agent proposes changes to its own identity, a human consents, and nothing is
lost, only superseded.

## Model

An identity block (persona, reinforcement, anti-sycophancy guard, self-improvement
scratch, presence voice) is a **linear parent-pointer version chain**. Exactly one
`active` version exists per `(agent_id, customer_id, user_id, block_key)`; each
version carries its `content`, a `rationale`, a `content_hash`, and its lineage.

Lifecycle: **propose (agent) → consent (user) → apply**, with non-destructive
rollback. Consent is **tiered** — tier-1 (identity-shaping) blocks require human
consent before applying; tier-2 (routine) blocks auto-apply with an async veto.

This differs from `knowledge`'s shadow-chain (which is scope-layering, not temporal
version history): identity versions are a temporal chain with rollback.

## Contents (v0.15.0)

- **T2.1a:** `identity_versions` table + `IdentityVersionsCollection` + the block /
  status / tier value types + migration v001.
- **T2.1b:** lifecycle ops (propose/consent/reject/rollback) + FrameworkEvents +
  owner RBAC + the `identity_propose` tool.

Consumers own the apply path (subscribe/authorize/write); this package provides the
store, the version chain, the lifecycle, and the events.
