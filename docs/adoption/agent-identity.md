# 3tears-agent-identity

`threetears.agent.identity` -- versioned identity blocks for LLM agents:
self-evolution via a linear version chain with propose/consent/apply,
rationale, and rollback.

## Problem

An agent that can improve how it presents itself or reasons about its own
behavior needs a safe way to change its own identity -- persona,
reinforcement rules, self-improvement notes -- without a human losing
visibility into what changed, why, or being unable to undo it.

## What it does

- Identity blocks (persona, reinforcement, anti-sycophancy guard,
  self-improvement scratch, presence voice) as a linear parent-pointer
  version chain.
- Exactly one `active` version per `(agent_id, customer_id, user_id,
  block_key)`.
- Every version carries its content, a rationale, a content hash, and its
  lineage.
- Non-destructive rollback -- nothing is lost, only superseded.

## Design philosophy

The substrate for agent self-evolution: the agent proposes a change to its
own identity, a human consents, and the change applies. Consent is tiered
-- identity-shaping blocks require explicit human consent before they apply;
routine blocks auto-apply immediately, with an async veto (a human can
`reject()` an already-applied version out of the pending queue after the
fact). This is explicitly contrasted with
`agent-knowledge`'s scope-layering shadow-chain: identity is a *temporal
history* of one agent's evolving self, not a layered merge across scopes.

## When to adopt

Any agent designed to adapt its own persona or behavior over time based on
its own deliberation, where a human needs an auditable, reversible record
of what changed.

## Composes with

- [`core`](core.md) -- the three-tier entity/collection base.
- [`agent-acl`](agent-acl.md) -- owner RBAC on propose/consent/apply.
- [`langgraph`](langgraph.md) -- identity lifecycle events integrate with
  LangGraph-built agents.
- [`agent-knowledge`](agent-knowledge.md) -- a related but distinct model;
  see "Design philosophy" for the contrast.

## Install

```bash
pip install 3tears-agent-identity
```
