# 3tears-agent-knowledge

`threetears.agent.knowledge` -- governed-knowledge retrieval and injection
for LLM agents: concepts, playbook entries, three-scope shadow merge, and a
`wrap_model_call` injection middleware.

## Problem

An agent needs access to governed knowledge -- concepts, playbook entries
-- that can be defined at multiple scopes (user, customer, platform) and
merged predictably, then injected into the model call without every
consumer reimplementing the merge and injection logic.

## What it does

- Playbook-entry and concept collections, accessed over the platform RBAC
  proxy.
- `KnowledgeIntegration` wiring.
- A `wrap_model_call` / `awrap_model_call` injection middleware that puts
  governed knowledge into the model call automatically. It deliberately
  does not use LangChain's `before_model` hook -- that hook crashes against
  real gateways when it produces multiple non-consecutive system messages.

## Design philosophy

Built on the shared three-scope shadow-merge authority that lives in core
`threetears.knowledge`, but lives in its own agent-side package solely
because its collections depend on `threetears.agent.acl`, which core
cannot depend on. The split exists to respect the dependency direction, not
because the two halves are conceptually separate -- read this alongside
`agent-identity`'s contrast: knowledge is a scope-layered shadow-merge, not
a temporal version history.

## When to adopt

Any LangGraph agent that needs governed, scope-aware knowledge injected
into its context automatically rather than hand-assembled per call.

## Composes with

- [`core`](core.md) -- the shadow-merge authority lives in core's
  `threetears.knowledge`.
- [`agent-acl`](agent-acl.md) -- the RBAC proxy this package's collections
  depend on.
- [`agent-identity`](agent-identity.md) -- a related but distinct model;
  see "Design philosophy."
- [`agent-memory`](agent-memory.md) -- the injection middleware attributes
  retrieved knowledge through memory's embedding-attribution scope.
- [`langgraph`](langgraph.md) -- the `wrap_model_call` middleware targets
  LangGraph agents.

## Install

```bash
pip install 3tears-agent-knowledge
```
