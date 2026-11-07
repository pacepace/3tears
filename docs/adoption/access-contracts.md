# 3tears-access-contracts

`threetears.access.contracts` -- dependency-free identity and authorization
vocabulary shared by `iam` and `agent-acl`.

## Problem

Authentication (`3tears-iam`) and authorization (`3tears-agent-acl`) guard the
same sensitive actions from two sides: agent-acl refuses them under
impersonation, and iam requires a recent sign-in for them. Each check is only
correct while both agree on what the actions are. Sharing the type by having
one package import the other made a service that only signs people in install
the RBAC evaluator and its database stack, just to name an action.

## What it does

- `ImpersonationCategory`, the fixed set of six sensitive actions both checks
  enforce.

## Design philosophy

The package holds only vocabulary both sides name, and depends on nothing, so
naming a shared value never costs either side the other's closure. A value one
side uses alone stays with that side. `threetears.agent.acl.ImpersonationCategory`
is this same class, so existing imports keep working.

## When to adopt

Code that names a sensitive-action category without needing the evaluator or
the login protocols: a policy table, an audit consumer, an admin surface.

## Composes with

- [`iam`](iam.md) -- the step-up check, which reads the categories.
- [`agent-acl`](agent-acl.md) -- the impersonation deny-list, which re-exports
  them.

## Install

```bash
pip install 3tears-access-contracts
```
