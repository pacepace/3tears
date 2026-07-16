# 3tears-agent-wake

`threetears.agent.wake` -- foundation for long-running agents. Wake
schedules, fires, and webhook subscriptions.

## Problem

A long-running agent needs to wake itself later -- on a schedule, or in
response to a webhook -- without every app hand-rolling that scheduling and
delivery logic per agent, and without infrastructure concerns (locking,
dispatch) getting tangled up with agent-specific behavior.

## What it does

- Wake schedule and wake fire entities, and webhook subscription records.
- A tick engine and dispatch handler that fire due wakes -- this package
  owns the full loop, not just the schema.
- Agent-facing tools for scheduling and managing webhooks.

## Design philosophy

Wake schedules and fires are entities on the same three-tier base as the
rest of the platform, dispatched by this package's own tick engine rather
than delegating firing to a generic scheduler. It also documents an
accepted trade-off: there is no DB-level foreign key to `conversations`,
because `conversations` has a composite primary key with no standalone
unique column. Orphan rows are possible in principle, but every read is
scoped by namespace, so an orphan is never surfaced to a caller -- a
trade-off shared with `agent-tools` and `agent-skills`.

## When to adopt

Any agent that needs to resume work later without a human re-invoking it --
scheduled check-ins, delayed follow-ups, webhook-triggered continuations.

## Composes with

- [`core`](core.md) -- the three-tier entity/collection base.
- [`agent-skills`](agent-skills.md) -- a real, FK-enforced dependency
  between wake tools and skill invocation.

## Install

```bash
pip install 3tears-agent-wake
```
