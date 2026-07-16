# 3tears-scheduled-jobs

`threetears.scheduled_jobs` -- payload-agnostic, multi-pod-safe
scheduled-jobs engine. Cross-pod-locked tick loop, reschedule math, and
store protocols.

## Problem

Scheduling recurring or delayed work across multiple pods needs a lock so
two pods don't both fire the same job, plus reschedule math that handles
timezones and DST correctly. Baking domain concepts (agent wake, webhook
delivery, skill invocation) into a scheduler makes it useless for the next
domain that needs the same locking and timing logic.

## What it does

- A cross-pod-locked tick engine: one pod fires a given due job, not all of
  them.
- Reschedule math for recurring jobs.
- Store protocols a consumer implements for its own job/payload shape.
- A default kind+payload store for the common case.

## Design philosophy

Every agent- or skill-specific concept is deliberately stripped out, leaving
only the pure scheduling machinery: locking, reschedule math, and
protocols. Any domain plugs in its own store and dispatch callback. It is
pure-async and fires one tick per call by design -- it does not own the
cadence or the scheduler loop itself; the host decides how often to call it.

## When to adopt

Any app that needs multi-pod-safe scheduled or delayed work and doesn't
want to couple its scheduler to a specific domain's job shape.

## Composes with

- [`nats`](nats.md) -- the distributed lock backing the cross-pod-locked
  tick engine.
- [`agent-wake`](agent-wake.md) -- a sibling scheduling package, not a
  direct dependent; `agent-wake` runs its own tick engine rather than
  building on this one today.

## Install

```bash
pip install 3tears-scheduled-jobs
```
