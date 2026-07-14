# 3tears-channels

`threetears.channels` -- a unified message protocol with Slack, Discord,
and WebSocket adapters.

## Problem

An agent that needs to talk to users over more than one channel -- Slack,
Discord, a custom WebSocket UI -- ends up with its core logic tangled
around whichever channel's SDK it was written against first, making a
second channel expensive to add.

## What it does

- A unified message protocol independent of any specific channel.
- Slack, Discord, and WebSocket adapters implementing that protocol.

## Design philosophy

Write your agent logic once and deliver it across channels. The adapters
absorb each channel's SDK-specific quirks; the agent-facing surface stays
the same regardless of which channel a message arrived on or is being sent
to. Adding a new channel means writing a new adapter, not touching agent
logic.

## When to adopt

Any agent or app that needs to be reachable from more than one chat
channel, or that wants to swap or add channels without rewriting core
logic.

## Composes with

- [`agent-tools`](agent-tools.md) -- typical consumer for delivering tool
  results back to a channel.
- [`agent-acl`](agent-acl.md) -- authorization for channel operations.
- [`nats`](nats.md) -- cross-pod presence fanout.
- [`agent-wake`](agent-wake.md) -- pulled in by the `webhook` extra for
  webhook signature verification.

## Install

```bash
pip install 3tears-channels
```
