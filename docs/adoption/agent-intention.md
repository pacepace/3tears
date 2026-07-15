# 3tears-agent-intention

`threetears.agent.intention` -- standing-wants corpus for LLM agents:
intention lifecycle, salience decay, embedding dedup, and private
deliberation tools.

## Problem

An agent that notices something worth following up on -- "ask about the
migration," "check whether the wake threads are firing" -- needs somewhere
to hold that want without either forgetting it or repeating it every single
turn. Without salience decay and dedup, a held want either nags constantly
or silently disappears.

## What it does

- Intentions: agent-authored deliberation output, each with its own status
  lifecycle.
- Salience decay so an intention fades rather than persisting indefinitely
  at full priority.
- Embedding dedup so the same want isn't recreated repeatedly.
- Restraint brakes so a want surfaces at most occasionally, not every turn.

## Design philosophy

Kept as a separate package rather than a memory `kind` in `agent-memory`
because memory is a hard-delete-only fact taxonomy with no lifecycle, and
reusing its enum would pollute a user-fact taxonomy with an agent-internal
control construct -- and would force an irreversible change on every one of
memory's six existing consumers. The separation is a direct application of
the platform's "compose existing primitives before building new ones"
principle applied in reverse: sometimes the right call is a new, small
package specifically *because* forcing the concept into an existing one
would corrupt that package's contract.

## When to adopt

Any agent that does private deliberation and needs to hold onto a want
across turns without either nagging or forgetting.

## Composes with

- [`core`](core.md) -- the three-tier entity/collection base.
- [`agent-acl`](agent-acl.md) -- authorization for intention access.
- [`langgraph`](langgraph.md) -- private deliberation tools integrate with
  LangGraph-built agents.
- [`agent-memory`](agent-memory.md) -- deliberately separate; see "Design
  philosophy."

## Install

```bash
pip install 3tears-agent-intention
```
