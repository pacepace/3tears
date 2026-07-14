# 3tears-agent-skills

`threetears.agent.skills` -- procedural memory. Skill definitions and
invocation history.

## Problem

An agent that can only follow instructions given fresh every time can't
build a reusable playbook. Procedural memory -- "here's how I do X" -- is a
different shape from factual memory and a different shape from tool
definitions; conflating any of them forces awkward workarounds.

## What it does

- Per-agent/per-user labeled markdown procedures.
- Tool-surface modifications tied to a skill.
- Invocation history.

## Design philosophy

Unlike its sibling `agent-memory`, this package takes no direct ACL/tools
dependency. Its tool factories accept a thin `SkillRegistryClient` Protocol
instead of importing `agent-acl`/`agent-tools` directly -- trading a small
consumer-side adapter for zero hard dependencies beyond core. This
establishes a platform-wide principle worth naming explicitly: use a direct
dependency when the types are genuinely part of the contract; use a
Protocol when the contract is method-shaped and the dependency is
incidental.

## When to adopt

Any agent that should learn and reuse procedures rather than being told the
same steps every time.

## Composes with

- [`core`](core.md) -- the three-tier collection base.
- [`agent-tools`](agent-tools.md) -- consumed through a `SkillRegistryClient`
  Protocol, not a direct dependency.

## Install

```bash
pip install 3tears-agent-skills
```
