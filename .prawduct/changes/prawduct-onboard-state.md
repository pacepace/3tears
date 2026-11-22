---
schema: 1
type: governance
scope: prawduct-onboard-state
date: 2026-11-22
title: Prawduct's governance state is tracked in the repo
---
The repo ran under the prawduct plugin with none of its core state: a blanket `.prawduct/` rule
in `.gitignore`, added in June to keep one session file out of a commit, hid everything under it.
This change narrows that rule (and `.claude/*`) so session files stay ignored while the governance
state is tracked, scaffolds `project-state.yaml`, `backlog.md`, the `changes/` fragment directory,
`project-preferences.md` and `.claude/rules/learnings/core.md`, moves the governance anchor into
`AGENTS.md` (imported by `CLAUDE.md`), adds the hook that warns when a session starts without the
plugin, records the API versioning decision and the testament evidence command, and archives every
build plan whose work has shipped.

### Local files under the newly tracked paths
Before this change a session could keep throwaway files under `.prawduct/artifacts/` and they
stayed invisible. After a worktree merges this, any such file shows up as untracked, and a
`git add -A` would commit it. At the time of the change only the `evals/main` worktree had any:
raw blind-read eval data, now ignored by `.prawduct/artifacts/blind-read-*/`, and 31 design notes,
second opinions and upgrade notes, which that line's owner decides to commit, move or delete.
Check `git status` for untracked `.prawduct/` files after merging this.

## Consumer note
none: repo governance only; no package code or published surface changed
