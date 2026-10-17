# 3tears-git-store

`threetears.git_store` -- a git working tree as the durable tier (L3): rows as files, every change
set one commit, one writer per repository.

## Problem

Some data is authored rather than generated -- stories, characters, recipes -- and wants git's
history, diffs and reverts, and hand edits in a text editor. Kept in git beside a database copy, the
two drift, and a save from a stale copy silently drops newer work. Kept only in git, many processes
writing one working tree race each other and every cache goes stale.

## What it does

- `GitDurableStore` implements the structured durable-store contract (`DurableStore`,
  `BulkDurableStore`, `BulkDeletingStore`), so a `DurableStoreCollection` runs on git with L1 and L2
  in front, unchanged.
- A product brings a `RowCodec` (`JsonCodec` ships) and a `Layout` (`TemplateLayout` places rows by
  path templates; a product can write its own).
- `GitRepo.change()` gathers collection saves into ONE commit. A save carries the version it read;
  a row changed since refuses the whole set (`ChangeConflictError`). A hand edit in the way is never
  written over (`HandEditPendingError`); `commit_paths` adopts hand edits.
- One writer per repository holds a NATS lease (`KVLease.hold`); other processes forward change
  sets to it and get the commit or the same typed refusal back.
- Pushes are plain, never forced. Clones that are not the writer follow origin on the writer's
  broadcast. A crash mid-change is put back by the next writer.
- `bind_collections` drops every row any commit changed -- hand commits and pulled commits
  included -- from the collections' caches. `GitRepos` / `RoutedGitStore` keep many repositories.

## Design philosophy

Git is the one copy of the truth; caches only ever evict, never guess. Every guard is a refusal
that writes nothing. The lease picks a writer cheaply; the fences that hold are git's own (the
index lock, the compare-and-swap check, the fast-forward-only push).

## When to adopt

Adopt it for human-authored data that should live in git and still be read through collections.
Not for high-rate machine data: every change set is a commit.

## Composes with

- [`core`](core.md) -- `DurableStoreCollection`, `CallerTransaction`, `KVLease`.
- [`nats`](nats.md) -- the writer lease, owner-routed forwarding, the advanced broadcast.
- [`observe`](observe.md) -- logging and the background loops.

## Install

```bash
pip install 3tears-git-store
```

Needs the `git` program on the path.
