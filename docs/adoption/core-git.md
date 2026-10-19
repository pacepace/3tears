# 3tears: the git backend

`threetears.core.backends.git` -- a git working tree as the durable tier (L3): rows as files, every change
set one commit, one writer per repository. Part of [`core`](core.md), an L3 beside
`threetears.core.backends.sql` and `threetears.core.backends.nats_proxy`; import it by its own path
(`threetears.core.backends` does not load it).

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

## Use

```python
from threetears.core.backends.git import (
    GitDurableStore, GitRepo, GitRepoConfig, JsonCodec, TableLayout, TemplateLayout, bind_collections,
)

layout = TemplateLayout([
    TableLayout("keepers", "{character}/keepers/{batch}/{tile}.json", ("character", "batch", "tile"), JsonCodec()),
    TableLayout("recipes", "{character}/recipe.json", ("character",), JsonCodec()),
])
repo = GitRepo(GitRepoConfig(key="characters", work_tree=path, remote_url=origin), layout, nats=nats)
bind_collections(repo, {"keepers": keepers, "recipes": recipes})
await repo.open()
store = GitDurableStore(repo)            # a DurableStore: DurableStoreCollection runs on it

async with repo.change("aria: bank 3 keepers") as cs:   # ONE commit
    await keepers.save_entity(k1, conn=cs)
    await recipes.save_entity(recipe, conn=cs)
```

- **Forwarded changes are sealed.** A request forwarded to the writer carries 3tears' service
  envelope, signed with the deployment's `forward_signing_seed` (required with NATS;
  `new_signing_seed()` makes one) and spent once in a `ReplayGuard`; the writer refuses anything else
  (`ForwardRefusedError`). Every process of a deployment shares the seed; grant the principal the
  `envelope_bucket` KV bucket alongside the lease bucket.
- **Saved now, committed later.** A product that commits only when the user means it gives a folder
  a `PendingTail`; `OpLogTail` keeps the saved edits in a NATS `OpLog`:

  ```python
  tail = OpLogTail(oplog, scope="stories/one", decode=decode_op, apply=rebuild_from_head)
  await repo.add_tail(tail)
  await tail.save(repo, payload=op_bytes, op_id=op_id)        # on the writer: logged + written, no commit
  await repo.commit_intent("stories/one", "edit")              # one commit, trailer "Op-Seq: <n>"
  ```

  The ops after the folder's newest `Op-Seq` trailer are uncommitted; a new writer rebuilds them on
  HEAD, so a pod that dies before the commit loses nothing.

Two things to get right when wiring it:

- **Bind before you open.** `bind_collections(repo, ...)` (or `on_rows_changed`) before
  `await repo.open()`: opening reports the commits made while no store was running.
- **Give new rows a `date_updated` key** (`None` is enough; the save stamps it). A row saved
  without one has no version, so later saves of it cannot be compare-and-swapped.

The NATS wire and disk names keep the name `git-store`: the forward family `git-store`, the subject
`{ns}.git-store.advanced`, the KV buckets `git-store-writers` and `git-store-envelopes`, and the
crash journal `.git/git-store/inflight.json`.

## Design philosophy

Git is the one copy of the truth; caches only ever evict, never guess. Every guard is a refusal
that writes nothing. The lease picks a writer cheaply; the fences that hold are git's own (the
index lock, the compare-and-swap check, the fast-forward-only push).

## When to adopt

Adopt it for human-authored data that should live in git and still be read through collections.
Not for high-rate machine data: every change set is a commit.

## Composes with

- [`core`](core.md) -- it is part of core: `DurableStoreCollection`, `CallerTransaction`, `KVLease`.
- [`nats`](nats.md) -- the writer lease, owner-routed forwarding, the advanced broadcast.
- [`observe`](observe.md) -- logging and the background loops.

## Install

```bash
pip install '3tears[nats]'
```

It ships with core; the `nats` extra brings the NATS client its writer lease, forwarding and
broadcast run over. Needs the `git` program on the path (no pygit2).
