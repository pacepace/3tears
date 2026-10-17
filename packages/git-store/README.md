# 3tears-git-store

A git working tree as a durable store (L3). Rows are files; every change set is one commit; one
process per repository writes, and every other process hands its change to that writer.

```python
from threetears.git_store import (
    GitDurableStore, GitRepo, GitRepoConfig, JsonCodec, TableLayout, TemplateLayout, bind_collections,
)

layout = TemplateLayout([
    TableLayout("keepers", "{character}/keepers/{batch}/{tile}.json", ("character", "batch", "tile"), JsonCodec()),
    TableLayout("recipes", "{character}/recipe.json", ("character",), JsonCodec()),
])
repo = GitRepo(GitRepoConfig(key="characters", work_tree=path, remote_url=origin), layout, nats=nats)
await repo.open()
store = GitDurableStore(repo)            # a DurableStore: DurableStoreCollection runs on it
bind_collections(repo, {"keepers": keepers, "recipes": recipes})

async with repo.change("aria: bank 3 keepers") as cs:   # ONE commit
    await keepers.save_entity(k1, conn=cs)
    await recipes.save_entity(recipe, conn=cs)
```

What it guarantees:

- **A save that would overwrite a newer change is refused, not applied.** A write carries the
  `date_updated` it read (compare-and-swap); if the file changed since, the whole change set is
  refused (`ChangeConflictError`, a `ConcurrentModificationError`) and nothing is written.
- **Hand edits are never overwritten.** A file someone edited and has not committed blocks a write
  to it (`HandEditPendingError`). `repo.commit_paths(paths, message)` commits hand edits; a raw
  `git commit` works too, and the writer notices HEAD moved.
- **Every cache drops what a commit changed**: this store's own commits through `CallerTransaction`,
  and every other commit (hand commits, commits pulled from origin) through `bind_collections`.
- **One writer per repository**, by a NATS lease (`KVLease.hold`). Other processes forward their
  change sets to it. Without NATS (`nats=None`) the process is always the writer.
- **Origin is never forced.** Pushes are plain; a refused push stops pushing and shows in
  `repo.status()` until a person reconciles.
- **A crash mid-change is recovered**: a journal in the git directory names the files being written,
  and the next writer puts them back as HEAD has them.

Many repositories at once (one per entity, one shared, or several): `GitRepos` opens them on first
use; `RoutedGitStore` is one store over them, routing each row by its key.

Two things to get right when wiring it:

- **Bind before you open.** `bind_collections(repo, ...)` (or `on_rows_changed`) before
  `await repo.open()`: opening reports the commits made while no store was running.
- **Give new rows a `date_updated` key** (`None` is enough; the save stamps it). A row saved
  without one has no version, so later saves of it cannot be compare-and-swapped.

Needs the `git` program. No pygit2.
