# How to add a migration

This guide is the single blessed path for adding a new schema migration to
any 3tears package or to the hub. There is one runner, one template,
one path. If you find yourself reaching for an alternative — a separate
runner, a feature flag, a "temporary" alembic file — stop and rethink:
the framework deliberately offers no second option.

## Decide the scope

| Scope | Owner | When to use |
|-------|-------|-------------|
| `MigrationScope.AGENT` | 3tears packages and hub features that ship per-agent tables | Anything stored in an agent's private YugabyteDB schema (workspaces, memories, checkpoints, etc.) |
| `MigrationScope.PLATFORM` | The hub | Tables in the shared `platform` or `platform_audit` schema (customers, agents, gateway state, audit events, etc.) |

If unsure: ask. Mis-scoping a migration is recoverable but disruptive.

## File layout

Put the migration in a `migrations/` subpackage of the owning Python
package, with one Python file per version:

```
my_package/
  migrations/
    __init__.py                  # exports register(runner) and the callables
    v001_create_widgets.py       # one async callable per file
    v002_add_color_column.py
```

The version prefix in the filename is informational; what matters is the
integer passed to `pkg.version(N)` in `__init__.py`.

## Migration body template

```python
# my_package/migrations/v003_add_priority_index.py
"""
my_package v003: add priority index to widgets.

short paragraph explaining why this migration exists and what it does
to the schema. include any non-obvious consequences (lock duration on
big tables, cross-schema FKs, etc.).
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

log = get_logger(__name__)


_CREATE_PRIORITY_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_widgets_priority "
    "ON widgets (priority)"
)


async def add_priority_index(store: DataStore) -> None:
    """
    create idx_widgets_priority for fast priority-ordered lookups.

    :param store: DataStore bound to per-agent schema
    :ptype store: DataStore
    """
    log.info("adding idx_widgets_priority")
    await store.execute(_CREATE_PRIORITY_IDX_SQL)
```

Rules:

- **Idempotent statements only.** Use `CREATE TABLE IF NOT EXISTS`,
  `ADD COLUMN IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`, etc. The
  runner will not re-run already-applied migrations under normal
  operation, but partial-failure replay must be safe.
- **No schema qualification for agent migrations.** The L3 broker sets
  `search_path` to the target agent schema before the runner runs, so
  unqualified DDL lands correctly. Hard-coding `agent_<hex>.` breaks
  the cross-schema reuse the runner exists to enable.
- **Schema qualification for platform-scope cross-schema writes.** If a
  platform migration writes into `platform_audit` (a different schema
  from the platform default), prefix the table reference. Examples
  live in `aibots/hub/migrations/v005_create_audit_schema.py`.
- **One migration body per file.** Multiple bodies in one file make
  partial-failure rollback ambiguous.
- **Single return.** Per CLAUDE.md, business-logic functions have one
  return statement. Guard clauses at the start are fine.

## Wire the version into the package's register function

```python
# my_package/migrations/__init__.py
"""my_package migrations entry point."""

from __future__ import annotations

from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

from my_package.migrations.v001_create_widgets import create_widgets
from my_package.migrations.v002_add_color_column import add_color_column
from my_package.migrations.v003_add_priority_index import add_priority_index

PACKAGE_NAME = "my_package"


def register(runner: MigrationRunner) -> PackageMigrations:
    """
    register my_package migrations with the runner.

    :param runner: canonical migration runner
    :ptype runner: MigrationRunner
    :return: populated PackageMigrations registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.AGENT,
        depends_on=("agent_workspace",),  # if relevant; otherwise omit
    )
    pkg.version(1)(create_widgets)
    pkg.version(2)(add_color_column)
    pkg.version(3)(add_priority_index)
    runner.register(pkg)
    return pkg
```

If your package depends on tables from another package (e.g. a foreign
key into `workspaces.id`), declare it via `depends_on=("agent_workspace",)`.
The runner topologically sorts so dependencies always apply first.

## Wire the package into the hub's runner builder

For agent-scope packages, add a single line to
`14-eng-ai-bot/src/aibots/hub/broker/migrations.py::build_agent_runner`:

```python
from my_package.migrations import register as register_my_package
...
def build_agent_runner() -> MigrationRunner:
    runner = MigrationRunner()
    register_workspace(runner)
    register_memory(runner)
    register_langgraph(runner)
    register_my_package(runner)  # <-- new line
    return runner
```

For platform-scope packages, add a similar line to
`14-eng-ai-bot/src/aibots/hub/common/migrations.py::build_platform_runner`.

## Test the migration

Every new migration ships with at least one unit test that drives the
callable against a `_CaptureStore` stub and asserts on the emitted SQL.
Pattern shown in `aibots/tests/unit/hub/broker/test_namespaces.py::TestAddNamespacesMigration`:

```python
class _CaptureStore:
    def __init__(self) -> None:
        self.executed: list[str] = []

    async def execute(self, sql: str, *params) -> str:
        self.executed.append(sql)
        return "EXECUTE"


@pytest.mark.asyncio
async def test_creates_idx_widgets_priority() -> None:
    store = _CaptureStore()
    await add_priority_index(store)
    sql = "\n".join(" ".join(s.split()) for s in store.executed)
    assert "idx_widgets_priority" in sql
```

For DDL that must run end-to-end (cross-schema writes, complex
constraints), add an integration test against a real Postgres or
YugabyteDB via testcontainers.

## Test in isolation

`MigrationRunner.apply_package(store, "my_package")` runs only the
named package's migrations. Useful in package-local test harnesses
that want the package's tables present without applying every other
package's tables. Caller must apply any depended-on packages first
(`apply_package` does not auto-resolve `depends_on`).

## What NOT to do

- **No back-compat shims.** When you change a migration's behavior,
  ship the change and update call sites in one commit. No
  `USE_NEW_X` runtime flags. The CLAUDE.md NO-SHIMS rule is enforced
  on review.
- **No editing applied migrations.** Once a migration is in `main`
  and any production schema has it recorded in `_schema_migrations`,
  treat it as immutable. Add a new migration to evolve.
- **No cross-package reaches.** Don't `INSERT INTO` or `SELECT` from
  another package's tables in a migration. Migrations create / alter
  schema, they do not seed data across boundaries.
- **No autogeneration.** Hand-write migrations. The framework offers
  no autogen and never will: silent destructive autogen is what
  alembic was retired for.

## Checksum verification

If your change touches existing translated migrations (i.e. you are
modifying one of the migrations that came from the alembic tree), run
the translation check script against a fresh Postgres pair to confirm
no schema drift was introduced. The script lives at
`14-eng-ai-bot/scripts/migration_translation_check.py` and requires
`POSTGRES_URL_OLD` + `POSTGRES_URL_NEW` env vars plus an
`--alembic-ref <pre-cutover commit SHA>` flag. See the script's
docstring for full usage.

## See also

- `migrations-task-01-canonical-runner.md` — task shard that
  established this pattern.
- `threetears.core.data.migrations.template.MIGRATION_FILE_TEMPLATE`
  — string template you can render programmatically.
- `aibots/src/aibots/hub/migrations/__init__.py` — biggest in-tree
  example: 13 platform-scope versions translated 1:1 from alembic.
