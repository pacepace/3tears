# Implementing a 3tears datasource driver

Companion to the `Driver` ABC in `src/threetears/datasources/drivers/base.py`.
Read this end-to-end before writing a new driver. Every contract here
is load-bearing; the enforcement tests under `tests/enforcement/` exist
to catch drift after merge.

If something below contradicts the ABC's docstring, the ABC's docstring
wins -- treat it as the canonical reference. Open an issue if you find
a mismatch.

## Quick orientation

Concrete drivers live alongside the ABC, one module per backend:

- `asyncpg_driver.py` -- Postgres / Yugabyte / agent-internal
- `redshift_driver.py` -- Amazon Redshift via `redshift_connector`
- `snowflake_driver.py` -- Snowflake via `snowflake-connector-python`
- `bigquery_driver.py` -- BigQuery via `google-cloud-bigquery`

Concrete classes are **never re-exported** from
`threetears.datasources` or `threetears.datasources.drivers`. Callers
construct them exclusively through `create_driver(config, hub_l3_pool=...)`.
The factory holds the lazy-import contract; re-exporting the classes at
either package root would break it.

## What the ABC promises callers

The ABC is the contract every backend honours. Surface:

```python
async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]
async def execute(self, sql: str, *params: Any) -> None
async def fetch_iter(self, sql: str, *params: Any) -> AsyncIterator[dict[str, Any]]
async def list_tables(self, schemas: list[str]) -> list[TableRow]
async def list_columns(self, schemas: list[str]) -> list[ColumnRow]
async def table_hashes(self, schemas: list[str]) -> dict[tuple[str, str], str]
async def test_connection(self) -> None
async def close(self) -> None
```

`fetch_iter` is the only non-abstract method -- it has a working
default impl that materializes via `fetch()` then yields. Drivers with
a native server-side cursor (asyncpg, redshift_connector) override it
to actually stream. Drivers without native streaming (BigQuery, the
Snowflake stub) inherit the default.

## Cancellation contract (the most-likely-to-be-violated invariant)

When the awaiting coroutine is cancelled, your driver MUST attempt to
cancel the in-flight query at the backend BEFORE re-raising
`CancelledError`. There is exactly one place this lives: the
`Driver._with_cancellation` helper in `base.py`. Every driver routes
every backend call through it:

```python
async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
    async with self._pool.acquire() as conn:
        return await self._with_cancellation(
            lambda: conn.fetch(sql, *params),
            cancel_callback=lambda: conn.cancel(),  # backend hook
        )
```

Anti-patterns the enforcement tests catch:

- Per-driver `try: ... except asyncio.CancelledError:` blocks. If we
  see one in a concrete driver, the AST walker fails the build.
- Swallowing `CancelledError`. The helper re-raises; your driver
  must not catch it again.
- Forgetting to await an async cancel callback. The helper handles
  that for you -- pass the callable directly.

Use the `DriverCancellationContractTest` mixin in
`tests/unit/_helpers/cancellation_contract.py` to prove your driver
propagates correctly. Subclass the mixin in your concrete-driver test
module, supply `make_slow_driver()` and `slow_sql()`, and the mixin
runs the canonical cancellation assertions.

## Placeholder translation

Callers always pass `$1`-style placeholders. Concrete drivers translate
to their dialect by calling
`_translate_placeholders(sql, target_style)` from
`drivers/_util.py`. Never roll your own regex -- the helper handles
the edge cases (`$10` vs `$1`, escaped `$$`, `'$1'` inside a string
literal) that bite a per-driver implementation.

Target styles:

| Style       | Mapping              | Use case                               |
|-------------|----------------------|----------------------------------------|
| `asyncpg`   | `$1` (no-op)         | asyncpg native -- skip translation     |
| `pyformat`  | `$1` -> `%s`         | psycopg2 / `redshift_connector`        |
| `numeric`   | `$1` -> `:1`         | Oracle-style numeric                   |
| `named-at`  | `$1` -> `@p1`        | BigQuery named query parameters        |

For BigQuery, the `@pN` naming encodes the original ordinal so you can
build the `bigquery.ScalarQueryParameter` list positionally.

## Sync-bridge for blocking backend libs

`redshift_connector`, `snowflake.connector`, and
`google.cloud.bigquery.Client` are all sync. Wrapping them in async
calls naively (e.g. `asyncio.to_thread` direct) loses cancellation
semantics and lets you accidentally exhaust the default executor.

Instead, every sync-backed driver owns an `AsyncSyncBridge`
(`drivers/_sync_bridge.py`):

```python
class RedshiftDriver(Driver):
    def __init__(self, config: RedshiftConnectionConfig) -> None:
        self._bridge = AsyncSyncBridge(
            max_workers=config.executor_max_workers,  # NEVER an inline literal
            name="rs-bridge",
        )

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        conn = await self._acquire_connection()
        return await self._bridge.to_thread_with_cancel(
            lambda: _sync_fetch(conn, sql, params),
            cancel_cb=conn.cancel,  # backend abort hook
        )

    async def close(self) -> None:
        await self._bridge.close()
        # then release any connection cache, etc.
```

`AsyncSyncBridge` enforces three rules:

1. Bounded `ThreadPoolExecutor` sized from the `ConnectionConfig`
   (the `executor_max_workers` field). The
   `test_no_hardcoded_pool_params` enforcement test fails the build if
   you inline a literal.
2. Cancellation fires `cancel_cb` before the `CancelledError` reaches
   the asyncio caller. Async callbacks are awaited automatically.
3. `close()` uses `shutdown(wait=False)`. `wait=True` deadlocks the
   asyncio event loop because the worker threads may be awaiting a
   coroutine that can't run while the loop is blocked.

AST enforcement: concrete drivers may not instantiate
`ThreadPoolExecutor` directly. The bridge is the single source of
truth.

## Row shapes (the Tier-2 hash hazard)

`list_tables` returns `list[TableRow]`; `list_columns` returns
`list[ColumnRow]`. The TypedDicts live in `base.py`:

```python
class TableRow(TypedDict):
    table_schema: str
    table_name: str

class ColumnRow(TypedDict):
    table_schema: str
    table_name: str
    column_name: str
    data_type: str
    is_nullable: str   # RAW warehouse string -- NOT a bool
    ordinal_position: int
```

The `is_nullable` field is the raw warehouse value (`'YES'`, `'NO'`,
or `''`). NOT a bool. The Tier-2 column hash in `datasource-task-02`
computes MD5 over a concatenation of column metadata using the raw
nullable string. If you convert to bool here, the Python-side hash
diverges from the warehouse-side MD5 and the change-probe breaks for
this datasource.

`data_type` is also the raw warehouse-reported type string. Don't
normalize it.

## Observability contract

Decorate your async query-emitting methods with `@_observed("<backend>")`
from `base.py`:

```python
class AsyncpgDriver(Driver):
    @_observed("asyncpg")
    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        ...
```

The decorator auto-emits:

- `datasource.driver.query.duration{driver_type, datasource_name}` -- histogram (seconds)
- `datasource.driver.error{driver_type, error_kind}` -- counter

`CancelledError` is NOT counted as an error -- it's a normal
propagation. The decorator skips the error counter on `CancelledError`
and re-raises.

Cache / saturation / cancellation-fired metrics are driver-specific
and emit-by-hand:

- `datasource.driver.cancellation.fired{driver_type}` -- bump in your
  cancel callback when you actually fire a backend cancel.
- `datasource.driver.executor.saturation{datasource_name}` -- gauge,
  sampled before / after each bridged call.
- `datasource.driver.cache.hit{datasource_name}` /
  `datasource.driver.cache.miss{datasource_name}` -- bumped in your
  connection cache code path.

Set `self._datasource_name` in your driver's `__init__` if you want
the auto-emitted metrics to carry the datasource name (otherwise they
carry `"unknown"`).

When `opentelemetry.metrics` isn't installed at runtime, the decorator
is a pure passthrough (single bool check per call). No-op cost.

## Lazy-import discipline

The lazy-import contract from DS-09-09 says: importing
`threetears.datasources`, `threetears.datasources.drivers`, or
`threetears.datasources.drivers.factory` MUST NOT load any of
`asyncpg`, `redshift_connector`, `snowflake.connector`, or
`google.cloud.bigquery` into `sys.modules`.

Concretely:

- The backend lib import lives inside the `case` arm of
  `create_driver` that dispatches to the driver using it. NOT at
  module top of `factory.py`. NOT at module top of the driver module
  if you can help it (deferred imports inside the driver's
  `__init__` or first-use method are fine).
- `drivers/__init__.py` exports ONLY the ABC, factory, and
  TypedDicts. No concrete driver class is re-exported here.
- `_sync_bridge.py` is the ONE place permitted to import
  `ThreadPoolExecutor` at module top. Everything else is lazy.

`tests/unit/test_lazy_imports.py` audits this in a fresh subprocess
on every test run. If your driver adds a backend-lib import somewhere
that pollutes the package roots, that test fails.

## close() concurrency

`close()` is single-shot. Concurrent calls and concurrent in-flight
`fetch`/`execute`/`fetch_iter` while `close()` is running are
undefined behaviour. Drivers SHOULD:

- Set `self._closed = False` in `__init__`.
- Set `self._closed = True` at the top of `close()`.
- Reject subsequent operations with `RuntimeError`.

```python
async def close(self) -> None:
    self._closed = True
    await self._bridge.close()  # bridge handles its own idempotency
    # release connection cache, etc.
```

```python
async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
    if self._closed:
        raise RuntimeError(f"{type(self).__name__} is closed")
    ...
```

The contract does NOT require concurrent-call safety. Caller's
responsibility.

## Enforcement tests

Live under `tests/enforcement/`:

- `test_secrets_typed.py` -- AST walker; credential-named pydantic
  fields must be `SecretStr` unless they end in `_env`.
- `test_no_hardcoded_pool_params.py` -- AST walker; pool / executor /
  timeout kwargs must be read from `ConnectionConfig` fields, not
  inlined literals.
- `tests/unit/test_lazy_imports.py` -- runtime audit; package roots
  do not load backend libs.
- (planned) AST walker forbidding direct `ThreadPoolExecutor`
  instantiation outside `_sync_bridge.py`.
- (planned) AST walker forbidding `try: ... except asyncio.CancelledError:`
  in driver modules outside `base.py`.

All four run as part of the default `pytest` invocation; they're not
slow.

## Env-gated live tests

Concrete-driver tests that connect to a real backend belong under
`tests/integration/` with the `live` pytest marker:

```python
@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("REDSHIFT_LIVE_TEST_URL"),
    reason="set REDSHIFT_LIVE_TEST_URL to enable",
)
async def test_against_real_redshift() -> None:
    ...
```

The `live` marker is registered in `pyproject.toml`. CI runs unit +
enforcement; live tests run on demand. The env-var pattern keeps
secrets out of the repo and out of CI defaults.

## Stub driver patterns (post-shard-12)

The Snowflake + BigQuery drivers ship as **stubs** in
`datasource-task-12` — every abstract method raises
`NotImplementedError`. They exist to prove the ABC handles two
different driver shapes (stateful-pooled DB-API vs stateless
HTTPS) before the concrete implementations land.

When adding a NEW backend that doesn't have a real implementation
yet, follow this shape:

### 1. Module docstring is the roadmap

The stub's module-level docstring MUST cover, in this order:

1. **Backend library** — PyPI URL + minimum version + extras key on
   this package's `pyproject.toml`.
2. **Connection lifecycle** — pool? client? per-call? what's the
   shape of the long-lived object?
3. **Placeholder style** — `%s` / `:N` / `@pN`. Reference
   `_translate_placeholders` with the target style; never reimplement
   the regex.
4. **Cancellation mechanism** — the specific API call. Wire it into
   `Driver._with_cancellation` (NOT a per-method try/except).
5. **Sync-to-async bridge** — almost always `AsyncSyncBridge` from
   `_sync_bridge.py`. The next implementer reads this and knows NOT
   to instantiate `ThreadPoolExecutor` directly.
6. **Row-shape pinning** — `TableRow` / `ColumnRow`. `is_nullable`
   MUST be the raw warehouse string (or document the mapping if
   the backend reports differently, like BigQuery's
   `NULLABLE`/`REQUIRED`/`REPEATED` modes).
7. **`information_schema`-equivalent source** — pg-style view? REST
   API? document the per-table column-shape hash strategy.
8. **Pool / executor / timeout knobs** — read from the
   `ConnectionConfig`. NO inline literals. Enforcement test catches.
9. **Secret handling** — `SecretStr` resolution at last moment +
   exception sanitization pattern.
10. **Observability** — same metric names; same `@_observed`
    decorator (`driver_type=` matches the backend slug).
11. **Anything that does NOT transfer** — backend-specific
    deviations (no `pg_sleep`, no `information_schema`-as-table,
    statement_timeout semantics, etc).
12. **CI-required live test** — env-gate pattern; CI fails when
    the live test can't run rather than silently skipping.

### 2. Stub body shape

```python
async def list_tables(self, schemas: list[str]) -> list[TableRow]:
    raise NotImplementedError(
        f"{type(self).__name__}.list_tables is not yet implemented. "
        "See module docstring + docs/datasource-task-NN-...md "
        "for the implementation roadmap."
    )
```

The message MUST name the method AND point at the doc/docstring.
"I'll fix this stub later" silent partial implementations are
caught by the stub tests' message-pattern assertion.

### 3. `__init__` does real work

The stub `__init__` MUST:
- accept and validate the corresponding `ConnectionConfig` member
- accept the `datasource_name: str = "unknown"` kwarg matching the
  AsyncpgDriver / RedshiftDriver contract
- store both as private attributes for the future impl

This proves:
- The config schema (shard 08) is consumable.
- The factory's dispatch returns a usable object.
- The `isinstance(driver, Driver)` invariant holds at construction.
- Type-checking `__init__` catches signature mismatches against
  the factory call site.

### 4. Stub tests are minimal but they exist

Each stub gets a sibling test file under `tests/unit/`:
- assert `isinstance(driver, Driver)`
- assert `driver.__abstractmethods__ == frozenset()`
- assert each method raises `NotImplementedError` when called
- assert the error message references the roadmap doc / docstring
- assert importing the stub module does NOT pull the backend lib

~30 LOC per stub file. The stubs from shard 12
(`test_snowflake_driver_stub.py`, `test_bigquery_driver_stub.py`)
are the canonical reference.

### 5. NO backend lib imports in the stub source

Stubs MUST NOT `import snowflake.connector` or `import
google.cloud.bigquery` or equivalent. They're stubs — they don't
need the backend lib. The real impl adds the import when it lands.
The lazy-import contract (`test_lazy_imports.py`) verifies this.
