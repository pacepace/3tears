"""the DDL lock, the session it takes, and the migration errors: below every package that needs them.

The SQL backend truncates under the DDL lock, the write-mark reconciler installs triggers under it, and the
migration runner applies under it. Those live in ``backends``, ``collections`` and ``data``, which import one
another, so the lock and what it needs are a leaf package of their own that imports none of them.
``threetears.core.data.migrations`` re-exports every name, so its import paths are unchanged.
"""
