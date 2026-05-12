"""shared test helpers for the workspace package.

every workspace test that exercises asyncpg-shaped code paths used to
declare its own ``_FakePool`` / ``_FakeConnection`` /
``_FakeTransaction`` / ``_FakeAcquireCM`` shell with a small per-test
``_FakeStore`` backing data layer. the shells were near-identical
across 22+ test files; the per-test variation lived only in the
store. centralising the shells here removes the duplication and
gives the fake-protocol-parity walker a single class to reason about
per shell type.

import sites use::

    from tests._helpers.asyncpg_shims import (
        FakeAsyncpgPool,
        FakeAsyncpgConnection,
        FakeAsyncpgTransaction,
        FakeAsyncpgAcquireCM,
    )

each test continues to define its own ``_FakeStore`` (or pass
``None`` to use the connection's built-in self-recording mode).
"""
