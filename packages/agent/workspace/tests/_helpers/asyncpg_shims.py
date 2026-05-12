"""asyncpg shell parity bases for the workspace test tree.

each workspace test that exercises asyncpg-shaped code paths
declares its own ``_FakePool`` / ``_FakeConnection`` /
``_FakeTransaction`` / ``_FakeAcquireCM`` shell with a small per-test
``_FakeStore`` backing data layer. the shells were near-identical
across 22+ test files; the per-test variation lived only in the
store.

these classes serve as the canonical PARITY BASES every per-test
shell subclasses. each base is a thin marker exposing the SUBSET of
the asyncpg surface workspace tests actually exercise (``acquire`` /
``transaction`` / ``execute`` / ``fetchrow``); the bases carry no
behaviour because every test customises ``execute`` and ``fetchrow``
with its own per-query routing logic. a subclass declaration
satisfies the fake-protocol-parity walker (subclass = parity
declared) without forcing each test to relocate its store logic into
a shared module.

# parity-with: asyncpg deliberately not declared
--------------------------------------------------

the asyncpg ``Pool`` / ``Connection`` carry 50+ public methods we
cannot meaningfully mock without a real Postgres server. the
fake-protocol-parity walker compares the fake to a SUBSET of asyncpg
intentionally embodied by these shell classes; the shell-as-parity-
target keeps each test's subclass declaration honest about WHICH
methods the test actually expects.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FakeAsyncpgAcquireCM",
    "FakeAsyncpgConnection",
    "FakeAsyncpgPool",
    "FakeAsyncpgTransaction",
]


class FakeAsyncpgTransaction:
    """asyncpg.Transaction subset shell.

    the asyncpg surface workspace tests rely on is just the async
    context-manager shape (``__aenter__`` / ``__aexit__``). subclasses
    provide their own implementations -- some flip a flag on the
    parent connection, some collect entered/exited timestamps for
    race-detection assertions. no public methods are declared here;
    the parity walker does not check dunders.
    """


class FakeAsyncpgConnection:
    """asyncpg.Connection subset shell.

    the asyncpg surface workspace tests rely on is ``transaction``,
    ``execute``, and (in most tests) ``fetchrow``. subclasses provide
    these methods with per-test logic -- some record calls into
    lists, some dispatch by SQL shape and return crafted rows, some
    delegate to a ``_FakeStore`` data layer. nothing is declared
    here because the walker only checks that subclasses exist as
    parity declarations; method-surface overlap with asyncpg.Connection
    is intentionally partial.
    """


class FakeAsyncpgAcquireCM:
    """asyncpg ``Pool.acquire()`` async-CM subset shell.

    only the ``__aenter__`` / ``__aexit__`` shape is needed; the
    inner connection comes from the test-specific subclass.
    """


class FakeAsyncpgPool:
    """asyncpg.Pool subset shell.

    workspace tests exercise only the ``acquire`` surface (returning
    an async-CM that yields a connection). subclasses provide
    ``acquire`` with whatever wiring they need.
    """

    def acquire(self) -> Any:
        """return the per-test async-CM. subclasses MUST override.

        the base raises :class:`NotImplementedError` to surface
        misconfigured subclasses fast (rather than letting tests
        await an empty CM and time out somewhere downstream).

        :return: the test's subclass-specific acquire context manager
        :rtype: Any
        :raises NotImplementedError: the subclass forgot to override
        """
        raise NotImplementedError("FakeAsyncpgPool subclass must override acquire()")
