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

import re
from typing import Any

__all__ = [
    "FakeAsyncpgAcquireCM",
    "FakeAsyncpgConnection",
    "FakeAsyncpgPool",
    "FakeAsyncpgTransaction",
    "asks_journal_max",
]

#: the largest-version aggregate however it is written: the hand-written ``COALESCE(MAX(version), 0)`` or the keyed
#: aggregate's ``MAX(t."version")``
_JOURNAL_MAX = re.compile(r'MAX\(\s*(?:t\.)?"?version"?\s*\)', re.IGNORECASE)


def asks_journal_max(query: str) -> bool:
    """whether ``query`` asks the journal's largest version of one path (``workspace_file_versions``, pinned by
    ``(workspace_id, relative_path)`` as ``$1, $2``), by its shape rather than its exact text: the fakes answer it the
    same whether the statement is hand-written or a keyed helper's.

    :param query: the statement a fake connection was sent
    :ptype query: str
    :return: whether it is the journal-max read
    :rtype: bool
    """
    return "workspace_file_versions" in query and _JOURNAL_MAX.search(query) is not None


# parity-exempt: parity bases ARE the canonical shims that per-test workspace fakes subclass; they have no further parity to declare against asyncpg
class FakeAsyncpgTransaction:
    """asyncpg.Transaction subset shell.

    the asyncpg surface workspace tests rely on is just the async
    context-manager shape (``__aenter__`` / ``__aexit__``). subclasses
    provide their own implementations -- some flip a flag on the
    parent connection, some collect entered/exited timestamps for
    race-detection assertions. no public methods are declared here;
    the parity walker does not check dunders.
    """


# parity-exempt: parity bases ARE the canonical shims that per-test workspace fakes subclass; they have no further parity to declare against asyncpg
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

    ``fetch`` answers as the subclass's ``fetchrow`` answers the same statement, as a list: the keyed helpers send
    every read through ``fetch`` and take its rows, where the hand-written reads they replaced called ``fetchrow``.
    """

    async def fetchrow(self, query: str, *args: Any) -> Any:
        """the subclass's per-statement answer. subclasses that are sent reads MUST override.

        :param query: the statement
        :ptype query: str
        :param args: its parameters
        :ptype args: Any
        :return: the row, or None
        :rtype: Any
        :raises NotImplementedError: the subclass answers no read
        """
        raise NotImplementedError("FakeAsyncpgConnection subclass must override fetchrow()")

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """``fetchrow``'s answer to the same statement, as the rows ``fetch`` returns.

        :param query: the statement
        :ptype query: str
        :param args: its parameters
        :ptype args: Any
        :return: no row, or the one row ``fetchrow`` answers
        :rtype: list[Any]
        """
        row = await self.fetchrow(query, *args)
        return [] if row is None else [row]


# parity-exempt: parity bases ARE the canonical shims that per-test workspace fakes subclass; they have no further parity to declare against asyncpg
class FakeAsyncpgAcquireCM:
    """asyncpg ``Pool.acquire()`` async-CM subset shell.

    only the ``__aenter__`` / ``__aexit__`` shape is needed; the
    inner connection comes from the test-specific subclass.
    """


# parity-exempt: parity bases ARE the canonical shims that per-test workspace fakes subclass; they have no further parity to declare against asyncpg
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
