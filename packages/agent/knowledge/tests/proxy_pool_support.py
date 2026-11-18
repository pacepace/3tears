"""the rbac-read proxy pool a knowledge listing reads through, for a test that scripts its answers."""

from __future__ import annotations

from typing import Any

from threetears.core.backends.protocol import RawL3Pool

__all__ = ["ProxyPoolDouble"]


class ProxyPoolDouble(RawL3Pool):
    """a raw pool whose only statement is a ``fetch`` carrying the caller's ``customer_scope``.

    A knowledge listing reads through the proxy pool with ``with_options(pool, customer_scope=...)``, so it only
    ever fetches; a subclass answers :meth:`fetch`, and any other statement reaching the pool is a test gone wrong.
    """

    def acquire(self) -> Any:
        raise AssertionError("a knowledge listing reads through fetch; nothing acquires a connection")

    async def fetch(self, query: str, /, *params: Any, customer_scope: Any = None) -> Any:
        raise NotImplementedError("a double answers its own reads")

    async def fetchrow(self, query: str, /, *params: Any) -> Any:
        raise AssertionError(f"a knowledge listing reads through fetch, not fetchrow: {query!r}")

    async def execute(self, query: str, /, *params: Any) -> Any:
        raise AssertionError(f"a knowledge listing writes nothing: {query!r}")
