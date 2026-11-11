"""a connection whose every statement carries the caller's keyword arguments (:func:`threetears.core.keyed.with_options`)."""

from __future__ import annotations

from typing import Any

from threetears.core.keyed import read_row, with_options


class _Conn:
    """records each statement with its keyword arguments."""

    tx_id = "tx-1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = []

    async def fetch(self, query: str, *params: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("fetch", query, params, kwargs))
        return [{"id": 1}]

    async def fetchrow(self, query: str, *params: Any, **kwargs: Any) -> None:
        self.calls.append(("fetchrow", query, params, kwargs))

    async def fetchval(self, query: str, *params: Any, **kwargs: Any) -> None:
        self.calls.append(("fetchval", query, params, kwargs))

    async def execute(self, query: str, *params: Any, **kwargs: Any) -> str:
        self.calls.append(("execute", query, params, kwargs))
        return "OK"

    async def executemany(self, query: str, args: Any, **kwargs: Any) -> None:
        self.calls.append(("executemany", query, (args,), kwargs))


async def test_a_helper_s_statement_carries_the_options() -> None:
    """Flips if a keyed helper's statement, sent through the wrapped connection, loses the caller's options."""
    conn = _Conn()
    row = await read_row(with_options(conn, namespace="workspace.x"), "files", by={"id": 1}, records=True)
    assert row == {"id": 1}
    [(method, _query, params, kwargs)] = conn.calls
    assert (method, params, kwargs) == ("fetch", (1,), {"namespace": "workspace.x"})


async def test_every_statement_method_carries_them_beside_its_own() -> None:
    """Flips if any statement method drops the options, or a statement's own keyword arguments."""
    conn = _Conn()
    wrapped = with_options(conn, namespace="n")
    await wrapped.fetch("q", 1, timeout=2)
    await wrapped.fetchrow("q", 1)
    await wrapped.fetchval("q", 1)
    await wrapped.execute("q", 1)
    await wrapped.executemany("q", [(1,)])
    assert [(method, kwargs) for method, _q, _p, kwargs in conn.calls] == [
        ("fetch", {"namespace": "n", "timeout": 2}),
        ("fetchrow", {"namespace": "n"}),
        ("fetchval", {"namespace": "n"}),
        ("execute", {"namespace": "n"}),
        ("executemany", {"namespace": "n"}),
    ]


def test_no_options_is_the_connection_itself_and_other_attributes_are_its_own() -> None:
    """Flips if no options wraps the connection, or a wrapped one hides the connection's own attributes."""
    conn = _Conn()
    assert with_options(conn) is conn
    assert with_options(conn, namespace="n").tx_id == "tx-1"


def test_it_binds_arrays_as_the_connection_does_and_has_only_its_methods() -> None:
    """Flips if wrapping changes how a batch read binds its keys, or adds a statement method the connection lacks."""

    class _Bare:
        async def fetch(self, query: str, *params: Any, **kwargs: Any) -> list[Any]:
            return []

    class _Native(_Bare):
        binds_native_arrays = True

    assert with_options(_Native(), namespace="n").binds_native_arrays is True
    bare = with_options(_Bare(), namespace="n")
    assert bare.binds_native_arrays is False
    assert not hasattr(bare, "fetch_arrow")
    assert not hasattr(bare, "executemany")
