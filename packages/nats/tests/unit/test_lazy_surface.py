"""unit tests for the lazy re-export surface of :mod:`threetears.nats`.

nine submodules reach ``nats-py`` or ``nkeys``; the rest are pure python.
the nats-py-backed names are resolved lazily (PEP 562) so that importing
this package -- which happens transitively for anyone touching
``threetears.core.collections`` -- does not drag the NATS client into a
process that only uses the L1 SQLite tier.

the value of that is entirely mechanical, so it gets a mechanical guard:
``test_l1_surface_imports_without_the_client`` blocks ``nats`` and
``nkeys`` at ``sys.meta_path`` and re-imports the package from scratch,
which fails whether an offending import sits at module scope or inside a
function body. the parity tests keep the lazy table honest against
``__all__`` and against what the submodules actually export.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator, Sequence
from types import ModuleType
from typing import Any

import pytest

import threetears.nats as nats_pkg

#: distributions the L1 tier must never load. ``nkeys`` is the load-bearing
#: one: it publishes no wheels, so an unconditional dependency turns every
#: install into a source build, including aarch64 targets that have prebuilt
#: wheels for the whole rest of the tree.
_CLIENT_DISTS = ("nats", "nkeys")


class _BlockedFinder:
    """meta_path finder that refuses the named top-level distributions."""

    def __init__(self, blocked: Sequence[str]) -> None:
        self._blocked = frozenset(blocked)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split(".")[0] in self._blocked:
            raise ImportError(f"blocked by test: {fullname}")
        return None


def _purge(prefixes: Sequence[str]) -> None:
    for name in [n for n in sys.modules if n.split(".")[0] in prefixes]:
        del sys.modules[name]


@pytest.fixture
def without_client() -> Iterator[None]:
    """re-import the world with the NATS client unimportable."""
    saved = sys.modules.copy()
    saved_path = sys.meta_path[:]
    _purge(("threetears", *_CLIENT_DISTS))
    sys.meta_path.insert(0, _BlockedFinder(_CLIENT_DISTS))  # type: ignore[arg-type]
    try:
        yield
    finally:
        sys.meta_path[:] = saved_path
        sys.modules.clear()
        sys.modules.update(saved)


class TestLazySurface:
    def test_l1_surface_imports_without_the_client(self, without_client: None) -> None:
        """the whole point: L1 collections import with nats-py absent."""
        importlib.import_module("threetears.core.collections")

        leaked = sorted({n.split(".")[0] for n in sys.modules if n.split(".")[0] in _CLIENT_DISTS})
        assert not leaked, f"client packages loaded on the L1 path: {leaked}"

    def test_eager_names_resolve_without_the_client(self, without_client: None) -> None:
        """errors, subjects, transport and permissions carry no client dependency."""
        pkg = importlib.import_module("threetears.nats")
        for name in ("KvError", "Subjects", "StreamTransport", "build_permissions"):
            assert getattr(pkg, name) is not None

    def test_lazy_name_without_the_client_names_the_extra(self, without_client: None) -> None:
        """a lazy import that cannot be satisfied must say how to fix it."""
        pkg = importlib.import_module("threetears.nats")
        with pytest.raises(ImportError, match=r"3tears-nats\[client\]"):
            _ = pkg.NatsClient

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        """__getattr__ must not turn a typo into an ImportError."""
        with pytest.raises(AttributeError, match="NoSuchName"):
            _ = nats_pkg.NoSuchName

    def test_lazy_table_matches_type_checking_block(self) -> None:
        """every lazily-declared name must actually live in its submodule."""
        for submod, attrs in nats_pkg._LAZY_SUBMOD_ATTRS.items():  # noqa: SLF001
            module: ModuleType = importlib.import_module(f"threetears.nats.{submod}")
            missing = [a for a in attrs if not hasattr(module, a)]
            assert not missing, f"threetears.nats.{submod} does not export {missing}"

    def test_lazy_names_are_declared_public(self) -> None:
        """the lazy table may not smuggle in names __all__ does not promise."""
        undeclared = sorted(set(nats_pkg._LAZY_ATTR_TO_SUBMOD) - set(nats_pkg.__all__))  # noqa: SLF001
        assert not undeclared, f"lazy names missing from __all__: {undeclared}"

    def test_every_public_name_resolves(self) -> None:
        """__all__ is a promise; eager and lazy names alike must honour it."""
        unresolvable = [n for n in nats_pkg.__all__ if not hasattr(nats_pkg, n)]
        assert not unresolvable, f"__all__ names that do not resolve: {unresolvable}"

    def test_dir_advertises_the_lazy_names(self) -> None:
        """tab-completion and introspection should see the deferred surface."""
        listed = set(dir(nats_pkg))
        assert set(nats_pkg._LAZY_ATTR_TO_SUBMOD) <= listed  # noqa: SLF001
