"""unit tests for the lazy re-export surface of :mod:`threetears.nats`.

some submodules reach ``nats-py`` or ``nkeys``; the rest are pure python.
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

import ast
import importlib
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
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
    """meta_path finder that refuses the named top-level distributions.

    Raises ``ModuleNotFoundError`` carrying ``name``, exactly as the real
    machinery does for an uninstalled distribution: the package's
    ``__getattr__`` reads ``name`` to decide whether to blame the missing
    extra or re-raise a genuine broken import from one of our own submodules,
    so a bare ``ImportError`` here would not exercise the branch that ships.
    """

    def __init__(self, blocked: Sequence[str]) -> None:
        self._blocked = frozenset(blocked)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split(".")[0] in self._blocked:
            raise ModuleNotFoundError(f"blocked by test: {fullname}", name=fullname)
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

    def test_lazy_table_matches_submodule_exports(self) -> None:
        """every lazily-declared name must actually live in its submodule.

        Named for what it checks. It was previously called
        ``test_lazy_table_matches_type_checking_block`` and never went near that block --
        which is the one of the three the package's own comment claims are kept in step
        that nothing could have caught, since those names do not exist at runtime.
        ``test_type_checking_block_matches_lazy_table`` below now covers it.
        """
        for submod, attrs in nats_pkg._LAZY_SUBMOD_ATTRS.items():  # noqa: SLF001
            module: ModuleType = importlib.import_module(f"threetears.nats.{submod}")
            missing = [a for a in attrs if not hasattr(module, a)]
            assert not missing, f"threetears.nats.{submod} does not export {missing}"

    def test_type_checking_block_matches_lazy_table(self) -> None:
        """the ``TYPE_CHECKING`` re-imports must name exactly the lazy table's entries.

        Those imports never execute, so a name that goes stale there is invisible to every
        runtime check: a type checker simply stops resolving one public name, and the
        package still imports and still serves it. AST-parsing the file is the only way to
        see them.
        """
        source = Path(nats_pkg.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)

        declared: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if not (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"):
                continue
            for statement in node.body:
                if not isinstance(statement, ast.ImportFrom) or statement.module is None:
                    continue
                submod = statement.module.removeprefix("threetears.nats.")
                declared.setdefault(submod, set()).update(alias.name for alias in statement.names)

        expected = {submod: set(attrs) for submod, attrs in nats_pkg._LAZY_SUBMOD_ATTRS.items()}  # noqa: SLF001
        assert declared == expected, (
            f"TYPE_CHECKING block and _LAZY_SUBMOD_ATTRS disagree; "
            f"only in the block: { {k: sorted(v - expected.get(k, set())) for k, v in declared.items() if v - expected.get(k, set())} }, "
            f"only in the table: { {k: sorted(v - declared.get(k, set())) for k, v in expected.items() if v - declared.get(k, set())} }"
        )

    def test_lazy_names_are_declared_public(self) -> None:
        """the lazy table may not smuggle in names __all__ does not promise."""
        undeclared = sorted(set(nats_pkg._LAZY_ATTR_TO_SUBMOD) - set(nats_pkg.__all__))  # noqa: SLF001
        assert not undeclared, f"lazy names missing from __all__: {undeclared}"

    def test_every_public_name_resolves(self) -> None:
        """__all__ is a promise; eager and lazy names alike must honour it."""
        unresolvable = [n for n in nats_pkg.__all__ if not hasattr(nats_pkg, n)]
        assert not unresolvable, f"__all__ names that do not resolve: {unresolvable}"

    def test_no_public_name_resolves_to_a_submodule(self) -> None:
        """``hasattr`` is not enough -- a shadowed name is present but wrong.

        ``forward`` is both a submodule and the function it exports, so the
        import machinery binding the submodule onto the package would satisfy
        ``hasattr`` while turning every call into ``TypeError: 'module' object
        is not callable``.
        """
        shadowed = [n for n in nats_pkg.__all__ if isinstance(getattr(nats_pkg, n, None), ModuleType)]
        assert not shadowed, f"public names shadowed by their submodule: {shadowed}"

    def test_dir_advertises_the_lazy_names(self) -> None:
        """tab-completion and introspection should see the deferred surface."""
        listed = set(dir(nats_pkg))
        assert set(nats_pkg._LAZY_ATTR_TO_SUBMOD) <= listed  # noqa: SLF001


class TestSubmoduleNameCollision:
    """``forward`` is exported under the same name as the submodule holding it.

    Both orderings below bind ``threetears.nats.forward`` to the submodule as a
    side effect of the import machinery. Under eager re-exports the function
    won because it was bound afterwards; under lazy resolution that order
    inverts, so the package must refuse the shadowing write explicitly.
    """

    @staticmethod
    def _reimported() -> ModuleType:
        """a clean ``threetears.nats``, unaffected by earlier tests' caching."""
        _purge(("threetears",))
        return importlib.import_module("threetears.nats")

    @pytest.fixture(autouse=True)
    def _restore_modules(self) -> Iterator[None]:
        saved = sys.modules.copy()
        try:
            yield
        finally:
            sys.modules.clear()
            sys.modules.update(saved)

    def test_survives_resolving_a_sibling_lazy_name(self) -> None:
        """resolving ``ForwardedHandlerError`` must not replace ``forward``."""
        pkg = self._reimported()

        _ = pkg.ForwardedHandlerError  # imports threetears.nats.forward

        assert callable(pkg.forward), f"forward became {type(pkg.forward).__name__}"

    def test_survives_a_direct_submodule_import(self) -> None:
        """the package's own tests import the submodule directly; that is fine."""
        pkg = self._reimported()

        importlib.import_module("threetears.nats.forward")

        assert callable(pkg.forward), f"forward became {type(pkg.forward).__name__}"
        assert "threetears.nats.forward" in sys.modules, "submodule must stay importable"
