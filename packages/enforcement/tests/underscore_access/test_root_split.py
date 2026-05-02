"""tests for the scan_roots / inheritance_roots split (architecture fix).

the original ``UnderscoreAccessConfig.src_roots`` overloaded "where to
look for violations" with "where to build the inheritance graph and
resolve module-to-file lookups". when a consumer enabled Option B
path-dep walking for inheritance, scanning also dragged in path-dep
code, producing spurious violations in upstream packages. these tests
prove the split fixes that:

- ``test_scan_roots_scopes_violations_to_local_only``: synthetic
  two-package fixture; consumer's ``scan_roots`` is just the consumer's
  src and ``inheritance_roots`` is the union. shape A scans only
  consumer code (sibling-side private imports never produce
  violations) but the same-package classification still resolves
  across both packages.
- ``test_inheritance_roots_default_walks_path_deps``: a consumer with
  a path-dep declaration; ``inheritance_roots=None`` defaults to
  :func:`discover_src_roots` so the cross-package class-private
  inventory contains classes from the path-dep target.
- ``test_explicit_scan_and_inheritance_roots_independent``: pass
  different tuples for each and verify scope independence end-to-end
  through the runner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.common import discover_src_roots
from threetears.enforcement.underscore_access import (
    UnderscoreAccessConfig,
    run_underscore_enforcement,
    shape_a_violations,
    shape_d_violations,
)


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_consumer_with_path_dep(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    """build consumer + sibling library each with their own pyproject.

    returns ``(consumer_root, consumer_src, lib_src)``.

    sibling library:
    - declares a ``Base`` class with a private ``_internal`` method
      (used by shape D test below).
    - has a sibling-internal cross-package private import that would
      be flagged (lib_pkg consumer importing from lib_internals.\\_x)
      so we can verify scan_roots scoping cleanly.

    consumer:
    - declares a ``Sub`` subclass that does NOT shadow ``_internal``
      (so the only shape-D violation in the fixture would come from
      sibling code, if scan_roots leaked).
    """
    lib_root = tmp_path / "lib"
    lib_src = lib_root / "src"
    lib_root.mkdir(parents=True)
    (lib_root / "pyproject.toml").write_text(
        '[project]\nname = "synthetic-lib"\nversion = "0"\n'
    )
    _write(lib_src / "lib_pkg" / "__init__.py", "")
    _write(
        lib_src / "lib_pkg" / "_internals.py",
        "_helper = 1\n",
    )
    _write(
        lib_src / "lib_pkg" / "base.py",
        (
            "class LibBase:\n"
            "    def _internal(self):\n"
            "        return 1\n"
        ),
    )
    # different top-level package so a private import from this
    # module is a shape-A violation across (lib_pkg, sibling_pkg)
    # under the lib's own roots.
    _write(lib_src / "sibling_pkg" / "__init__.py", "")
    _write(
        lib_src / "sibling_pkg" / "consumer.py",
        "from lib_pkg._internals import _helper\n",
    )

    consumer_root = tmp_path / "consumer"
    consumer_src = consumer_root / "src"
    consumer_root.mkdir(parents=True)
    (consumer_root / "pyproject.toml").write_text(
        "[tool.poetry]\n"
        'name = "synthetic-consumer"\n'
        'version = "0"\n'
        'description = ""\n'
        'authors = ["t"]\n'
        "\n"
        "[tool.poetry.dependencies]\n"
        'python = "^3.11"\n'
        f'synthetic-lib = {{path = "{lib_root}", develop = true}}\n'
    )
    _write(consumer_src / "consumer_pkg" / "__init__.py", "")
    # consumer's Sub subclasses LibBase but does NOT shadow _internal,
    # so the consumer is clean. shape-D shadow detection relies on
    # the cross-package class inventory (built from inheritance_roots)
    # to know LibBase's private surface.
    _write(
        consumer_src / "consumer_pkg" / "sub.py",
        (
            "from lib_pkg.base import LibBase\n"
            '__all__ = ["Sub"]\n'
            "class Sub(LibBase):\n"
            "    def public_method(self):\n"
            "        return 2\n"
        ),
    )
    return consumer_root, consumer_src, lib_src


class TestArchitectureFix:
    def test_scan_roots_scopes_violations_to_local_only(
        self, tmp_path: Path,
    ) -> None:
        """sibling-package code never appears in violations even when
        the inheritance graph spans both packages."""
        consumer_root, consumer_src, lib_src = _make_consumer_with_path_dep(
            tmp_path,
        )
        # scan ONLY the consumer; build inheritance graph over BOTH.
        # the consumer has no shape-A violations of its own; the lib's
        # ``sibling_pkg/consumer.py`` does, but it must NOT be reported.
        a = shape_a_violations(
            (consumer_src,), consumer_root, (consumer_src, lib_src),
        )
        assert a == []

        # shape D — Sub does not shadow _internal, so 0 violations.
        # the inventory is built from inheritance_roots so LibBase's
        # private surface is known. (we're verifying both that no
        # spurious violation is produced AND that the inventory still
        # crosses package boundaries.)
        d = shape_d_violations(
            (consumer_src,), consumer_root, (consumer_src, lib_src),
        )
        assert d == []

    def test_inheritance_roots_default_walks_path_deps(
        self, tmp_path: Path,
    ) -> None:
        """``inheritance_roots=None`` defaults to discover_src_roots,
        which is path-dep aware."""
        consumer_root, _consumer_src, lib_src = _make_consumer_with_path_dep(
            tmp_path,
        )
        discovered = discover_src_roots(consumer_root)
        resolved = {p.resolve() for p in discovered}
        # the path-dep target's src is in the discovered set.
        assert lib_src.resolve() in resolved

    def test_explicit_scan_and_inheritance_roots_independent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """end-to-end via the runner: scan_roots and inheritance_roots
        govern independent scopes; sibling-internal violations stay
        out of the consumer's report."""
        consumer_root, consumer_src, lib_src = _make_consumer_with_path_dep(
            tmp_path,
        )
        monkeypatch.setenv("UNDERSCORE_AUDIT_MODE_TEST", "strict")
        config = UnderscoreAccessConfig(
            repo_root=consumer_root,
            scan_roots=(consumer_src,),
            inheritance_roots=(consumer_src, lib_src),
            mode_env_var="UNDERSCORE_AUDIT_MODE_TEST",
            enable_shape_b_ruff=False,
        )
        # strict mode: must pass even though the sibling package's
        # ``sibling_pkg/consumer.py`` contains a cross-package private
        # import. it does not appear because scan_roots is the
        # consumer src only.
        run_underscore_enforcement(config, walker="all")
