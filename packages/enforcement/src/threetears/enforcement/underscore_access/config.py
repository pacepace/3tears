"""configuration dataclass for underscore-access enforcement.

the underscore-access domain is universal — there is no per-repo
allowlist or convention dictionary. the only knobs a consumer needs
are the repo root, optional explicit src-root overrides, optional
exemption file, and the env-var name controlling strict / report
mode. shape-C's "module is allowed to have no ``__all__``" carve-out
is parameterised here too because some projects keep additional
no-public-surface basenames (test conftests, runtime shims).

src-root responsibilities are split across two fields:

- :attr:`scan_roots` is *where to look for violations* (defaults to
  :func:`find_local_src_roots
  <threetears.enforcement.common.repo_layout.find_local_src_roots>`,
  the consumer's own code only).
- :attr:`inheritance_roots` is *where to build the inheritance graph
  and resolve module-to-file lookups* (used by shape A's
  :func:`same_package` classification and shape D's cross-package
  shadow detection). defaults to :func:`discover_src_roots
  <threetears.enforcement.common.pyproject_discovery.discover_src_roots>`,
  which is path-dep aware.

splitting them prevents Option B path-dep walking from dragging
upstream code into the violation scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["UnderscoreAccessConfig"]


_DEFAULT_SKIP_BASENAMES: frozenset[str] = frozenset({
    "conftest.py",
    "__main__.py",
    "_version.py",
})


@dataclass(frozen=True)
class UnderscoreAccessConfig:
    """per-repo config for the underscore-access enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root (the
        directory containing its top-level ``pyproject.toml``).
    :ivar scan_roots: where to LOOK for violations. when ``None``, the
        runner uses :func:`find_local_src_roots
        <threetears.enforcement.common.repo_layout.find_local_src_roots>`
        — always scoped to the consumer's own code. set explicitly only
        to override discovery in tests or specialised harnesses.
    :ivar inheritance_roots: where to BUILD the inheritance graph and
        resolve module-to-file lookups for shape A's ``same_package``
        check and shape D's cross-package shadow detection. when
        ``None``, the runner uses :func:`discover_src_roots
        <threetears.enforcement.common.pyproject_discovery.discover_src_roots>`
        — path-dep aware so private-name semantics are evaluated
        correctly across package boundaries.
    :ivar exemptions_path: path to ``_underscore_exemptions.txt``;
        ``None`` means "no exemptions file" (used by tests).
    :ivar mode_env_var: environment variable controlling strict vs
        report mode. defaults to ``UNDERSCORE_AUDIT_MODE`` to match
        the canonical convention.
    :ivar skip_basenames: file basenames where shape C (missing
        ``__all__``) does not apply. defaults to the canonical set
        (``conftest.py``, ``__main__.py``, ``_version.py``).
    :ivar enable_shape_b_ruff: whether to run ruff for shape B. set
        ``False`` on hosts without ruff installed (in which case
        shape B yields zero violations). defaults to ``True``.
    """

    repo_root: Path
    scan_roots: tuple[Path, ...] | None = None
    inheritance_roots: tuple[Path, ...] | None = None
    exemptions_path: Path | None = None
    mode_env_var: str = "UNDERSCORE_AUDIT_MODE"
    skip_basenames: frozenset[str] = field(default_factory=lambda: _DEFAULT_SKIP_BASENAMES)
    enable_shape_b_ruff: bool = True
