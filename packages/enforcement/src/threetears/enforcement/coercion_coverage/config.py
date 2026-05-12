"""configuration dataclass for coercion-coverage enforcement.

the coercion-coverage domain enforces a single contract: subclasses of
``TearsTool`` (or any class whose base name ends with ``Tool`` by
default) must override ``execute``, never ``run``. ``TearsTool.run``
calls ``normalize_kwargs`` and then dispatches to ``execute``;
overriding ``run`` silently bypasses the input-coercion path and
re-introduces the empty-string / JSON-encoded-string 422 bug class.

this domain is universal — there is no per-repo allowlist or
convention dictionary. the only knobs a consumer needs are the repo
root, optional explicit src-root override, optional exemption file,
the env-var name controlling strict / report mode, and the set of
base-class name suffixes that mark a class as Tool-ish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CoerceCoverageConfig"]


_DEFAULT_BASE_CLASS_SUFFIXES: frozenset[str] = frozenset({"Tool"})


@dataclass(frozen=True)
class CoerceCoverageConfig:
    """per-repo config for the coercion-coverage enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root (the
        directory containing its top-level ``pyproject.toml``).
    :ivar src_roots: optional explicit src-trees to scan. when
        ``None``, the runner calls
        :func:`threetears.enforcement.common.pyproject_discovery.discover_src_roots`
        so the walker sees every transitively-reachable path-dep src
        tree. set this to override discovery in tests or specialised
        harnesses.
    :ivar exemptions_path: path to ``_coercion_coverage_exemptions.txt``;
        ``None`` means "no exemptions file" (used by tests).
    :ivar mode_env_var: environment variable controlling strict vs
        report mode. defaults to ``COERCE_ENFORCEMENT_MODE``.
    :ivar base_class_suffixes: a class is considered Tool-ish when any
        base name (last textual segment) ends with one of these
        suffixes. defaults to ``{"Tool"}`` matching the canonical
        contract.
    """

    repo_root: Path
    src_roots: tuple[Path, ...] | None = None
    exemptions_path: Path | None = None
    mode_env_var: str = "COERCE_ENFORCEMENT_MODE"
    base_class_suffixes: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_BASE_CLASS_SUFFIXES,
    )
