"""configuration dataclass for logger-coverage enforcement.

the logger-coverage domain enforces a single contract: every
production module declares a module-level
``log = get_logger(__name__)`` (or the legacy ``_logger`` alias). a
silent module is the most expensive class of operability defect: there
is nothing to grep, nothing to correlate, nothing to alert on.

the contract is universal — the rule does not vary per repo — but
several knobs let consumers handle their own legitimate escape
hatches without forking the walker:

- :attr:`exempt_files`: relative-posix-path -> rationale. modules
  that legitimately produce no observable behaviour (re-export shims,
  pure pydantic models, constants tables, generator output) are
  listed here. each repo curates its own dict.
- :attr:`logger_factory_names`: function names that legitimately
  produce a logger. defaults to ``{"get_logger"}``. matches both the
  bare-name callee (``get_logger(__name__)``) and the namespaced
  callee (``observe.get_logger(__name__)``) — the comparison is on
  the attribute name on the right of the dot.
- :attr:`expected_var_names`: module-level variable names that
  satisfy the contract. defaults to ``{"log", "_logger"}``. the
  legacy ``_logger`` form is accepted because some pre-platform
  modules predate the ``log`` convention; new code should use
  ``log``.
- :attr:`skip_basenames`: file basenames the walker skips outright.
  defaults to ``{"__init__.py"}`` since package init modules are
  re-export shims by convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["LoggerCoverageConfig"]


@dataclass(frozen=True)
class LoggerCoverageConfig:
    """per-repo config for the logger-coverage enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root (the
        directory containing its top-level ``pyproject.toml``).
    :ivar src_roots: optional explicit src-trees to scan. when
        ``None``, the runner calls
        :func:`threetears.enforcement.common.pyproject_discovery.discover_src_roots`
        so the walker sees every transitively-reachable path-dep src
        tree. set this to override discovery in tests or specialised
        harnesses.
    :ivar exemptions_path: path to
        ``_logger_coverage_exemptions.txt``; ``None`` means "no
        exemptions file" (this domain leans on :attr:`exempt_files`
        instead, but the path-based exemption machinery is retained
        for symmetry with sibling domains).
    :ivar mode_env_var: environment variable controlling strict vs
        report mode. defaults to ``LOGGER_COVERAGE_ENFORCEMENT_MODE``.
    :ivar exempt_files: relative-posix-path -> rationale mapping. a
        file in this dict is skipped entirely (the walker does not
        check it). the path key is the module's path relative to the
        repo root with forward slashes; the value is a short
        operator-supplied rationale (re-export shim, pure pydantic
        models, generator output, constants table, etc.).
    :ivar logger_factory_names: function names whose call is
        recognised as logger construction. defaults to
        ``{"get_logger"}``. matches both ``get_logger(...)`` (Name
        callee) and ``observe.get_logger(...)`` (Attribute callee) —
        the comparison is on the attribute name in both cases.
    :ivar expected_var_names: module-level variable names accepted as
        the assignment target for a module logger. defaults to
        ``{"log", "_logger"}``. the legacy ``_logger`` alias is
        retained for compatibility with pre-platform modules.
    :ivar skip_basenames: file basenames the walker skips entirely
        before any AST work. defaults to ``{"__init__.py"}`` because
        package init modules are re-export shims by convention.
    """

    repo_root: Path
    src_roots: tuple[Path, ...] | None = None
    exemptions_path: Path | None = None
    mode_env_var: str = "LOGGER_COVERAGE_ENFORCEMENT_MODE"
    exempt_files: dict[str, str] = field(default_factory=dict)
    logger_factory_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"get_logger"}),
    )
    expected_var_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"log", "_logger"}),
    )
    skip_basenames: frozenset[str] = field(
        default_factory=lambda: frozenset({"__init__.py"}),
    )
