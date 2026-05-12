"""configuration dataclass for codebase-conventions enforcement.

the codebase-conventions domain enforces four AST-level structural
contracts that universally apply across every consumer repo:

- no bare ``print(...)`` calls in production source — observability
  flows through the shared logger, not stdout.
- no stdlib ``logging.getLogger(...)`` calls — observability flows
  through ``threetears.observe.get_logger`` so correlation tags and
  the ``ContextFormatter`` aren't bypassed. legitimate third-party
  logger configuration sites can be exempted via per-line marker or
  the file-level allowlist.
- every production module must have ``from __future__ import
  annotations`` at module top — the project depends on PEP 563-style
  evaluation for all type hints.
- every non-dunder, non-test function definition must declare a
  return type annotation. ``__init__``-style dunders and pytest test
  functions (``def test_*``) are exempt by construction.

the rules are universal, but four configurable knobs let consumers
manage legitimate escape hatches without forking the walkers:

- :attr:`print_exempt_files`, :attr:`getlogger_exempt_files`,
  :attr:`future_annotations_exempt_files`, :attr:`return_type_exempt_files`:
  per-check relative-posix-path -> rationale dicts. files in a dict
  are skipped entirely for that check. paths are forward-slash
  relative to ``repo_root``.
- :attr:`getlogger_marker`: per-line comment marker that justifies
  a single ``logging.getLogger`` call (default
  ``# stdlib-getlogger: ok``). scoped to one line — adjacent calls
  without the marker are still flagged.
- :attr:`skip_basenames`: filenames (basename only) that skip the
  future-annotations and return-type checks. defaults to
  ``{"__init__.py"}`` since ``__init__`` files are typically pure
  re-export shims with no executable definitions to annotate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CodebaseConventionsConfig"]


_DEFAULT_SKIP_BASENAMES: frozenset[str] = frozenset({"__init__.py"})


@dataclass(frozen=True)
class CodebaseConventionsConfig:
    """per-repo config for the codebase-conventions enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root (the
        directory containing its top-level ``pyproject.toml``).
    :ivar src_roots: optional explicit src-trees to scan. when
        ``None``, the runner calls
        :func:`threetears.enforcement.common.pyproject_discovery.discover_src_roots`
        so the walkers see every transitively-reachable path-dep src
        tree. set this to override discovery in tests or specialised
        harnesses.
    :ivar exemptions_path: path to
        ``_codebase_conventions_exemptions.txt``; ``None`` means "no
        exemptions file" (this domain leans on the per-walker
        allowlists, but the path-based exemption machinery is retained
        for symmetry with sibling domains).
    :ivar mode_env_var: environment variable controlling strict vs
        report mode. defaults to
        ``CODEBASE_CONVENTIONS_ENFORCEMENT_MODE``.
    :ivar print_exempt_files: relative-posix-path -> rationale mapping.
        files listed here skip the no-print check entirely.
    :ivar getlogger_exempt_files: relative-posix-path -> rationale
        mapping. files listed here skip the no-stdlib-getlogger check
        entirely (e.g. bootstrap modules that load before
        ``threetears.observe`` is initialised).
    :ivar getlogger_marker: per-line comment marker that justifies a
        single ``logging.getLogger(...)`` call. defaults to
        ``# stdlib-getlogger: ok``. the substring must appear on the
        offending line; placement on the line above is not accepted
        (per-line marker is intentionally narrow — file-level escape
        hatches go through :attr:`getlogger_exempt_files`).
    :ivar future_annotations_exempt_files: relative-posix-path ->
        rationale mapping. files listed here may omit
        ``from __future__ import annotations``.
    :ivar return_type_exempt_files: relative-posix-path -> rationale
        mapping. functions in these files may omit return-type
        annotations.
    :ivar skip_basenames: file basenames that skip both the
        future-annotations and return-type checks regardless of the
        per-check exempt sets. default is ``{"__init__.py"}``;
        ``__init__`` files are typically pure re-exports.
    """

    repo_root: Path
    src_roots: tuple[Path, ...] | None = None
    exemptions_path: Path | None = None
    mode_env_var: str = "CODEBASE_CONVENTIONS_ENFORCEMENT_MODE"
    print_exempt_files: dict[str, str] = field(default_factory=dict)
    getlogger_exempt_files: dict[str, str] = field(default_factory=dict)
    getlogger_marker: str = "# stdlib-getlogger: ok"
    future_annotations_exempt_files: dict[str, str] = field(default_factory=dict)
    return_type_exempt_files: dict[str, str] = field(default_factory=dict)
    skip_basenames: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_SKIP_BASENAMES,
    )
