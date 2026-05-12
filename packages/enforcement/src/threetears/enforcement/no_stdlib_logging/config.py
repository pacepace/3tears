"""configuration dataclass for no-stdlib-logging enforcement.

the no-stdlib-logging domain enforces a single contract: every
production module must use ``threetears.observe.get_logger`` instead
of stdlib ``logging``. a stray ``import logging`` followed by
``logging.getLogger(...)`` silently drops correlation tags, call-site
info, and ``extra_data`` rendering — bypassing the
``ContextFormatter`` that ``threetears.observe`` installs.

the contract is universal — there is no per-repo override of the rule
itself — but two configurable knobs let consumers manage legitimate
escape hatches without forking the walker:

- :attr:`exempt_files`: relative-posix-path -> rationale. modules
  that load before ``threetears.observe`` is initialised (bootstrap
  config, app composition that quiets uvicorn) need a transient stdlib
  reference; listing them here skips them entirely.
- :attr:`line_marker`: the per-line comment marker (default
  ``# stdlib-logging: ok``) that justifies a single import line.
  scoped to one line, so adding it to one import does not silence
  others in the same file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["NoStdlibLoggingConfig"]


@dataclass(frozen=True)
class NoStdlibLoggingConfig:
    """per-repo config for the no-stdlib-logging enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root (the
        directory containing its top-level ``pyproject.toml``).
    :ivar src_roots: optional explicit src-trees to scan. when
        ``None``, the runner calls
        :func:`threetears.enforcement.common.pyproject_discovery.discover_src_roots`
        so the walker sees every transitively-reachable path-dep src
        tree. set this to override discovery in tests or specialised
        harnesses.
    :ivar exemptions_path: path to
        ``_no_stdlib_logging_exemptions.txt``; ``None`` means "no
        exemptions file" (this domain leans on
        :attr:`exempt_files` instead, but the path-based exemption
        machinery is retained for symmetry with sibling domains).
    :ivar mode_env_var: environment variable controlling strict vs
        report mode. defaults to ``STDLIB_LOGGING_ENFORCEMENT_MODE``.
    :ivar exempt_files: relative-posix-path -> rationale mapping. a
        file in this dict is skipped entirely (its imports are not
        scanned). the path key is the file's path relative to the
        repo root with forward slashes; the value is a short
        operator-supplied rationale (bootstrap, third-party logger
        config, etc.).
    :ivar line_marker: per-line comment marker that justifies a
        single import. defaults to ``# stdlib-logging: ok``. the
        substring must appear on the offending import line; placement
        on the line above is not accepted (per-line marker is
        intentionally narrow — file-level escape hatches go through
        :attr:`exempt_files`).
    """

    repo_root: Path
    src_roots: tuple[Path, ...] | None = None
    exemptions_path: Path | None = None
    mode_env_var: str = "STDLIB_LOGGING_ENFORCEMENT_MODE"
    exempt_files: dict[str, str] = field(default_factory=dict)
    line_marker: str = "# stdlib-logging: ok"
