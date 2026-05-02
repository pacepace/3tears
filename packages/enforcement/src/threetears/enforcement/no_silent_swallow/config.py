"""configuration dataclass for no-silent-swallow enforcement.

the no-silent-swallow domain enforces a single contract: every
exception handler (and every ``contextlib.suppress(...)`` invocation)
in production code must either log, re-raise, or carry an explicit
``# NOSILENT: <reason>`` marker justifying silence. bare ``except:``
clauses (which catch ``SystemExit`` / ``KeyboardInterrupt`` and are
never correct in production) are flagged unconditionally.

the policy is universal — there is no per-repo override of the
contract itself — but logger conventions vary: most repos use ``log``
or ``logger`` as the receiver name, some use ``LOGGER``, some use
``_log`` / ``_logger``, and a future repo may use a different
convention entirely. the configurable ``logger_names`` /
``logger_methods`` knobs let a consumer extend the recogniser without
forking the walker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["NoSilentSwallowConfig"]


_DEFAULT_LOGGER_NAMES: frozenset[str] = frozenset({
    "log",
    "logger",
    "_log",
    "_logger",
    "LOGGER",
})

_DEFAULT_LOGGER_METHODS: frozenset[str] = frozenset({
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
})

_DEFAULT_NOSILENT_MARKER: str = "# NOSILENT:"


@dataclass(frozen=True)
class NoSilentSwallowConfig:
    """per-repo config for the no-silent-swallow enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root (the
        directory containing its top-level ``pyproject.toml``).
    :ivar src_roots: optional explicit src-trees to scan. when
        ``None``, the runner calls
        :func:`threetears.enforcement.common.pyproject_discovery.discover_src_roots`
        so the walker sees every transitively-reachable path-dep src
        tree. set this to override discovery in tests or specialised
        harnesses.
    :ivar exemptions_path: path to
        ``_no_silent_swallow_exemptions.txt``; ``None`` means "no
        exemptions file" (used by tests).
    :ivar mode_env_var: environment variable controlling strict vs
        report mode. defaults to ``NO_SILENT_SWALLOW_MODE``.
    :ivar logger_names: known receiver names for a logger reference
        (``log``, ``logger``, ``_log``, ``_logger``, ``LOGGER``). a
        bare-name receiver matching one of these counts as a logger
        call when paired with one of the configured methods.
    :ivar logger_methods: method names recognised on a logger. matches
        the python stdlib + ``exception`` (which is ``logging.Logger``
        idiomatic) by default.
    :ivar nosilent_marker: the comment-marker substring that justifies
        a silent handler / suppress. defaults to ``# NOSILENT:``. the
        text after the colon (the "reason") must be non-empty.
    """

    repo_root: Path
    src_roots: tuple[Path, ...] | None = None
    exemptions_path: Path | None = None
    mode_env_var: str = "NO_SILENT_SWALLOW_MODE"
    logger_names: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_LOGGER_NAMES,
    )
    logger_methods: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_LOGGER_METHODS,
    )
    nosilent_marker: str = _DEFAULT_NOSILENT_MARKER
