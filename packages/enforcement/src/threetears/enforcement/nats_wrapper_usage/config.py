"""configuration dataclass for nats-wrapper-usage enforcement.

the nats-wrapper-usage domain enforces a single contract: every
``nats-py`` reference in production code (and, optionally, in test
code) must route through the ``threetears.nats.NatsClient`` wrapper.
direct ``import nats`` / ``from nats import ...`` / ``from nats.X
import ...`` lines bypass the wrapper, leak transport-layer details
into call sites, and break the abstraction the wrapper exists to
provide. flagged at the import boundary, the contract is cheap to
audit and easy to fix.

the policy is universal — there is no per-repo override of the rule
itself — but the configurable knobs let consumers manage repo-specific
layout (separate src / tests trees) and legitimate escape hatches
without forking the walker:

- :attr:`src_roots`: when ``None``, the runner uses
  :func:`discover_src_roots
  <threetears.enforcement.common.pyproject_discovery.discover_src_roots>`
  for the production scan so the walker sees every transitively-
  reachable path-dep src tree. set this to override discovery in
  tests or specialised harnesses.
- :attr:`tests_root`: root of the consumer's ``tests/`` tree,
  scanned separately because the test surface is allowed a different
  exemption posture during the wrapper-migration cleanup window.
  ``None`` means "skip the tests check".
- :attr:`exemptions_path`: repo-relative ``_nats_exemptions.txt``
  file. format is one repo-relative path per line preceded by a
  ``# rationale: <reason>`` line; entries listed there are skipped
  whole-file (every direct nats import inside that file is allowed).
- :attr:`forbidden_module`: top-level module name forbidden from
  direct import. defaults to ``"nats"``. matches the literal name
  exactly and any submodule (``nats.aio``, ``nats.errors``, etc.);
  unrelated names that merely contain the substring ``nats``
  (``natural``, ``natalie``) are not matched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["NatsWrapperConfig"]


@dataclass(frozen=True)
class NatsWrapperConfig:
    """per-repo config for the nats-wrapper-usage enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root (the
        directory containing its top-level ``pyproject.toml``).
    :ivar src_roots: optional explicit production src-trees to scan.
        when ``None``, the runner calls
        :func:`threetears.enforcement.common.pyproject_discovery.discover_src_roots`
        so the walker sees every transitively-reachable path-dep src
        tree. set this to override discovery in tests or specialised
        harnesses.
    :ivar tests_root: root of the consumer's ``tests/`` tree. when
        set, the runner scans it as a separate pass and emits
        ``nats_wrapper_usage.test_import`` violations. ``None``
        skips the tests check entirely (used by repos that don't
        ship a tests tree, and by the unit tests in this package).
    :ivar exemptions_path: path to ``_nats_exemptions.txt``;
        ``None`` means "no exemptions file". the format is one
        repo-relative path per line preceded by a
        ``# rationale: <reason>`` line. listed files are exempted
        whole-file (every direct nats import inside is allowed).
    :ivar mode_env_var: environment variable controlling strict vs
        report mode. defaults to ``NATS_ENFORCEMENT_MODE``.
    :ivar forbidden_module: top-level module name forbidden from
        direct import. defaults to ``"nats"``. matches the literal
        name and any dotted submodule (``nats.aio``, ``nats.errors``);
        unrelated names that merely contain the substring (``natural``,
        ``natalie``) are not matched.
    """

    repo_root: Path
    src_roots: tuple[Path, ...] | None = None
    tests_root: Path | None = None
    exemptions_path: Path | None = None
    mode_env_var: str = "NATS_ENFORCEMENT_MODE"
    forbidden_module: str = "nats"
