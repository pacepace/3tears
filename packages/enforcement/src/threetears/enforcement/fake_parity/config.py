"""configuration dataclass for fake-protocol-parity enforcement.

the fake-parity domain checks every test fake declared via the
``Fake<Name>`` / ``_Fake<Name>`` naming convention against the
production protocol it claims to mock. the only knobs a consumer
needs are the repo root, the test roots to scan, and the path to the
exemptions file.

src-root responsibilities are split:

- :attr:`scan_roots` is *where to look for fakes* (the consumer's
  test trees). defaults to every ``tests`` directory directly under
  the repo root or under any ``packages/*/`` subtree.
- the walker imports production classes named on ``# parity-with:``
  markers via :mod:`importlib`; the importer relies on the regular
  python path the test runner already set up. there is no separate
  inheritance-roots concept the way there is for shape A / shape D
  in the underscore-access domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["FakeParityConfig"]


@dataclass(frozen=True)
class FakeParityConfig:
    """per-repo config for the fake-protocol-parity enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root (the
        directory containing its top-level ``pyproject.toml``).
    :ivar scan_roots: directories to walk for ``Fake*`` / ``_Fake*``
        class declarations. when ``None``, the runner discovers every
        ``tests`` directory under the repo root and under
        ``packages/**`` (sufficient for both single-package layouts
        and the 3tears workspace shape).
    :ivar exemptions_path: path to ``_fake_parity_exemptions.txt``.
        ``None`` means "no exemptions file" (used by tests).
    :ivar mode_env_var: environment variable controlling strict vs
        report mode. defaults to ``FAKE_PARITY_ENFORCEMENT_MODE``.
    """

    repo_root: Path
    scan_roots: tuple[Path, ...] | None = None
    exemptions_path: Path | None = None
    mode_env_var: str = "FAKE_PARITY_ENFORCEMENT_MODE"
