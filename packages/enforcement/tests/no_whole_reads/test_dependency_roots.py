"""3tears trees a consumer's gate reads for the declarations it depends on (``installed_dependency_roots``).

A consumer's environment is usually inside its checkout (``uv sync`` makes ``.venv`` there, in CI as on a laptop), so
an installed 3tears inside the repo root is still the dependency's source; only 3tears' own repo, whose packages are
installed from its own trees, reads no tree as a dependency. Found at reports' 0.95 relock (2026-11-16): with the venv
skipped, every 3tears index a consumer's table list copies (``scope_epochs_pending``, ``source_tokens_by_signal_c``)
read as the consumer's own and was refused for naming no purpose.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import threetears

from threetears.enforcement.no_whole_reads.schema import installed_dependency_roots


def _tree(root: Path, *parts: str) -> Path:
    """a ``threetears`` namespace directory at ``root/parts``.

    :param root: where
    :ptype root: Path
    :return: the directory
    :rtype: Path
    """
    tree = root.joinpath(*parts, "threetears")
    tree.mkdir(parents=True)
    return tree


def test_a_consumers_venv_inside_its_checkout_is_the_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = _tree(tmp_path, ".venv", "lib", "python3.14", "site-packages")
    monkeypatch.setattr(threetears, "__path__", [str(installed)])
    assert installed_dependency_roots(tmp_path) == (installed,)


def test_3tears_own_trees_are_never_its_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    own = _tree(tmp_path, "packages", "core", "src")
    monkeypatch.setattr(threetears, "__path__", [str(own)])
    assert installed_dependency_roots(tmp_path) == ()


def test_an_install_outside_the_checkout_is_the_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    installed = _tree(tmp_path, "env", "lib", "python3.14", "site-packages")
    monkeypatch.setattr(threetears, "__path__", [str(installed)])
    assert installed_dependency_roots(repo) == (installed,)
