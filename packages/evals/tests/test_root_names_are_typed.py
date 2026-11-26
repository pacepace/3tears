"""A type checker sees the package root's names exactly: each exported name typed, and a misspelt one refused.

The root serves its exports lazily through a module ``__getattr__``. A checker that saw that function would type every
``from threetears.evals import <name>`` as its return, ``Any``, so a misspelt or removed name would pass strict mypy
and fail only at run time in the host. The root keeps the function out of the checker's sight and declares each name
to it instead, ``__version__`` as a ``str``.

Each public root that serves its names this way is held the same: the package root,
:mod:`~threetears.evals.providers` and :mod:`~threetears.evals.testing`. The list of them is kept by hand, so a second
test holds it to every public root that has a module ``__getattr__`` at run time: a new lazy root cannot be left out.

This runs strict mypy, with the repository's configuration, over one program that imports every name of each such
root's ``__all__``, imports one misspelt name from each, and assigns ``__version__`` to an ``int``. The refusals must be
exactly those lines, so the test cannot pass on a checker that sees nothing: a root typed as ``Any`` refuses none of
them, and a root whose declarations are gone refuses the real names too. One mypy run, from a cold cache in a
temporary directory.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
from mypy import api

import threetears.evals
import threetears.evals.providers
import threetears.evals.testing

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: One error line of mypy's output: the line it names and the error code it closes on.
_ERROR = re.compile(r"^[^:]+:(\d+): error: .*\[([a-z-]+)\]$", re.MULTILINE)


#: Each root whose exports a module ``__getattr__`` serves, and a misspelling of one of its names.
_LAZY_ROOTS = (
    (threetears.evals, "Evall"),
    (threetears.evals.providers, "ProviderErrorr"),
    (threetears.evals.testing, "Modell"),
)


def _program() -> tuple[str, dict[int, str]]:
    """The program mypy checks, and each line it must refuse mapped to the error code it must give."""
    lines: list[str] = []
    refused: dict[int, str] = {}
    for root, misspelt in _LAZY_ROOTS:
        lines.append(f"from {root.__name__} import {', '.join(sorted(root.__all__))}")
        lines.append(f"from {root.__name__} import {misspelt}")
        refused[len(lines)] = "attr-defined"
    lines += ["", "version: str = __version__", "not_a_version: int = __version__"]
    refused[len(lines)] = "assignment"
    return "\n".join(lines) + "\n", refused


def test_a_misspelt_root_name_fails_strict_mypy_and_every_real_one_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program, refused = _program()
    path = tmp_path / "imports_the_root.py"
    path.write_text(program, encoding="utf-8")
    # The configuration's mypy_path is relative to the repository root, and names this checkout's source trees.
    monkeypatch.chdir(_REPO_ROOT)
    stdout, stderr, status = api.run(
        [
            "--config-file",
            str(_REPO_ROOT / "pyproject.toml"),
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            "--follow-imports=silent",
            str(path),
        ]
    )
    errors = {int(line): code for line, code in _ERROR.findall(stdout)}
    assert status == 1 and errors == refused, f"mypy gave status {status}:\n{stdout}{stderr}"


def test_every_public_root_that_serves_its_names_lazily_is_checked_above() -> None:
    lazy = {
        name
        for name in ("threetears.evals", *threetears.evals.PUBLIC_ROOTS)
        if hasattr(importlib.import_module(name), "__getattr__")
    }
    listed = {root.__name__ for root, _ in _LAZY_ROOTS}
    assert lazy <= listed, (
        f"public roots with a module __getattr__ that the misspelt-name check leaves out: {sorted(lazy - listed)}"
    )
