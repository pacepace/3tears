"""A type checker sees the package root's names exactly: each exported name typed, and a misspelt one refused.

The root serves its exports lazily through a module ``__getattr__``. A checker that saw that function would type every
``from threetears.evals import <name>`` as its return, ``Any``, so a misspelt or removed name would pass strict mypy
and fail only at run time in the host. The root keeps the function out of the checker's sight and declares each name
to it instead, ``__version__`` as a ``str``.

This runs strict mypy, with the repository's configuration, over one program that imports every name of the root's
``__all__``, imports a misspelt name, and assigns ``__version__`` to an ``int``. The two refusals must be exactly those
two lines, so the test cannot pass on a checker that sees nothing: a root typed as ``Any`` refuses neither, and a
root whose declarations are gone refuses the real names too. One mypy run, from a cold cache in a temporary directory.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from mypy import api

import threetears.evals

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: One error line of mypy's output: the line it names and the error code it closes on.
_ERROR = re.compile(r"^[^:]+:(\d+): error: .*\[([a-z-]+)\]$", re.MULTILINE)


def _program() -> tuple[str, dict[int, str]]:
    """The program mypy checks, and each line it must refuse mapped to the error code it must give."""
    real = ", ".join(sorted(threetears.evals.__all__))
    lines = [
        f"from threetears.evals import {real}",
        "from threetears.evals import Evall",
        "",
        "version: str = __version__",
        "not_a_version: int = __version__",
    ]
    return "\n".join(lines) + "\n", {2: "attr-defined", 5: "assignment"}


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
