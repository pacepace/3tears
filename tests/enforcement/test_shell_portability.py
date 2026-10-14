"""Enforcement: the scripts stay bash 3.2-clean and parse.

The remote test lane runs ``scripts/threetears-testlane-shim`` and
``scripts/remote-setup.sh`` on macOS, whose ``/bin/bash`` is 3.2, and any
script here may be run by hand on a Mac. bash 4+ constructs there are not a
warning: the shim is a forced command, and a syntax error in it locks the lane
out silently. This walker refuses the constructs 3.2 lacks, with comments
stripped, and runs ``bash -n`` over every file.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = sorted([*(_REPO_ROOT / "scripts").glob("*.sh"), _REPO_ROOT / "scripts" / "threetears-testlane-shim"])

# construct -> why bash 3.2 cannot run it
_BASH4_ONLY: dict[str, re.Pattern[str]] = {
    "case-modifying expansion ${v,,} ${v^^}": re.compile(r"\$\{[#!]?[A-Za-z_][A-Za-z0-9_]*(\[[^]]*\])?(,,?|\^\^?)"),
    "mapfile / readarray": re.compile(r"(^|[\s;&|(])(mapfile|readarray)\b"),
    "associative array (declare -A)": re.compile(r"\b(declare|local|typeset)\s+(-\w*\s+)*-\w*A"),
    "nameref (declare -n)": re.compile(r"\b(declare|local|typeset)\s+(-\w*\s+)*-\w*n\b"),
    "transform expansion ${v@Q}": re.compile(r"\$\{[^}]*@[QEPAaKkUuL]\}"),
    "case fallthrough ;& or ;;&": re.compile(r";;&|(?<!;);&"),
    "|& and &>> redirections": re.compile(r"\|&|&>>"),
    "negative array subscript": re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\[-[0-9]"),
    "coproc": re.compile(r"(^|[\s;&|])coproc\b"),
    "wait -n": re.compile(r"\bwait\s+-n\b"),
}


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """every line that is not a whole-line comment, numbered from 1."""
    return [
        (number, line)
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if not line.lstrip().startswith("#")
    ]


def test_the_walker_sees_the_lane_scripts() -> None:
    names = {path.name for path in _SCRIPTS}
    assert {"test.sh", "test-remote.sh", "remote-setup.sh", "remote-test.sh", "threetears-testlane-shim"} <= names


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda path: path.name)
def test_no_bash4_constructs(script: Path) -> None:
    findings = [
        f"{script.name}:{number}: {label}: {line.strip()}"
        for number, line in _code_lines(script)
        for label, pattern in _BASH4_ONLY.items()
        if pattern.search(line)
    ]
    assert not findings, "bash 3.2 (macOS) cannot run:\n" + "\n".join(findings)


@pytest.mark.parametrize(
    "line",
    ['echo "${name,,}"', "mapfile -t lines < f", "declare -A seen", "local -A m", 'echo "${v@Q}"', "cmd |& tee x"],
)
def test_the_walker_catches_each_construct(line: str) -> None:
    assert any(pattern.search(line) for pattern in _BASH4_ONLY.values()), line


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda path: path.name)
def test_bash_parses_it(script: Path) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr
