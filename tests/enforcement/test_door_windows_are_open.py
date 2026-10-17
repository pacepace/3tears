"""enforcement: a door's dated window past its date fails the build until its optional path is deleted.

A door adopting a proof declares a :class:`~threetears.core.security.door_proof.DoorWindow` in code,
beside the door, as a module-level constant. From the window's ``required_from`` date the door refuses
a sender without the proof, so the optional path is dead code: the commit that closes the window
deletes the constant and that path together. This gate finds every such constant in the workspace's
source and fails on any whose date has come.
"""

from __future__ import annotations

import ast
import importlib
from datetime import UTC, datetime
from pathlib import Path

from threetears.enforcement.common import parse_python_path
from threetears.core.security.door_proof import DoorWindow, expired_windows

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _declared_windows() -> list[tuple[str, str]]:
    """every module-level ``NAME = DoorWindow(...)`` (annotated or not), as (module, name).

    :return: the declarations found
    :rtype: list[tuple[str, str]]
    """
    found: list[tuple[str, str]] = []
    for src in sorted(_REPO_ROOT.glob("packages/**/src")):
        for path in sorted(src.rglob("*.py")):
            tree = parse_python_path(path)
            for node in tree.body:
                target: ast.expr | None = None
                value: ast.expr | None = None
                if isinstance(node, ast.AnnAssign):
                    target, value = node.target, node.value
                elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target, value = node.targets[0], node.value
                if (
                    isinstance(target, ast.Name)
                    and isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "DoorWindow"
                ):
                    module = ".".join(path.relative_to(src).with_suffix("").parts)
                    found.append((module, target.id))
    return found


def test_the_walker_finds_the_declared_windows() -> None:
    """non-vacuity: the hub JWKS statement window is declared, so the walk reaches real source."""
    assert ("threetears.core.security.jwks_trust", "HUB_JWKS_STATEMENT_WINDOW") in _declared_windows()


def test_no_door_window_is_past_its_date() -> None:
    windows: list[DoorWindow] = []
    for module, name in _declared_windows():
        window = getattr(importlib.import_module(module), name)
        assert isinstance(window, DoorWindow), f"{module}.{name} is not a DoorWindow"
        windows.append(window)
    expired = expired_windows(windows, today=datetime.now(UTC).date())
    assert not expired, (
        "these doors now refuse a sender without their proof, so their optional path is dead code: "
        "delete each window and its optional path in one commit, and drop the door's open row "
        f"and allowlist line: {[w.door for w in expired]}"
    )
