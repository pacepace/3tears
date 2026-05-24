"""Package-side enforcement: no ``uuid4`` in agent-skills production source.

The workspace walker (``tests/enforcement/test_uuidv7_enforcement.py``)
already pins this contract across every ``packages/*/src/`` tree --
this per-package walker fails fast at the local-test boundary so a
regression in this package is caught without running the full
workspace enforcement pass.

UUIDv7's timestamp prefix is load-bearing for the entity-id paths:
cursor-paged queries on ``agent_skills`` / ``agent_skill_invocations``
rely on the byte-lex ordering matching creation time. Mixing in
uuid4 would silently poison that ordering.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "agent" / "skills"


def _collect_source_files() -> list[Path]:
    """Walk every ``.py`` under the package src, skipping migrations.

    Migrations may legitimately use ``gen_random_uuid()`` (server-side
    backfill) -- they are exempt from the UUIDv7 walker, matching the
    workspace convention documented in
    ``packages/agent/memory/src/threetears/agent/memory/migrations/
    v016_backfill_memory_ids.py``.
    """
    return sorted(p for p in _SRC_ROOT.rglob("*.py") if "migrations" not in p.parts and "__pycache__" not in p.parts)


_SRC_FILES = _collect_source_files()
_SRC_IDS = [str(p.relative_to(_SRC_ROOT)) for p in _SRC_FILES]


class TestNoUuid4InSkillsPackage:
    """Pin the UUIDv7 invariant inside the agent-skills package."""

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_no_uuid4_imports(self, src_file: Path) -> None:
        """No production source imports stdlib ``uuid4``."""
        tree = ast.parse(src_file.read_text(encoding="utf-8"), filename=str(src_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "uuid":
                for alias in node.names:
                    if alias.name == "uuid4":
                        pytest.fail(
                            f"{src_file.relative_to(_SRC_ROOT)}:{node.lineno} -- "
                            f"imports stdlib uuid4. Use uuid_utils.uuid7().",
                        )

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_no_uuid4_call_sites(self, src_file: Path) -> None:
        """No production source calls ``uuid4()``."""
        tree = ast.parse(src_file.read_text(encoding="utf-8"), filename=str(src_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "uuid4":
                pytest.fail(
                    f"{src_file.relative_to(_SRC_ROOT)}:{node.lineno} -- uuid4() call site",
                )
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "uuid4"
                and isinstance(func.value, ast.Name)
                and func.value.id == "uuid"
            ):
                pytest.fail(
                    f"{src_file.relative_to(_SRC_ROOT)}:{node.lineno} -- uuid.uuid4() call site",
                )
