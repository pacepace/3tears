"""Workspace-wide UUIDv7 enforcement: every persisted entity ID must
be chronologically sortable (UUIDv7), never a random uuid4.

Why this matters: UUIDv7's timestamp prefix lets cursor-paged queries
use the PK directly (``WHERE chunk_id > $cursor ORDER BY chunk_id``).
Mixing in uuid4 IDs poisons that ordering — paged queries can skip
rows or return them out of creation order. The metallm side has a
known historical bug where ``from uuid import uuid4 as uuid7``
silently fell back to uuid4 when ``uuid_utils`` was unavailable; this
test pins every 3tears framework src root against the same regression.

Scope: walks ``packages/*/src/`` across the workspace. Migration
modules under ``migrations/`` are exempt because their inserts are
migration-time synthesis (one-shot data backfill); they use Postgres'
core ``gen_random_uuid()`` and the carve-out is documented in
``packages/agent/memory/src/threetears/agent/memory/migrations/v016_backfill_memory_ids.py``.

Two layers of pin:

1. **Static scan** — no Python file under any ``packages/*/src/``
   subtree may import ``uuid4`` from the stdlib.
2. **Dynamic check** — ``uuid_utils.uuid7()`` itself produces version
   byte 7 and stays monotonic within a millisecond burst. Fails loudly
   if a downstream lib bump regresses the contract.
"""

from __future__ import annotations

import ast
from pathlib import Path
from uuid import UUID

import pytest
from uuid_utils import uuid7


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PACKAGES_ROOT = _REPO_ROOT / "packages"


def _collect_entity_source_files() -> list[Path]:
    """Collect production source files across every workspace package.

    Skips ``migrations/`` subtrees — those are migration-time inserts
    that may use ``gen_random_uuid()`` per the documented carve-out.
    """
    return sorted(
        p
        for p in _PACKAGES_ROOT.rglob("src/**/*.py")
        if "migrations" not in p.parts and "__pycache__" not in p.parts
    )


_ENTITY_SOURCE_FILES = _collect_entity_source_files()
_ENTITY_SOURCE_IDS = [
    str(p.relative_to(_REPO_ROOT)) for p in _ENTITY_SOURCE_FILES
]


def _imports_uuid4(tree: ast.Module) -> list[int]:
    """Return line numbers where ``uuid4`` is imported from the stdlib."""
    hits: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "uuid":
                for alias in node.names:
                    if alias.name == "uuid4":
                        hits.append(node.lineno)
    return hits


def _calls_uuid4(tree: ast.Module) -> list[tuple[int, str]]:
    """Return (line, call-shape) for every ``uuid4()`` or ``uuid.uuid4()``."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "uuid4":
            hits.append((node.lineno, "uuid4()"))
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "uuid4"
            and isinstance(func.value, ast.Name)
            and func.value.id == "uuid"
        ):
            hits.append((node.lineno, "uuid.uuid4()"))
    return hits


class TestUUIDv7Enforcement:
    """Pin the UUIDv7 invariant across every workspace package."""

    @pytest.mark.parametrize(
        "src_file", _ENTITY_SOURCE_FILES, ids=_ENTITY_SOURCE_IDS
    )
    def test_no_uuid4_imports(self, src_file: Path) -> None:
        """No production source under packages/*/src/ imports stdlib uuid4."""
        tree = ast.parse(
            src_file.read_text(encoding="utf-8"), filename=str(src_file)
        )
        hits = _imports_uuid4(tree)
        if hits:
            relpath = src_file.relative_to(_REPO_ROOT)
            lines = ", ".join(str(line) for line in hits)
            pytest.fail(
                f"{relpath}: imports stdlib uuid4 at line(s) {lines}. "
                f"All entity-ID generation must use uuid_utils.uuid7 — "
                f"see learnings on the metallm uuid4-fallback regression."
            )

    @pytest.mark.parametrize(
        "src_file", _ENTITY_SOURCE_FILES, ids=_ENTITY_SOURCE_IDS
    )
    def test_no_uuid4_call_sites(self, src_file: Path) -> None:
        """No production source under packages/*/src/ calls uuid4()."""
        tree = ast.parse(
            src_file.read_text(encoding="utf-8"), filename=str(src_file)
        )
        hits = _calls_uuid4(tree)
        if hits:
            relpath = src_file.relative_to(_REPO_ROOT)
            details = ", ".join(f"line {line}: {shape}" for line, shape in hits)
            pytest.fail(
                f"{relpath}: calls uuid4 ({details}). "
                f"Use uuid_utils.uuid7() for every entity-ID generation."
            )

    def test_uuid_utils_uuid7_returns_version_7(self) -> None:
        """uuid_utils.uuid7() must produce a UUID whose version byte is 7."""
        sampled = [UUID(str(uuid7())) for _ in range(10)]
        assert all(u.version == 7 for u in sampled), [u.version for u in sampled]

    def test_uuid_utils_uuid7_monotonic_within_burst(self) -> None:
        """UUIDv7 IDs in a tight burst must be chronologically ordered.

        Cursor-paged queries rely on byte-level lexicographic ordering
        matching creation order.
        """
        ids = [str(uuid7()) for _ in range(50)]
        sorted_ids = sorted(ids)
        assert ids == sorted_ids, "uuid_utils.uuid7() lost monotonicity"
