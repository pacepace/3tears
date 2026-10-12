"""a row a read returned enters L1 only under the fence taken before that read.

A scan reads L3 with no per-key ticket, so a write that commits while it is in flight, and is
evicted here, is followed by the scan caching the row as it was before the write. Nothing ages in
L1, so that row would be served until the key is written again. Every ``self.write_to_cache_sync``
in an ``async`` method -- a method that reads -- passes ``read_since=`` the
:meth:`~threetears.core.collections.base.BaseCollection.scan_ticket` taken before its first await.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: (repo-relative path, method) -> why its L1 write is not a read's row
_NOT_A_READ = {
    (
        "packages/agent/tools/src/threetears/agent/tools/collections.py",
        "_upsert_keyed",
    ): "caches the row its own upsert committed, not a row a read returned",
}


def unfenced_read_caches(roots: list[Path], repo_root: Path) -> list[str]:
    """every ``self.write_to_cache_sync`` in an async method under ``roots`` that passes no ``read_since=``.

    :param roots: the source trees to scan
    :ptype roots: list[Path]
    :param repo_root: what the reported paths are relative to
    :ptype repo_root: Path
    :return: ``path:line method`` for each
    :rtype: list[str]
    """
    found: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(repo_root).as_posix()
            if "/tests/" in relative or "/testing/" in relative:
                continue
            # a hidden directory is nobody's source: the sidecar's own .venv lands under packages/
            # once its tests have run, holding third-party files that are not even UTF-8
            if any(part.startswith(".") for part in path.relative_to(repo_root).parts[:-1]):
                continue
            source = path.read_text(encoding="utf-8")
            if "write_to_cache_sync(" not in source:
                continue
            for function in ast.walk(ast.parse(source)):
                if not isinstance(function, ast.AsyncFunctionDef) or (relative, function.name) in _NOT_A_READ:
                    continue
                for node in ast.walk(function):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "write_to_cache_sync"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "self"
                        and not any(keyword.arg == "read_since" for keyword in node.keywords)
                    ):
                        found.append(f"{relative}:{node.lineno} {function.name}")
    return found


def test_every_read_caches_its_rows_under_its_fence() -> None:
    assert unfenced_read_caches([_REPO_ROOT / "packages"], _REPO_ROOT) == []


def test_an_unfenced_read_is_caught(tmp_path: Path) -> None:
    module = tmp_path / "pkg" / "collections.py"
    module.parent.mkdir()
    module.write_text(
        "class C:\n"
        "    async def list_all(self):\n"
        "        for row in await self.l3_pool.fetch('SELECT 1'):\n"
        "            self.write_to_cache_sync(dict(row))\n",
        encoding="utf-8",
    )
    assert unfenced_read_caches([tmp_path], tmp_path) == ["pkg/collections.py:4 list_all"]


def test_a_virtualenv_under_the_tree_is_not_scanned(tmp_path: Path) -> None:
    """the sidecar's test run leaves ``packages/scrape/sidecar/.venv``; its files are not the repo's."""
    vendored = tmp_path / "pkg" / ".venv" / "lib" / "site-packages" / "cdp.py"
    vendored.parent.mkdir(parents=True)
    vendored.write_bytes(b"# generated\r\nx = '\xb1'\r\n")
    unfenced = tmp_path / "pkg" / ".venv" / "lib" / "site-packages" / "other.py"
    unfenced.write_text(
        "class C:\n    async def f(self):\n        self.write_to_cache_sync({})\n",
        encoding="utf-8",
    )
    assert unfenced_read_caches([tmp_path], tmp_path) == []
