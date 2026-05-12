"""unit tests for ``threetears.agent.workspace.materialize.materialize``.

covers the happy path (nested relative paths faithfully reproduced to
disk), the empty-workspace path (tempdir created, nothing inside), the
``parent_dir`` kwarg (tempdir lives under the caller-supplied parent),
and verification that :func:`atomic_write` is the write primitive
(spied via module-level monkey-patch). all tests rely on real
filesystem I/O under :func:`pytest.TempPathFactory`-backed ``tmp_path``
so the atomic-write durability contract is exercised for real.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest

import importlib

materialize_module = importlib.import_module("threetears.agent.workspace.materialize")
from threetears.agent.workspace.materialize import materialize  # noqa: E402
from _helpers.workspace_shims import (
    FakeWorkspaceCollection,
    FakeWorkspaceContext,
    FakeWorkspaceEntity,
    FakeWorkspaceFile,
    FakeWorkspaceFileCollection,
    FakeWorkspaceFileVersionCollection,
    FakeWorkspaceSandbox,
)


@dataclass
class _FakeFile(FakeWorkspaceFile):
    """stand-in for :class:`WorkspaceFile` exposing just the fields materialize reads."""

    relative_path: str
    content: bytes
    sha256: str
    version: int


class _FakeFileCollection(FakeWorkspaceFileCollection):
    """fake :class:`WorkspaceFileCollection` exposing :meth:`find_by_workspace`."""

    def __init__(self, files: list[_FakeFile]) -> None:
        self._files = files
        self.calls: list[UUID] = []

    async def find_by_workspace(self, workspace_id: UUID) -> list[_FakeFile]:
        """return the pre-seeded file list, recording the call for assertions."""
        self.calls.append(workspace_id)
        return list(self._files)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@pytest.mark.asyncio
async def test_materialize_writes_three_files_across_nested_paths(
    tmp_path: Path,
) -> None:
    """workspace with three files at nested paths -> all present with exact bytes."""
    ws_id = uuid4()
    files = [
        _FakeFile("a.txt", b"alpha", _sha(b"alpha"), 1),
        _FakeFile("sub/b.yaml", b"beta\n", _sha(b"beta\n"), 1),
        _FakeFile(
            "sub/deep/c.sql",
            b"SELECT 1;\n",
            _sha(b"SELECT 1;\n"),
            1,
        ),
    ]
    coll = _FakeFileCollection(files)
    result = await materialize(
        workspace_id=ws_id,
        workspace_file_collection=coll,  # type: ignore[arg-type]
        parent_dir=tmp_path,
    )
    assert result.exists()
    assert result.is_dir()
    assert (result / "a.txt").read_bytes() == b"alpha"
    assert (result / "sub" / "b.yaml").read_bytes() == b"beta\n"
    assert (result / "sub" / "deep" / "c.sql").read_bytes() == b"SELECT 1;\n"
    assert coll.calls == [ws_id]


@pytest.mark.asyncio
async def test_materialize_empty_workspace_creates_empty_tempdir(
    tmp_path: Path,
) -> None:
    """workspace with no files -> tempdir exists and is empty."""
    ws_id = uuid4()
    coll = _FakeFileCollection([])
    result = await materialize(
        workspace_id=ws_id,
        workspace_file_collection=coll,  # type: ignore[arg-type]
        parent_dir=tmp_path,
    )
    assert result.exists()
    assert result.is_dir()
    assert list(result.iterdir()) == []


@pytest.mark.asyncio
async def test_materialize_honors_parent_dir_kwarg(tmp_path: Path) -> None:
    """tempdir lives under the caller-supplied ``parent_dir``."""
    ws_id = uuid4()
    coll = _FakeFileCollection([])
    result = await materialize(
        workspace_id=ws_id,
        workspace_file_collection=coll,  # type: ignore[arg-type]
        parent_dir=tmp_path,
    )
    assert result.parent == tmp_path


@pytest.mark.asyncio
async def test_materialize_default_parent_dir_uses_system_temp() -> None:
    """no ``parent_dir`` -> tempdir allocated in system temp location."""
    import tempfile as _tempfile

    ws_id = uuid4()
    coll = _FakeFileCollection([])
    result = await materialize(
        workspace_id=ws_id,
        workspace_file_collection=coll,  # type: ignore[arg-type]
    )
    try:
        system_temp = Path(_tempfile.gettempdir()).resolve()
        assert result.resolve().parent == system_temp
    finally:
        result.rmdir()


@pytest.mark.asyncio
async def test_materialize_creates_parent_dirs_for_nested_paths(
    tmp_path: Path,
) -> None:
    """nested paths -> intermediate directories created before write."""
    ws_id = uuid4()
    files = [
        _FakeFile(
            "a/b/c/d/e.txt",
            b"deep",
            _sha(b"deep"),
            1,
        ),
    ]
    coll = _FakeFileCollection(files)
    result = await materialize(
        workspace_id=ws_id,
        workspace_file_collection=coll,  # type: ignore[arg-type]
        parent_dir=tmp_path,
    )
    target = result / "a" / "b" / "c" / "d" / "e.txt"
    assert target.exists()
    assert target.read_bytes() == b"deep"


@pytest.mark.asyncio
async def test_materialize_uses_atomic_write_primitive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """every file write goes through :func:`atomic_write` (spied via monkeypatch)."""
    calls: list[tuple[Path, bytes]] = []

    async def _spy(path: Path, content: bytes | str) -> None:
        payload = content if isinstance(content, bytes) else content.encode("utf-8")
        calls.append((path, payload))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    monkeypatch.setattr(materialize_module, "atomic_write", _spy)
    ws_id = uuid4()
    files = [
        _FakeFile("x.txt", b"X", _sha(b"X"), 1),
        _FakeFile("y/z.txt", b"Z", _sha(b"Z"), 1),
    ]
    coll = _FakeFileCollection(files)
    await materialize(
        workspace_id=ws_id,
        workspace_file_collection=coll,  # type: ignore[arg-type]
        parent_dir=tmp_path,
    )
    assert len(calls) == 2
    assert calls[0][1] == b"X"
    assert calls[1][1] == b"Z"


@pytest.mark.asyncio
async def test_materialize_tempdir_prefix_includes_workspace_hex_prefix(
    tmp_path: Path,
) -> None:
    """tempdir name starts with ``workspace-{first-8-hex}-`` per the design."""
    ws_id = UUID("019470a8-b5c3-7def-8123-456789abcdef")
    coll = _FakeFileCollection([])
    result = await materialize(
        workspace_id=ws_id,
        workspace_file_collection=coll,  # type: ignore[arg-type]
        parent_dir=tmp_path,
    )
    assert result.name.startswith("workspace-019470a8-")


@pytest.mark.asyncio
async def test_materialize_byte_identical_roundtrip_for_binary_content(
    tmp_path: Path,
) -> None:
    """binary content (not UTF-8 clean) is round-tripped byte-identically."""
    ws_id = uuid4()
    payload = bytes(range(256))
    files = [
        _FakeFile("bin.dat", payload, _sha(payload), 1),
    ]
    coll = _FakeFileCollection(files)
    result = await materialize(
        workspace_id=ws_id,
        workspace_file_collection=coll,  # type: ignore[arg-type]
        parent_dir=tmp_path,
    )
    assert (result / "bin.dat").read_bytes() == payload


# ---------------------------------------------------------------------------
# snapshot_disk: symlink-escape rejection + asyncio.to_thread dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def testsnapshot_disk_skips_symlink_that_escapes_root(
    tmp_path: Path,
) -> None:
    """symlinks whose target resolves outside the root are dropped silently.

    defensive guard: a symlink planted inside the bind root pointing at
    ``/etc/hosts`` (or anywhere outside the root) must not smuggle bytes
    into L3 via the capture-back path.
    """
    root = tmp_path / "root"
    root.mkdir()
    # legit file inside the root
    (root / "ok.txt").write_bytes(b"inside")
    # symlink inside root -> target outside root
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"SECRET")
    link = root / "escape.txt"
    link.symlink_to(outside)

    result = await materialize_module.snapshot_disk(root)
    # only the legit file is present
    assert "ok.txt" in result
    assert result["ok.txt"][0] == b"inside"
    # the escaping symlink was skipped
    assert "escape.txt" not in result


@pytest.mark.asyncio
async def testsnapshot_disk_includes_in_root_symlink(tmp_path: Path) -> None:
    """symlink whose resolved target stays inside the root is still read."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.txt").write_bytes(b"inside")
    # symlink pointing at an in-root sibling resolves inside the root -> kept
    link = root / "alias.txt"
    link.symlink_to(root / "ok.txt")

    result = await materialize_module.snapshot_disk(root)
    assert "ok.txt" in result
    assert "alias.txt" in result
    assert result["alias.txt"][0] == b"inside"
