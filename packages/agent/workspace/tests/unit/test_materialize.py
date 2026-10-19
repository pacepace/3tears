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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

import importlib

materialize_module = importlib.import_module("threetears.agent.workspace.materialize")
from threetears.agent.workspace.materialize import materialize  # noqa: E402
from packages.agent.workspace.tests.helpers.workspace_shims import (
    FakeWorkspaceFile,
    FakeWorkspaceFileCollection,
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
async def test_materialize_tempdir_prefix_includes_workspace_hex(
    tmp_path: Path,
) -> None:
    """tempdir name starts with ``workspace-{hex}-``: the whole id, since a uuid7's head is its timestamp."""
    ws_id = UUID("019470a8-b5c3-7def-8123-456789abcdef")
    coll = _FakeFileCollection([])
    result = await materialize(
        workspace_id=ws_id,
        workspace_file_collection=coll,  # type: ignore[arg-type]
        parent_dir=tmp_path,
    )
    assert result.name.startswith("workspace-019470a8b5c37def8123456789abcdef-")


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


_ROOT_READS_EVERYTHING = pytest.mark.skipif(
    os.geteuid() == 0, reason="root reads a mode-000 directory, so the unreadable directory cannot be made"
)


@pytest.mark.asyncio
@_ROOT_READS_EVERYTHING
async def test_a_directory_the_snapshot_cannot_list_fails_it_rather_than_reading_as_empty(tmp_path: Path) -> None:
    """``rglob`` passed an unreadable directory over, so every file under it read as deleted."""
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "ok.txt").write_bytes(b"inside")
    (root / "sub" / "kept.txt").write_bytes(b"still here")
    (root / "sub").chmod(0)
    try:
        with pytest.raises(PermissionError):
            await materialize_module.snapshot_disk(root)
    finally:
        (root / "sub").chmod(0o755)


@pytest.mark.asyncio
@_ROOT_READS_EVERYTHING
async def test_a_bind_snapshot_records_an_unlistable_directory_and_reads_the_rest(tmp_path: Path) -> None:
    """with somewhere to record them, paths below the root that cannot be read are named, not raised."""
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "ok.txt").write_bytes(b"inside")
    (root / "sub" / "kept.txt").write_bytes(b"still here")
    (root / "sub").chmod(0)
    unreadable: dict[str, OSError] = {}
    try:
        result = await materialize_module.snapshot_disk(root, unreadable=unreadable)
    finally:
        (root / "sub").chmod(0o755)
    assert set(result) == {"ok.txt"}
    assert list(unreadable) == ["sub"]
    assert isinstance(unreadable["sub"], PermissionError)


@pytest.mark.asyncio
async def test_a_bind_snapshot_records_a_file_it_cannot_read_or_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    for name in ("ok.txt", "unreadable.txt", "unresolvable.txt"):
        (root / name).write_bytes(b"x")
    real_read_bytes, real_resolve = Path.read_bytes, Path.resolve

    def _read_bytes(self: Path) -> bytes:
        if self.name == "unreadable.txt":
            raise PermissionError(13, "Permission denied", str(self))
        return real_read_bytes(self)

    def _resolve(self: Path, strict: bool = False) -> Path:
        if self.name == "unresolvable.txt":
            raise PermissionError(13, "Permission denied", str(self))
        return real_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "read_bytes", _read_bytes)
    monkeypatch.setattr(Path, "resolve", _resolve)
    unreadable: dict[str, OSError] = {}
    assert set(await materialize_module.snapshot_disk(root, unreadable=unreadable)) == {"ok.txt"}
    assert sorted(unreadable) == ["unreadable.txt", "unresolvable.txt"]


@pytest.mark.asyncio
@_ROOT_READS_EVERYTHING
async def test_a_bind_snapshot_still_raises_for_a_root_it_cannot_list(tmp_path: Path) -> None:
    """the root is never a subtree to skip: without it there is nothing left to capture."""
    root = tmp_path / "root"
    root.mkdir()
    root.chmod(0)
    unreadable: dict[str, OSError] = {}
    try:
        with pytest.raises(PermissionError):
            await materialize_module.snapshot_disk(root, unreadable=unreadable)
    finally:
        root.chmod(0o755)
    assert unreadable == {}


@pytest.mark.asyncio
async def test_a_file_whose_path_cannot_be_resolved_fails_the_snapshot_rather_than_reading_as_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """an unresolvable path was skipped like an escaping symlink, and capture-back deletes every file it misses."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.txt").write_bytes(b"inside")
    (root / "unresolvable.txt").write_bytes(b"still here")
    real_resolve = Path.resolve

    def _resolve(self: Path, strict: bool = False) -> Path:
        if self.name == "unresolvable.txt":
            raise PermissionError(13, "Permission denied", str(self))
        return real_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", _resolve)
    with pytest.raises(PermissionError):
        await materialize_module.snapshot_disk(root)


@pytest.mark.asyncio
async def test_a_directory_removed_from_disk_reads_as_its_files_gone(tmp_path: Path) -> None:
    """a directory that no longer exists holds no files: only an unreadable one fails the snapshot."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.txt").write_bytes(b"inside")
    real_walk = Path.walk

    def _walk_with_a_vanished_directory(
        self: Path, top_down: bool = True, on_error: Any = None, follow_symlinks: bool = False
    ) -> Any:
        assert on_error is not None
        on_error(FileNotFoundError(2, "No such file or directory", str(self / "gone")))
        return real_walk(self, top_down=top_down, on_error=on_error, follow_symlinks=follow_symlinks)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "walk", _walk_with_a_vanished_directory)
        result = await materialize_module.snapshot_disk(root)
    assert set(result) == {"ok.txt"}


@pytest.mark.asyncio
async def test_a_missing_root_fails_the_snapshot_rather_than_reading_as_empty(tmp_path: Path) -> None:
    """``Path.walk`` hands the root's own FileNotFoundError to on_error, which read it as an empty workspace."""
    with pytest.raises(FileNotFoundError):
        await materialize_module.snapshot_disk(tmp_path / "never-made")


@pytest.mark.asyncio
async def test_a_root_on_another_device_than_recorded_fails_the_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.txt").write_bytes(b"inside")
    recorded = root.stat().st_dev
    with pytest.raises(materialize_module.DiskRootChangedError):
        await materialize_module.snapshot_disk(root, root_device=recorded + 1)
    assert set(await materialize_module.snapshot_disk(root, root_device=recorded)) == {"ok.txt"}


@pytest.mark.asyncio
async def test_a_file_removed_between_listing_and_reading_reads_as_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """consistent with a vanished directory: it is gone, so capture-back deleting its row is right."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.txt").write_bytes(b"inside")
    (root / "lstat-gone.txt").write_bytes(b"x")
    (root / "read-gone.txt").write_bytes(b"x")
    real_lstat, real_read_bytes = Path.lstat, Path.read_bytes

    def _lstat(self: Path) -> os.stat_result:
        if self.name == "lstat-gone.txt":
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_lstat(self)

    def _read_bytes(self: Path) -> bytes:
        if self.name == "read-gone.txt":
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "lstat", _lstat)
    monkeypatch.setattr(Path, "read_bytes", _read_bytes)
    assert set(await materialize_module.snapshot_disk(root)) == {"ok.txt"}


@pytest.mark.asyncio
async def test_an_in_root_symlink_whose_target_cannot_be_stated_fails_the_snapshot(tmp_path: Path) -> None:
    """``is_file()`` read a looping symlink as "not a file", so capture-back deleted its row."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "loop").symlink_to(root / "loop")
    with pytest.raises(OSError):
        await materialize_module.snapshot_disk(root)


@pytest.mark.asyncio
async def test_a_dangling_in_root_symlink_names_no_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.txt").write_bytes(b"inside")
    (root / "dangling").symlink_to(root / "missing.txt")
    assert set(await materialize_module.snapshot_disk(root)) == {"ok.txt"}


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
