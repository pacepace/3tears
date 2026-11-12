"""Column tokens: which columns of a part a write moved, known from the file's footer without reading a row.

The contracts (``stamped``, ``stamping``, ``file_tokens``, ``column_print``). A column keeps its base file's token
only when the base holds tokens, as many rows, and the column equal position by position with the same type; any other
column, and every column of a file with no readable base, takes a fresh token. So every column holding a token holds
exactly what the token was minted for. A reader, an opener, a file and parts are written with no tokens. The tokens
survive the slot's write and the publish that links the file into a version. A reader of tokens opens a file's footer
once per process, and a file with no tokens prints as its identity: any write moves it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from threetears.core.shared_arrow import (
    COLUMN_TOKENS_KEY,
    FileTokens,
    Parts,
    SharedArrowStore,
    column_print,
    file_tokens,
    stamped,
    stamping,
    version_files,
)


def _rows(**columns: Any) -> pa.Table:
    """a table of the given columns.

    :return: the table
    :rtype: pa.Table
    """
    return pa.table(columns)


def _base() -> pa.Table:
    return _rows(
        a=pa.array([1, 2, 3], pa.int64()), s=pa.array(["x", "y", "z"], pa.string()), n=pa.array([1.0, 2.0, 3.0])
    )


def _write(path: Path, table: pa.Table) -> Path:
    """``table`` as an uncompressed Arrow IPC file at ``path``.

    :return: the path
    :rtype: Path
    """
    with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return path


def _tokens(table: pa.Table) -> FileTokens:
    """the tokens a stamped table's schema holds.

    :return: the tokens
    :rtype: FileTokens
    """
    held = json.loads(table.schema.metadata[COLUMN_TOKENS_KEY])
    return FileTokens(held["rows"], held["columns"])


@pytest.fixture
def base(tmp_path: Path) -> tuple[Path, FileTokens]:
    """a stamped base file and its tokens.

    :return: (the file, its tokens)
    :rtype: tuple[Path, FileTokens]
    """
    table = stamped(_base(), None)
    return _write(tmp_path / "base.arrow", table), _tokens(table)


class TestAColumnKeepsItsTokenOnlyWhenItIsEqual:
    def test_an_unchanged_column_keeps_its_token_and_a_changed_one_takes_a_fresh_one(
        self, base: tuple[Path, FileTokens]
    ) -> None:
        path, before = base
        table = _base().set_column(0, "a", pa.array([1, 2, 4], pa.int64()))
        after = _tokens(stamped(table, path))
        assert after.columns["s"] == before.columns["s"]
        assert after.columns["n"] == before.columns["n"]
        assert after.columns["a"] != before.columns["a"]
        assert after.rows == before.rows

    def test_a_value_turned_null_takes_a_fresh_token(self, base: tuple[Path, FileTokens]) -> None:
        path, before = base
        table = _base().set_column(1, "s", pa.array(["x", None, "z"], pa.string()))
        after = _tokens(stamped(table, path))
        assert after.columns["s"] != before.columns["s"]
        assert after.columns["a"] == before.columns["a"]

    def test_a_column_of_another_type_takes_a_fresh_token(self, base: tuple[Path, FileTokens]) -> None:
        path, before = base
        table = _base().set_column(0, "a", pa.array([1, 2, 3], pa.int32()))
        assert _tokens(stamped(table, path)).columns["a"] != before.columns["a"]

    def test_rows_reordered_take_fresh_tokens(self, base: tuple[Path, FileTokens]) -> None:
        path, before = base
        after = _tokens(stamped(_base().take([2, 1, 0]), path))
        assert after.columns["a"] != before.columns["a"]
        assert after.columns["s"] != before.columns["s"]
        # the row count did not move
        assert after.rows == before.rows

    @pytest.mark.parametrize("rows", [[0, 1], [0, 1, 2, 2]], ids=["a row removed", "a row added"])
    def test_a_row_count_moved_makes_every_token_fresh(self, base: tuple[Path, FileTokens], rows: list[int]) -> None:
        path, before = base
        after = _tokens(stamped(_base().take(rows), path))
        assert after.rows != before.rows
        assert set(after.columns.values()).isdisjoint(before.columns.values())

    def test_a_new_column_takes_a_fresh_token(self, base: tuple[Path, FileTokens]) -> None:
        path, before = base
        after = _tokens(stamped(_base().append_column("b", pa.array([7, 8, 9])), path))
        assert after.columns["b"] not in before.columns.values()
        assert after.columns["a"] == before.columns["a"]

    def test_a_base_holding_no_tokens_or_none_makes_every_token_fresh(self, tmp_path: Path) -> None:
        plain = _write(tmp_path / "plain.arrow", _base())
        for given in (plain, tmp_path / "gone.arrow", tmp_path / "not-arrow.txt", None):
            if given is not None and given.suffix == ".txt":
                given.write_text("not an Arrow file")
            one, two = _tokens(stamped(_base(), given)), _tokens(stamped(_base(), given))
            assert one.rows != two.rows
            assert set(one.columns.values()).isdisjoint(two.columns.values())

    def test_other_metadata_is_kept(self, base: tuple[Path, FileTokens]) -> None:
        path, _ = base
        table = _base().replace_schema_metadata({b"mine": b"kept"})
        assert stamped(table, path).schema.metadata[b"mine"] == b"kept"


class TestAWriteStepStampsOnlyTables:
    def _step(self) -> tuple[Any, dict[str, Any]]:
        given: dict[str, Any] = {}

        def write(tables: Mapping[str, Any]) -> int:
            given.update(tables)
            return 0

        return write, given

    def test_a_table_is_stamped_against_its_base_by_name(self, base: tuple[Path, FileTokens]) -> None:
        path, before = base
        write, given = self._step()
        asked: list[str] = []

        def base_of(name: str) -> Path | None:
            asked.append(name)
            return path if name == "t.p1" else None

        stamping(write, base_of)({"t.p1": _base()})
        assert asked == ["t.p1"]
        assert _tokens(given["t.p1"]) == before

    def test_a_reader_an_opener_a_file_and_parts_are_written_without_tokens(
        self, base: tuple[Path, FileTokens]
    ) -> None:
        path, _ = base
        carrying = stamped(_base(), None)
        parts = Parts(schema=_base().schema, parts={"p": _base()})
        write, given = self._step()
        stamping(write, lambda name: path)(
            {
                "reader": carrying.to_reader(),
                "opener": lambda: carrying,
                "file": path,
                "parts": parts,
            }
        )
        assert COLUMN_TOKENS_KEY not in (given["reader"].schema.metadata or {})
        assert given["reader"].read_all().equals(_base())
        opened = given["opener"]()
        assert COLUMN_TOKENS_KEY not in (opened.schema.metadata or {})
        assert given["file"] is path
        assert given["parts"] is parts


class TestTokensAreReadFromTheFooterOncePerFile:
    def test_a_file_without_tokens_has_none(self, tmp_path: Path) -> None:
        assert file_tokens(_write(tmp_path / "plain.arrow", _base())) is None

    def test_a_cached_file_is_not_opened_again_and_a_new_file_at_its_name_is(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = stamped(_base(), None)
        path = _write(tmp_path / "f.arrow", first)
        opened: list[str] = []
        real = pa.memory_map

        def counting(where: str, mode: str = "r") -> Any:
            opened.append(where)
            return real(where, mode)

        monkeypatch.setattr(pa, "memory_map", counting)
        assert file_tokens(path) == _tokens(first)
        assert file_tokens(path) == _tokens(first)
        assert opened == [str(path)]
        # a new file under the same name: another inode (the old one held open so its inode is not reused)
        with open(path, "rb"):
            os.unlink(path)
            second = stamped(_base(), None)
            _write(path, second)
            assert file_tokens(path) == _tokens(second)
        assert opened == [str(path), str(path)]

    def test_a_file_without_tokens_prints_as_its_identity(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "plain.arrow", _base())
        status = os.stat(path)
        assert column_print(path, ["a"]) == f"file:{path}:{status.st_ino}:{status.st_size}:{status.st_mtime_ns}"

    def test_a_print_names_the_row_token_and_the_columns_read(self, base: tuple[Path, FileTokens]) -> None:
        path, tokens = base
        assert column_print(path, ["s", "a"]) == f"{tokens.rows}|a={tokens.columns['a']};s={tokens.columns['s']}"
        assert column_print(path, ["missing"]) == f"{tokens.rows}|missing="


class TestTokensSurviveTheSlotAndThePublish:
    def test_a_stamped_part_written_by_a_slot_and_linked_by_a_publish_keeps_its_tokens(self, tmp_path: Path) -> None:
        store = SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)
        schema = _base().schema
        store.publish("v1", {"t": Parts(schema=schema, parts={"p": _base()})})
        held = version_files(store.root, "v1")["t@p.arrow"].path

        slot = store.reserve(1 << 20)
        with slot.writing() as write:
            stamping(write, lambda name: held)({"t.p": _base().set_column(0, "a", pa.array([9, 9, 9], pa.int64()))})
        claimed = store.claim(slot)
        assert claimed is not None
        written = file_tokens(claimed.files["t.p"])
        assert written is not None
        # the slot's write checks each batch's columns against the table's, its metadata aside
        store.publish("v2", {"t": Parts(schema=schema, parts={"p": claimed.files["t.p"]})}, base="v1")
        claimed.close()
        assert file_tokens(version_files(store.root, "v2")["t@p.arrow"].path) == written
