"""A keys-unknown epoch record carries its commit's notes bracket, so a follower can resolve the keys from the notes.

The contracts. The bracket is stored and read back exactly; it is named only on a keys-unknown record and never ends
before it starts; a source that takes it is handed it and one built before it is not; and a reader built before the
field reads the same stored record as keys unknown (behind, never misled).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import BaseModel

from threetears.core.collections.epoch_change import (
    ChangedRow,
    EpochChange,
    advance_recording,
    epoch_change_body,
    read_epoch_change_body,
)

_SINCE = datetime(2026, 11, 3, 20, 0, tzinfo=UTC)
_BRACKET = (_SINCE, _SINCE + timedelta(seconds=3))


class TestTheBracketIsStored:
    def test_a_keys_unknown_body_reads_back_its_bracket(self) -> None:
        body = epoch_change_body("lsf_runs", None, keys_unknown=True, noted_between=_BRACKET)
        record = read_epoch_change_body(body, "lsf_runs", "inc:7")
        assert record.keys_unknown and record.noted_between == _BRACKET

    def test_a_reader_built_before_the_field_reads_keys_unknown(self) -> None:
        class _OldBody(BaseModel):
            table: str
            whole_table: bool = False
            keys_unknown: bool = False
            rows: list[ChangedRow] = []

        old = _OldBody.model_validate_json(
            epoch_change_body("lsf_runs", None, keys_unknown=True, noted_between=_BRACKET)
        )
        assert old.keys_unknown and not old.whole_table and not old.rows

    def test_a_bracket_on_a_record_that_names_its_reach_is_refused(self) -> None:
        with pytest.raises(ValueError, match="keys-unknown"):
            epoch_change_body("lsf_runs", None, noted_between=_BRACKET)
        with pytest.raises(ValueError, match="keys-unknown"):
            EpochChange(table="t", token="i:1", rows=[ChangedRow(ids=["1"])], noted_between=_BRACKET)

    def test_a_bracket_that_ends_before_it_starts_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ends before"):
            epoch_change_body("t", None, keys_unknown=True, noted_between=(_BRACKET[1], _BRACKET[0]))


class _Recording:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def advance(self, table_name: str) -> str:
        return "i:1"


class _Takes(_Recording):
    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
        noted_between: tuple[datetime, datetime] | None = None,
    ) -> str:
        self.calls.append({"keys_unknown": keys_unknown, "noted_between": noted_between})
        return "i:1"


class _Older(_Recording):
    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
    ) -> str:
        self.calls.append({"keys_unknown": keys_unknown})
        return "i:1"


class TestTheBracketReachesTheSource:
    async def test_a_source_that_takes_it_is_handed_it(self) -> None:
        source = _Takes()
        await advance_recording(source, "t", None, keys_unknown=True, noted_between=_BRACKET)  # type: ignore[arg-type]
        assert source.calls == [{"keys_unknown": True, "noted_between": _BRACKET}]

    async def test_a_source_built_before_it_records_keys_unknown(self) -> None:
        source = _Older()
        await advance_recording(source, "t", None, keys_unknown=True, noted_between=_BRACKET)  # type: ignore[arg-type]
        assert source.calls == [{"keys_unknown": True}]
