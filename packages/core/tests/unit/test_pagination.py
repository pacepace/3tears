"""Unit tests for the keyset (seek) paginator (threetears.core.pagination)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from threetears.core import CursorError, Keyset, Page, decode_cursor, encode_cursor


class TestCursorCodec:
    def test_round_trips_strings_and_ints(self) -> None:
        token = encode_cursor(["2026-06-03T12:00:00+00:00", 42])
        assert decode_cursor(token) == ["2026-06-03T12:00:00+00:00", 42]

    def test_stringifies_datetime_and_uuid(self) -> None:
        dt = datetime(2026, 6, 3, 12, 0, tzinfo=UTC)
        uid = UUID("019e3e26-9870-7a03-8f04-8cc6a4f5f418")
        decoded = decode_cursor(encode_cursor([dt, uid]))
        # default=str -> str(dt) ('2026-06-03 12:00:00+00:00', a valid timestamptz literal)
        # and str(uid); the column ``casts`` restore the types at query time.
        assert decoded == [str(dt), str(uid)]

    def test_token_is_opaque_urlsafe(self) -> None:
        token = encode_cursor(["a/b+c", 1])
        assert "/" not in token and "+" not in token  # url-safe base64

    def test_decode_rejects_garbage(self) -> None:
        with pytest.raises(CursorError):
            decode_cursor("!!!not-base64!!!")

    def test_decode_rejects_non_array(self) -> None:
        # base64 of a JSON object, not an array
        token = encode_cursor(["x"]).replace("=", "")
        with pytest.raises(CursorError):
            decode_cursor("eyJhIjogMX0=")  # {"a": 1}
        del token


class TestKeysetValidation:
    def test_requires_at_least_one_column(self) -> None:
        with pytest.raises(ValueError, match="at least one column"):
            Keyset(columns=())

    def test_casts_arity_must_match(self) -> None:
        with pytest.raises(ValueError, match="casts arity"):
            Keyset(columns=("a", "b"), casts=("uuid",))


class TestOrderBy:
    def test_descending(self) -> None:
        assert Keyset(("date_created", "message_id")).order_by() == ("date_created DESC, message_id DESC")

    def test_ascending(self) -> None:
        assert Keyset(("id",), descending=False).order_by() == "id ASC"


class TestPredicate:
    def test_first_page_has_no_predicate(self) -> None:
        ks = Keyset(("date_created", "message_id"))
        sql, params = ks.predicate(cursor=None, first_param=4)
        assert sql == "" and params == []

    def test_single_column_descending(self) -> None:
        ks = Keyset(("message_id",), casts=("uuid",))
        cursor = encode_cursor(["019e3e26-9870-7a03-8f04-8cc6a4f5f418"])
        sql, params = ks.predicate(cursor, first_param=1)
        assert sql == "(message_id) < ($1::uuid)"
        assert params == ["019e3e26-9870-7a03-8f04-8cc6a4f5f418"]

    def test_composite_with_casts_and_offset_params(self) -> None:
        ks = Keyset(("date_created", "message_id"), casts=("timestamptz", "uuid"))
        cursor = encode_cursor(["2026-06-03T12:00:00+00:00", "019e-..."])
        sql, params = ks.predicate(cursor, first_param=5)
        assert sql == "(date_created, message_id) < ($5::timestamptz, $6::uuid)"
        assert params == ["2026-06-03T12:00:00+00:00", "019e-..."]

    def test_ascending_uses_gt(self) -> None:
        ks = Keyset(("id",), descending=False)
        sql, _ = ks.predicate(encode_cursor([10]), first_param=1)
        assert sql == "(id) > ($1)"

    def test_cursor_arity_mismatch_raises(self) -> None:
        ks = Keyset(("a", "b"))
        with pytest.raises(CursorError, match="arity"):
            ks.predicate(encode_cursor([1]), first_param=1)


class TestPage:
    def _ks(self) -> Keyset:
        return Keyset(("date_created", "message_id"), casts=("timestamptz", "uuid"))

    def test_trims_sentinel_and_emits_next_cursor(self) -> None:
        # page_size=2, fetched 3 (the +1 sentinel) -> there IS a next page
        rows = [
            {"date_created": "t3", "message_id": "m3"},
            {"date_created": "t2", "message_id": "m2"},
            {"date_created": "t1", "message_id": "m1"},  # sentinel
        ]
        page = self._ks().page(rows, page_size=2, key_of=lambda r: (r["date_created"], r["message_id"]))
        assert [r["message_id"] for r in page.items] == ["m3", "m2"]  # sentinel dropped
        assert page.next_cursor is not None
        # cursor anchors on the LAST KEPT row (m2), not the sentinel
        assert decode_cursor(page.next_cursor) == ["t2", "m2"]

    def test_no_sentinel_means_last_page(self) -> None:
        rows = [{"date_created": "t2", "message_id": "m2"}, {"date_created": "t1", "message_id": "m1"}]
        page = self._ks().page(rows, page_size=2, key_of=lambda r: (r["date_created"], r["message_id"]))
        assert len(page.items) == 2
        assert page.next_cursor is None

    def test_empty(self) -> None:
        page: Page[dict] = self._ks().page([], page_size=2, key_of=lambda r: (r["x"],))
        assert page.items == [] and page.next_cursor is None
