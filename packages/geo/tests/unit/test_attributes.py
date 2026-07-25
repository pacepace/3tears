"""unit tests for SQL-to-MVT attribute coercion.

the NULL case carries the most weight: MVT has no null, so absence is
encoded by omitting the key, and a client that assumes presence will read a
missing measurement as zero.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

import pytest

from threetears.geo.attributes import (
    UnsupportedAttributeError,
    coerce_attribute,
    coerce_attributes,
    validate_attribute_value,
)


class TestNullHandling:
    def test_null_omits_the_key_entirely(self) -> None:
        # the whole point: "no data" must stay distinguishable from zero, or
        # a choropleth shades unmeasured regions as though they were measured.
        row = {"total_unreg": None, "trump_approval": 0.0}
        attributes = coerce_attributes(row, ("total_unreg", "trump_approval"))
        assert "total_unreg" not in attributes
        assert attributes["trump_approval"] == 0.0

    def test_missing_column_is_treated_as_null(self) -> None:
        assert coerce_attributes({}, ("absent",)) == {}


class TestScalars:
    def test_bool_stays_bool_rather_than_becoming_one(self) -> None:
        # bool is an int subclass; encoding True as 1 loses the type style
        # expressions match on.
        assert coerce_attribute(True) is True
        assert coerce_attribute(False) is False

    @pytest.mark.parametrize("value", [0, -5, 42, 3.5, -0.25])
    def test_numbers_pass_through(self, value: float) -> None:
        assert coerce_attribute(value) == value

    def test_strings_pass_through(self) -> None:
        assert coerce_attribute("Phoenix") == "Phoenix"

    def test_decimal_becomes_float(self) -> None:
        assert coerce_attribute(Decimal("1.25")) == 1.25

    def test_uuid_becomes_string(self) -> None:
        value = UUID("00000000-0000-0000-0000-0000000000ff")
        assert coerce_attribute(value) == "00000000-0000-0000-0000-0000000000ff"


class TestTemporal:
    def test_aware_datetime_becomes_utc_iso8601(self) -> None:
        value = datetime(2026, 7, 24, 12, 0, tzinfo=timezone(timedelta(hours=-7)))
        assert coerce_attribute(value) == "2026-07-24T19:00:00+00:00"

    def test_naive_datetime_is_read_as_utc(self) -> None:
        # a tile is served to every timezone, so local time is never the
        # right reading for a naive stamp.
        assert coerce_attribute(datetime(2026, 7, 24, 12, 0)) == "2026-07-24T12:00:00+00:00"

    def test_date_becomes_iso8601(self) -> None:
        assert coerce_attribute(date(2026, 7, 24)) == "2026-07-24"

    def test_utc_datetime_round_trips(self) -> None:
        value = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
        assert coerce_attribute(value) == "2026-07-24T12:00:00+00:00"


class TestStructuredRejection:
    @pytest.mark.parametrize("value", [{"a": 1}, [1, 2], (1, 2), {1, 2}, b"bytes"])
    def test_structured_values_are_rejected_not_stringified(self, value: object) -> None:
        # silently stringifying a JSONB blob produces a large attribute no
        # style expression can use; failing is the honest outcome.
        with pytest.raises(UnsupportedAttributeError):
            coerce_attribute(value)

    def test_validation_names_the_offending_column(self) -> None:
        with pytest.raises(UnsupportedAttributeError, match="'places_payload'"):
            validate_attribute_value("places_payload", {"rating": 4.5})

    def test_validation_accepts_null_since_type_is_unknowable_from_it(self) -> None:
        validate_attribute_value("maybe", None)

    def test_validation_accepts_scalars(self) -> None:
        for value in (1, 1.5, "x", True):
            validate_attribute_value("ok", value)


class TestProjection:
    def test_only_declared_columns_are_emitted(self) -> None:
        # a tile carries what the layer declared, not whatever the query
        # happened to select -- undeclared columns are payload weight on
        # every request.
        row = {"name": "Site A", "score": 80, "internal_note": "do not ship"}
        assert coerce_attributes(row, ("name", "score")) == {"name": "Site A", "score": 80}
