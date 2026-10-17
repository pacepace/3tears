"""the plug-in boundary: JsonCodec's exact round trip, and TemplateLayout's path <-> key mapping."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import PurePosixPath
from uuid import UUID

import pytest

from threetears.git_store import JsonCodec, Layout, RowCodec, TableLayout, TemplateLayout
from threetears.git_store.layout import escape_value, unescape_value


class TestJsonCodec:
    def test_a_file_it_wrote_decodes_and_encodes_back_to_the_same_bytes(self) -> None:
        codec = JsonCodec()
        row = {
            "zeta": 1,
            "alpha": "ő ✓ 名前",
            "date_updated": datetime(2026, 10, 17, 9, 30, tzinfo=UTC),
            "nested": [1, {"b": 2}],
        }
        data = codec.encode(row)
        assert codec.encode(codec.decode(data)) == data
        assert data.endswith(b"\n")
        assert "ő ✓ 名前".encode() in data  # unicode is written as itself, not escaped

    def test_keys_keep_the_rows_order(self) -> None:
        text = JsonCodec().encode({"zeta": 1, "alpha": 2}).decode()
        assert text.index("zeta") < text.index("alpha")
        ordered = JsonCodec(sort_keys=True).encode({"zeta": 1, "alpha": 2}).decode()
        assert ordered.index("alpha") < ordered.index("zeta")

    def test_version_columns_come_back_as_instants(self) -> None:
        when = datetime(2026, 10, 17, 9, 30, 1, 5, tzinfo=UTC)
        decoded = JsonCodec().decode(JsonCodec().encode({"date_updated": when, "label": "2026-10-17"}))
        assert decoded["date_updated"] == when
        assert decoded["label"] == "2026-10-17"  # only the named columns are parsed

    def test_uuid_and_decimal_have_a_written_form(self) -> None:
        data = JsonCodec().encode({"id": UUID(int=1), "price": Decimal("1.50")})
        assert b"00000000-0000-0000-0000-000000000001" in data and b'"1.50"' in data

    def test_a_naive_datetime_is_refused(self) -> None:
        with pytest.raises(ValueError):
            JsonCodec().encode({"date_updated": datetime(2026, 10, 17)})

    def test_a_file_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(ValueError):
            JsonCodec().decode(b"[1, 2]\n")

    def test_it_is_a_row_codec(self) -> None:
        assert isinstance(JsonCodec(), RowCodec)


def _layout() -> TemplateLayout:
    codec = JsonCodec()
    return TemplateLayout(
        [
            TableLayout("keepers", "{character}/keepers/{batch}/{tile}.json", ("character", "batch", "tile"), codec),
            TableLayout(
                "rejects", "{character}/rejects/{tile}-{n}.json", ("character", "tile", "n"), codec, pk_types={"n": int}
            ),
            TableLayout("whole", "{character}/whole.json", ("character",), codec, pk_in_file=True),
        ]
    )


class TestTemplateLayout:
    def test_path_and_key_round_trip(self) -> None:
        layout = _layout()
        pk = {"character": "aria", "batch": "V3", "tile": "12"}
        path = layout.path_for("keepers", pk)
        assert path == PurePosixPath("aria/keepers/V3/12.json")
        assert layout.identify(path, None) == ("keepers", pk)

    @pytest.mark.parametrize("value", ["a/b", "..x", ".hidden", "50%", "back\\slash", "tab\there", "dash-in-tile"])
    def test_awkward_values_stay_one_component_and_read_back(self, value: str) -> None:
        layout = _layout()
        pk = {"character": "aria", "tile": value, "n": 3}
        path = layout.path_for("rejects", pk)
        assert len(path.parts) == 3  # character / rejects / file: nothing climbed or split
        assert not path.parts[-1].startswith(".")
        assert layout.identify(path, None) == ("rejects", pk)

    @pytest.mark.parametrize("value", ["", ".", ".."])
    def test_a_value_that_names_no_file_is_refused(self, value: str) -> None:
        with pytest.raises(ValueError):
            _layout().path_for("keepers", {"character": value, "batch": "b", "tile": "t"})

    def test_escaping_reverses(self) -> None:
        for value in ["plain", "a/b%c", ".dot", "ünï"]:
            assert unescape_value(escape_value(value)) == value

    def test_the_value_ending_character_is_escaped_so_dashed_tiles_parse(self) -> None:
        layout = _layout()
        path = layout.path_for("rejects", {"character": "aria", "tile": "V3-12", "n": 0})
        assert path.name == "V3%2D12-0.json"
        assert layout.identify(path, None) == ("rejects", {"character": "aria", "tile": "V3-12", "n": 0})

    def test_a_filter_on_a_path_column_narrows_the_glob(self) -> None:
        layout = _layout()
        assert layout.candidates("keepers", None) == "*/keepers/*/*.json"
        assert layout.candidates("keepers", {"character": "aria", "kind": "x"}) == "aria/keepers/*/*.json"
        assert layout.candidates("keepers", {"batch": "V[3]"}) == "*/keepers/V[[]3]/*.json"

    def test_a_path_no_template_matches_holds_no_row(self) -> None:
        layout = _layout()
        assert layout.identify(PurePosixPath("aria/training-log.md"), None) is None
        assert layout.identify(PurePosixPath("aria/rejects/t-notanumber.json"), None) is None

    def test_key_columns_live_in_the_path_not_the_file(self) -> None:
        layout = _layout()
        row = {"character": "aria", "batch": "V3", "tile": "12", "score": 9}
        data = layout.encode("keepers", row)
        assert b"character" not in data and b"score" in data
        assert layout.decode("keepers", PurePosixPath("aria/keepers/V3/12.json"), data) == row

    def test_a_table_may_keep_its_key_in_the_file(self) -> None:
        layout = _layout()
        row = {"character": "aria", "x": 1}
        assert b"character" in layout.encode("whole", row)

    @pytest.mark.parametrize(
        ("path", "pk"),
        [
            ("{character}/x.json", ("character", "batch")),  # a pk column missing from the path
            ("{character}/{character}.json", ("character",)),  # named twice
            ("{character}{batch}.json", ("character", "batch")),  # two fields touching
            ("../{character}.json", ("character",)),  # climbs out
            ("/{character}.json", ("character",)),  # absolute
        ],
    )
    def test_a_template_that_cannot_name_each_key_once_is_refused(self, path: str, pk: tuple[str, ...]) -> None:
        with pytest.raises(ValueError):
            TableLayout("t", path, pk, JsonCodec())

    def test_a_table_laid_out_twice_is_refused(self) -> None:
        table = TableLayout("t", "{a}.json", ("a",), JsonCodec())
        with pytest.raises(ValueError):
            TemplateLayout([table, table])

    def test_it_is_a_layout(self) -> None:
        assert isinstance(_layout(), Layout)
