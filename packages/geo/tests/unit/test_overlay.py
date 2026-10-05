from __future__ import annotations

import pytest
from shapely.errors import GEOSException
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from threetears.geo import (
    KeyedFeature,
    LayerOverlayError,
    area_km2,
    cut,
    mean_width_m,
    missing_features,
    replace_features,
)

#: the thresholds measured on US counties cut by congressional districts
WIDTH_M, AREA_KM2 = 50.0, 1.0


def square(feature_id: str, key: str, x: float, y: float, size: float = 1.0) -> KeyedFeature:
    return KeyedFeature(feature_id, key, box(x, y, x + size, y + size))


def _cut(coarse: list[KeyedFeature], fine: list[KeyedFeature]):  # type: ignore[no-untyped-def]
    return cut(coarse, fine, min_width_m=WIDTH_M, min_area_km2=AREA_KM2)


class TestMeasures:
    def test_width_is_meters_at_the_shape_latitude(self) -> None:
        # area over half the perimeter: half a square's side, a long strip's width
        assert mean_width_m(box(0, 0, 0.001, 0.001)) == pytest.approx(55.7, rel=0.01)
        assert mean_width_m(box(0, 60, 0.001, 60.001)) == pytest.approx(37.1, rel=0.01)

    def test_area_is_square_kilometers_at_the_shape_latitude(self) -> None:
        assert area_km2(box(0, 0, 0.01, 0.01)) == pytest.approx(1.239, rel=0.01)
        assert area_km2(box(0, 60, 0.01, 60.01)) == pytest.approx(0.620, rel=0.01)


class TestCut:
    def test_features_are_cut_only_within_their_key(self) -> None:
        county = square("48001", "48", 0, 0, 2)
        west, east = square("4801", "48", -1, -1, 2), square("4802", "48", 1, -1, 2)
        elsewhere = square("0401", "04", 0, 0, 2)  # overlaps in space, another key
        result = _cut([county], [west, east, elsewhere])
        assert [(p.feature_id, round(p.geometry.area, 6)) for p in result.pieces] == [
            ("48001-4801", 1.0),
            ("48001-4802", 1.0),
        ]
        assert (result.slivers_dropped, result.without_pieces) == (0, ())

    def test_a_thin_sliver_is_dropped(self) -> None:
        county = square("48001", "48", 0, 0, 1)
        result = _cut([county], [square("4801", "48", 0.9999, 0, 1)])  # about 11 m wide
        assert (result.pieces, result.slivers_dropped, result.without_pieces) == ((), 1, ("48001",))

    def test_a_small_compact_fragment_is_dropped(self) -> None:
        # about 200 m square: wide enough, but 0.04 km², like a coastline fragment
        result = _cut([square("48001", "48", 0, 0, 0.1)], [square("4801", "48", 0.098, 0.098, 0.1)])
        assert result.pieces == ()

    def test_a_narrow_real_piece_is_kept(self) -> None:
        # a 330 m strip down a whole side: narrow, but 3.7 km²
        result = _cut([square("48001", "48", 0, 0, 0.1)], [square("4801", "48", 0.097, 0, 0.1)])
        assert [p.feature_id for p in result.pieces] == ["48001-4801"]

    def test_each_part_is_judged_on_its_own(self) -> None:
        county = square("48001", "48", 0, 0, 0.1)
        # a real square and, apart from it, a 1 m strip on the far edge
        district = KeyedFeature("4801", "48", box(-0.01, -0.01, 0.05, 0.05).union(box(0.09999, 0.06, 0.2, 0.1)))
        result = _cut([county], [district])
        (piece,) = result.pieces
        assert piece.geometry.bounds[2] == pytest.approx(0.05)
        assert result.slivers_dropped == 1

    def test_neighbours_that_only_touch_are_not_slivers(self) -> None:
        county = square("48001", "48", 0, 0, 1)
        result = _cut([county], [square("4801", "48", 0, 0, 1), square("4802", "48", 1, 0, 1)])
        assert ([p.feature_id for p in result.pieces], result.slivers_dropped) == (["48001-4801"], 0)

    def test_a_shape_the_engine_cannot_cut_names_both_features(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(self: BaseGeometry, other: BaseGeometry) -> BaseGeometry:
            raise GEOSException("TopologyException: side location conflict")

        monkeypatch.setattr(BaseGeometry, "intersection", refuse)
        with pytest.raises(LayerOverlayError, match="cannot cut '48001' by '4801'"):
            _cut([square("48001", "48", 0, 0)], [square("4801", "48", 0, 0)])


class TestReplace:
    def test_replaces_exactly_the_given_keys(self) -> None:
        base = [square("4801", "48", 0, 0), square("0401", "04", 5, 5)]
        new = [square("4801", "48", 0, 0, 2), square("4802", "48", 2, 0)]
        result = replace_features(base, new, frozenset({"48"}))
        assert [(f.feature_id, f.geometry.bounds[2]) for f in result] == [("0401", 6.0), ("4801", 2.0), ("4802", 3.0)]

    def test_refuses_replacements_that_disagree_with_the_keys(self) -> None:
        with pytest.raises(LayerOverlayError, match="replacements cover keys"):
            replace_features([], [square("4801", "48", 0, 0)], frozenset({"48", "06"}))


def test_missing_features_are_the_needed_ids_with_no_feature() -> None:
    assert missing_features(["48001", "48003"], ["48001", "48005", "48005"]) == ("48005",)
    assert missing_features(["48001"], ["48001"]) == ()
