"""A host's palette, read from an artifact with :func:`~threetears.evals.vega.load_palette`.

The loader a host declares its colours through, beside ``load_chart_font``: an artifact of the packaged
one's shape, held to the packaged artifact's checks — every colour resolved sRGB hex, and slot widths the
chart vocabulary assigns. The measurement is the check that matters: an artifact validated over fewer than
``VALIDATED_SLOTS`` slots would carry a separation promise the engine makes and nothing checked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.kernel.host import SERIES_SLOTS, VALIDATED_SLOTS, ChartPalette
from threetears.evals.vega import PaletteError, load_palette, packaged_palette
from threetears.evals.vega.palette import packaged_artifact


def _artifact() -> dict[str, Any]:
    """A host artifact: the packaged one's shape and colours, a deep copy to edit."""
    copied: dict[str, Any] = json.loads(json.dumps(packaged_artifact()))
    return copied


def _write(tmp_path: Path, artifact: Any) -> Path:
    path = tmp_path / "chart_palette.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_an_artifact_of_the_packaged_shape_reads_as_the_palette_it_declares(tmp_path: Path, theme: str):
    palette = load_palette(_write(tmp_path, _artifact()), theme)  # type: ignore[arg-type]
    assert isinstance(palette, ChartPalette)
    assert palette == packaged_palette(theme)  # type: ignore[arg-type]


def test_a_host_measured_over_more_slots_than_the_engine_validates_is_admitted(tmp_path: Path):
    artifact = _artifact()
    artifact["validated_slots"] = SERIES_SLOTS
    assert load_palette(_write(tmp_path, artifact), "dark").series == packaged_palette("dark").series


def test_an_artifact_measured_over_fewer_slots_than_the_engine_validates_is_refused(tmp_path: Path):
    artifact = _artifact()
    artifact["validated_slots"] = VALIDATED_SLOTS - 1
    with pytest.raises(PaletteError, match=rf"validated_slots={VALIDATED_SLOTS - 1}.*assigns {VALIDATED_SLOTS} slots"):
        load_palette(_write(tmp_path, artifact), "dark")


def test_an_artifact_validated_past_its_own_width_is_refused(tmp_path: Path):
    artifact = _artifact()
    artifact["validated_slots"] = SERIES_SLOTS + 1
    with pytest.raises(PaletteError, match=rf"validated_slots={SERIES_SLOTS + 1}"):
        load_palette(_write(tmp_path, artifact), "dark")


def test_an_artifact_of_another_width_is_refused(tmp_path: Path):
    artifact = _artifact()
    artifact["series_slots"] = SERIES_SLOTS - 1
    with pytest.raises(PaletteError, match=rf"series_slots={SERIES_SLOTS - 1}.*exactly the {SERIES_SLOTS}"):
        load_palette(_write(tmp_path, artifact), "dark")


@pytest.mark.parametrize("width", [None, True, 4.0, "4"])
def test_a_slot_width_that_is_not_a_whole_number_is_refused(tmp_path: Path, width: object):
    artifact = _artifact()
    if width is None:
        del artifact["validated_slots"]
    else:
        artifact["validated_slots"] = width
    with pytest.raises(PaletteError, match="validated_slots=.*must be a whole number"):
        load_palette(_write(tmp_path, artifact), "dark")


def test_a_colour_a_rasteriser_draws_black_is_refused(tmp_path: Path):
    artifact = _artifact()
    artifact["light"]["ink"] = "oklch(0.30 0.02 260)"
    with pytest.raises(PaletteError, match=r"light\.ink.*not resolved sRGB hex"):
        load_palette(_write(tmp_path, artifact), "dark")


def test_a_theme_the_artifact_has_no_block_for_is_refused(tmp_path: Path):
    artifact = _artifact()
    del artifact["light"]
    with pytest.raises(PaletteError, match="has no 'light' block"):
        load_palette(_write(tmp_path, artifact), "light")


def test_a_block_missing_a_role_is_refused_naming_it(tmp_path: Path):
    artifact = _artifact()
    del artifact["dark"]["on_fill"], artifact["dark"]["grid"]
    with pytest.raises(PaletteError, match="'dark' block names no grid, on_fill"):
        load_palette(_write(tmp_path, artifact), "dark")


def test_a_block_that_is_not_a_palette_is_refused(tmp_path: Path):
    artifact = _artifact()
    artifact["dark"]["series"] = artifact["dark"]["series"][:-1]
    with pytest.raises(PaletteError, match="'dark' block is not a palette"):
        load_palette(_write(tmp_path, artifact), "dark")


def test_a_missing_file_is_refused(tmp_path: Path):
    with pytest.raises(PaletteError, match="missing at"):
        load_palette(tmp_path / "absent.json", "dark")


def test_a_file_that_is_not_json_is_refused(tmp_path: Path):
    path = tmp_path / "chart_palette.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PaletteError, match="is not JSON"):
        load_palette(path, "dark")


def test_a_file_that_is_not_an_object_is_refused(tmp_path: Path):
    with pytest.raises(PaletteError, match="is not an object"):
        load_palette(_write(tmp_path, ["#000000"]), "dark")
