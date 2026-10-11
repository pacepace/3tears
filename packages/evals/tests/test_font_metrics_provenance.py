"""A font table is held to the engine that reads it and to the files it was measured from.

The measurer ships in the package (``python -m threetears.evals.vega.measure_font``) and writes through the
same gate it always did; what it adds is provenance. A table records its metrics format and the SHA-256 of each
font file that declared the measured family, and two refusals read them: turning a table into a font refuses
another format, and a renderer given a font directory refuses a font whose files there have changed. Each
refusal names the command that re-measures the face.

The font files here are built in the test, as the smallest sfnt a font database reads a family from (a table
directory and a ``name`` table), because the check reads names and bytes and never draws them.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.kernel.host import (
    CHART_FONT_CHARACTERS,
    ChartFont,
    StyleError,
    StyleProfile,
)
from threetears.evals.kernel.host.style import assert_no_style_text
from threetears.evals.vega import (
    TextMetricsError,
    VegaRenderer,
    load_chart_font,
    packaged_font,
    packaged_palette,
    render_svg,
    write_font_metrics,
)
from threetears.evals.vega.measure_font import (
    EMBEDDED_FACE,
    FontMeasureError,
    face_files,
    font_families,
    main,
    resolving_files,
)
from threetears.evals.vega.text_metrics import (
    MEASURE_COMMAND,
    METRICS_FORMAT,
    METRICS_PATH,
    chart_font_from_artifact,
    check_font_files,
    font_file_digest,
    load_metrics,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _name_table(names: list[tuple[int, int, int, str]]) -> bytes:
    """A ``name`` table holding ``(platform, encoding, name_id, text)`` records."""
    storage = b""
    records = b""
    for platform, encoding, name_id, text in names:
        raw = text.encode("mac_roman") if platform == 1 else text.encode("utf-16-be")
        records += struct.pack(">HHHHHH", platform, encoding, 0x409, name_id, len(raw), len(storage))
        storage += raw
    return struct.pack(">HHH", 0, len(names), 6 + 12 * len(names)) + records + storage


def _sfnt(names: list[tuple[int, int, int, str]], *, at: int = 0) -> bytes:
    """One sfnt face whose table directory starts at ``at`` in the file, holding only a ``name`` table."""
    table = _name_table(names)
    start = at + 12 + 16
    header = struct.pack(">IHHHH", 0x00010000, 1, 16, 0, 0)
    return header + struct.pack(">4sIII", b"name", 0, start, len(table)) + table


def _font(family: str, *, typographic: str | None = None) -> bytes:
    """A font file declaring ``family`` on the Windows platform, and ``typographic`` as name ID 16."""
    names = [(3, 1, 1, family), (3, 1, 2, "Regular")]
    if typographic is not None:
        names.append((3, 1, 16, typographic))
    return _sfnt(names)


def _collection(*families: str) -> bytes:
    """A font collection holding one face per family."""
    header_size = 12 + 4 * len(families)
    faces: list[bytes] = []
    offsets: list[int] = []
    at = header_size
    for family in families:
        face = _sfnt([(3, 1, 1, family)], at=at)
        offsets.append(at)
        faces.append(face)
        at += len(face)
    return struct.pack(f">4sII{len(families)}I", b"ttcf", 0x00010000, len(families), *offsets) + b"".join(faces)


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _measured(**update: Any) -> dict[str, Any]:
    """What the measurer hands ``write_font_metrics``, for a face measured from ``host.ttf``."""
    return {
        "advances": dict.fromkeys(CHART_FONT_CHARACTERS, 0.6),
        "fallback_advance": 0.6,
        "worst_label": "WWWW",
        "worst_ratio": 1.0,
        "font": "Host Face, sans-serif",
        "measured_with": "vl-convert-python test",
        "probe_size": 1000,
        "weights": [400, 600],
        "font_files": {"host.ttf": "0" * 64},
    } | update


def _host_font(font_dir: Path) -> ChartFont:
    """A host font measured from ``font_dir/host.ttf`` as it is now."""
    face = _write(font_dir / "host.ttf", _font("Host Face"))
    return ChartFont(
        family="Host Face, sans-serif",
        advances=dict.fromkeys(CHART_FONT_CHARACTERS, 0.6),
        fallback_advance=0.6,
        font_files={"host.ttf": font_file_digest(face)},
    )


# --- the table records its format and its files ------------------------------------------------------


def test_a_written_table_records_its_format_and_files_and_reads_back_with_them(tmp_path: Path) -> None:
    path = write_font_metrics(**_measured(), path=tmp_path / "host.json")

    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["format"] == METRICS_FORMAT
    assert artifact["font_files"] == {"host.ttf": "0" * 64}
    assert load_chart_font(path).font_files == {"host.ttf": "0" * 64}


def test_the_packaged_table_is_this_format_and_records_no_files() -> None:
    """The packaged face is embedded in the rasteriser, so no file provided it and none is recorded."""
    assert load_metrics()["format"] == METRICS_FORMAT
    assert packaged_font().font_files == {}


@pytest.mark.parametrize("written", [None, 1, METRICS_FORMAT + 1, "2"])
def test_a_table_of_another_format_is_refused_naming_the_command(tmp_path: Path, written: object) -> None:
    path = write_font_metrics(**_measured(), path=tmp_path / "host.json")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if written is None:
        del artifact["format"]
    else:
        artifact["format"] = written
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(TextMetricsError) as refused:
        load_chart_font(path)

    message = str(refused.value)
    assert f"this engine reads format {METRICS_FORMAT}" in message
    assert f"{MEASURE_COMMAND} --family 'Host Face, sans-serif' --font-dir " in message
    assert f"--out {path}" in message


def test_the_packaged_table_s_refusal_names_the_bare_command() -> None:
    """The packaged face is re-measured by the command with no arguments, into the packaged table."""
    stale = load_metrics() | {"format": 1}

    with pytest.raises(TextMetricsError, match=rf"Re-measure the face: {MEASURE_COMMAND}$"):
        chart_font_from_artifact(stale, where=str(METRICS_PATH))


# --- the contract holds the files' shape --------------------------------------------------------------


@pytest.mark.parametrize("name", ["/abs/host.ttf", "../host.ttf", "a/./host.ttf", "host\\font.ttf", "host;rm.ttf", ""])
def test_a_font_file_that_is_not_a_relative_path_is_refused(name: str) -> None:
    with pytest.raises(StyleError, match="is not a relative path"):
        ChartFont(
            family="Host Face",
            advances=dict.fromkeys(CHART_FONT_CHARACTERS, 0.6),
            fallback_advance=0.6,
            font_files={name: "0" * 64},
        )


@pytest.mark.parametrize("digest", ["0" * 63, "A" * 64, "g" * 64, 7])
def test_a_font_file_digest_that_is_not_sha256_hex_is_refused(digest: object) -> None:
    with pytest.raises(StyleError, match="SHA-256"):
        ChartFont(
            family="Host Face",
            advances=dict.fromkeys(CHART_FONT_CHARACTERS, 0.6),
            fallback_advance=0.6,
            font_files={"host.ttf": digest},  # type: ignore[dict-item]
        )


def test_a_font_file_name_is_a_host_string_the_prompt_check_walks() -> None:
    font = ChartFont(
        family="Host Face",
        advances=dict.fromkeys(CHART_FONT_CHARACTERS, 0.6),
        fallback_advance=0.6,
        font_files={"fonts/HostFace-Regular.ttf": "0" * 64},
    )

    with pytest.raises(StyleError, match="HostFace-Regular"):
        assert_no_style_text("set in fonts/HostFace-Regular.ttf", StyleProfile(chart_font=font))


# --- the renderer refuses files that changed ----------------------------------------------------------


def test_files_as_measured_pass_and_an_unrelated_face_beside_them_changes_nothing(tmp_path: Path) -> None:
    font = _host_font(tmp_path)
    _write(tmp_path / "Other-Regular.ttf", _font("Other Face"))

    check_font_files(font, [tmp_path])
    (tmp_path / "Other-Regular.ttf").write_bytes(_font("Other Face Updated"))
    check_font_files(font, [tmp_path])


def test_a_font_file_changed_in_place_is_refused_naming_the_command(tmp_path: Path) -> None:
    font = _host_font(tmp_path)
    (tmp_path / "host.ttf").write_bytes(_font("Host Face", typographic="Host Face Variable"))

    with pytest.raises(TextMetricsError, match="hashes differently") as refused:
        check_font_files(font, [tmp_path])

    assert f"{MEASURE_COMMAND} --family 'Host Face, sans-serif' --font-dir {tmp_path} --out" in str(refused.value)


def test_a_font_file_missing_from_every_directory_is_refused(tmp_path: Path) -> None:
    font = _host_font(tmp_path / "measured")
    (tmp_path / "elsewhere").mkdir()

    with pytest.raises(TextMetricsError, match="host.ttf is in none of"):
        check_font_files(font, [tmp_path / "elsewhere"])


def test_one_matching_directory_among_several_is_enough(tmp_path: Path) -> None:
    font = _host_font(tmp_path / "measured")
    _write(tmp_path / "stale" / "host.ttf", b"another face entirely")

    check_font_files(font, [tmp_path / "stale", tmp_path / "measured"])


def test_a_font_that_records_no_files_has_nothing_to_check(tmp_path: Path) -> None:
    check_font_files(packaged_font(), [tmp_path])


def test_a_renderer_built_over_changed_files_is_refused_when_built(tmp_path: Path) -> None:
    font = _host_font(tmp_path)
    VegaRenderer(palette=packaged_palette("dark"), font=font, font_dir=tmp_path)

    (tmp_path / "host.ttf").write_bytes(b"replaced")

    with pytest.raises(TextMetricsError, match="hashes differently"):
        VegaRenderer(palette=packaged_palette("dark"), font=font, font_dir=tmp_path)
    with pytest.raises(TextMetricsError, match="hashes differently"):
        VegaRenderer.for_style(StyleProfile(chart_font=font), font_dir=tmp_path)


def test_a_render_naming_a_directory_without_the_files_is_refused(tmp_path: Path) -> None:
    """The render path's own refusal, for a caller of ``render_svg`` that never built a renderer."""
    font = _host_font(tmp_path / "measured")
    (tmp_path / "empty").mkdir()

    with pytest.raises(TextMetricsError, match="in none of"):
        render_svg({}, palette=packaged_palette("dark"), font_dir=tmp_path / "empty", font=font)


# --- the measurer finds the files that provide the face ----------------------------------------------


def test_a_font_s_families_are_read_from_its_name_table() -> None:
    assert font_families(_font("Host Face")) == {"Host Face"}
    assert font_families(_font("Host Face Light", typographic="Host Face")) == {"Host Face Light", "Host Face"}
    assert font_families(_sfnt([(1, 0, 1, "Mac Face"), (3, 1, 4, "Mac Face Regular")])) == {"Mac Face"}
    assert font_families(_collection("First Face", "Second Face")) == {"First Face", "Second Face"}


@pytest.mark.parametrize("data", [b"", b"not a font at all", b"ttcf\x00\x01\x00\x00\x00\x00\x00\x05"])
def test_bytes_that_are_not_a_font_declare_no_family(data: bytes) -> None:
    assert font_families(data) == frozenset()


def test_only_the_files_declaring_the_face_are_recorded(tmp_path: Path) -> None:
    regular = _write(tmp_path / "HostFace-Regular.ttf", _font("Host Face"))
    _write(tmp_path / "bold" / "HostFace-Bold.OTF", _font("host face"))
    _write(tmp_path / "Other-Regular.ttf", _font("Other Face"))
    _write(tmp_path / "Broken.ttf", b"garbage")
    _write(tmp_path / "HostFace.woff2", _font("Host Face"))
    _write(tmp_path / "Family.ttc", _collection("Other Face", "Host Face"))

    files = resolving_files("Host Face", tmp_path)

    assert sorted(files) == ["Family.ttc", "HostFace-Regular.ttf", "bold/HostFace-Bold.OTF"]
    assert files["HostFace-Regular.ttf"] == font_file_digest(regular)


def test_a_face_no_file_declares_is_refused(tmp_path: Path) -> None:
    _write(tmp_path / "Other-Regular.ttf", _font("Other Face"))

    with pytest.raises(FontMeasureError, match="declares the family 'Host Face'"):
        face_files("Host Face, sans-serif", tmp_path)


def test_a_font_dir_that_is_not_a_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FontMeasureError, match="is not a directory"):
        face_files("Host Face", tmp_path / "missing")


def test_a_face_other_than_the_embedded_one_needs_a_font_dir() -> None:
    with pytest.raises(FontMeasureError, match="Pass --font-dir"):
        face_files("Host Face, sans-serif", None)


def test_the_embedded_face_needs_no_directory_and_records_no_files() -> None:
    assert face_files(f"{EMBEDDED_FACE}, Arial", None) == {}


def test_measuring_the_packaged_face_reproduces_the_packaged_table(tmp_path: Path) -> None:
    """End to end through the command: the shipped measurer, the shipped gate, the shipped table."""
    out = tmp_path / "packaged.json"

    assert main(["--out", str(out)]) == 0

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == load_metrics()


def test_the_measurer_runs_as_a_module() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "threetears.evals.vega.measure_font", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--font-dir" in proc.stdout
