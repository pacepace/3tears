"""Move a frozen campaign bundle, or a stored analysis record, from aspect bundle shape 65 to 66, and check it.

Shape 66 takes names out of the evidence (owner ruling, 2026-11-11): a bundle freezes ids and facts only, and a page
words and names them when it is rendered. Over unchanged evidence it drops two sentences the bundle worded itself,
each a function of facts the bundle keeps:

- each lockstep group's ``aliased_factors[].sentence`` (its ``factors``, ``n_runs`` and ``n_levels`` stay);
- the factor-pair scan's ``factor_pairs.completeness`` (its counts and pivots stay);
- each aspect bundle's ``schema_version`` becomes 66.

A campaign's derived arm is labelled by its levers' ids read as words from 66 on, where a build between the lever
names and 66 spelled it with their reader names. Such a label cannot be told from a person's by what the frame holds,
so a bundle that carries lever names (``lever_names`` on any aspect) is refused: re-assemble it from its runs instead.

A stored analysis record (``{"analysis": ..., ...}``, as ``tests/fixtures/retired_writers`` holds them) moves each
aspect's frozen bundle the same way, and every fingerprint it records over them with it: each aspect's
``bundle_fingerprint``, each frame entry's, the generation's campaign fingerprint and its ``bundle_schema_version``.

Nothing else is written. Before it writes, the script compares every leaf before and after and refuses when any path
but those moved, and it reads the result back (:meth:`CampaignBundle.from_dict`, or :meth:`CampaignBundle.from_frame`
for an analysis), which refuses a bundle of any shape but this build's. Run from the repository root, once per
fixture::

    uv run python packages/evals/tests/fixtures/report_inputs/migrate_bundle_shape_66.py <file.json | file.json.gz>...

A file is written in the serialization it was read in (compact or indented by two, every character past ASCII escaped
or not, as it was read; a gzipped one at gzip level 9 with no timestamp), so running it on a shape-66 file refuses
and writes nothing.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

from threetears.evals.analysis import AnalysisContextBundle, CampaignBundle
from threetears.evals.analysis.bundle.schema import evidence_digest

__all__ = ["FROM_SHAPE", "TO_SHAPE", "main", "migrate", "moved_paths"]

#: The aspect bundle shape this reads, and the one it writes.
FROM_SHAPE, TO_SHAPE = 65, 66

Document = dict[str, Any]


def _migrate_aspect(bundle: Document) -> Document:
    """One aspect bundle at shape 66."""
    if bundle.get("schema_version") != FROM_SHAPE:
        raise SystemExit(f"an aspect bundle is at shape {bundle.get('schema_version')}, not {FROM_SHAPE}")
    if bundle.get("lever_names"):
        raise SystemExit(
            f"aspect {bundle.get('aspect')!r} carries lever names, so its campaign's derived arms may be labelled with "
            "them, which shape 66 labels by id; re-assemble this bundle from its runs instead"
        )
    moved = deepcopy(bundle)
    moved["schema_version"] = TO_SHAPE
    for group in moved.get("aliased_factors") or []:
        group.pop("sentence", None)
    if isinstance(pairs := moved.get("factor_pairs"), dict):
        pairs.pop("completeness", None)
    return moved


def migrate(data: Document) -> Document:
    """A campaign bundle with every aspect bundle at shape 66, or a stored analysis record with its frozen bundles
    and the fingerprints over them at 66; everything else as it was."""
    if "analysis" not in data:
        return {**data, "aspects": [_migrate_aspect(aspect) for aspect in data["aspects"]]}
    analysis = deepcopy(data["analysis"])
    for aspect in analysis["aspects"]:
        aspect["bundle"] = _migrate_aspect(aspect["bundle"])
        aspect["bundle_fingerprint"] = evidence_digest(aspect["bundle"])
    fingerprints = {aspect["label"]: aspect["bundle_fingerprint"] for aspect in analysis["aspects"]}
    frame = analysis["campaign_bundle"]
    frame["aspects"] = [{**entry, "fingerprint": fingerprints[entry["aspect"]]} for entry in frame["aspects"]]
    bundles = [AnalysisContextBundle.from_dict(aspect["bundle"]) for aspect in analysis["aspects"]]
    analysis["generation"]["bundle_fingerprint"] = CampaignBundle.from_frame(frame, bundles).fingerprint()
    analysis["generation"]["bundle_schema_version"] = TO_SHAPE
    return {**data, "analysis": analysis}


def _leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, f"{path}/{key}")
    elif isinstance(value, list) and value and isinstance(value[0], (dict, list)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}[{index}]")
    else:
        yield path, value


def moved_paths(before: Document, after: Document) -> list[str]:
    """Every leaf path whose value differs between two documents, or that only one holds."""
    old, new = dict(_leaves(before)), dict(_leaves(after))
    return sorted(path for path in old.keys() | new.keys() if old.get(path, KeyError) != new.get(path, KeyError))


#: The leaves shape 66 moves: an aspect bundle's shape, a group's sentence and the scan's completeness (both dropped),
#: and, in an analysis record, each fingerprint over its bundles and the shape its generation recorded.
_BUNDLE = r"(?:/aspects\[\d+\]|/analysis/aspects\[\d+\]/bundle)"
_MOVED = re.compile(
    rf"{_BUNDLE}/(?:schema_version|aliased_factors\[\d+\]/sentence|factor_pairs/completeness)"
    r"|/analysis/aspects\[\d+\]/bundle_fingerprint"
    r"|/analysis/campaign_bundle/aspects\[\d+\]/fingerprint"
    r"|/analysis/generation/(?:bundle_fingerprint|bundle_schema_version)"
)


def _allowed(path: str) -> bool:
    """Whether shape 66 may move this path (:data:`_MOVED`)."""
    return _MOVED.fullmatch(path) is not None


def _read(path: Path) -> tuple[Document, bool, bool]:
    """A file's document, whether it was written compact, and whether it escaped every character past ASCII."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            text = stream.read()
    else:
        text = path.read_text(encoding="utf-8")
    return dict(json.loads(text)), "\n" not in text.strip(), text.isascii()


def _write(path: Path, data: Document, *, compact: bool, ascii_only: bool) -> None:
    if compact:
        text = json.dumps(data, separators=(",", ":"), ensure_ascii=ascii_only)
    else:
        text = json.dumps(data, indent=2, ensure_ascii=ascii_only)
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0))
    else:
        path.write_text(text, encoding="utf-8")


def _reads(after: Document) -> None:
    """Read the result back as this build reads it, which refuses a bundle of any shape but its own."""
    if "analysis" not in after:
        CampaignBundle.from_dict(after)
        return
    analysis = after["analysis"]
    bundles = [AnalysisContextBundle.from_dict(aspect["bundle"]) for aspect in analysis["aspects"]]
    CampaignBundle.from_frame(analysis["campaign_bundle"], bundles)


def main(paths: list[str]) -> None:
    """Migrate each file in place, after checking that only what shape 66 changes moved and the result reads."""
    for name in paths:
        path = Path(name)
        before, compact, ascii_only = _read(path)
        after = migrate(before)
        stray = [moved for moved in moved_paths(before, after) if not _allowed(moved)]
        if stray:
            raise SystemExit(f"{path}: shape 66 moves no other field, but these moved: {stray}")
        _reads(after)
        _write(path, after, compact=compact, ascii_only=ascii_only)
        moved = [moved for moved in moved_paths(before, after) if not moved.endswith("/schema_version")]
        print(f"{path}: aspect bundle(s) at shape {TO_SHAPE}; moved: {moved}")


if __name__ == "__main__":
    main(sys.argv[1:])
