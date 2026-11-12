"""Move a frozen campaign bundle, a stored analysis record, a writer case or a captured store from aspect bundle shape 65
to 66, and check it.

Shape 66 takes names out of the evidence (owner ruling, 2026-11-11): a bundle freezes ids and facts only, and a page
words and names them when it is rendered. Over unchanged evidence it moves only what a name or a sentence held, each a
function of facts the bundle keeps:

- each lockstep group's ``aliased_factors[].sentence`` is dropped (its ``factors``, ``n_runs`` and ``n_levels`` stay);
- the factor-pair scan's ``factor_pairs.completeness`` is dropped (its counts and pivots stay);
- each trial's failed check holds its measure's key where it held the catalog's reader name
  (``trials[].failed_checks[].name``), and each judged dimension its id where a reserved axis held its word
  (``trials[].judge[].dim``, the keys of ``trials[].judged``: ``outcome`` is ``__outcome__``);
- a declared crossing's cell holds each level's content hash where it held the level's declared display
  (``declared_crossing.cells[].levels``), read off the aspect's own declared design;
- each aspect bundle's ``schema_version`` becomes 66;
- a campaign's readings hold each bar's decision where they held its reason (``readings.bars[].reason`` becomes
  ``.decision``, read off the reason's last clause, which says it) and no measure's name (``readings.measures[].name``).

A measure's ``reader_name`` and ``reader_prose``, every level's ``display`` and each trial's ``case_label`` stay where
they are: presentation, left out of every fingerprint, so a page built from the moved file prints the names it printed.

A file already at shape 66 is re-fingerprinted: a trial's ``case_label`` left the fingerprint within shape 66, so a file
moved before that records fingerprints this build no longer computes over the same bundles. Only the fingerprints it
records move (each aspect's, each frame entry's and the campaign's), and a file whose fingerprints are current refuses.

A campaign's derived arm is labelled by its levers' ids read as words from 66 on, where a build between the lever
names and 66 spelled it with their reader names. Such a label cannot be told from a person's by what the frame holds,
so a bundle that carries lever names (``lever_names`` on any aspect) is refused: re-assemble it from its runs instead.

A stored analysis record (``{"analysis": ..., ...}``, as ``tests/fixtures/retired_writers`` holds them) moves each
aspect's frozen bundle and its frame's readings the same way, and every fingerprint it records over them with it: each
aspect's ``bundle_fingerprint``, each frame entry's, the generation's campaign fingerprint and its
``bundle_schema_version``. A writer case (``{"format": "writer-case/1", "snapshot": ...}``) moves its snapshot's
campaign bundle and the fingerprint it records over it; its report input, which a page was built from, is not touched. A
captured store (a JSON list of stored documents, as ``tests/fixtures/hot_path_oracle/inputs`` holds them) moves each
stored analysis and report snapshot that holds a bundle of shape 65, and nothing else.

Nothing else is written. Before it writes, the script compares every leaf before and after and refuses when any path
but those moved, and it reads the result back (:meth:`CampaignBundle.from_dict`, or :meth:`CampaignBundle.from_frame`
for an analysis), which refuses a bundle of any shape but this build's. Run from the repository root, once per
fixture::

    uv run python packages/evals/tests/fixtures/report_inputs/migrate_bundle_shape_66.py <file.json | file.json.gz>...

A file is written in the serialization it was read in (compact or indented by one or two, every character past ASCII
escaped or not, as it was read; a gzipped one at gzip level 9 with no timestamp), so running it twice on one file
refuses the second time and writes nothing.
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

#: The word a trial at shape 65 gave each reserved judge axis, and its id.
_AXIS_IDS = {"transcript": "__transcript__", "outcome": "__outcome__"}

#: The last clause of a bar's reason at shape 65, which says the bundle's decision it was read from.
_BAR_WHYS = {
    "its whole interval is on the good side of the bar": "cleared",
    "its whole interval is on the bad side of the bar": "missed",
    "its interval straddles the bar": "undecided",
    "too few observations for an interval": "no_interval",
    "no observation of it": "no_data",
}


def _check_keys(bundle: Document) -> dict[str, str]:
    """Each name a shape-65 trial gave a failed check, to the check's measure key: the catalog's reader name, else the
    key itself."""
    keys: dict[str, str] = {}
    for key, descriptor in (bundle.get("measure_catalog") or {}).items():
        name = descriptor.get("reader_name") or key
        if keys.get(name, key) != key:
            raise SystemExit(f"aspect {bundle.get('aspect')!r}: measures {keys[name]!r} and {key!r} share a name")
        keys[name] = key
    return keys


def _trial(trial: Document, keys: dict[str, str], aspect: str) -> Document:
    """One trial row at shape 66: its checks by measure key, its judged dimensions by id."""
    moved = dict(trial)
    checks = []
    for check in trial.get("failed_checks") or []:
        if check["name"] not in keys:
            raise SystemExit(f"aspect {aspect!r}: a trial's failed check {check['name']!r} names no catalogued measure")
        checks.append({**check, "name": keys[check["name"]]})
    if "failed_checks" in trial:
        moved["failed_checks"] = checks
    if "judge" in trial:
        moved["judge"] = [{**entry, "dim": _AXIS_IDS.get(entry["dim"], entry["dim"])} for entry in trial["judge"]]
    if "judged" in trial:
        judged = trial["judged"]
        if any(word in judged and axis in judged for word, axis in _AXIS_IDS.items()):
            raise SystemExit(f"aspect {aspect!r}: a trial holds a reserved axis by its word and its id")
        moved["judged"] = {_AXIS_IDS.get(dim, dim): score for dim, score in judged.items()}
    return moved


def _crossing(crossing: Document, bundle: Document) -> Document:
    """A declared crossing at shape 66: each cell's levels by content hash, read off the aspect's declared design."""
    hashes: dict[str, dict[str, str]] = {}
    for axis in ((bundle.get("declared") or {}).get("design") or {}).get("axes") or []:
        held = hashes.setdefault(axis["axis_id"], {})
        for value in axis["values"]:
            if held.get(value["display"], value["content_hash"]) != value["content_hash"]:
                raise SystemExit(f"axis {axis['axis_id']!r} declares two levels as {value['display']!r}")
            held.setdefault(value["display"], value["content_hash"])
    cells = []
    for cell in crossing["cells"]:
        try:
            levels = {axis: hashes[axis][display] for axis, display in cell["levels"].items()}
        except KeyError as missing:
            raise SystemExit(f"a crossing cell names a level its design does not declare: {missing}") from missing
        cells.append({**cell, "levels": levels})
    return {**crossing, "cells": cells}


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
    keys = _check_keys(bundle)
    moved["trials"] = [_trial(trial, keys, str(bundle.get("aspect"))) for trial in moved.get("trials") or []]
    if isinstance(crossing := moved.get("declared_crossing"), dict):
        moved["declared_crossing"] = _crossing(crossing, bundle)
    return moved


def _bar(status: Document) -> Document:
    """A bar status at shape 66: its decision, read off its reason's last clause, in place of the reason."""
    moved = {key: value for key, value in status.items() if key != "reason"}
    why = str(status.get("reason", "")).rsplit(": ", 1)[-1]
    if why not in _BAR_WHYS:
        raise SystemExit(f"a bar's reason {status.get('reason')!r} ends in no decision's words")
    moved["decision"] = _BAR_WHYS[why]
    return moved


def _migrate_readings(readings: Document) -> Document:
    """A campaign's readings at shape 66: each bar's decision for its reason, no measure's name."""
    return {
        **readings,
        "bars": [_bar(status) for status in readings.get("bars") or []],
        "measures": [
            {key: value for key, value in measure.items() if key != "name"}
            for measure in readings.get("measures") or []
        ],
    }


def _at_66(aspects: list[Document]) -> bool:
    """Whether every aspect bundle is at shape 66 already, so only the fingerprints over them are moved."""
    return bool(aspects) and all(aspect.get("schema_version") == TO_SHAPE for aspect in aspects)


def _migrate_campaign(data: Document) -> Document:
    """A whole campaign bundle (``CampaignBundle.to_dict()``) at shape 66; one there already as it is."""
    if _at_66(data["aspects"]):
        return data
    return {
        **data,
        "readings": _migrate_readings(data["readings"]),
        "aspects": [_migrate_aspect(aspect) for aspect in data["aspects"]],
    }


def _migrate_analysis(stored: Document) -> Document:
    """A stored analysis with its frozen bundles, its frame's readings and the fingerprints over them at 66; one at 66
    already with only its fingerprints computed again."""
    analysis = deepcopy(stored)
    moving = not _at_66([aspect["bundle"] for aspect in analysis["aspects"]])
    for aspect in analysis["aspects"]:
        if moving:
            aspect["bundle"] = _migrate_aspect(aspect["bundle"])
        aspect["bundle_fingerprint"] = evidence_digest(aspect["bundle"])
    fingerprints = {aspect["label"]: aspect["bundle_fingerprint"] for aspect in analysis["aspects"]}
    frame = analysis["campaign_bundle"]
    if moving:
        frame["readings"] = _migrate_readings(frame["readings"])
    frame["aspects"] = [{**entry, "fingerprint": fingerprints[entry["aspect"]]} for entry in frame["aspects"]]
    bundles = [AnalysisContextBundle.from_dict(aspect["bundle"]) for aspect in analysis["aspects"]]
    analysis["generation"]["bundle_fingerprint"] = CampaignBundle.from_frame(frame, bundles).fingerprint()
    analysis["generation"]["bundle_schema_version"] = TO_SHAPE
    return dict(analysis)


def _migrate_snapshot(snapshot: Document) -> Document:
    """A report snapshot with its campaign bundle and the fingerprint it records over it at 66; its report input as it
    was."""
    bundle = _migrate_campaign(snapshot["bundle"])
    return {**snapshot, "bundle": bundle, "bundle_fingerprint": CampaignBundle.from_dict(bundle).fingerprint()}


def _frozen(document: Document) -> bool:
    """Whether a stored document holds an aspect bundle of shape 65 or 66."""
    shapes = (FROM_SHAPE, TO_SHAPE)
    if document.get("doc_type") == "eval_analysis":
        return any(aspect["bundle"].get("schema_version") in shapes for aspect in document.get("aspects") or [])
    if document.get("doc_type") == "eval_report_snapshot" and isinstance(document.get("bundle"), dict):
        return any(aspect.get("schema_version") in shapes for aspect in document["bundle"].get("aspects") or [])
    return False


def _migrate_stored(document: Document) -> Document:
    """One stored document of a captured store at 66, or as it was where it holds no bundle of shape 65 or 66."""
    if not _frozen(document):
        return document
    if document["doc_type"] == "eval_analysis":
        return _migrate_analysis(document)
    return _migrate_snapshot(document)


def migrate(data: Document | list[Document]) -> Document | list[Document]:
    """A campaign bundle, a stored analysis record, a writer case or a captured store with every shape-65 aspect bundle
    at shape 66 and the fingerprints over them moved with it, or every fingerprint over a shape-66 one computed again;
    everything else as it was."""
    if isinstance(data, list):
        if not any(_frozen(document) for document in data):
            raise SystemExit(f"the captured store holds no bundle of shape {FROM_SHAPE} or {TO_SHAPE}")
        return [_migrate_stored(document) for document in data]
    if "analysis" in data:
        return {**data, "analysis": _migrate_analysis(data["analysis"])}
    if "snapshot" in data:
        if not isinstance(data["snapshot"].get("bundle"), dict):
            raise SystemExit("the writer case freezes a bare report input and holds no bundle to move")
        return {**data, "snapshot": _migrate_snapshot(data["snapshot"])}
    return _migrate_campaign(data)


def _leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, f"{path}/{key}")
    elif isinstance(value, list) and value and isinstance(value[0], (dict, list)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}[{index}]")
    else:
        yield path, value


def moved_paths(before: Any, after: Any) -> list[str]:
    """Every leaf path whose value differs between two documents, or that only one holds."""
    old, new = dict(_leaves(before)), dict(_leaves(after))
    return sorted(path for path in old.keys() | new.keys() if old.get(path, KeyError) != new.get(path, KeyError))


#: The leaves shape 66 moves: an aspect bundle's shape, a group's sentence and the scan's completeness (both dropped),
#: a trial's checks and judged dimensions, a crossing cell's levels, a bar's reason and decision and a measure reading's
#: name, and each fingerprint over the bundles and the shape a generation recorded. Each under any of the places a file
#: holds a bundle: whole (``/aspects[i]``), in an analysis record or a captured store's analysis (``.../aspects[i]/bundle``)
#: or in a writer case or a captured store's report snapshot (``.../bundle/aspects[i]``).
_AT = r"(?:\[\d+\])?(?:/analysis|/snapshot)?"
_BUNDLE = rf"{_AT}(?:/aspects\[\d+\]/bundle|/bundle/aspects\[\d+\]|/aspects\[\d+\])"
_READINGS = rf"{_AT}(?:/campaign_bundle|/bundle)?/readings"
_MOVED = re.compile(
    rf"{_BUNDLE}/(?:schema_version|aliased_factors\[\d+\]/sentence|factor_pairs/completeness"
    r"|trials\[\d+\]/failed_checks\[\d+\]/name|trials\[\d+\]/judge\[\d+\]/dim"
    r"|trials\[\d+\]/judged/(?:__)?(?:outcome|transcript)(?:__)?"
    r"|declared_crossing/cells\[\d+\]/levels/[^/]+)"
    rf"|{_READINGS}/(?:bars\[\d+\]/(?:reason|decision)|measures\[\d+\]/name)"
    rf"|{_AT}/aspects\[\d+\]/bundle_fingerprint"
    rf"|{_AT}/campaign_bundle/aspects\[\d+\]/fingerprint"
    rf"|{_AT}/generation/(?:bundle_fingerprint|bundle_schema_version)"
    rf"|{_AT}/bundle_fingerprint"
)


def _allowed(path: str) -> bool:
    """Whether shape 66 may move this path (:data:`_MOVED`)."""
    return _MOVED.fullmatch(path) is not None


def _read(path: Path) -> tuple[Any, int | None, bool]:
    """A file's document, its indent (None for compact), and whether it escaped every character past ASCII."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            text = stream.read()
    else:
        text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    indent = None if len(lines) < 2 else len(lines[1]) - len(lines[1].lstrip(" "))
    return json.loads(text), indent, text.isascii()


def _write(path: Path, data: Any, *, indent: int | None, ascii_only: bool, newline: bool) -> None:
    if indent is None:
        text = json.dumps(data, separators=(",", ":"), ensure_ascii=ascii_only)
    else:
        text = json.dumps(data, indent=indent, ensure_ascii=ascii_only)
    text += "\n" if newline else ""
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0))
    else:
        path.write_text(text, encoding="utf-8")


def _reads(after: Any) -> None:
    """Read the result back as this build reads it, which refuses a bundle of any shape but its own."""
    if isinstance(after, list):
        for document in after:
            if document.get("doc_type") == "eval_analysis":
                _reads({"analysis": document})
            elif document.get("doc_type") == "eval_report_snapshot" and isinstance(document.get("bundle"), dict):
                CampaignBundle.from_dict(document["bundle"])
        return
    if "snapshot" in after:
        CampaignBundle.from_dict(after["snapshot"]["bundle"])
        return
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
        before, indent, ascii_only = _read(path)
        after = migrate(before)
        if not moved_paths(before, after):
            raise SystemExit(f"{path}: every bundle is at shape {TO_SHAPE} and every fingerprint it records is current")
        stray = [moved for moved in moved_paths(before, after) if not _allowed(moved)]
        if stray:
            raise SystemExit(f"{path}: shape 66 moves no other field, but these moved: {stray[:20]}")
        _reads(after)
        raw = gzip.open(path, "rt", encoding="utf-8").read() if path.suffix == ".gz" else path.read_text("utf-8")
        _write(path, after, indent=indent, ascii_only=ascii_only, newline=raw.endswith("\n"))
        moved = sorted({re.sub(r"\[\d+\]", "[]", moved) for moved in moved_paths(before, after)})
        print(f"{path}: aspect bundle(s) at shape {TO_SHAPE}; moved: {moved}")


if __name__ == "__main__":
    main(sys.argv[1:])
