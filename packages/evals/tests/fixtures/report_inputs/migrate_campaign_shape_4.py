"""Move a frozen campaign bundle, a stored analysis record, a writer case or a captured store from campaign bundle shape 2
or 3 to 4, and check it.

Campaign shape 4 takes names out of the readings (owner ruling, 2026-11-11), as aspect bundle shape 66 took them out of
the evidence: a page words a floor's reason and spells a derived arm when it is rendered. Over unchanged evidence it
moves only what a sentence held, each read again off facts the bundle keeps:

- each floor status holds its decision as facts where it held its reason (``readings.floors[].reason`` becomes
  ``.decision``): the floor is read again on its aspect's frozen bundle
  (:func:`~threetears.evals.analysis.report.decision.evaluate_floor`), every other field of its status must come back
  as stored, and the decision, worded as a page words it, must say the stored reason word for word;
- each derived arm whose label code spelled from several levers carries its spelling (``arm_spellings``), read off its
  lever map as :func:`~threetears.evals.analysis.campaigns.campaign_arms_spelled` reads it, and refused unless it spells
  the stored label; an arm spelled from one level, or one a person named, carries none (``arm_spellings`` is ``{}``);
- a shape-2 bundle gains its readings' ``aggregate``, read off its own readings and aspects' bundles
  (:func:`~threetears.evals.analysis.campaign_readings.campaign_aggregate`), as the build that read shape 2 as 3 did;
- the campaign bundle's ``schema_version`` becomes 4.

No aspect bundle moves, so no aspect fingerprint moves; the campaign fingerprint does, and every place a file records it
with it: a report snapshot's ``bundle_fingerprint`` and a stored analysis's ``generation.bundle_fingerprint``. A writer
case's report input, which a page was built from, is not touched, and neither is its snapshot id, which hashes only
that.

A stored analysis record (``{"analysis": ..., ...}``, as ``tests/fixtures/retired_writers`` holds them) moves its frame.
A writer case (``{"format": "writer-case/1", "snapshot": ...}``) moves its snapshot's campaign bundle. A captured store
(a JSON list of stored documents, as ``tests/fixtures/hot_path_oracle/inputs`` holds them) moves each stored analysis
and report snapshot that holds a campaign bundle of shape 2 or 3, and nothing else.

Nothing else is written. Before it writes, the script compares every leaf before and after and refuses when any path
but those moved, and it reads the result back (:meth:`CampaignBundle.from_dict`, or :meth:`CampaignBundle.from_frame`
for an analysis), which refuses a campaign bundle of any shape but this build's. Run from the repository root, once per
fixture::

    uv run python -m packages.evals.tests.fixtures.report_inputs.migrate_campaign_shape_4 <file.json | file.json.gz>...

A file is written in the serialization it was read in (as ``migrate_bundle_shape_66.py`` writes it), so running it
twice on one file refuses the second time and writes nothing.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from packages.evals.tests.fixtures.report_inputs.migrate_bundle_shape_66 import moved_paths
from threetears.evals.analysis import AnalysisContextBundle, CampaignBundle
from threetears.evals.analysis.arms import (
    ArmLevel,
    arm_names,
    distinguishing_axes,
    elide_level,
    naming_levels,
    short_digest,
)
from threetears.evals.analysis.campaign_bundle import CAMPAIGN_BUNDLE_SHAPE, ArmSpelling
from threetears.evals.analysis.campaign_readings import CampaignReadings, campaign_aggregate
from threetears.evals.analysis.report.decision import evaluate_floor, floor_readings, floor_subject
from threetears.evals.kernel.campaign import CampaignDeclaration, VariantIndexEntry
from threetears.evals.kernel.host.sweepables import CANDIDATE_KIND_LEVER
from threetears.evals.kernel.identity import compute_variant_key

__all__ = ["FROM_SHAPES", "TO_SHAPE", "main", "migrate"]

#: The campaign bundle shapes this reads, and the one it writes.
FROM_SHAPES, TO_SHAPE = frozenset({2, 3}), CAMPAIGN_BUNDLE_SHAPE

Document = dict[str, Any]


def _floor(
    status: Document,
    declaration: CampaignDeclaration,
    bundles: Mapping[str, AnalysisContextBundle],
    control_label: str,
) -> Document:
    """One floor status at shape 4: its decision, read again on its aspect's frozen bundle, in place of its reason.

    :raises SystemExit: The floor reads back other than stored, or its decision, worded, is not the stored reason.
    """
    (floor,) = [one for one in declaration.floors if one.key == status["floor"] and one.aspect == status["aspect"]]
    bundle = bundles[status["aspect"]]
    control = bundle.design.control_arm
    if control is None:
        raise SystemExit(f"floor {status['floor']!r}: its aspect's bundle names no control")
    read = evaluate_floor(
        floor,
        floor_readings(bundle, control.variant_key, status["arm"]),
        arm=status["arm"],
        arm_label=status["arm_label"],
        control_label=control_label,
    )
    fresh = read.model_dump(mode="json")
    held = {key: value for key, value in status.items() if key != "reason"}
    if (again := {key: value for key, value in fresh.items() if key != "decision"}) != held:
        raise SystemExit(f"floor {status['floor']!r} on {status['arm_label']!r} reads back {again}, stored {held}")
    worded = read.reason(floor_subject(read.decision, bundle), control=control_label)
    if worded != status["reason"]:
        raise SystemExit(
            f"floor {status['floor']!r} on {status['arm_label']!r}: its decision words {worded!r}, and it stored "
            f"{status['reason']!r}"
        )
    moved: Document = {}
    for key, value in status.items():
        if key == "reason":
            moved["decision"] = fresh["decision"]
        else:
            moved[key] = value
    return moved


def _taken(label: str, spelled: str) -> int | None:
    """The ``(n)`` a label carries past its spelling, where its spelling was taken; None for none."""
    if label == spelled:
        return None
    if (match := re.fullmatch(re.escape(spelled) + r" \((\d+)\)", label)) is None:
        raise SystemExit(f"arm {label!r} is spelled {spelled!r}, which its label does not extend by a number")
    return int(match.group(1))


def _spellings(bundle: CampaignBundle) -> dict[str, Document]:
    """Each derived arm's spelling where code spelled its label from several levers, as ``campaign_arms_spelled``
    reads it: each arm's index entry from its lever map where the map digests to its key, the levels the arms differ
    on, the candidate's kind dropped where one other lever names it.

    :raises SystemExit: A spelling does not spell its arm's stored label.
    """
    given = {key for arm in bundle.declaration.arms for key in bundle.arm_keys(arm)}
    derived = [arm for arm in bundle.arms if arm.variant_key not in given]
    if not derived:
        return {}
    entries = [
        VariantIndexEntry(variant_key=arm.variant_key, levers=dict(arm.levers))
        if arm.levers is not None and compute_variant_key(arm.levers) == arm.variant_key
        else VariantIndexEntry(variant_key=arm.variant_key, levers={}, levels_unavailable="no lever map")
        for arm in [*bundle.declaration.arms, *derived]
    ]
    names = arm_names(entries)
    distinguishing = distinguishing_axes(entries)
    by_key = {entry.variant_key: entry for entry in entries}
    spelled: dict[str, Document] = {}
    for arm in derived:
        entry = by_key[arm.variant_key]
        levels: Sequence[ArmLevel] = naming_levels(entry, distinguishing) if entry.levels_unavailable is None else []
        if len(levels) > 1:
            others = [level for level in levels if level.axis_id != CANDIDATE_KIND_LEVER]
            levels = others if len(others) == 1 else levels
        if len(levels) < 2:
            continue
        name = names[arm.variant_key]
        digest = next(
            (shown for shown in (short_digest(arm.variant_key), arm.variant_key) if name.endswith(f" (arm {shown})")),
            None,
        )
        spelling = ArmSpelling(
            levers=[level.axis_id for level in levels],
            levels=[elide_level(level.display) for level in levels],
            digest=digest,
        )
        spelling = spelling.model_copy(update={"taken": _taken(arm.label, spelling.label())})
        if spelling.label() != arm.label:
            raise SystemExit(f"arm {arm.label!r} is spelled {spelling.label()!r}")
        spelled[arm.variant_key] = spelling.model_dump(mode="json")
    return spelled


def _migrate_frame(frame: Document, bundles: Sequence[AnalysisContextBundle]) -> Document:
    """A campaign bundle (whole, or as a stored analysis's frame) at shape 4, read over its aspects' bundles."""
    shape = frame.get("schema_version")
    if shape not in FROM_SHAPES:
        raise SystemExit(f"a campaign bundle is at shape {shape}, not one of {sorted(FROM_SHAPES)}")
    by_label = {bundle.aspect: bundle for bundle in bundles}
    declaration = CampaignDeclaration.model_validate(frame["declaration"])
    readings = deepcopy(frame["readings"])
    readings["floors"] = [
        _floor(status, declaration, by_label, readings["control_label"]) for status in readings.get("floors") or []
    ]
    if shape == 2:
        if "aggregate" in readings:
            raise SystemExit("a shape-2 campaign bundle already holds readings.aggregate")
        aggregate = campaign_aggregate(CampaignReadings.model_validate(readings), by_label)
        readings["aggregate"] = None if aggregate is None else aggregate.model_dump(mode="json")
    moved: Document = {}
    for key, value in frame.items():
        if key == "schema_version":
            moved[key] = TO_SHAPE
        elif key == "readings":
            moved[key] = readings
        elif key == "aspects":
            moved["arm_spellings"] = {}
            moved[key] = value
        else:
            moved[key] = value
    if "arm_spellings" in frame:
        raise SystemExit("a campaign bundle of shape 2 or 3 already holds arm_spellings")
    whole = (
        CampaignBundle.from_frame(moved, bundles)
        if moved["aspects"] and isinstance(moved["aspects"][0], dict) and "fingerprint" in moved["aspects"][0]
        else CampaignBundle.from_dict(moved)
    )
    moved["arm_spellings"] = _spellings(whole)
    return moved


def _bundles_of(aspects: Sequence[Document]) -> list[AnalysisContextBundle]:
    return [AnalysisContextBundle.from_dict(dict(aspect)) for aspect in aspects]


def _migrate_campaign(data: Document) -> Document:
    """A whole campaign bundle (``CampaignBundle.to_dict()``) at shape 4."""
    return _migrate_frame(data, _bundles_of(data["aspects"]))


def _migrate_analysis(stored: Document) -> Document:
    """A stored analysis with its frame at shape 4 and the campaign fingerprint it records over it."""
    analysis = deepcopy(stored)
    bundles = _bundles_of([aspect["bundle"] for aspect in analysis["aspects"]])
    analysis["campaign_bundle"] = _migrate_frame(analysis["campaign_bundle"], bundles)
    analysis["generation"]["bundle_fingerprint"] = CampaignBundle.from_frame(
        analysis["campaign_bundle"], bundles
    ).fingerprint()
    return dict(analysis)


def _migrate_snapshot(snapshot: Document) -> Document:
    """A report snapshot with its campaign bundle and the fingerprint it records over it at shape 4; its report input
    and id as they were."""
    bundle = _migrate_campaign(snapshot["bundle"])
    return {**snapshot, "bundle": bundle, "bundle_fingerprint": CampaignBundle.from_dict(bundle).fingerprint()}


def _frozen(document: Document) -> bool:
    """Whether a stored document holds a campaign bundle of shape 2 or 3."""
    if document.get("doc_type") == "eval_analysis":
        return (document.get("campaign_bundle") or {}).get("schema_version") in FROM_SHAPES
    if document.get("doc_type") == "eval_report_snapshot" and isinstance(document.get("bundle"), dict):
        return document["bundle"].get("schema_version") in FROM_SHAPES
    return False


def _migrate_stored(document: Document) -> Document:
    """One stored document of a captured store at shape 4, or as it was where it holds no earlier campaign bundle."""
    if not _frozen(document):
        return document
    if document["doc_type"] == "eval_analysis":
        return _migrate_analysis(document)
    return _migrate_snapshot(document)


def migrate(data: Document | list[Document]) -> Document | list[Document]:
    """A campaign bundle, a stored analysis record, a writer case or a captured store with every campaign bundle of
    shape 2 or 3 at shape 4 and the fingerprints recorded over it moved with it; everything else as it was."""
    if isinstance(data, list):
        if not any(_frozen(document) for document in data):
            raise SystemExit(f"the captured store holds no campaign bundle of shape {sorted(FROM_SHAPES)}")
        return [_migrate_stored(document) for document in data]
    if "analysis" in data:
        return {**data, "analysis": _migrate_analysis(data["analysis"])}
    if "snapshot" in data:
        if not isinstance(data["snapshot"].get("bundle"), dict):
            raise SystemExit("the writer case freezes a bare report input and holds no bundle to move")
        return {**data, "snapshot": _migrate_snapshot(data["snapshot"])}
    return _migrate_campaign(data)


#: The leaves shape 4 moves: the campaign bundle's shape, a floor status's reason and decision, a derived arm's
#: spelling, a shape-2 bundle's aggregate, and each campaign fingerprint a file records. Each under any of the places a
#: file holds a campaign bundle: whole (the root), a writer case's or a captured store's report snapshot
#: (``.../bundle``), or an analysis record's or a captured store's analysis frame (``.../campaign_bundle``).
_AT = r"(?:\[\d+\])?(?:/analysis|/snapshot)?"
_CAMPAIGN = rf"{_AT}(?:/campaign_bundle|/bundle)?"


def _allowed(path: str, *, aggregate: bool) -> bool:
    """Whether shape 4 may move this path; the aggregate only where a shape-2 bundle gained it."""
    moved = (
        rf"{_CAMPAIGN}/(?:schema_version|readings/floors\[\d+\]/(?:reason|decision/[a-z_]+)|arm_spellings/.+"
        + (r"|readings/aggregate(?:/.*|\[\d+\].*)?" if aggregate else "")
        + ")"
        + rf"|{_AT}/bundle_fingerprint|{_AT}/generation/bundle_fingerprint"
    )
    return re.fullmatch(moved, path) is not None


def _shape_2(data: Any) -> bool:
    """Whether a file holds a campaign bundle of shape 2, which gains its aggregate."""
    if isinstance(data, list):
        return any(_shape_2(document) for document in data)
    if not isinstance(data, dict):
        return False
    for held in (data, data.get("snapshot"), data.get("analysis"), data.get("bundle"), data.get("campaign_bundle")):
        if isinstance(held, dict) and "readings" in held and held.get("schema_version") == 2:
            return True
    return any(_shape_2(data.get(key)) for key in ("snapshot", "analysis") if isinstance(data.get(key), dict))


def _read(path: Path) -> tuple[Any, int | None, bool, bool]:
    """A file's document, its indent (None for compact), whether it escaped every character past ASCII, and whether
    it ends in a newline."""
    raw = path.read_bytes()
    text = (gzip.decompress(raw) if path.suffix == ".gz" else raw).decode("utf-8")
    lines = text.splitlines()
    indent = None if len(lines) < 2 else len(lines[1]) - len(lines[1].lstrip(" "))
    return json.loads(text), indent, text.isascii(), text.endswith("\n")


def _write(path: Path, data: Any, *, indent: int | None, ascii_only: bool, newline: bool) -> None:
    """Write a document in the serialization it was read in; a gzipped one at level 9 with no timestamp."""
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
    """Read the result back as this build reads it, which refuses a campaign bundle of any shape but its own."""
    if isinstance(after, list):
        for document in after:
            if document.get("doc_type") == "eval_analysis":
                _reads({"analysis": document})
            elif document.get("doc_type") == "eval_report_snapshot" and isinstance(document.get("bundle"), dict):
                _reads({"snapshot": document})
        return
    if "snapshot" in after:
        bundle = CampaignBundle.from_dict(after["snapshot"]["bundle"])
        if bundle.fingerprint() != after["snapshot"]["bundle_fingerprint"]:
            raise SystemExit("a snapshot's recorded fingerprint is not its bundle's")
        return
    if "analysis" not in after:
        CampaignBundle.from_dict(after)
        return
    analysis = after["analysis"]
    bundles = _bundles_of([aspect["bundle"] for aspect in analysis["aspects"]])
    bundle = CampaignBundle.from_frame(analysis["campaign_bundle"], bundles)
    if bundle.fingerprint() != analysis["generation"]["bundle_fingerprint"]:
        raise SystemExit("an analysis's recorded fingerprint is not its frame's")
    if bundle.frame() != analysis["campaign_bundle"]:
        raise SystemExit("an analysis's frame is not the one this build writes for its bundle")


def main(paths: list[str]) -> None:
    """Migrate each file in place, after checking that only what shape 4 changes moved and the result reads."""
    for name in paths:
        path = Path(name)
        before, indent, ascii_only, newline = _read(path)
        after = migrate(before)
        paths_moved = moved_paths(before, after)
        stray = [moved for moved in paths_moved if not _allowed(moved, aggregate=_shape_2(before))]
        if stray:
            raise SystemExit(f"{path}: shape 4 moves no other field, but these moved: {stray[:20]}")
        _reads(after)
        _write(path, after, indent=indent, ascii_only=ascii_only, newline=newline)
        shown = sorted({re.sub(r"\[\d+\]", "[]", moved) for moved in paths_moved})
        print(f"{path}: campaign bundle(s) at shape {TO_SHAPE}; moved: {shown}")


if __name__ == "__main__":
    main(sys.argv[1:])
