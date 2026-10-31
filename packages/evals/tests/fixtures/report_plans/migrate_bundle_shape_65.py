"""Move a frozen campaign bundle's aspect bundles from shape 64 to 65, from what each bundle itself holds, and check it.

Shape 65 changes one thing over unchanged evidence: a rate's family contrast is drawn on its bounded interval, and its
verdict needs that interval to exclude zero. A rate is the overall pass rate (a ``trial`` reading); a check, a goal
check's pass rate or a classifier's accuracy, by the bundle's measure catalog; or a judged dimension scored pass or
fail. Every comparison a shape-64 family corrected already carries that
interval, ``bounded_interval``, at the family's level, so the move is a recomputation over stored values and never a
re-assembly:

- a rate comparison in its family (one carrying an adjusted p) takes ``interval := bounded_interval``, and its verdict
  is read again off its stored adjusted p's, its delta and that interval, by the rule shape 65 assembles by;
- each aspect bundle's ``schema_version`` becomes 65.

Nothing else is written. Before it writes, the script compares every leaf of the bundle before and after and refuses
when any path but those moved, and it reads the result back through :meth:`CampaignBundle.from_dict`, which refuses a
bundle of any shape but this build's. Run from the repository root, once per fixture::

    uv run python packages/evals/tests/fixtures/report_plans/migrate_bundle_shape_65.py <bundle.json | bundle.json.gz>...

A file is written in the serialization it was read in (a gzipped one compact, at gzip level 9 with no timestamp; a
plain one indented by two), so running it on a shape-65 file refuses and writes nothing.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from threetears.evals.analysis import CampaignBundle
from threetears.evals.analysis.stats import interval_permits_separation
from threetears.evals.kernel.metrics import ACCURACY_MEASURE, MetricDescriptor, goal_check_of

__all__ = ["FROM_SHAPE", "TO_SHAPE", "main", "migrate", "moved_paths"]

#: The aspect bundle shape this reads, and the one it writes.
FROM_SHAPE, TO_SHAPE = 64, 65

Bundle = dict[str, Any]


def _is_rate(comparison: Mapping[str, Any], bundle: Mapping[str, Any]) -> bool:
    """Whether a stored comparison reads a rate, as shape 65 assembles it: the overall pass rate; a check, a goal
    check's pass rate or a classifier's accuracy, by the bundle's measure catalog; or a judged dimension scored pass or
    fail, by the bundle's judged measures."""
    kind, name = comparison["reading"], comparison["name"]
    if kind == "trial":
        return True
    if kind == "judged":
        judged = {measure["name"]: measure.get("value_range") for measure in bundle.get("judged_measures") or []}
        return judged.get(name) in ([0.0, 1.0], (0.0, 1.0))
    entry = (bundle.get("measure_catalog") or {}).get(name)
    if entry is None:
        return False
    return (
        MetricDescriptor.model_validate(entry).is_check or goal_check_of(name) is not None or name == ACCURACY_MEASURE
    )


def _verdict(comparison: Mapping[str, Any], interval: list[float] | None, alpha: float) -> str:
    """A corrected comparison's verdict read off its stored adjusted p's and delta, beside a rate's bounded interval."""
    delta = comparison["delta"]
    if comparison["p_adjusted"] < alpha and delta and interval is not None and interval_permits_separation(interval):
        return "improved" if (delta > 0) == comparison["higher_is_better"] else "regressed"
    equivalence = comparison.get("equivalence_p_adjusted")
    if equivalence is not None and equivalence < alpha:
        return "equivalent"
    return "not_separated"


def _migrate_aspect(bundle: Bundle) -> Bundle:
    """One aspect bundle at shape 65."""
    if bundle.get("schema_version") != FROM_SHAPE:
        raise SystemExit(f"an aspect bundle is at shape {bundle.get('schema_version')}, not {FROM_SHAPE}")
    moved = deepcopy(bundle)
    moved["schema_version"] = TO_SHAPE
    for family in moved["multiple_comparisons"]["families"]:
        for comparison in family["comparisons"]:
            if comparison.get("p_adjusted") is None or not _is_rate(comparison, moved):
                continue
            interval = comparison["bounded_interval"]
            comparison["interval"] = interval
            comparison["verdict"] = _verdict(comparison, interval, family["alpha"])
    return moved


def migrate(data: Bundle) -> Bundle:
    """A campaign bundle with every aspect bundle at shape 65; the campaign's own fields as they were."""
    return {**data, "aspects": [_migrate_aspect(aspect) for aspect in data["aspects"]]}


def _leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, f"{path}/{key}")
    elif isinstance(value, list) and value and isinstance(value[0], (dict, list)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}[{index}]")
    else:
        yield path, value


def moved_paths(before: Bundle, after: Bundle) -> list[str]:
    """Every leaf path whose value differs between two bundles, or that only one holds."""
    old, new = dict(_leaves(before)), dict(_leaves(after))
    return sorted(path for path in old.keys() | new.keys() if old.get(path, KeyError) != new.get(path, KeyError))


def _allowed(path: str, after: Bundle) -> bool:
    """Whether shape 65 may move this path: an aspect's shape, or a rate comparison's interval or verdict."""
    parts = path.split("/")
    if len(parts) == 3 and parts[2] == "schema_version":
        return True
    if parts[-1] not in ("interval", "verdict") or len(parts) != 6 or parts[2] != "multiple_comparisons":
        return False
    aspect = after["aspects"][int(parts[1].removeprefix("aspects[").rstrip("]"))]
    family = aspect["multiple_comparisons"]["families"][int(parts[3].removeprefix("families[").rstrip("]"))]
    comparison = family["comparisons"][int(parts[4].removeprefix("comparisons[").rstrip("]"))]
    return _is_rate(comparison, aspect)


def _read(path: Path) -> Bundle:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return dict(json.load(stream))
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path, data: Bundle) -> None:
    if path.suffix == ".gz":
        text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        path.write_bytes(gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0))
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main(paths: list[str]) -> None:
    """Migrate each file in place, after checking that only what shape 65 changes moved and the result reads."""
    for name in paths:
        path = Path(name)
        before = _read(path)
        after = migrate(before)
        stray = [moved for moved in moved_paths(before, after) if not _allowed(moved, after)]
        if stray:
            raise SystemExit(f"{path}: shape 65 moves no other field, but these moved: {stray}")
        CampaignBundle.from_dict(after)
        _write(path, after)
        moved = [moved for moved in moved_paths(before, after) if not moved.endswith("/schema_version")]
        print(f"{path}: {len(after['aspects'])} aspect bundle(s) at shape {TO_SHAPE}; rate fields moved: {moved}")


if __name__ == "__main__":
    main(sys.argv[1:])
