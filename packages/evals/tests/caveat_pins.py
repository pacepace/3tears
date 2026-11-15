"""The report inputs the caveat list is pinned over, and what code's caveat reads said of each before the list existed.

Every report input a fixture holds, stored or built: the frozen writer cases (rung 5's and the held-out ones, each a
real campaign's snapshot), the stored inputs of ``fixtures/report_inputs`` and the hot-path captures, and the inputs
built fresh from the hot-path campaigns' bundles. For each, :func:`caveat_reads` is what the earlier reads returned:
the stated caveats with their reach, the implied ones with theirs, each stated caveat's materiality and code's own
called options. ``fixtures/caveat_list/before.json`` holds them as the tree before the list computed them, so a test
can hold that the list, and the thin reads over it, return exactly what was returned before.

``python -m packages.evals.tests.caveat_pins`` rewrites the file, which is right only on a tree before the list.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from threetears.evals.analysis.report_input import ReportInput

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PACKAGE = Path(__file__).resolve().parents[1]
PINS = FIXTURES / "caveat_list" / "before.json"

#: The frozen writer cases: rung 5's, then the held-out ones.
CASE_DIRS = (PACKAGE / "examples" / "5-writer" / "cases", FIXTURES / "writer_cases" / "held_out")

#: The stored report inputs that are not inside a writer case.
STORED = (
    *(FIXTURES / "report_inputs" / rung / "input.json" for rung in ("r1", "r2", "r3", "r4", "r5")),
    FIXTURES / "hot_path_oracle" / "inputs" / "toyhost_doc.json",
)


def read_fixture(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _input_of(document: Any) -> dict[str, Any] | None:
    """The report input a fixture document holds, wherever it holds it."""
    if isinstance(document, dict):
        if document.get("schema") == "report-input/1":
            return document
        for key in ("snapshot", "report_input"):
            if key in document:
                found = _input_of(document[key])
                if found is not None:
                    return found
    return None


def stored_inputs() -> Iterator[tuple[str, ReportInput]]:
    """Each stored report input a fixture holds, by a name the pins key it by."""
    for directory in CASE_DIRS:
        for path in sorted(directory.glob("*.json.gz")):
            held = _input_of(read_fixture(path))
            if held is not None:
                yield f"case:{path.name.removesuffix('.json.gz')}", ReportInput.model_validate(held)
    for path in STORED:
        held = _input_of(read_fixture(path))
        if held is not None:
            yield f"stored:{path.relative_to(FIXTURES)}", ReportInput.model_validate(held)


def built_inputs() -> Iterator[tuple[str, ReportInput]]:
    """Each report input built fresh from a hot-path campaign's bundle, as a page reads it."""
    from packages.evals.tests.test_hot_path_oracle import WITH_BUNDLES, subject
    from threetears.evals.analysis.report_input_build import report_input
    from threetears.evals.analysis.service import assemble_campaign_bundle

    for name in WITH_BUNDLES:
        read = subject(name)
        yield f"built:{name}", report_input(assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id))


def caveat_reads(report_input: ReportInput) -> dict[str, Any]:
    """What code's caveat reads say of one input: stated and implied caveats with their reach, each stated caveat's
    materiality, and the options code's own read calls."""
    from threetears.evals.analysis.report.call import (
        caveat_materiality,
        decision_at,
        implied_caveats,
        report_caveats,
    )

    return {
        "stated": [[caveat.id, caveat.kind, caveat.reach] for caveat in report_caveats(report_input)],
        "implied": [[caveat.id, caveat.kind, caveat.reach] for caveat in implied_caveats(report_input)],
        "immaterial": sorted(key for key, held in caveat_materiality(report_input).items() if held.immaterial),
        "called": list(decision_at(report_input).dominant),
    }


def main() -> None:
    reads = {name: caveat_reads(held) for name, held in (*stored_inputs(), *built_inputs())}
    PINS.parent.mkdir(exist_ok=True)
    PINS.write_text(json.dumps(reads, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
