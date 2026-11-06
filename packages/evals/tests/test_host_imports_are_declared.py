"""Every name a host imports is declared: ``host-contract`` or ``stable``, never ``provisional``.

A host's production code binds to what it imports, so a package that exports a name a host imports promises it. Each
file under ``tests/fixtures/host_imports/`` is one host's import list, one fully qualified
``threetears.evals.<module>.<name>`` per line, as the host reported it. A line naming a module rather than a name in
one (``threetears.evals``, ``threetears.evals.ops``) is skipped. Every other line must be a public name the snapshot
records, tiered ``host-contract`` or ``stable``; the failure names the host's file and the name.

A name the package removed while a host's recorded list still carries it is listed in :data:`REMOVED_BEFORE_REPIN`,
by host file, with the date and the host's word that it drops each import before it moves its pin. Each listed name must still be in
that host's file and must be gone from the snapshot, so the set empties as the host re-records its list.

A name only a host's tests import, which the host accepted may move, is provisional and listed in
:data:`PROVISIONAL_FOR_HOST_TESTS`, by host file, with the date and the host's word. Each listed name must still be in
that host's file and must be provisional in the snapshot. A name leaves the set when the host's list no longer holds
it or when it is declared again; a name the host's tests keep importing stays listed, by the host's own word.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from packages.evals.tests.script_loader import load_script

surface = load_script("public_surface.py")

HOST_IMPORTS = Path(__file__).resolve().parent / "fixtures" / "host_imports"

#: Names removed from the package that a host's recorded import list still carries, by host file. 2026-11-06:
#: Discodon confirmed it drops every import of the retired generations' shims on its re-pin branch before it moves
#: its pin (removal wave R4), so these leave the package before its list is recorded again.
REMOVED_BEFORE_REPIN: dict[str, frozenset[str]] = {
    "discodon.txt": frozenset(
        {
            "threetears.evals.analysis.AsRecordedReporterKind",
            "threetears.evals.analysis.EVAL_ANALYSIS_GEN_DEFAULT",
            "threetears.evals.analysis.REPORTER_KIND",
            "threetears.evals.analysis.ReporterKind",
            "threetears.evals.analysis.ReporterOverlays",
            "threetears.evals.analysis.analysis_gen_request_settings_for",
            "threetears.evals.analysis.generate_analysis",
            "threetears.evals.analysis.refuse_undeliverable_reporter_template",
            "threetears.evals.analysis.reporter_cell_timeout_s",
            "threetears.evals.analysis.reporter_kind_contract",
            "threetears.evals.kernel.host.StyleProfile",
            "threetears.evals.ops.REPORTER_UNHONOURED_LAUNCH_ARGUMENTS",
            "threetears.evals.ops.ReporterAdmitted",
            "threetears.evals.ops.ReporterModelResolver",
            "threetears.evals.ops.ReporterModelRole",
            "threetears.evals.ops.admit_reporter_launch",
            "threetears.evals.ops.plan_reporter_arm",
            "threetears.evals.ops.price_largest_reporter_call",
        }
    ),
}


#: Names a host's recorded import list carries that only the host's tests import, provisional with the host's word, by
#: host file. 2026-11-06: Discodon confirmed that a provisional name its tests import may move (removal wave R5): its
#: tests assert against the package's definition. Its production code imports none of them, and its re-pin branch
#: imports ``assemble_campaign_bundle``, ``resolve_variant_identity`` and ``CampaignBundle`` nowhere.
PROVISIONAL_FOR_HOST_TESTS: dict[str, frozenset[str]] = {
    "discodon.txt": frozenset(
        {
            "threetears.evals.analysis.ABSENT",
            "threetears.evals.analysis.CampaignBundle",
            "threetears.evals.analysis.MIN_CASES",
            "threetears.evals.analysis.assemble_campaign_bundle",
            "threetears.evals.analysis.serialize_report",
            "threetears.evals.kernel.ACCURACY_MEASURE",
            "threetears.evals.kernel.ENGINE_FAMILIES",
            "threetears.evals.kernel.FIND_CAMPAIGN",
            "threetears.evals.kernel.FIND_RUN",
            "threetears.evals.kernel.INCOMPLETE_STOP_REASONS",
            "threetears.evals.kernel.MeasureSummary",
            "threetears.evals.kernel.TURN_BUDGET_ENDED_KEY",
            "threetears.evals.kernel.classifier_label_measure",
            "threetears.evals.kernel.count_substituted",
            "threetears.evals.kernel.count_substituted_deliveries",
            "threetears.evals.kernel.host.CANDIDATE_MODEL_LEVER",
            "threetears.evals.kernel.host.UNSEATED_LEVEL",
            "threetears.evals.kernel.judge_case_of",
            "threetears.evals.kernel.resolve_variant_identity",
            "threetears.evals.ops.run_job_id",
            "threetears.evals.run.callers_missing_the_constructor",
            "threetears.evals.run.rejudge_result",
            "threetears.evals.run.resolve_constructor",
            "threetears.evals.testing.READER_CONFORMANCE_CASES",
            "threetears.evals.testing.ReaderConformanceCase",
            "threetears.evals.testing.ReaderSample",
        }
    ),
}


def _is_module(line: str) -> bool:
    try:
        importlib.import_module(line)
    except ImportError:
        return False
    return True


def host_imports() -> list[tuple[str, str]]:
    """Each ``(host file, name)`` the host import lists carry, modules skipped."""
    found: list[tuple[str, str]] = []
    for path in sorted(HOST_IMPORTS.glob("*.txt")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                found.append((path.name, line))
    return found


def test_there_is_a_host_import_list() -> None:
    assert host_imports(), f"no host import list under {HOST_IMPORTS}, so the test below says nothing"


def test_no_name_a_host_imports_is_provisional_or_missing() -> None:
    recorded = surface.load_snapshot()
    problems: list[str] = []
    for host_file, name in host_imports():
        if name in REMOVED_BEFORE_REPIN.get(host_file, frozenset()):
            continue
        if name in PROVISIONAL_FOR_HOST_TESTS.get(host_file, frozenset()):
            continue
        if name in recorded:
            if not surface.declared(recorded[name].tier):
                problems.append(f"{host_file}: {name} is {recorded[name].tier}")
        elif not _is_module(name):
            problems.append(f"{host_file}: {name} is not a public name of any root")
    assert not problems, (
        "a host imports these, so each must be host-contract or stable: add it to HOST_CONTRACT or STABLE in "
        f"packages/evals/scripts/public_surface.py and run `{surface.REGENERATE_SNAPSHOT}`:\n  " + "\n  ".join(problems)
    )


def test_a_name_removed_before_its_host_re_pins_is_gone_and_still_on_the_host_s_list() -> None:
    """The exception covers only what it says: a name the package no longer exports that the host's file still lists."""
    recorded = surface.load_snapshot()
    listed = set(host_imports())
    stale = [
        f"{host_file}: {name} ({'still public' if name in recorded else 'no longer on the host list'})"
        for host_file, names in REMOVED_BEFORE_REPIN.items()
        for name in sorted(names)
        if name in recorded or (host_file, name) not in listed
    ]
    assert not stale, "drop these from REMOVED_BEFORE_REPIN:\n  " + "\n  ".join(stale)


def test_a_name_only_a_host_s_tests_import_is_provisional_and_still_on_the_host_s_list() -> None:
    """The exception covers only what it says: a provisional name the host's file still lists."""
    recorded = surface.load_snapshot()
    listed = set(host_imports())
    stale: list[str] = []
    for host_file, names in PROVISIONAL_FOR_HOST_TESTS.items():
        for name in sorted(names):
            if (host_file, name) not in listed:
                stale.append(f"{host_file}: {name} (no longer on the host list)")
            if name not in recorded:
                stale.append(f"{host_file}: {name} (not a public name of any root)")
            elif surface.declared(recorded[name].tier):
                stale.append(f"{host_file}: {name} (declared {recorded[name].tier} again)")
    assert not stale, "drop these from PROVISIONAL_FOR_HOST_TESTS:\n  " + "\n  ".join(stale)
