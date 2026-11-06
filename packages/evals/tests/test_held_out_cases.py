"""The held-out writer cases (``tests/fixtures/writer_cases/held_out/``) read, stay apart from rung 5's, and all sit
on the held-out side.

An acceptance run reads one directory holding rung 5's eleven cases and these fourteen, at a held-out share of 0.99
(the directory's ``README.md``). These tests hold what that run relies on: every file reads as a writer case, no name
is two cases, no two files freeze one source (the mint would count them as one case), every group lands held-out at
0.99, and the cases the export rebuilds from the package's fixtures are the ones committed: every figure the same to
twelve places, and every snapshot id the same but where :data:`PLATFORM_BOUND` says.
"""

from __future__ import annotations

import json
import platform

import pytest

from packages.evals.tests.fixtures.writer_cases.export_held_out_cases import (
    DOW_RUNGS,
    HELD_OUT_DIR,
    DowRung,
    held_out_path,
    reproducible_cases,
)
from packages.evals.tests.fixtures.writer_cases.export_writer_cases import CASES_DIR
from threetears.evals.harness import WriterCaseFile, read_writer_cases
from threetears.evals.ops.report_writer.writer_bank import writer_group_side
from threetears.evals.schema.hashing import identity_json

#: The held-out share the acceptance run is launched at.
ACCEPTANCE_FRACTION = 0.99

#: How many held-out cases the acceptance run needs, rung 5's and these together.
ACCEPTANCE_CASES = 25

#: Where the reproducible cases were minted: the machine and its C library.
MINTED_ON = "aarch64-glibc-2.39"

#: This platform, as :data:`MINTED_ON` names one.
PLATFORM_KEY = f"{platform.machine()}-{'-'.join(platform.libc_ver())}"

#: The cases whose snapshot id may move off the platform that minted them. A snapshot's id hashes its report input with
#: every float in full (``report_input_document_digest``), and these two inputs hold statistics whose last digits the
#: platform's maths library decides (the hot-path oracle's platform note); every other figure agrees to twelve. Another
#: platform's library may agree with the minting one's, so they may move there, and no other case may.
PLATFORM_BOUND = frozenset({"s1", "hot-path-scaled"})


def _held_out() -> tuple[WriterCaseFile, ...]:
    return read_writer_cases(HELD_OUT_DIR)


def _union() -> tuple[WriterCaseFile, ...]:
    return (*read_writer_cases(CASES_DIR), *_held_out())


def test_every_held_out_file_reads_as_a_writer_case_under_its_own_name() -> None:
    cases = _held_out()
    assert len(cases) == 14
    assert {case.format for case in cases} == {"writer-case/1"}
    assert sorted(path.name for path in HELD_OUT_DIR.glob("*.json.gz")) == sorted(
        held_out_path(case.name).name for case in cases
    )


def test_no_name_is_two_cases_across_rung_five_and_the_held_out_cases() -> None:
    names = [case.name for case in _union()]
    assert len(names) == len(set(names)), sorted(name for name in names if names.count(name) > 1)


def test_no_two_cases_freeze_one_source() -> None:
    # A mint keys a case by the campaign its snapshot froze, or a bare input's digest: two files of one source would
    # be minted as one case, and the run would read fewer cases than the directory holds.
    sources = [case.snapshot.campaign_id or case.snapshot.report_input_digest for case in _union()]
    assert len(sources) == len(set(sources))


def test_cases_of_one_campaign_are_one_group_withholding_one_note() -> None:
    # The mint's key misses a bare input copied from a campaign (it keys one by campaign id, the other by digest), so
    # the campaign each report input names is read too. Cases naming one campaign are a reference and its reshapings:
    # they sit in one group, so one side, and withhold the same notes, so none reads the note reasoned on its campaign.
    seen: dict[str, WriterCaseFile] = {}
    clashes = []
    for case in _union():
        campaign = str(case.snapshot.report_input["meta"]["campaign"])
        first = seen.setdefault(campaign, case)
        if (first.group or first.name, sorted(first.withhold)) != (case.group or case.name, sorted(case.withhold)):
            clashes.append(f"{first.name!r} and {case.name!r} both report on {campaign!r}")
    assert clashes == []


def test_every_held_out_group_lands_held_out_and_the_union_holds_enough() -> None:
    dev = sorted(
        {
            case.group or case.name
            for case in _held_out()
            if writer_group_side(case.group or case.name, ACCEPTANCE_FRACTION) != "held_out"
        }
    )
    assert dev == []
    held_out = [
        case for case in _union() if writer_group_side(case.group or case.name, ACCEPTANCE_FRACTION) == "held_out"
    ]
    assert len(held_out) >= ACCEPTANCE_CASES


def _figures(case: WriterCaseFile) -> str:
    """The case as the house rule compares it, every float to twelve figures, without the snapshot's id and its
    report input's digest, which hash the input's floats in full."""
    document = json.loads(case.to_json())
    for derived in ("id", "report_input_digest"):
        del document["snapshot"][derived]
    return identity_json(document)


async def test_the_reproducible_cases_are_the_committed_ones() -> None:
    cases = await reproducible_cases()
    assert len(cases) == len(_held_out()) - len(DOW_RUNGS)
    moved, ids_moved = [], []
    for case in cases:
        committed = WriterCaseFile.read(held_out_path(case.name))
        if _figures(case) != _figures(committed):
            moved.append(case.name)
        if case.snapshot.id != committed.snapshot.id:
            ids_moved.append(case.name)
    assert moved == [], "rebuilt differently: run export_held_out_cases.py and review the diff"
    may_move = frozenset() if PLATFORM_KEY == MINTED_ON else PLATFORM_BOUND
    assert set(ids_moved) <= may_move


@pytest.mark.parametrize("rung", DOW_RUNGS, ids=[rung.name for rung in DOW_RUNGS])
def test_each_dungeons_of_wagons_case_is_its_rungs_campaign(rung: DowRung) -> None:
    case = WriterCaseFile.read(held_out_path(rung.name))
    assert (case.group, case.stratum, case.withhold, case.snapshot.source) == (rung.group, "dow", [], "campaign")
