"""Write the held-out writer case files (``tests/fixtures/writer_cases/held_out/``): minted offline, no model called.

Run from the repository root, with no key set (every replay reads its recordings, and a call they do not answer is
refused), the clone's ``src`` on ``PYTHONPATH`` so its rungs import the ``dow`` package::

    PYTHONPATH=<clone>/src uv run python -m packages.evals.tests.fixtures.writer_cases.export_held_out_cases --dow <clone>

The clone is dungeons-of-wagons on branch ``evals-ladder``; the environment is this worktree's, not the clone's.

Fifteen cases, each a :class:`~threetears.evals.harness.WriterCaseFile`, none of them a rung-5 case
(``examples/5-writer/cases/``, :mod:`~packages.evals.tests.fixtures.writer_cases.export_writer_cases`) and none of them
dev: they are the acceptance run's held-out cases.

- **the five reference inputs** r1 to r5 of ``fixtures/report_inputs/``, as the rung-5 export builds Kairo and the
  misfits: each in its :data:`~packages.evals.tests.writer_case_support.SEED` group, withholding its note (r5
  withholds none), in stratum ``reference``;
- **the four report shapes** s1 to s4 (``report_shapes_support``), each frozen with its bundle, in its own group and
  stratum ``shape``;
- **three dungeons-of-wagons rungs**, from the clone ``--dow`` names: rung 3 (``router_prompts.py``, router-v1 against
  router-v2), rung 2 (``forge.py``, judged) and rung 1 (``router.py``), each replayed from the recordings beside it
  into a store in memory and frozen through its own host, in stratum ``dow``; the two router rungs share a group;
- **three hot-path host campaigns**, built as ``test_hot_path_oracle.py`` builds them: the toy host's corpus campaign
  (``ops_fixture``), the courier's planner bake-off (its capture, ``hot_path_oracle/inputs/courierhost.json``) and the
  production-shaped ``scaled`` campaign (``hot_path_corpus``, scale 1), each frozen through its own host, in stratum
  ``host``. The courier's group is ``courier-planner``: the literal ``courierhost`` lands on dev at a held-out share
  of 0.99.

**Which runs reproduce their bytes.** The reference, shape and host cases are pure functions of the package's
fixtures: every host campaign is stored under fixed ids (the courier's from its capture), and each case's freeze time
is fixed, so every run on one platform rewrites them to the same bytes, and ``test_held_out_cases.py`` holds that. On
another platform the last digits of some statistics differ, and two cases (s1, hot-path-scaled) hash those into their
snapshot ids, so there they rebuild under other ids: commit what the minting platform wrote. The
dungeons-of-wagons cases are frozen once and kept: a replay mints a new campaign id, so its snapshot id would move with
every export. This export writes one only when its file is missing (it refuses to overwrite one); to freeze one again,
delete its file on purpose. ``--dow`` is needed only while a dungeons-of-wagons case is missing. A directory holding a
case file this export does not write is refused, naming the file.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from packages.evals.tests.analyst_support import fixture_snapshots, shape_snapshots
from packages.evals.tests.fixtures.report_inputs.misfits import NAMES as REFERENCES
from packages.evals.tests.fixtures.writer_cases.export_writer_cases import (
    FROZEN_AT,
    FROZEN_BY,
    PACKAGE,
    SCOPE,
    case_path,
    frozen_case,
    replay_only,
    replayed_case,
    seeded_case,
)
from threetears.evals.harness import WriterCaseFile
from threetears.evals.harness.writer_cases import WRITER_CASE_SUFFIX, writer_case_freeze
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops import OpsHost

#: Where the held-out case files are written.
HELD_OUT_DIR: Final = Path(__file__).resolve().parent / "held_out"

#: The report shapes minted, by short name.
SHAPES: Final = ("s1", "s2", "s3", "s4")

#: Each source's stratum.
REFERENCE: Final = "reference"
SHAPE: Final = "shape"
DOW: Final = "dow"
HOST: Final = "host"


@dataclass(frozen=True)
class DowRung:
    """One dungeons-of-wagons rung whose campaign becomes a case.

    :ivar file: Its eval file, under the clone's ``evals/``.
    :ivar name: The case's name.
    :ivar group: Its split group.
    """

    file: str
    name: str
    group: str


#: The dungeons-of-wagons rungs minted: rung 3, rung 2 and rung 1.
DOW_RUNGS: Final = (
    DowRung("router_prompts.py", "dow-router-prompts", "dow-router"),
    DowRung("forge.py", "dow-forge", "dow-forge"),
    DowRung("router.py", "dow-router", "dow-router"),
)


@dataclass(frozen=True)
class Campaign:
    """A stored campaign a host reads: the host, the campaign and its scope."""

    host: EvalHost
    campaign_id: str
    scope_id: str


def _toyhost() -> Campaign:
    from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture

    fixture = ops_fixture(generation=False)
    return Campaign(fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE)


def _courierhost() -> Campaign:
    import json

    from packages.evals.tests.fixtures.courierhost import COURIER_SCOPE
    from packages.evals.tests.hot_path_support import campaign_id_of, courierhost_from

    documents = json.loads(
        (PACKAGE / "tests" / "fixtures" / "hot_path_oracle" / "inputs" / "courierhost.json").read_text()
    )
    return Campaign(courierhost_from(documents), campaign_id_of(documents), COURIER_SCOPE)


def _scaled() -> Campaign:
    from packages.evals.tests.hot_path_support import HOT_PATH_SCOPE, hot_path_corpus

    corpus = hot_path_corpus(scale=1)
    return Campaign(corpus.host, corpus.campaign.id, HOT_PATH_SCOPE)


@dataclass(frozen=True)
class HostCampaign:
    """One hot-path host campaign that becomes a case: its name, its group and how it is built.

    The builders import the hot-path oracle's support (and through it the toy host's operations fixture) when called,
    so a reader of the reference and shape cases loads neither.

    :ivar name: The case's name.
    :ivar group: Its split group.
    :ivar build: Stores the campaign and returns where it is read.
    """

    name: str
    group: str
    build: Callable[[], Campaign]


#: The hot-path host campaigns minted (``scaled_two_aspects`` holds the same runs as ``scaled``, so it is not one).
HOST_CAMPAIGNS: Final = (
    HostCampaign("toyhost", "toyhost", _toyhost),
    HostCampaign("courierhost", "courier-planner", _courierhost),
    HostCampaign("hot-path-scaled", "hot-path-scaled", _scaled),
)


def synthetic_cases() -> list[WriterCaseFile]:
    """The reference and shape cases: pure functions of the package's fixtures, the same bytes on every run."""
    references = fixture_snapshots(SCOPE)
    shapes = shape_snapshots(SCOPE, SHAPES)
    return [
        *(seeded_case(references[name], name, REFERENCE) for name in REFERENCES),
        *(frozen_case(shapes[name], name=name, group=name, stratum=SHAPE) for name in SHAPES),
    ]


async def dow_case(clone: Path, rung: DowRung) -> WriterCaseFile:
    """One dungeons-of-wagons rung replayed from its recordings, in the clone's ``evals/`` directory.

    :raises ValueError: The ``dow`` package does not import: the clone's ``src`` is not on ``PYTHONPATH``.
    """
    if importlib.util.find_spec("dow") is None:
        raise ValueError(f"the dow package does not import: run with PYTHONPATH={clone / 'src'}")
    with contextlib.chdir(clone / "evals"):
        return await replayed_case(
            clone / "evals" / rung.file, name=rung.name, group=rung.group, stratum=DOW, scope=DOW
        )


async def host_case(campaign: HostCampaign) -> WriterCaseFile:
    """One host campaign, stored off the loop (its store refuses a call on it) and frozen through its own host.

    Its bundle's assembly time and its freeze time are both put at
    :data:`~packages.evals.tests.fixtures.writer_cases.export_writer_cases.FROZEN_AT`, the clock's two readings in a
    freeze (neither is in the snapshot's id), so the case's bytes are the same on every run.
    """
    built = await asyncio.to_thread(campaign.build)
    case = await writer_case_freeze(
        OpsHost.reading(built.host),
        built.scope_id,
        campaign_id=built.campaign_id,
        name=campaign.name,
        created_by=FROZEN_BY,
    )
    assembled = case.snapshot.model_copy(update={"bundle_assembled_at": FROZEN_AT})
    return frozen_case(assembled, name=campaign.name, group=campaign.group, stratum=HOST)


async def reproducible_cases() -> list[WriterCaseFile]:
    """Every case whose bytes are the same on every run: the reference and shape cases, then the host cases."""
    return [*synthetic_cases(), *[await host_case(campaign) for campaign in HOST_CAMPAIGNS]]


def held_out_path(name: str) -> Path:
    """The file a held-out case is written to."""
    return case_path(name, HELD_OUT_DIR)


async def export(*, dow: Path | None) -> list[tuple[Path, str]]:
    """Write every held-out case file; each written or kept, with which.

    :param dow: The dungeons-of-wagons clone; needed only while one of its cases is missing.
    :ptype dow: Path | None
    :return: Each case file and ``written`` or ``kept``.
    :rtype: list[tuple[Path, str]]
    :raises ValueError: The directory holds a case file this export does not write, or a dungeons-of-wagons case is
        missing and no clone is named.
    """
    reproducible = await reproducible_cases()
    expected = {held_out_path(case.name) for case in reproducible} | {held_out_path(rung.name) for rung in DOW_RUNGS}
    if stray := sorted(path.name for path in HELD_OUT_DIR.glob(f"*{WRITER_CASE_SUFFIX}") if path not in expected):
        raise ValueError(f"{HELD_OUT_DIR} holds case files this export does not write: {', '.join(stray)}")
    done = [(case.write(held_out_path(case.name)), "written") for case in reproducible]
    for rung in DOW_RUNGS:
        path = held_out_path(rung.name)
        if path.is_file():
            done.append((path, "kept"))
            continue
        done.append(((await dow_case(_clone(dow), rung)).write(path), "written"))
    return done


def _clone(dow: Path | None) -> Path:
    if dow is None:
        raise ValueError("a dungeons-of-wagons case is missing: name the clone its rungs replay from with --dow")
    return dow.resolve()


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="export_held_out_cases.py", description=__doc__.splitlines()[0])
    parser.add_argument("--dow", type=Path, help="the dungeons-of-wagons clone (branch evals-ladder)")
    options = parser.parse_args(list(argv))
    replay_only()  # a replay or nothing: a recording that no longer serves its rung is refused, never sent
    for path, what in asyncio.run(export(dow=options.dow)):
        print(f"{what} {path.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
