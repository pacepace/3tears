"""Write rung 5's eleven writer case files (``examples/5-writer/cases/``): a one-off export, run again only on purpose.

Run from the repository root, with no key set (every rung replays its recordings)::

    uv run python -m packages.evals.tests.fixtures.writer_cases.export_writer_cases
    uv run python -m packages.evals.tests.fixtures.writer_cases.export_writer_cases --refreeze

The eleven cases, each a :class:`~threetears.evals.harness.WriterCaseFile`:

- **the four ladder campaigns**, rungs 1 to 4: each rung's eval run through the quick path into a store of its own
  in memory, replayed from the recordings beside it, its campaign frozen through its own host. The case's name and
  split group are the rung's directory, and its stratum the rung's domain; it withholds no note, since none was
  reasoned on it;
- **the package's frozen campaign** and **the six misfit inputs** of ``fixtures/report_docs/prototype5/``, as the
  writer bank's seed builds them (``tests/writer_case_support.py``): each in its reference's split group and
  withholding its reference's note, the campaign in the stratum its group names and every misfit in ``misfit``.

**The bytes are the same on every run.** The seven fixture cases are pure functions of the package's fixtures, their
freeze time fixed, so each is rewritten to the same bytes. A ladder campaign is not: each run of a rung is a new run,
with a new id, its cells started in an order derived from that id (so a recorded call can answer another repeat),
and a judge's wall-clock time among its readings. Its snapshot id, which every writer recording of the case is keyed
by, would move with every export, so a ladder case is frozen once and kept: the export writes one only when its file
is missing, or when ``--refreeze`` says to freeze every rung again (which strands every recording made over the old
ones). A directory holding a case file this export does not write is refused, naming the file.

**The ``3-invoices`` case is the pre-split campaign.** It was frozen from the single file rung 3 was before it split
into one concept a file (``extract.py``, 2026-11-03). ``guardrail.py`` declares that same eval and replays it byte for
byte, so it is the file named below; but the case was never frozen from it, and ``--refreeze`` would strand the $7.53
of rung-5 writer recordings made over the frozen one. Never refreeze it.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.fixtures.report_inputs.misfits import MISFITS
from packages.evals.tests.writer_case_support import SEED
from threetears.evals.harness import WriterCaseFile
from threetears.evals.harness.writer_cases import WRITER_CASE_SUFFIX, writer_case_freeze
from threetears.evals.kernel.report_snapshot import EvalReportSnapshot
from threetears.evals.ops import OpsHost
from threetears.evals.quick.files import load_evals

#: The package directory.
PACKAGE: Final = Path(__file__).resolve().parents[3]

#: Where the case files are written.
CASES_DIR: Final = PACKAGE / "examples" / "5-writer" / "cases"

#: Who froze every case, as a stored snapshot names it.
FROZEN_BY: Final = "export_writer_cases"

#: When every fixture case was frozen: fixed, so its bytes never move.
FROZEN_AT: Final = "2026-11-03T00:00:00+00:00"

#: The scope the fixture snapshots are built in; a mint restores each into its own.
SCOPE: Final = "writer-cases"

#: The stratum every misfit input is minted in.
MISFIT: Final = "misfit"

#: The variables a recording run reads: none is set here, so a rung's recording that no longer serves it is refused.
_KEYS: Final = ("RECORD", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")


def replay_only() -> None:
    """Unset every variable a recording run reads, so a replay whose recording no longer serves it is refused, never
    sent to a provider."""
    for variable in _KEYS:
        os.environ.pop(variable, None)


@dataclass(frozen=True)
class Rung:
    """One ladder rung whose campaign becomes a case: its directory, its eval file and its domain.

    :ivar directory: The rung's directory under ``examples/``: the case's name and split group.
    :ivar file: Its eval file.
    :ivar stratum: Its domain, the case's stratum.
    """

    directory: str
    file: str
    stratum: str


#: The four ladder rungs a case is frozen from.
LADDER: Final = (
    Rung("1-triage", "triage.py", "support-triage"),
    Rung("2-replies", "replies.py", "support-replies"),
    Rung("3-invoices", "guardrail.py", "invoices"),  # the pre-split campaign, replayed byte for byte (see above)
    Rung("4-booking", "agent.py", "booking"),
)


def case_path(name: str, directory: Path = CASES_DIR) -> Path:
    """The file a case is written to in ``directory``: its name, spaces as dashes."""
    return directory / f"{name.replace(' ', '-')}{WRITER_CASE_SUFFIX}"


def frozen_case(
    snapshot: EvalReportSnapshot, *, name: str, group: str, stratum: str, withhold: Sequence[str] = ()
) -> WriterCaseFile:
    """A fixture snapshot as a case, frozen at :data:`FROZEN_AT` so its bytes never move."""
    frozen = snapshot.model_copy(update={"created_at": FROZEN_AT})
    return WriterCaseFile(name=name, group=group, stratum=stratum, withhold=list(withhold), snapshot=frozen)


def seeded_case(snapshot: EvalReportSnapshot, name: str, stratum: str) -> WriterCaseFile:
    """A fixture snapshot as a case in its :data:`SEED` group, withholding its :data:`SEED` notes."""
    group, withhold = SEED[name]
    return frozen_case(snapshot, name=name, group=group, stratum=stratum, withhold=withhold)


def fixture_cases() -> list[WriterCaseFile]:
    """The package's frozen campaign and the six misfits, each in its reference's group with its note withheld."""
    snapshots = fixture_snapshots(SCOPE)
    return [
        seeded_case(snapshots[name], name, stratum)
        for name, stratum in ((KAIRO, SEED[KAIRO][0]), *((misfit, MISFIT) for misfit in MISFITS))
    ]


async def replayed_case(eval_file: Path, *, name: str, group: str, stratum: str, scope: str) -> WriterCaseFile:
    """An eval file's one eval replayed in memory through the quick path, its campaign frozen through its own host.

    Nothing is sent when :func:`replay_only` has run: a call its recordings do not answer is refused.

    :param eval_file: The eval file, its recordings beside it.
    :ptype eval_file: Path
    :param name: The case's name.
    :ptype name: str
    :param group: Its split group.
    :ptype group: str
    :param stratum: Its stratum.
    :ptype stratum: str
    :param scope: The scope its run is stored in.
    :ptype scope: str
    :return: The case, unwritten.
    :rtype: WriterCaseFile
    """
    (evaluation,) = load_evals(eval_file)
    ran = await evaluation.run(store=":memory:", scope=scope)
    return await writer_case_freeze(
        OpsHost.reading(ran.host),
        ran.scope_id,
        campaign_id=ran.campaign_id,
        name=name,
        group=group,
        stratum=stratum,
        created_by=FROZEN_BY,
    )


async def ladder_case(rung: Rung) -> WriterCaseFile:
    """One rung's eval replayed in memory through the quick path, its campaign frozen through its own host."""
    return await replayed_case(
        PACKAGE / "examples" / rung.directory / rung.file,
        name=rung.directory,
        group=rung.directory,
        stratum=rung.stratum,
        scope=rung.directory,
    )


async def export(*, refreeze: bool) -> list[tuple[Path, str]]:
    """Write every case file; each written or kept, with which.

    :param refreeze: Freeze every ladder rung again, over the files already there.
    :ptype refreeze: bool
    :return: Each case file and ``written`` or ``kept``.
    :rtype: list[tuple[Path, str]]
    :raises ValueError: The directory holds a case file this export does not write.
    """
    fixtures = fixture_cases()
    expected = {case_path(case.name) for case in fixtures} | {case_path(rung.directory) for rung in LADDER}
    if stray := sorted(path.name for path in CASES_DIR.glob(f"*{WRITER_CASE_SUFFIX}") if path not in expected):
        raise ValueError(f"{CASES_DIR} holds case files this export does not write: {', '.join(stray)}")
    done = [(case.write(case_path(case.name)), "written") for case in fixtures]
    for rung in LADDER:
        path = case_path(rung.directory)
        if path.is_file() and not refreeze:
            done.append((path, "kept"))
            continue
        done.append(((await ladder_case(rung)).write(path), "written"))
    return done


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="export_writer_cases.py", description=__doc__.splitlines()[0])
    parser.add_argument("--refreeze", action="store_true", help="freeze every ladder rung again (strands recordings)")
    options = parser.parse_args(list(argv))
    replay_only()  # a replay or nothing: a rung whose recording no longer serves it is refused, never sent
    for path, what in asyncio.run(export(refreeze=options.refreeze)):
        print(f"{what} {path.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
