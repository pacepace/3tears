"""Write the re-root's map: the old-to-new spelling of every public name of ``3tears-evals`` that moves (#1199).

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed. It renders what
``role_roots.plan`` decides as two generated files, never edited by hand:

- ``api/reroot-map.txt``, which the rewrite script (``scripts/reroot.py``) reads to carry every consumer across;
- ``api/reroot-left.txt``, every provisional name that leaves the public roots because nothing declared reaches it,
  each with why, and which hosts' tests import it. A host's production code importing one would mean the rule is
  wrong for it, so this refuses to write the files when one does.

    uv run python packages/evals/scripts/reroot_map.py              # write both, the epoch being HEAD
    uv run python packages/evals/scripts/reroot_map.py --check      # fail when either is not what it would write

**The map's format** (design-reroot-cutover.md §3.2, as ``scripts/reroot.py`` reads it), UTF-8, ``\n`` line ends:

- The header is the leading lines that start with ``#``. The first is exactly ``# reroot-map 1``; then one
  ``# epoch <SHA> <title>`` per epoch, oldest first; then ``# body-sha256 <hex>``, the SHA-256 of every byte after the
  header. Any other header line is prose.
- Each body line is five tab-separated fields, ``epoch``, ``record``, ``old``, ``new``, ``note``; ``epoch`` is the SHA
  the line takes effect at. The records, in the order the body lists them:

  - ``module``: an internal module path renamed; matched as a dotted prefix.
  - ``root``: a former root; ``new`` is the role roots its names went to, comma-separated, in root order; matched
    whole only.
  - ``renamed``: a public name moves and changes name; ``note`` says how its shape changed.
  - ``name``: a public name moves to a role root; ``note`` is its home spelling (the module that defines it).
  - ``demoted``: it leaves the roots; ``new`` is its home spelling, ``note`` what to use instead.
  - ``left``: an unreached provisional name leaves the roots; ``new`` is its home spelling, ``note`` says to import
    it from there and why it is unreached.

- **Invariant**, checked here on write and by the script on load: no new value (each root of a ``root`` line, and a
  ``name`` line's home) equals, or has as a dotted prefix, the ``old`` of a line other than a ``root`` line at or
  before its epoch. One ``old`` has at most one line of each class (``module``; ``root``; every other record), so the
  six renamed submodules (§1.5) share their path only with the public function they shadowed.
- A name whose spelling does not change has no line: its new value would equal its old one.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, get_args

from public_surface import PACKAGE_ROOT
from role_roots import UNREACHED, Plan, Unreached, host_import_names, plan

#: Where the map is written; the rewrite script reads it.
MAP_PATH = PACKAGE_ROOT / "api" / "reroot-map.txt"

#: Where the names that leave unreached are listed.
LEFT_PATH = PACKAGE_ROOT / "api" / "reroot-left.txt"

#: The first header line of every map this writes and reads.
FORMAT = "reroot-map 1"

#: The title of the cut-over's epoch.
TITLE = "re-root by role (#1199)"

#: A map line's record.
MapRecord = Literal["module", "root", "renamed", "name", "demoted", "left"]

#: The records, in the order the body lists them.
RECORDS: tuple[MapRecord, ...] = get_args(MapRecord)

#: The columns of a body line.
COLUMNS = ("epoch", "record", "old", "new", "note")


@dataclasses.dataclass(frozen=True)
class Line:
    """One body line of the map."""

    epoch: str
    record: MapRecord
    old: str
    new: str
    note: str = ""

    def news(self) -> tuple[str, ...]:
        """Each spelling it can write: the roots of a ``root`` line; else its ``new``, and a ``name`` line's home."""
        if self.record == "root":
            return tuple(self.new.split(","))
        return (self.new, self.note) if self.record == "name" and self.note else (self.new,)

    def render(self) -> str:
        return "\t".join((self.epoch, self.record, self.old, self.new, self.note))


@dataclasses.dataclass(frozen=True)
class Epoch:
    """One epoch of the map: the SHA its lines take effect at, and a title."""

    sha: str
    title: str


@dataclasses.dataclass(frozen=True)
class ReRootMap:
    """A parsed map: its epochs, oldest first, and its body lines."""

    epochs: tuple[Epoch, ...]
    lines: tuple[Line, ...]


def map_lines(planned: Plan, epoch: str) -> list[Line]:
    """The body of one epoch, in record order and then by old spelling."""
    found: list[Line] = [Line(epoch, "module", old, new) for old, new in planned.modules.items()]
    found += [Line(epoch, "root", old, ",".join(new)) for old, new in planned.roots.items()]
    for placement in planned.placements:
        if placement.record == "kept":
            continue
        note = placement.note
        if placement.record == "name":
            note = placement.home
        elif placement.record == "left":
            note = f"import it from its home, the new column: unreached, {placement.note}"
        found.append(Line(epoch, placement.record, placement.old, placement.new, note))
    return sorted(found, key=lambda line: (RECORDS.index(line.record), line.old))


def _class(record: MapRecord) -> str:
    return record if record in ("module", "root") else "name"


def invariant_problems(lines: Sequence[Line], epochs: Sequence[str]) -> list[str]:
    """Each way the lines break the map's invariant (the module docstring states it), or nothing."""
    order = {sha: index for index, sha in enumerate(epochs)}
    problems = [f"{line.old}: epoch {line.epoch} is not in the header" for line in lines if line.epoch not in order]
    if problems:
        return problems
    seen: set[tuple[str, str, str]] = set()
    for line in lines:
        key = (line.epoch, _class(line.record), line.old)
        if key in seen:
            problems.append(f"{line.old} has two {_class(line.record)} lines")
        seen.add(key)
        if line.record in ("demoted", "left") and not line.note.strip():
            problems.append(f"{line.record} {line.old} names no replacement")
    prefixed = [line for line in lines if line.record != "root"]
    for line in lines:
        for new in line.news():
            for other in prefixed:
                if order[other.epoch] <= order[line.epoch] and (new == other.old or new.startswith(f"{other.old}.")):
                    how = "is" if new == other.old else "has as a dotted prefix"
                    problems.append(f"{line.record} {line.old} -> {new}: {how} the old spelling {other.old}")
    return problems


def _body_digest(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def render_map(planned: Plan, epoch: str, title: str = TITLE) -> str:
    """The map file for one epoch.

    :raises ValueError: The lines break the invariant.
    """
    lines = map_lines(planned, epoch)
    if problems := invariant_problems(lines, [epoch]):
        raise ValueError("the map breaks its invariant:\n" + "\n".join(problems))
    body = "".join(f"{line.render()}\n" for line in lines)
    counts = {record: sum(1 for line in lines if line.record == record) for record in RECORDS}
    header = [
        f"# {FORMAT}",
        "# The old-to-new spelling of each public name of 3tears-evals the re-root moves (issue #1199).",
        "# GENERATED by packages/evals/scripts/reroot_map.py from api/surface.json and the kinds: do not edit by hand.",
        f"# Planned at {epoch[:12]}; the cut-over regenerates it at its own base. Unreached provisional names: "
        f"{planned.unreached}.",
        "# Lines by record: " + ", ".join(f"{record} {count}" for record, count in counts.items()) + ".",
        f"# epoch {epoch} {title}",
        f"# body-sha256 {_body_digest(body)}",
    ]
    return "\n".join(header) + "\n" + body


def parse_map(text: str) -> ReRootMap:
    """Read a map file.

    :raises ValueError: Its first line is not ``# reroot-map 1``, it names no epoch, its body does not match its
        digest, a line is malformed, or the lines break the invariant.
    """
    rows = text.split("\n")
    if not text.endswith("\n") or rows[0] != f"# {FORMAT}":
        raise ValueError(f"not a map: its first line is not '# {FORMAT}', or its last has no newline")
    epochs: list[Epoch] = []
    digest = None
    header = 0
    for row in rows:
        if not row.startswith("#"):
            break
        header += 1
        words = row[1:].strip().split(" ", 2)
        if words[0] == "epoch" and len(words) == 3:
            epochs.append(Epoch(words[1], words[2]))
        elif words[0] == "body-sha256" and len(words) == 2:
            digest = words[1]
    if not epochs:
        raise ValueError("the map names no epoch")
    if digest != _body_digest("\n".join(rows[header:])):
        raise ValueError("the map's body does not match its body-sha256: it was edited by hand or cut short")
    lines = []
    for raw in rows[header:-1]:
        fields = raw.split("\t")
        if len(fields) != len(COLUMNS) or fields[1] not in RECORDS:
            raise ValueError(f"a malformed map line: {raw!r}")
        lines.append(Line(fields[0], fields[1], fields[2], fields[3], fields[4]))  # type: ignore[arg-type]
    if problems := invariant_problems(lines, [epoch.sha for epoch in epochs]):
        raise ValueError("the map breaks its invariant:\n" + "\n".join(problems))
    return ReRootMap(tuple(epochs), tuple(lines))


def leaving_production(planned: Plan, production: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    """Each host whose production code imports a name that leaves unreached, with those names."""
    left = {placement.old for placement in planned.placements if placement.record == "left"}
    return {host: sorted(left & set(names)) for host, names in production.items() if left & set(names)}


def render_left(planned: Plan, epoch: str, host_tests: Mapping[str, Iterable[str]]) -> str:
    """The list of names that leave unreached, each with its home spelling, why, and which hosts' tests import it."""
    left = [placement for placement in planned.placements if placement.record == "left"]
    tests = {host: set(names) for host, names in host_tests.items()}
    names = {placement.old.rsplit(".", 1)[1] for placement in left}
    demoted = sum(1 for placement in planned.placements if placement.record == "demoted")
    imported = {host: sorted(placement.old for placement in left if placement.old in found) for host, found in tests.items()}
    header = [
        "# The provisional names of 3tears-evals that leave the public roots at the re-root (issue #1199), because no",
        "# declared name reaches them and they are neither a stored document class nor an action's result (owner ruling",
        "# on Q1, 2026-11-15). Each is then imported from its home module, the second column.",
        "# GENERATED by packages/evals/scripts/reroot_map.py: do not edit by hand.",
        f"# Planned at {epoch[:12]}: {len(left)} spellings of {len(names)} names leave; {demoted} demoted spellings "
        "are in reroot-map.txt with their replacements.",
        "# A host's production code imports none of them (reroot_map.py refuses to write this list otherwise).",
        *(
            f"# {host}'s tests import {len(found)} of them (marked in the last column)."
            for host, found in sorted(imported.items())
        ),
        "# old\thome\twhy\thost tests",
    ]
    rows = []
    for placement in left:
        hosts = ",".join(sorted(host for host, found in tests.items() if placement.old in found))
        rows.append("\t".join((placement.old, placement.new, placement.note, hosts)))
    return "\n".join([*header, *rows]) + "\n"


def head_sha() -> str:
    """The checkout's ``HEAD``, the epoch a planned map is generated at."""
    return subprocess.run(
        ["git", "-C", str(PACKAGE_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def render_all(planned: Plan, epoch: str) -> dict[Path, str]:
    """Both files, by path.

    :raises ValueError: A host's production code imports a name that leaves unreached: the rule is wrong for it.
    """
    if wrong := leaving_production(planned, host_import_names()):
        raise ValueError(
            "a host's production code imports a name the rule finds unreached, so the rule is wrong for it: "
            f"{wrong}"
        )
    from role_roots import HOST_TEST_IMPORTS

    return {
        MAP_PATH: render_map(planned, epoch),
        LEFT_PATH: render_left(planned, epoch, host_import_names(HOST_TEST_IMPORTS)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--unreached", choices=get_args(Unreached), default=UNREACHED)
    parser.add_argument("--epoch", help="the SHA the lines take effect at (default: HEAD)")
    parser.add_argument("--check", action="store_true", help="fail when a file differs from what would be written")
    arguments = parser.parse_args(argv)
    epoch = arguments.epoch or head_sha()
    if arguments.check:
        epoch = arguments.epoch or _written_epoch()
    files = render_all(plan(arguments.unreached), epoch)
    if arguments.check:
        stale = [path for path, text in files.items() if not path.exists() or path.read_text("utf-8") != text]
        for path in stale:
            print(f"{path.relative_to(PACKAGE_ROOT)} is not what the rule writes: re-run reroot_map.py", file=sys.stderr)
        return 1 if stale else 0
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
        print(path.relative_to(PACKAGE_ROOT))
    return 0


def _written_epoch() -> str:
    """The epoch the written map names, so a check compares like with like."""
    return parse_map(MAP_PATH.read_text("utf-8")).epochs[0].sha


if __name__ == "__main__":
    raise SystemExit(main())
