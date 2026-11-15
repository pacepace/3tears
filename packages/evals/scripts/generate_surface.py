"""Generate ``packages/evals/docs/surface.md`` and ``packages/evals/llms.txt`` from the tier map's kinds.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed.

    # Rewrite both files:
    uv run python packages/evals/scripts/generate_surface.py

    # Exit 1 when either committed file is stale, without writing:
    uv run python packages/evals/scripts/generate_surface.py --check

Both are read from what the snapshot generator records: each declared name's kind (``surface_kinds.py``) and, for a
``typed`` name, its rung. ``docs/surface.md`` lists every declared name once, by role: the names an eval is written
with, rung by rung, then the host's contract, the operations, the action catalogue, a host's test kits, and the
shapes, each under the name that hands it over. Each line is the name and the first sentence of its own docstring.
``llms.txt`` is what an AI agent adopting the package reads first: what the package is, the names a first eval
needs with a few lines of the first ladder example verbatim, where a later task's names are by rung and by role,
and the actions. The wheel ships it as ``threetears/evals/llms.txt``. ``tests/test_surface_docs.py`` fails while
either committed file is not what this makes of the package.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import textwrap
import typing
from collections.abc import Iterator, Mapping
from pathlib import Path

from public_surface import (
    PACKAGE_ROOT,
    SurfaceEntry,
    describe,
    first_sentence,
    installed_surface,
    md_text,
    paragraphs,
)
from surface_kinds import (
    FIRST_EVAL_CEILING,
    KIND_CEILINGS,
    KINDS,
    TYPED,
    Kind,
    counts,
    declared_kinds,
    objects_of,
    rung_titles,
    shape_reach,
)

#: The page this writes.
SURFACE_PATH = PACKAGE_ROOT / "docs" / "surface.md"

#: The agent's first read this writes, at the package's root; the wheel ships it beside the code.
AGENT_PATH = PACKAGE_ROOT / "llms.txt"

#: The command that rewrites both, as the staleness test tells a reader.
REGENERATE = "uv run python packages/evals/scripts/generate_surface.py"

#: The hold-back ledger, whose counts the page gives.
LEDGER_PATH = PACKAGE_ROOT / "tests" / "fixtures" / "declared_holdbacks.txt"

#: Each ledger reason, as the page's table names what holds a name listed under it.
LEDGER_REASONS: Mapping[str, str] = {
    "discodon-tests-only": "Discodon's tests import it, and nothing else imports, names or reaches it",
    "port-host-implements": "A protocol a host implements and hands the engine",
    "protocol-engine-provides": "A protocol the engine implements and hands a host, or its storage satisfies",
    "unreached": "Nothing",
}

#: What each kind asks of the reader who meets it.
KIND_MEANINGS: Mapping[Kind, str] = {
    "typed": "a newcomer writes it in an eval file",
    "bound": "a host implements it or hands it to the engine",
    "invoked": "an operation called by name, from host code or through an action",
    "received": "a value someone is handed and reads",
    "tests": "a kit, recorder or guard a host runs in its own tests",
}

#: The first ladder example, and the lines of it the agent's file shows, each the start of exactly one line.
FIRST_EXAMPLE = PACKAGE_ROOT / "examples" / "1-triage" / "triage.py"
EXAMPLE_LINES: tuple[str, ...] = (
    "from threetears.evals import ",
    "model = Model(",
    "async def triage(",
    "    return await model.json(",
    "triage_eval = Eval(",
)

PREFIX = "threetears.evals."


def short(qualified: str) -> str:
    """A name as both files print it: ``threetears.evals.`` dropped."""
    return qualified.removeprefix(PREFIX)


def root_of(qualified: str) -> str:
    """The root a name is exported from, ``threetears.evals.`` dropped, or ``threetears.evals`` for the package's own."""
    root = qualified.rsplit(".", 1)[0]
    return short(root) if root != "threetears.evals" else root


def summary(qualified: str, obj: object) -> str:
    """The first sentence of a name's own docstring, as Markdown on one line."""
    root, name = qualified.rsplit(".", 1)
    text = describe(root, name, obj).description
    return md_text(first_sentence(text)) if text else "*(no description)*"


class Surface:
    """The declared names, their kinds and objects, and what reaches each shape: read once for both files."""

    def __init__(self) -> None:
        self.entries: dict[str, SurfaceEntry] = {
            name: entry for name, entry in installed_surface().items() if entry.tier != "provisional"
        }
        kinds = declared_kinds(self.entries)
        if kinds.problems:
            raise SystemExit("declared names without one kind:\n  " + "\n  ".join(kinds.problems))
        self.kinds: Mapping[str, Kind] = kinds.of
        self.objects = objects_of(self.entries)
        self.reach = shape_reach(self.entries, self.kinds)
        self.rungs = rung_titles()

    def of_kind(self, kind: Kind) -> list[str]:
        return [name for name in self.entries if self.kinds[name] == kind]

    def line(self, qualified: str) -> str:
        return f"- `{short(qualified)}`: {summary(qualified, self.objects[qualified])}"

    def typed_by_rung(self) -> dict[int, list[str]]:
        found: dict[int, list[str]] = {rung: [] for rung in sorted(self.rungs)}
        for name in TYPED:
            found.setdefault(TYPED[name], []).append(name)
        return found


def _by_root(names: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(root_of(name), []).append(name)
    return grouped


def _ledger() -> dict[str, str]:
    found: dict[str, str] = {}
    for raw in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.startswith("#"):
            name, reason = (part.strip() for part in raw.split("|"))
            found[name] = reason
    return found


def _actions() -> list[tuple[str, list[tuple[str, str]]]]:
    from threetears.evals.actions import eval_catalogue

    grouped: dict[str, list[tuple[str, str]]] = {}
    for action in eval_catalogue().actions:
        grouped.setdefault(action.workflow, []).append((action.name, action.summary))
    return list(grouped.items())


# --- docs/surface.md ---------------------------------------------------------------------------------------------


def surface_page(surface: Surface) -> str:
    """``docs/surface.md``: every declared name once, by role, then rung, then the name that reaches it."""
    return "\n".join(_surface_lines(surface)).rstrip() + "\n"


def _surface_lines(surface: Surface) -> Iterator[str]:
    total = counts(surface.kinds)
    yield "# The surface: every declared name, by role"
    yield ""
    yield from textwrap.wrap(
        f"Generated by `{REGENERATE}` from each declared name's kind; change a docstring or "
        "`scripts/surface_kinds.py`, never this page. Each line is a name, `threetears.evals.` dropped, and the "
        "first sentence of its docstring; the [reference](reference.md#public-api) has the rest.",
        120,
        break_on_hyphens=False,
        break_long_words=False,
    )
    yield ""
    yield "| Kind | Who meets it | Names | Ceiling |"
    yield "|---|---|---|---|"
    for kind in KINDS:
        ceiling = KIND_CEILINGS[kind]
        yield f"| `{kind}` | {KIND_MEANINGS[kind]} | {total[kind]} | {ceiling if ceiling is not None else 'none'} |"
    yield f"| all | | {sum(total.values())} | |"
    yield ""
    by_rung = surface.typed_by_rung()
    first = [name for rung, names in by_rung.items() if rung <= 1 for name in names]
    yield "## Writing an eval"
    yield ""
    yield f"### Your first eval: these {len(first)}"
    yield ""
    yield from (surface.line(name) for name in first)
    yield ""
    for rung, names in by_rung.items():
        if rung <= 1:
            continue
        added = f"adds {len(names)}" if names else "adds no name"
        yield f"### Rung {rung}: {surface.rungs[rung]} ({added})"
        yield ""
        if names:
            yield from (surface.line(name) for name in names)
            yield ""
    ledger = _ledger()
    bound = surface.of_kind("bound")
    yield f"## The host's contract: {len(bound)} names a host implements or hands the engine"
    yield ""
    for reason, title in (
        ("port-host-implements", "Ports a host implements"),
        ("protocol-engine-provides", "Protocols the engine provides"),
    ):
        names = [name for name in bound if ledger.get(name) == reason]
        yield f"### {title} ({len(names)})"
        yield ""
        yield from (surface.line(name) for name in names)
        yield ""
    rest = [name for name in bound if ledger.get(name) not in ("port-host-implements", "protocol-engine-provides")]
    yield f"### What a host builds and hands the engine ({len(rest)})"
    yield ""
    yield from (surface.line(name) for name in rest)
    yield ""
    invoked = surface.of_kind("invoked")
    yield f"## Operations: {len(invoked)} names called from host code"
    yield ""
    for root, names in _by_root(invoked).items():
        yield f"### `{root}` ({len(names)})"
        yield ""
        yield from (surface.line(name) for name in names)
        yield ""
    actions = _actions()
    yield f"## The action catalogue: {sum(len(each) for _, each in actions)} actions an agent calls"
    yield ""
    yield "`actions.eval_catalogue()` holds them; `transports.fastmcp.mount_fastmcp` serves them over MCP."
    yield ""
    for workflow, each in actions:
        yield f"### {workflow} ({len(each)})"
        yield ""
        yield from (f"- `{name}`: {md_text(first_sentence(text))}" for name, text in each)
        yield ""
    kits = surface.of_kind("tests")
    yield f"## A host's own tests: {len(kits)} kits, recorders and guards"
    yield ""
    yield from (surface.line(name) for name in kits)
    yield ""
    received = surface.of_kind("received")
    yield f"## Shapes: {len(received)} values a reader is handed"
    yield ""
    groups: dict[str | None, list[str]] = {}
    constants: list[str] = []
    for name in received:
        if name in surface.reach.by:
            groups.setdefault(surface.reach.by[name], []).append(name)
        else:
            constants.append(name)
    order = {name: index for index, name in enumerate(n for kind in KINDS for n in surface.of_kind(kind))}
    for start in sorted((s for s in groups if s is not None), key=order.__getitem__):
        yield f"### From `{short(start)}` ({len(groups[start])})"
        yield ""
        yield from (surface.line(name) for name in groups[start])
        yield ""
    yield f"### Constants ({len(constants)})"
    yield ""
    yield from (surface.line(name) for name in constants)
    yield ""
    unreached = groups.get(None, [])
    yield f"### Reached by no declared signature ({len(unreached)})"
    yield ""
    yield from (surface.line(name) for name in unreached)
    yield ""
    reasons = list(ledger.values())
    yield "## Held back"
    yield ""
    yield "The hold-back ledger, [`tests/fixtures/declared_holdbacks.txt`](../tests/fixtures/declared_holdbacks.txt),"
    yield "lists every declared name the tree alone does not hold; `tests/test_declared_holdbacks.py` holds it."
    yield ""
    yield "| Reason | What holds the name | Names |"
    yield "|---|---|---|"
    for reason, meaning in LEDGER_REASONS.items():
        yield f"| `{reason}` | {meaning} | {reasons.count(reason)} |"


# --- llms.txt ----------------------------------------------------------------------------------------------------

_WIDTH = 100


def _plain(text: str) -> str:
    """Markdown on one line as plain text: emphasis dropped, code spans kept."""
    return re.sub(r"\*\*(.+?)\*\*", r"\1", md_text(text)).replace("&lt;", "<")


def _sentences(text: str, count: int) -> list[str]:
    found: list[str] = []
    while text and len(found) < count:
        sentence = first_sentence(text)
        found.append(sentence)
        text = text[len(sentence) :].lstrip()
    return found


def what_it_is() -> list[str]:
    """Three sentences: the README's first, then the two after the package docstring's "Start here" lead."""
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    opening = next(p for p in readme.split("\n\n")[1:] if not p.startswith(("#", "`")))
    import threetears.evals

    start = next(p for p in paragraphs(threetears.evals.__doc__) if p.startswith("**Start here"))
    rest = _plain(start).partition(". ")[2]
    return [_sentences(" ".join(opening.split()), 1)[0], *_sentences(rest, 2)]


def example_lines() -> list[str]:
    """The lines of the first ladder example the agent's file shows, verbatim, in the file's order."""
    lines = FIRST_EXAMPLE.read_text(encoding="utf-8").splitlines()
    shown: list[int] = []
    for start in EXAMPLE_LINES:
        matches = [index for index, line in enumerate(lines) if line.startswith(start)]
        if len(matches) != 1:
            raise SystemExit(f"{FIRST_EXAMPLE} has {len(matches)} lines starting {start!r}, not one")
        shown.append(matches[0])
    return [lines[index] for index in sorted(shown)]


def agent_file(surface: Surface) -> str:
    """``llms.txt``: what an AI agent adopting the package reads first."""
    return "\n".join(_agent_lines(surface)).rstrip() + "\n"


def _wrapped(text: str, indent: str = "") -> Iterator[str]:
    yield from textwrap.wrap(text, _WIDTH, initial_indent=indent, subsequent_indent=f"{indent}  ")


def _names(names: list[str]) -> str:
    return ", ".join(short(name) for name in names)


def _agent_lines(surface: Surface) -> Iterator[str]:
    by_rung = surface.typed_by_rung()
    first = [name for rung, names in by_rung.items() if rung <= 1 for name in names]
    yield "3tears-evals: read this first"
    yield ""
    yield from _wrapped(f"Generated by {REGENERATE}; do not edit. Names drop the threetears.evals. prefix.")
    yield ""
    yield "WHAT IT IS"
    yield from _wrapped(" ".join(what_it_is()))
    yield ""
    yield f"A FIRST EVAL: THESE {len(first)} NAMES (at most {FIRST_EVAL_CEILING})"
    for name in first:
        yield from _wrapped(_plain(f"{short(name)}: {summary(name, surface.objects[name])}"), "  ")
    yield ""
    relative = FIRST_EXAMPLE.relative_to(PACKAGE_ROOT)
    run = _sentences(paragraphs(_module_doc(FIRST_EXAMPLE))[0], 2)[-1]
    yield f"EXAMPLE: lines of {relative}, verbatim"
    yield from (f"  {line}" for line in example_lines())
    yield from _wrapped(_plain(run))
    yield ""
    yield "LATER TASKS, BY RUNG (docs/ladder.md teaches each)"
    for rung, names in by_rung.items():
        if rung > 1:
            yield from _wrapped(f"Rung {rung}, {surface.rungs[rung]}: {_names(names) or 'no new name'}", "  ")
    yield ""
    total = counts(surface.kinds)
    yield "LATER TASKS, BY ROLE (docs/surface.md lists every name with one line)"
    for kind in KINDS:
        if kind == "typed":
            continue
        roots = _by_root(surface.of_kind(kind))
        where = ", ".join(f"{root} {len(names)}" for root, names in roots.items())
        yield from _wrapped(f"{kind} ({total[kind]}), {KIND_MEANINGS[kind]}: {where}", "  ")
    yield ""
    actions = _actions()
    yield f"ACTIONS: {sum(len(each) for _, each in actions)}, no Python needed"
    yield from _wrapped(
        "actions.eval_catalogue() holds them; transports.fastmcp.mount_fastmcp serves them over MCP; "
        "docs/serving.md and docs/reference.md describe each.",
        "  ",
    )
    for workflow, each in actions:
        yield from _wrapped(f"{workflow}: {', '.join(name for name, _ in each)}", "  ")


def _module_doc(path: Path) -> str:
    return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""


# --- main --------------------------------------------------------------------------------------------------------


def render() -> dict[Path, str]:
    """Both files, as this package makes them."""
    surface = Surface()
    return {SURFACE_PATH: surface_page(surface), AGENT_PATH: agent_file(surface)}


def main(argv: list[str] | None = None) -> int:
    """Write both files, or check them."""
    parser = argparse.ArgumentParser(description=typing.cast(str, __doc__).splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when a committed file is stale; write nothing")
    args = parser.parse_args(argv)
    stale = False
    for path, text in render().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                print(f"{path} is stale: run `{REGENERATE}`", file=sys.stderr)
                stale = True
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
