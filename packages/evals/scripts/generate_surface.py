"""Generate ``packages/evals/docs/surface.md`` and ``packages/evals/llms.txt`` from the declared names' kinds.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed.

    # Rewrite both files:
    uv run python packages/evals/scripts/generate_surface.py

    # Exit 1 when either committed file is stale, without writing:
    uv run python packages/evals/scripts/generate_surface.py --check

Both are read from each declared name's kind (``surface_kinds.py``) and, for a ``typed`` name, the rung that teaches it.
``docs/surface.md`` opens with the names a first eval needs and what each rung adds, then the host's contract, the
functions a host's code calls, the functions behind actions, the action catalogue, a host's test kits, the shapes by the
concept each belongs to, the two registers issue #1199 settles (awaiting demotion, and needing a shipped example), each
name in its register's section and in no other, the reading states with their reasons and remedies, the near-duplicate
shapes, and last the counts by kind, each kind with its meaning in one line, reported as a trend and never a budget;
each line is a name and the first sentence of its own docstring. ``llms.txt`` is what an AI agent adopting the package
reads first: what it is, the first example whole, how to get and run the examples, how to read what comes back, where a
later task's names are, the names that need a shipped example (marked so), the actions, and the kinds, one line each;
a name awaiting demotion is never offered to the agent. The wheel ships it as ``threetears/evals/llms.txt``, so it cites only what an
install holds or the repository by URL. ``tests/test_surface_docs.py`` fails while either committed file is not what
this makes of the package.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import textwrap
import tomllib
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
    AWAITING_DEMOTION,
    AWAITING_SHIPPED_USE,
    ENGINE_PROVIDES,
    FIRST_EVAL_GOAL,
    HOST_IMPLEMENTS,
    KIND_MEANINGS,
    KINDS,
    REGISTER_ISSUE,
    THROUGH_RUNG_3_GOAL,
    TYPED,
    TYPED_IN_DOCS,
    Kind,
    counts,
    declared_kinds,
    near_duplicate_shapes,
    objects_of,
    protocol_sides,
    rung_anchors,
    rung_titles,
    shape_reach,
    typed_through,
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
    "port-host-implements": "A protocol a host implements and hands the engine",
    "protocol-engine-provides": "A protocol the engine implements and hands a host, or its storage satisfies",
    "unreached": "Nothing",
}

#: The concepts a reader already knows, in the order the shapes are grouped by: a shape belongs to the first whose
#: words its own name holds. An exception is an error, and a shape no concept's words name is "other".
CONCEPTS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "Runs, results and traces",
        frozenset(
            {"Run", "Runs", "Runner", "Result", "Results", "Trace", "Cell", "Trial", "Attempt", "Launch", "Delivery"}
            | {"Candidate", "Variant"}
        ),
    ),
    (
        "Judges, graders and labels",
        frozenset({"Judge", "Judged", "Rubric", "Calibration", "Rating", "Label", "Blind", "Verdict", "Grader"}),
    ),
    (
        "Campaigns and comparisons",
        frozenset({"Campaign", "Aspect", "Floor", "Bar", "Arm", "Control", "Swept", "Contrast", "Gate", "Frontier"})
        | frozenset({"Sweepable", "Axis", "Pass", "Threshold", "Pins", "Merit"}),
    ),
    (
        "Reports and analyses",
        frozenset({"Report", "Chart", "Reading", "Readings", "Insight", "Analysis", "Analyst", "Summary", "Bundle"}),
    ),
    (
        "Cost and budgets",
        frozenset({"Cost", "Spend", "Price", "Pricing", "Budget", "Ceiling", "Ceilings", "Usage", "Metered", "Rate"}),
    ),
    (
        "Cases, templates and seeds",
        frozenset({"Case", "Template", "Seed", "Corpus", "Definition", "Prompt", "Variation"}),
    ),
    (
        "Worlds, agents and conversations",
        frozenset({"World", "Goal", "Precondition", "Tool", "Turn", "Conversation", "Actor", "Simulator", "Subject"}),
    ),
    (
        "Model calls",
        frozenset({"Completion", "Settings", "Setting", "Wire", "Refusal", "Provider", "Thinking", "Call", "Model"})
        | frozenset({"Stop", "Failure", "Negotiated"}),
    ),
    ("Measures", frozenset({"Measure", "Metric", "Descriptor", "Latency", "Interval", "Scale", "Value", "Phase"})),
    ("Actions and jobs", frozenset({"Action", "Catalogue", "Job", "Jobs", "Lease"})),
    ("Storage", frozenset({"Store", "Doc", "Document", "Route", "Routing", "Reset"})),
)

#: The first ladder example, which the agent's file shows whole.
FIRST_EXAMPLE = PACKAGE_ROOT / "examples" / "1-triage" / "triage.py"

#: The examples' own guide, whose "Running them" section says how to fetch them and what their exit codes mean.
EXAMPLES_GUIDE = PACKAGE_ROOT / "examples" / "README.md"

PREFIX = "threetears.evals."

#: Every name either register holds, which no healthy section lists.
REGISTERED: frozenset[str] = AWAITING_DEMOTION | frozenset(AWAITING_SHIPPED_USE)


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


def repository() -> str:
    """The repository's URL, as the package's metadata states it."""
    config = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return typing.cast(str, config["project"]["urls"]["Repository"])


def at_tag(path: str) -> str:
    """A file of the package in the repository, at the tag of the installed version."""
    return f"{repository()}/blob/<tag>/packages/evals/{path}"


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
        self.anchors = rung_anchors()
        self._first: dict[int, str] = {}

    def of_kind(self, kind: Kind) -> list[str]:
        """The names of one kind, less those a register holds: each register has its own section."""
        return [name for name in self.entries if self.kinds[name] == kind and name not in REGISTERED]

    def line(self, qualified: str) -> str:
        """One name's line; the second spelling of one class or function says whose it is."""
        obj = self.objects[qualified]
        if isinstance(obj, type) or callable(obj):
            first = self._first.setdefault(id(obj), qualified)
            if first != qualified:
                return f"- `{short(qualified)}`: the same as `{short(first)}`."
        return f"- `{short(qualified)}`: {summary(qualified, obj)}"

    def typed_by_rung(self) -> dict[int, list[str]]:
        found: dict[int, list[str]] = {rung: [] for rung in sorted(self.rungs)}
        for name in TYPED:
            found.setdefault(TYPED[name], []).append(name)
        return found

    def first_eval(self) -> list[str]:
        return [name for name in TYPED if TYPED[name] <= 1]


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


def concept_of(qualified: str, entry: SurfaceEntry) -> str:
    """The concept a shape belongs to: the first of :data:`CONCEPTS` whose words its name holds."""
    if entry.kind == "exception":
        return "Errors"
    words = set(re.findall(r"[A-Z][a-z]+|[A-Z]+(?![a-z])", qualified.rsplit(".", 1)[1]))
    return next((concept for concept, names in CONCEPTS if words & names), "Other")


def reading_states() -> list[tuple[str, str, str]]:
    """Each state a reading can be in that is not counted, each of its reasons, and the act that closes it."""
    from threetears.evals.analysis.caveats import REMEDY_OF_REASON
    from threetears.evals.schema.reading_states import REASONS_OF

    return [
        (state, reason, REMEDY_OF_REASON[reason]) for state, reasons in REASONS_OF.items() for reason in sorted(reasons)
    ]


# --- docs/surface.md ---------------------------------------------------------------------------------------------


def surface_page(surface: Surface) -> str:
    """``docs/surface.md``: every declared name once, the newcomer's first."""
    return "\n".join(_surface_lines(surface)).rstrip() + "\n"


def _rung_heading(surface: Surface, rung: int, added: str) -> str:
    return f"### [Rung {rung}: {surface.rungs[rung]}](ladder.md#{surface.anchors[rung]}) ({added})"


def _surface_lines(surface: Surface) -> Iterator[str]:
    by_rung = surface.typed_by_rung()
    first = surface.first_eval()
    yield "# The surface: every declared name, by role"
    yield ""
    yield "Each line is a name, `threetears.evals.` dropped, and the first sentence of its docstring; the"
    yield "[reference](reference.md#public-api) has the rest."
    yield ""
    yield "## Writing an eval"
    yield ""
    yield f"### Your first eval: these {len(first)}"
    yield ""
    yield from (surface.line(name) for name in first)
    yield ""
    for rung, names in by_rung.items():
        if rung <= 1:
            continue
        yield _rung_heading(surface, rung, f"adds {len(names)}" if names else "adds no name")
        yield ""
        if names:
            yield from (surface.line(name) for name in names)
            yield ""
    off_ladder = [name for name in surface.entries if name in TYPED_IN_DOCS]
    yield f"### Written in an eval, taught on a page rather than a rung ({len(off_ladder)})"
    yield ""
    for name in off_ladder:
        page = TYPED_IN_DOCS[name]
        yield f"{surface.line(name)} Taught in [{Path(page).stem}]({Path(page).relative_to('docs')})."
    yield ""
    sides = protocol_sides()
    bound = surface.of_kind("bound")
    yield f"## The host's contract: {len(bound)} names a host implements or hands the engine"
    yield ""
    ports = [name for name in bound if sides.get(name) == HOST_IMPLEMENTS]
    yield f"### Ports a host implements ({len(ports)})"
    yield ""
    yield from (surface.line(name) for name in ports)
    yield ""
    rest = [name for name in bound if name not in ports]
    yield f"### What a host builds and hands the engine ({len(rest)})"
    yield ""
    yield from (surface.line(name) for name in rest)
    yield ""
    invoked = surface.of_kind("invoked")
    yield f"## Functions a host's code calls: {len(invoked)}"
    yield ""
    yield "Each is `invoked`: your code imports it and calls it. None is an action; the actions follow."
    yield ""
    for root, names in _by_root(invoked).items():
        yield f"### `{root}` ({len(names)})"
        yield ""
        yield from (surface.line(name) for name in names)
        yield ""
    behind = surface.of_kind("action")
    yield f"## The functions behind actions: {len(behind)}"
    yield ""
    yield "Each is `action`: the function an action of the catalogue (below) runs, under the same name. An agent or an"
    yield "operator invokes the action by name; a host's code may also call the function."
    yield ""
    yield from (surface.line(name) for name in behind)
    yield ""
    actions = _actions()
    yield f"## The action catalogue: {sum(len(each) for _, each in actions)} actions an agent calls"
    yield ""
    yield "`actions.eval_catalogue()` holds them; `transports.fastmcp.mount_fastmcp` serves them over MCP. An agent or"
    yield "an operator invokes each by its name, with no Python, and the action does the work."
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
    provided = [name for name in received if sides.get(name) == ENGINE_PROVIDES]
    yield f"### Protocols the engine implements and hands a host ({len(provided)})"
    yield ""
    yield from (surface.line(name) for name in provided)
    yield ""
    groups: dict[str, list[str]] = {}
    for name in received:
        if name in provided:
            continue
        concept = "Constants" if name not in surface.reach.by else concept_of(name, surface.entries[name])
        groups.setdefault(concept, []).append(name)
    for concept in (*(c for c, _ in CONCEPTS), "Errors", "Other", "Constants"):
        if concept in groups:
            yield f"### {concept} ({len(groups[concept])})"
            yield ""
            yield from (surface.line(name) for name in groups[concept])
            yield ""
    demoting = [name for name in surface.entries if name in AWAITING_DEMOTION]
    yield f"## Awaiting demotion ({REGISTER_ISSUE}): {len(demoting)}"
    yield ""
    yield "Declared, and used by no production code and no shipped example (admission criteria 2 and 3): only tests, the"
    yield "toy host, a page's prose or the action catalogue reach each. Where an action runs one, the action is the door"
    yield f"to use. Issue {REGISTER_ISSUE} demotes them together with the namespace re-root, so a host migrates once; do"
    yield "not start using one."
    yield ""
    yield from (f"{surface.line(name)} ({surface.kinds[name]})" for name in demoting)
    yield ""
    needing = [name for name in surface.entries if name in AWAITING_SHIPPED_USE]
    yield f"## Needs a shipped example ({REGISTER_ISSUE}): {len(needing)}"
    yield ""
    yield "Each is the only door to something a host must be able to do, and no shipped example uses it yet: use it as"
    yield f"its line says. Issue {REGISTER_ISSUE} adds the example."
    yield ""
    yield from (f"- `{short(name)}`: {md_text(AWAITING_SHIPPED_USE[name])}" for name in needing)
    yield ""
    states = reading_states()
    yield f"## Reading states: {len(states)} reasons a reading is not counted, and what closes each"
    yield ""
    yield "| State | Reason | Remedy |"
    yield "|---|---|---|"
    yield from (f"| `{state}` | `{reason}` | `{remedy}` |" for state, reason, remedy in states)
    yield ""
    pairs = near_duplicate_shapes(surface.kinds)
    yield f"## Near-duplicate shapes: {len(pairs)}, each saying why both exist"
    yield ""
    yield from (f"- `{short(one)}` and `{short(other)}`" for one, other in pairs)
    yield ""
    total = counts(surface.kinds)
    yield "## The counts"
    yield ""
    yield "A name is public when it meets the admission criteria, not to meet a number: these counts are reported each"
    yield "wave as a trend. The goals are the adopter's, a first eval's names and rung 3's."
    yield ""
    yield "| Kind | Names | Who meets it |"
    yield "|---|---|---|"
    for kind in KINDS:
        yield f"| `{kind}` | {total[kind]} | {KIND_MEANINGS[kind]} |"
    yield f"| all | {sum(total.values())} | |"
    yield ""
    yield f"Of these, {len(AWAITING_DEMOTION)} await demotion and {len(AWAITING_SHIPPED_USE)} need a shipped example"
    yield f"(issue {REGISTER_ISSUE}), each listed in its register's section above."
    yield ""
    yield "| Typed names | Names | Goal |"
    yield "|---|---|---|"
    yield f"| A first eval (rungs 0 and 1) | {typed_through(1)} | {FIRST_EVAL_GOAL} |"
    yield f"| Through rung 3 | {typed_through(3)} | {THROUGH_RUNG_3_GOAL} |"
    yield f"| The whole ladder | {len(TYPED)} | |"
    yield f"| Taught on a page rather than a rung | {len(TYPED_IN_DOCS)} | |"
    yield ""
    reasons = list(_ledger().values())
    yield "The hold-back ledger, [`tests/fixtures/declared_holdbacks.txt`](../tests/fixtures/declared_holdbacks.txt),"
    yield "lists every declared name the tree alone does not hold; `tests/test_declared_holdbacks.py` holds it."
    yield ""
    yield "| Reason | What holds the name | Names |"
    yield "|---|---|---|"
    for reason, meaning in LEDGER_REASONS.items():
        yield f"| `{reason}` | {meaning} | {reasons.count(reason)} |"
    yield ""
    yield f"Generated by `{REGENERATE}`: change a docstring or `scripts/surface_kinds.py`, never this page."


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


def _module_doc(path: Path) -> str:
    return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""


def example_program() -> list[str]:
    """The first ladder example whole, its module docstring left out: a complete program, verbatim."""
    source = FIRST_EXAMPLE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    first = tree.body[0]
    skip = first.end_lineno if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) else 0
    lines = source.splitlines()[skip or 0 :]
    while lines and not lines[0].strip():
        lines.pop(0)
    return lines


def example_notes() -> list[str]:
    """What running an example costs, from the examples' guide, and how the first one replays and records again,
    from its own docstring."""
    guide = paragraphs(_section(EXAMPLES_GUIDE, "Running them").partition("\n\n")[2])[0]
    doc = paragraphs(_module_doc(FIRST_EXAMPLE))
    return [_plain(_sentences(guide, 1)[0]), _plain(_sentences(doc[1], 2)[1])]


def _section(path: Path, heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else len(text)]


def fetch_commands() -> list[str]:
    """The commands the examples' guide fetches them with, verbatim."""
    block = re.search(r"```bash\n(.*?)```", _section(EXAMPLES_GUIDE, "Running them"), re.S)
    assert block is not None, f"{EXAMPLES_GUIDE} shows no fetch commands under Running them"
    return block[1].rstrip("\n").splitlines()


def exit_codes() -> list[tuple[str, str]]:
    """The exit codes the command line uses, from the examples' guide."""
    return re.findall(r"^\| (\d) \| (.+?) \|$", _section(EXAMPLES_GUIDE, "Running them"), re.M)


def printed_first_line(rung_heading: str) -> str:
    """The first line a rung's run prints, as ``docs/ladder.md`` shows it."""
    block = re.search(r"```text\n(.*?)\n", _section(PACKAGE_ROOT / "docs" / "ladder.md", rung_heading), re.S)
    assert block is not None, f"docs/ladder.md shows no output under {rung_heading}"
    return block[1]


def agent_file(surface: Surface) -> str:
    """``llms.txt``: what an AI agent adopting the package reads first."""
    return "\n".join(_agent_lines(surface)).rstrip() + "\n"


def _wrapped(text: str, indent: str = "") -> Iterator[str]:
    yield from textwrap.wrap(
        text, _WIDTH, initial_indent=indent, subsequent_indent=f"{indent}  ", break_on_hyphens=False
    )


def _names(names: list[str]) -> str:
    return ", ".join(short(name) for name in names)


def _agent_lines(surface: Surface) -> Iterator[str]:
    by_rung = surface.typed_by_rung()
    first = surface.first_eval()
    yield "3tears-evals: read this first"
    yield ""
    yield from _wrapped(
        f"Generated by {REGENERATE}; do not edit. Names drop the threetears.evals. prefix. <tag> below is v and "
        "the installed version (threetears.evals.__version__)."
    )
    yield ""
    yield "WHAT IT IS"
    yield from _wrapped(" ".join(what_it_is()))
    yield ""
    yield f"A FIRST EVAL NEEDS AT MOST {len(first)} NAMES"
    for name in first:
        yield from _wrapped(_plain(f"{short(name)}: {summary(name, surface.objects[name])}"), "  ")
    yield from _wrapped("An app that calls no model needs Eval alone: its runner is the app's own function.", "  ")
    yield ""
    yield "YOUR APP'S CASES"
    yield from _wrapped(
        "Start from the cases the app already has: the data files its tests read, the tables of inputs and expected "
        "outputs in its test code, the examples its README or docs give. A .jsonl file (an object a line), a .json "
        "list of objects, a .csv with a header row or a .toml [[cases]] table loads as the eval's cases. Take each "
        "expected value from the app's documentation or specification, never by running the app or reading its code "
        "and copying what it gives: a case that expects what the app does today cannot catch what it does wrong. One "
        'case, whole: {"id": "double-charge", "email": "I was billed twice this month.", "category": "billing"}. The '
        "package keeps a case's id, stratum and world keys: it reads them and hands the runner and every grader the "
        "case without them."
    )
    yield ""
    relative = FIRST_EXAMPLE.relative_to(PACKAGE_ROOT / "examples")
    yield f"THE FIRST EXAMPLE, WHOLE ({relative} of the examples fetched below; its docstring left out)"
    yield from (f"  {line}" if line else "" for line in example_program())
    for note in example_notes():
        yield from _wrapped(note)
    yield ""
    yield "GETTING THE EXAMPLES: they are not in the installed package; fetch them from the repository"
    yield from (f"  {line}" for line in fetch_commands())
    yield from _wrapped(
        "Then run one from that fetched directory: python -m threetears.evals 1-triage/triage.py. Each rung reads "
        "its cases and replays its recordings from beside itself, so keep its directory together."
    )
    yield ""
    yield "READING WHAT COMES BACK"
    yield from _wrapped("The first line printed is the answer. Rung 1 prints:")
    yield from _wrapped(printed_first_line("Rung 1: your own function"), "  ")
    yield from _wrapped("A comparison's first line: where v2 stands, and how strong the evidence is. Rung 3a prints:")
    yield from _wrapped(printed_first_line("Rung 3a: two prompts, compared"), "  ")
    yield from _wrapped(
        "Run again into its store with cases appended, a comparison continues (only new cases run) and its pass-rate "
        "and ranged-score verdicts keep their error at every reading; a runner or grader is known by its name (a "
        "judge by model and criteria), not its code: rename the eval to compare an edit."
    )
    yield from _wrapped(
        "Below it: each measure's rate, the top misses with why each missed, and where the report page is. A "
        "refusal before anything runs goes to stderr with exit 2; an action that refuses answers with refused: "
        "and a remedy: saying what to do instead."
    )
    yield from _wrapped(f"Exit codes: {'; '.join(f'{code} {meaning}' for code, meaning in exit_codes())}.", "  ")
    yield ""
    yield f"LATER TASKS, BY RUNG ({at_tag('docs/ladder.md')})"
    for rung, names in by_rung.items():
        if rung > 1:
            yield from _wrapped(f"Rung {rung}, {surface.rungs[rung]}: {_names(names) or 'no new name'}", "  ")
    off_ladder = [name for name in surface.entries if name in TYPED_IN_DOCS]
    yield from _wrapped(
        f"Off the ladder, taught on a page: {'; '.join(f'{short(n)} ({at_tag(TYPED_IN_DOCS[n])})' for n in off_ladder)}",
        "  ",
    )
    yield ""
    ports = [name for name in surface.of_kind("bound") if protocol_sides().get(name) == HOST_IMPLEMENTS]
    yield f"EMBEDDING IT IN AN APP ({at_tag('docs/adopting-a-host.md')})"
    yield from _wrapped(
        f"A product builds one kernel.host.EvalHost and hands it to every entrypoint. The {len(ports)} ports a host "
        f"may implement: {_names(ports)}.",
        "  ",
    )
    yield from _wrapped(f"Every declared name, with one line each: {at_tag('docs/surface.md')}", "  ")
    yield ""
    yield f"NEEDS A SHIPPED EXAMPLE ({REGISTER_ISSUE}): the only door to each; no shipped example uses it yet"
    for name in AWAITING_SHIPPED_USE:
        yield from _wrapped(_plain(f"{short(name)}: {AWAITING_SHIPPED_USE[name]}"), "  ")
    yield ""
    actions = _actions()
    yield f"ACTIONS: {sum(len(each) for _, each in actions)}, called without writing Python"
    yield from _wrapped(
        f"actions.eval_catalogue() holds them; transports.fastmcp.mount_fastmcp serves them over MCP "
        f"({at_tag('docs/serving.md')}).",
        "  ",
    )
    for workflow, each in actions:
        yield from _wrapped(f"{workflow}: {', '.join(name for name, _ in each)}", "  ")
    yield ""
    # The surface page's address is given once, under EMBEDDING IT IN AN APP.
    yield from _wrapped("KINDS: every declared name is one; the surface page named above lists each under its kind")
    for kind in KINDS:
        yield from _wrapped(f"{kind}: {KIND_MEANINGS[kind]}.", "  ")


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
