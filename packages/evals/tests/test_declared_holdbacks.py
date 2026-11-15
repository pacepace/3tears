"""The hold-back ledger is held to the tree, and every declared name is used (admission criterion 2).

Criterion 2 (owner ruling, 2026-11-15): production code or a shipped example uses a public name; a test alone does
not count, and neither does a page's prose. A declared name is used when code names it:

- another package's source in this repository (``packages/*/src``, the evals package's own left out), and the
  shipped completion client (``threetears.evals.providers``), production code that ``tests/test_providers_boundary.py``
  holds to the public roots exactly as a host is held;
- a shipped example: the ladder's examples under ``examples/`` and the courier host (``tests/fixtures/courierhost/``,
  the reference host the README and the host guide point a reader at). The toy host does not count: it exercises
  every shape of the contract on purpose (:mod:`test_api_tiers` says why);
- a Python code block on a shipped page that imports the name, reads it off its module, calls it, decorates with it
  or otherwise refers to it as code. A word in prose, in a comment, in a string or in a text block is not use;
- a host's production code, as ``tests/fixtures/host_imports/`` records it (generated from a checkout of the host by
  ``scripts/record_host_imports.py``, which states what production is there; a host's tests never count).

A name whose kind needs no caller of its own is held by criterion 1 instead: a ``received`` name (reached through a
signature, which :mod:`test_api_tiers` holds), a protocol a host implements, a type a declared signature names (a
shape, an alias, an exception or a constant comes with the operation or port that names it), and the function behind
an action of the catalogue, which an agent invokes through the catalogue a host serves.

The ledger, ``tests/fixtures/declared_holdbacks.txt``, records of each declared protocol who implements it, one
``<name> | <reason>`` per line:

- ``port-host-implements``: a declared protocol or abstract class a host's code implements and hands the engine (a
  store, a client, a kind, a pricer).
- ``protocol-engine-provides``: a declared protocol the engine implements and hands a host's code (a cell's sink, a
  delivery's recorder), or that the engine's own storage satisfies over the host's store. Which of the two a protocol
  is, is the ledger's judgement; this holds that every declared protocol is one or the other.
- ``unreached``: declared, and used by nothing. There are none, and this holds that.

A declared signature reaches a name when the name is a word of another declared name's signature, of one of its
public methods or properties, or of a ``:raises:`` line in their docstrings.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import io
import re
import subprocess
import textwrap
import tokenize
import typing
import warnings
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

from packages.evals.tests import test_guide_names_are_declared as pages
from packages.evals.tests.script_loader import load_script
from packages.evals.tests.test_host_imports_are_declared import host_imports

surface = load_script("public_surface.py")
host_record = load_script("record_host_imports.py")
kinds_of = load_script("surface_kinds.py")

PACKAGE = Path(__file__).resolve().parents[1]

#: The repository, whose other packages' sources are read for imports.
REPO = PACKAGE.parents[1]

#: The ledger this holds.
LEDGER = PACKAGE / "tests" / "fixtures" / "declared_holdbacks.txt"

HOST_IMPLEMENTS = "port-host-implements"
ENGINE_PROVIDES = "protocol-engine-provides"
UNREACHED = "unreached"

#: Every reason a ledger line may give.
REASONS = (HOST_IMPLEMENTS, ENGINE_PROVIDES, UNREACHED)

#: The reasons a declared protocol is listed under.
PROTOCOL_REASONS = (HOST_IMPLEMENTS, ENGINE_PROVIDES)

#: The pages generated from the declared names themselves, which list every one by construction: never a use.
GENERATED_PAGES = ("reference.md", "surface.md")
#: A fenced code block on a page: its language, and its body.
_BLOCK = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.M | re.S)
#: The languages a block of Python code is fenced with (an unfenced language is read only when it parses).
_PYTHON = ("python", "py")
#: One ``from threetears.evals… import …`` statement, parenthesised across lines or not.
_IMPORT = re.compile(r"^[ \t]*from threetears\.evals(?:\.\w+)* import (?:\([^)]*\)|[^\n]*)", re.M)
#: An exception a docstring says a callable raises.
_RAISES = re.compile(r":raises\s+~?([\w.]+)\s*:")
#: The snapshot kinds of a name that does work when called or built: it needs a caller of its own.
_OPERATIONS = ("function", "async function", "class")
#: The function that builds the action catalogue: an action is used wherever it is.
CATALOGUE = "threetears.evals.actions.eval_catalogue"

#: Declared names that fail criterion 2 and await a ruling on the scope of their demotion (package V, fix round 2,
#: 2026-11-15): each is a function or class that only tests (the package's or a host's), the toy host or a page's
#: prose use. Demoting them rewrites what six pages teach and the imports of 188 test files, past that round's brief.
#: The set only shrinks: :func:`test_a_name_awaiting_demotion_is_still_unused` fails once one is used or no longer
#: declared, and a name leaves it by being demoted. Nothing tells an author to list a name here.
AWAITING_DEMOTION: frozenset[str] = frozenset(
    {
        "threetears.evals.actions.render_jobs_started",
        "threetears.evals.analysis.assemble_context_bundle",
        "threetears.evals.analysis.gate_verdicts",
        "threetears.evals.analysis.report_html",
        "threetears.evals.kernel.CappedCompletion",
        "threetears.evals.kernel.failed_call_output",
        "threetears.evals.kernel.host.MemberActsOn",
        "threetears.evals.kernel.host.Ordinal",
        "threetears.evals.kernel.host.load_bar_registry",
        "threetears.evals.negotiation.closest_settings",
        "threetears.evals.ops.await_job",
        "threetears.evals.ops.case_rename",
        "threetears.evals.ops.start_action_job",
        "threetears.evals.quick.run_cli",
        "threetears.evals.run.JudgeService",
        "threetears.evals.run.LaunchGroup",
        "threetears.evals.run.admit_template",
        "threetears.evals.run.launch_as_group",
        "threetears.evals.run.load_seed_corpus",
        "threetears.evals.run.mint_case_set",
        "threetears.evals.run.quote_launch",
        "threetears.evals.run.regrade",
        "threetears.evals.run.start_run",
        "threetears.evals.testing.RecordedPrompt",
        "threetears.evals.testing.check_completion_conformance",
        "threetears.evals.transports.fastmcp.CatalogueTool",
    }
)


def read_ledger() -> dict[str, str]:
    """Each name the ledger lists, with its reason."""
    found: dict[str, str] = {}
    for raw in LEDGER.read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.startswith("#"):
            name, reason = (part.strip() for part in raw.split("|"))
            found[name] = reason
    return found


@cache
def declared_names() -> frozenset[str]:
    """Every name the snapshot records ``host-contract`` or ``stable``."""
    return frozenset(name for name, entry in surface.load_snapshot().items() if surface.declared(entry.tier))


def public_object(name: str) -> Any:
    """What ``root.name`` is, read off its root. A one-minor shim warns on the read; the warning is not this test's."""
    root, attribute = name.rsplit(".", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return getattr(importlib.import_module(root), attribute)


def code_files() -> Iterator[Path]:
    """The code whose use counts, besides the pages and a host's list: another package's source, the shipped
    completion client, the ladder's examples and the courier host."""
    yield from sorted((PACKAGE / "src" / "threetears" / "evals" / "providers").rglob("*.py"))
    yield from (
        path
        for path in sorted((REPO / "packages").glob("**/src/**/*.py"))
        if PACKAGE not in path.parents and "threetears.evals" in path.read_text(encoding="utf-8")
    )
    yield from sorted((PACKAGE / "examples").rglob("*.py"))
    yield from sorted((PACKAGE / "tests" / "fixtures" / "courierhost").rglob("*.py"))


@cache
def used_in_code() -> frozenset[str]:
    """Every package name another package's source, the shipped client, an example or the courier host imports or
    reads off a module."""
    return frozenset(
        name for path in code_files() for name in host_record.used_names(path.read_text(encoding="utf-8"), str(path))
    )


def _by_word(names: Iterable[str]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for name in names:
        found.setdefault(name.rsplit(".", 1)[1], set()).add(name)
    return found


class _PageReader:
    """What a page's Python code blocks use, read in page order: an import binds a name (or a module) for every block
    after it, and a name the page spells qualified or attributed in prose binds it too."""

    def __init__(self, declared: frozenset[str], taught: Iterable[str]) -> None:
        self.by_word = _by_word(declared)
        self.names: dict[str, str] = {name.rsplit(".", 1)[1]: name for name in taught}
        self.modules: dict[str, str] = {}
        self.used: set[str] = set()

    def _bind(self, statement: ast.ImportFrom | ast.Import) -> None:
        if isinstance(statement, ast.ImportFrom):
            if not (statement.module or "").startswith("threetears.evals"):
                return
            for alias in statement.names:
                qualified = f"{statement.module}.{alias.name}"
                self.used.add(qualified)
                if host_record.is_module(qualified):
                    self.modules[alias.asname or alias.name] = qualified
                else:
                    self.names[alias.asname or alias.name] = qualified
        else:
            for alias in statement.names:
                if alias.name.startswith("threetears.evals"):
                    self.modules[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )

    def _refer(self, word: str) -> None:
        """A word used as code: the name the page bound to it, else every declared name it spells."""
        self.used |= {self.names[word]} if word in self.names else self.by_word.get(word, set())

    def _attribute(self, base: str, attribute: str) -> None:
        head, _, rest = base.partition(".")
        module = f"{self.modules[head]}{'.' + rest if rest else ''}" if head in self.modules else base
        if module.startswith("threetears.evals"):
            self.used.add(f"{module}.{attribute}")

    def read(self, language: str, body: str) -> None:
        """Read one fenced block. A block that parses is read whole; a Python block that does not (an elision a
        reader fills in) is read by its tokens: its imports, its calls, its decorators and what it reads off a
        module. A block in another language, or an unlabelled one that is not Python, is not code."""
        if language not in (*_PYTHON, ""):
            return
        try:
            tree = ast.parse(body)
        except SyntaxError:
            if language in _PYTHON:
                self._read_tokens(body)
            return
        local: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._bind(node)
            elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                local.add(node.id)
            elif isinstance(node, ast.arg):
                local.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local.add(node.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in local:
                self._refer(node.id)
            elif isinstance(node, ast.Attribute) and (base := host_record.dotted(node.value)) is not None:
                self._attribute(base, node.attr)

    def _read_tokens(self, body: str) -> None:
        for statement in _IMPORT.finditer(body):
            for node in ast.walk(ast.parse(textwrap.dedent(statement[0]))):
                if isinstance(node, ast.ImportFrom):
                    self._bind(node)
        tokens = [
            token
            for token in tokenize.generate_tokens(io.StringIO(body).readline)
            if token.type in (tokenize.NAME, tokenize.OP)
        ]
        for index, token in enumerate(tokens):
            before = tokens[index - 1].string if index else ""
            after = tokens[index + 1].string if index + 1 < len(tokens) else ""
            if token.type != tokenize.NAME or before == ".":
                continue
            if after == "." and index + 2 < len(tokens) and token.string in self.modules:
                self._attribute(token.string, tokens[index + 2].string)
            elif after == "(" or before == "@":
                self._refer(token.string)


def used_in_page(text: str, *, declared: frozenset[str], taught: Iterable[str] = ()) -> frozenset[str]:
    """Every package name one page's Python code blocks use. ``taught`` is what the page spells qualified or
    attributed in prose, which a block's bare word then refers to."""
    reader = _PageReader(declared, taught)
    for block in _BLOCK.finditer(text):
        reader.read(block[1], textwrap.dedent(block[2]))
    return frozenset(reader.used)


@cache
def used_on_a_page() -> frozenset[str]:
    """Every package name a shipped page's code blocks use."""
    declared = declared_names()
    return frozenset(
        name
        for page in pages.PAGES
        if page.name not in GENERATED_PAGES
        for name in used_in_page(page.read_text(encoding="utf-8"), declared=declared, taught=pages.taught_names(page))
    )


@cache
def used_by_a_host() -> frozenset[str]:
    """Every package name a host's production code uses, as its generated list records it."""
    return frozenset(name for _, name in host_imports())


def _raised(member: Any) -> str:
    return " ".join(_RAISES.findall(inspect.getdoc(member) or ""))


def signature_words(name: str) -> set[str]:
    """Every word of one declared name's signature: the snapshot's, its public methods' and properties', and the
    exceptions their docstrings say they raise."""
    obj = public_object(name)
    texts = [surface.load_snapshot()[name].signature]
    if inspect.isclass(obj):
        for owner in obj.__mro__:
            if not owner.__module__.startswith("threetears."):
                continue
            for attribute, member in vars(owner).items():
                if attribute.startswith("_") and attribute not in ("__init__", "__call__"):
                    continue
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if isinstance(member, property) and member.fget is not None:
                    texts += [str(inspect.signature(member.fget).return_annotation), _raised(member.fget)]
                elif inspect.isfunction(member):
                    texts += [surface.signature(member, name=attribute), _raised(member)]
    elif callable(obj):
        texts.append(_raised(obj))
    return set(re.findall(r"\w+", " ".join(texts)))


@cache
def reached_by_a_signature() -> frozenset[str]:
    """Every declared name another declared name's signature reaches."""
    declared = declared_names()
    reachers: dict[str, set[str]] = {}
    for name in declared:
        for word in signature_words(name):
            reachers.setdefault(word, set()).add(name)
    return frozenset(name for name in declared if reachers.get(name.rsplit(".", 1)[1], set()) - {name})


@cache
def ports() -> frozenset[str]:
    """Every declared protocol or abstract class: what a host implements."""
    return frozenset(
        name
        for name in declared_names()
        if inspect.isclass(obj := public_object(name)) and (typing.is_protocol(obj) or inspect.isabstract(obj))
    )


@cache
def needs_no_caller() -> frozenset[str]:
    """Every declared name criterion 1 holds instead of criterion 2, by its kind: a received name, a protocol a host
    implements, a type a declared signature names, and an action's function wherever the catalogue is used."""
    kinds = kinds_of.declared_kinds(surface.load_snapshot()).of
    recorded = surface.load_snapshot()
    sides = read_ledger()
    catalogue_used = CATALOGUE in used_in_code() | used_on_a_page() | used_by_a_host()
    return frozenset(
        name
        for name in declared_names()
        if kinds.get(name) == "received"
        or sides.get(name) == HOST_IMPLEMENTS
        or (recorded[name].kind not in _OPERATIONS and name in reached_by_a_signature())
        or (kinds.get(name) == "action" and catalogue_used)
    )


def unused() -> set[str]:
    """Every declared name nothing uses and nothing holds instead (criterion 2)."""
    return set(declared_names() - used_in_code() - used_on_a_page() - used_by_a_host() - needs_no_caller())


def test_the_ledger_is_one_sorted_line_a_name() -> None:
    lines = [line for line in LEDGER.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    assert lines, f"{LEDGER} lists nothing, so the tests below say nothing"
    assert lines == sorted(lines), f"{LEDGER} is not sorted"
    ledger = read_ledger()
    assert len(ledger) == len(lines), f"{LEDGER} lists a name twice"
    unknown = {name: reason for name, reason in ledger.items() if reason not in REASONS}
    assert not unknown, f"{LEDGER} gives reasons outside {REASONS}: {unknown}"


def test_every_name_the_ledger_lists_is_declared() -> None:
    gone = sorted(set(read_ledger()) - declared_names())
    assert not gone, (
        f"{LEDGER} lists names that are no longer declared (provisional or removed): drop their lines\n  "
        + "\n  ".join(gone)
    )


def test_the_readers_find_the_names_they_are_written_for() -> None:
    assert "threetears.evals.testing.NegotiationConformanceFailure" in reached_by_a_signature(), (
        "the signature reader missed a name only a :raises: line reaches"
    )
    assert "threetears.evals.quick.WorldRunner" in reached_by_a_signature(), (
        "the signature reader missed a name only a method's return reaches"
    )
    assert "threetears.evals.testing.STORE_CONFORMANCE_CASES" in used_on_a_page(), (
        "the page reader missed a name a code block passes as a value"
    )
    assert "threetears.evals.testing.StoreConformanceCase" in used_on_a_page(), (
        "the page reader missed a name a code block annotates with"
    )
    assert "threetears.evals.ops.RegressionWatch" in used_on_a_page(), (
        "the page reader missed a name a block that is not whole Python calls"
    )
    assert "threetears.evals.kernel.host.EvalHost" in used_in_code(), "the code reader found nothing"
    assert "threetears.evals.kernel.extract_json" in used_by_a_host(), "the host's list is empty or stale"
    assert "threetears.evals.schema.DocumentStore" in ports(), "the port reader missed the store port"


def test_a_page_s_prose_and_words_are_not_use() -> None:
    """A name a page spells in prose, writes in a comment, a string or a text block, or that only a test imports, is
    not used; a block that imports it, calls it or reads it off its module uses it."""
    declared = frozenset({"threetears.evals.kernel.extract_json", "threetears.evals.quick.measure"})
    words = textwrap.dedent(
        """
        Call `threetears.evals.kernel.extract_json` (or `measure`) on the reply.

        ```text
        measure extract_json
        ```

        ```python
        # extract_json is how a host would measure it
        reply = "extract_json(measure)"
        ```
        """
    )
    assert used_in_page(words, declared=declared, taught=declared) == frozenset()
    for code in (
        "from threetears.evals.kernel import extract_json\n",
        "from threetears.evals import kernel\nkernel.extract_json(reply)\n",
        "value = extract_json(reply)\n",
        "@measure(higher_is_better=True)\ndef score(answer: str) -> float: ...\n",
    ):
        used = used_in_page(f"```python\n{code}```\n", declared=declared)
        assert used & declared, f"the page reader missed the use in {code!r}"


def test_a_host_s_tests_are_not_its_production(tmp_path: Path) -> None:
    """The host list's generator records a name its package source imports and never one only its tests import."""
    checkout = tmp_path / "host"
    (checkout / "app").mkdir(parents=True)
    (checkout / "tests").mkdir()
    (checkout / "app" / "wiring.py").write_text("from threetears.evals.kernel import extract_json\n")
    (checkout / "app" / "test_wiring.py").write_text("from threetears.evals.kernel import seed_digest\n")
    (checkout / "tests" / "test_app.py").write_text("from threetears.evals.run import start_run\n")
    (checkout / "conftest.py").write_text("from threetears.evals.kernel import confusion_cell\n")
    for command in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "host"],
    ):
        subprocess.run(["git", "-C", str(checkout), *command], check=True)
    assert host_record.host_uses(checkout) == {"threetears.evals.kernel.extract_json"}


def test_the_ledger_lists_every_declared_protocol_once() -> None:
    listed = {name for name, reason in read_ledger().items() if reason in PROTOCOL_REASONS}
    assert listed == ports(), (
        f"{LEDGER}'s {' and '.join(PROTOCOL_REASONS)} lines are not the declared protocols and abstract classes. "
        f"Not listed: {sorted(ports() - listed)}. Not a declared protocol: {sorted(listed - ports())}."
    )


def test_every_declared_name_is_used() -> None:
    """Criterion 2: production code or a shipped example uses each declared name; a test alone does not count."""
    listed = {name for name, reason in read_ledger().items() if reason == UNREACHED}
    unused_now = unused() - AWAITING_DEMOTION
    assert not unused_now and not listed, (
        "no production code or shipped example uses these declared names, and a test or a page's prose does not "
        "count (admission criterion 2, owner ruling 2026-11-15). Make each private: drop it from its root's "
        f"__all__, record it removed in MIGRATIONS.toml and run `{surface.REGENERATE_SNAPSHOT}`. Or show its real "
        "use: production code in another package, a shipped example, a code block on a shipped page that uses it, or "
        "a host's production code (re-record the host's list with packages/evals/scripts/record_host_imports.py)\n  "
        + "\n  ".join(sorted(unused_now | listed))
    )


def test_a_name_awaiting_demotion_is_still_unused() -> None:
    """The set awaiting a ruling only shrinks: a listed name something now uses, or that is no longer declared,
    leaves it."""
    stale = sorted(AWAITING_DEMOTION - unused())
    assert not stale, "drop these from AWAITING_DEMOTION: each is used now, or no longer declared\n  " + "\n  ".join(
        stale
    )


def test_the_surface_page_gives_the_ledgers_counts() -> None:
    page = (PACKAGE / "docs" / "surface.md").read_text(encoding="utf-8")
    assert "../tests/fixtures/declared_holdbacks.txt" in page, "docs/surface.md does not point at the ledger"
    reasons = list(read_ledger().values())
    rows = dict(re.findall(r"^\| `([a-z-]+)` \|[^|\n]*\| (\d+) \|$", page, re.M))
    assert rows == {reason: str(reasons.count(reason)) for reason in REASONS}, (
        f"docs/surface.md's hold-back table is not the ledger's counts: {rows}"
    )
