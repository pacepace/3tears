"""The hold-back ledger is held to the tree, so it cannot rot silently.

A declared name stays declared while something imports it, a shipped page names it or a declared signature reaches
it (``docs/surface.md`` states the rule). The ledger, ``tests/fixtures/declared_holdbacks.txt``, records the declared
names whose reason the tree alone does not show, one ``<name> | <reason>`` per line:

- ``discodon-tests-only``: Discodon's tests import it and nothing else does; no shipped page names it and no
  declared signature reaches it. Which of Discodon's imports are its tests' is read from Discodon, not from here
  (``tests/fixtures/host_imports/discodon.txt`` holds its imports from both), so this holds the rest: nothing in this
  repository imports the name, no page names it, no declared signature reaches it, and it is not a port.
- ``port``: a protocol or an abstract class, which a host implements. Every declared one is listed, whatever else
  holds it.
- ``unreached``: declared, and nothing imports it, names it or reaches it. There are none, and this holds that.

A page is read as :mod:`test_guide_names_are_declared` reads it, qualified, attributed and bare, without the lines
that only list the declared names (the reference's entries and ``docs/surface.md``'s rungs), since those name every
declared name by construction. A declared signature reaches a name when the name is a word of another declared
name's signature, of one of its public methods or properties, or of a ``:raises:`` line in their docstrings.
"""

from __future__ import annotations

import importlib
import inspect
import re
import typing
import warnings
from functools import cache
from pathlib import Path
from typing import Any

from packages.evals.tests import test_api_tiers as tiers
from packages.evals.tests import test_guide_names_are_declared as pages
from packages.evals.tests.script_loader import load_script
from packages.evals.tests.test_host_imports_are_declared import host_imports

surface = load_script("public_surface.py")

PACKAGE = Path(__file__).resolve().parents[1]

#: The repository, whose other packages' sources are read for imports.
REPO = PACKAGE.parents[1]

#: The ledger this holds.
LEDGER = PACKAGE / "tests" / "fixtures" / "declared_holdbacks.txt"

DISCODON_TESTS_ONLY = "discodon-tests-only"
PORT = "port"
UNREACHED = "unreached"

#: Every reason a ledger line may give.
REASONS = (DISCODON_TESTS_ONLY, PORT, UNREACHED)

#: A reference entry's own name, which the reference prints for every declared name.
_REFERENCE_ENTRY = re.compile(r"^- \*\*`[^`]+`\*\*", re.M)
#: A reference entry's own signature, on the line under it.
_REFERENCE_SIGNATURE = re.compile(r"^\s*<br>`.*$", re.M)
#: One root's line on ``docs/surface.md``, which lists that root's declared names.
_SURFACE_ROOT = re.compile(r"^- `[\w.]+`: .*$", re.M)
#: An exception a docstring says a callable raises.
_RAISES = re.compile(r":raises\s+~?([\w.]+)\s*:")


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


@cache
def imported_in_repo() -> frozenset[str]:
    """Every public name something in this repository imports from outside the package's own source: a shipped page,
    an example, the courier and toy hosts, another package's source."""
    readers = {*pages.PAGES, *tiers.HOST_GUIDES, *tiers.SHOWN, *tiers.EXERCISER}
    others = (
        path
        for path in (REPO / "packages").glob("**/src/**/*.py")
        if PACKAGE not in path.parents and "threetears.evals" in path.read_text(encoding="utf-8")
    )
    return frozenset(name for path in (*readers, *others) for name in tiers.imported_names(path))


def page_text(page: Path) -> str:
    """A page's text with the lines that only list declared names blanked, every line kept in place."""
    text = page.read_text(encoding="utf-8")
    if page.name == "reference.md":
        text = _REFERENCE_SIGNATURE.sub("", _REFERENCE_ENTRY.sub("-", text))
    if page.name == "surface.md":
        text = _SURFACE_ROOT.sub("", text)
    return text


@cache
def named_on_a_page() -> frozenset[str]:
    """Every declared name a shipped page names: spelled in full, attributed to its root, or spelled bare."""
    declared = declared_names()
    by_word: dict[str, set[str]] = {}
    for name in declared:
        by_word.setdefault(name.rsplit(".", 1)[1], set()).add(name)
    named: set[str] = set()
    for page in pages.PAGES:
        text = page_text(page)
        named |= declared & set(pages.taught_names(page, text=text))
        for word in pages.bare_names(page, text=text):
            named |= by_word.get(word, set())
    return frozenset(named)


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
    assert "threetears.evals.quick.EXIT_OK" in named_on_a_page(), "the page reader missed a bare name"
    assert "threetears.evals.kernel.host.EvalHost" in imported_in_repo(), "the import reader found nothing"
    assert "threetears.evals.schema.DocumentStore" in ports(), "the port reader missed the store port"


def test_the_ledger_lists_every_declared_port() -> None:
    listed = {name for name, reason in read_ledger().items() if reason == PORT}
    assert listed == ports(), (
        f"{LEDGER}'s ports are not the declared protocols and abstract classes. "
        f"Not listed: {sorted(ports() - listed)}. Not a declared port: {sorted(listed - ports())}."
    )


def test_a_name_only_discodons_tests_import_is_held_by_nothing_else() -> None:
    """Otherwise it is held by what holds it, and Discodon's re-pin would not free it."""
    held = {
        "imported in this repository": imported_in_repo(),
        "named on a shipped page": named_on_a_page(),
        "reached by a declared signature": reached_by_a_signature(),
        "a port": ports(),
    }
    wrong = [
        f"{name}: {why}"
        for name, reason in sorted(read_ledger().items())
        if reason == DISCODON_TESTS_ONLY
        for why, names in held.items()
        if name in names
    ]
    assert not wrong, (
        f"{LEDGER} gives these {DISCODON_TESTS_ONLY}, but something else holds them: drop their lines\n  "
        + "\n  ".join(wrong)
    )


def test_no_declared_name_is_unreached() -> None:
    ledger = read_ledger()
    discodon = {name for _, name in host_imports()} | {n for n, r in ledger.items() if r == DISCODON_TESTS_ONLY}
    unreached = (
        declared_names() - imported_in_repo() - discodon - named_on_a_page() - reached_by_a_signature() - ports()
    )
    listed = {name for name, reason in ledger.items() if reason == UNREACHED}
    assert not unreached and not listed, (
        "these declared names are imported by nothing, named on no shipped page and reached by no declared signature: "
        f"demote each (drop it from HOST_CONTRACT or STABLE in packages/evals/scripts/public_surface.py and run "
        f"`{surface.REGENERATE_SNAPSHOT}`), or, when only Discodon's tests import it, list it "
        f"{DISCODON_TESTS_ONLY} in {LEDGER}\n  " + "\n  ".join(sorted(unreached | listed))
    )


def test_the_surface_page_gives_the_ledgers_counts() -> None:
    page = (PACKAGE / "docs" / "surface.md").read_text(encoding="utf-8")
    assert "../tests/fixtures/declared_holdbacks.txt" in page, "docs/surface.md does not point at the ledger"
    reasons = list(read_ledger().values())
    rows = dict(re.findall(r"^\| `([a-z-]+)` \|[^|\n]*\| (\d+) \|$", page, re.M))
    assert rows == {reason: str(reasons.count(reason)) for reason in REASONS}, (
        f"docs/surface.md's hold-back table is not the ledger's counts: {rows}"
    )
