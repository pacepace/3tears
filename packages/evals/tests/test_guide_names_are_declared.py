"""Every name a shipped page tells a reader to use is declared: ``host-contract`` or ``stable``, never ``provisional``.

The pages are every page under ``docs/``, the package README and the examples' README. Each names things in prose
that a reader is to call, build, catch or match, and a package that teaches a name promises it.

Two spellings are read:

- **Qualified or attributed.** A name spelled in full (`` `threetears.evals.run.definition_drift` ``) or attributed
  to a module in parentheses (`` `strict_response_format` (`threetears.evals.kernel`) ``, `` `A` and `B` (in
  `threetears.evals.kernel`) ``) must be declared.
- **Bare.** A name spelled alone (`` `run_cli(argv, ...)` ``, `` `@grader(...)` ``, `` `EXIT_OK` ``) may also be a
  field, a parameter or another root's name. So a bare name is held to stay declared rather than to be declared: if
  the newest release declared it, it must not go provisional while a page still names it.

A provisional name a page marks provisional where it names it (`` `@measure(...)`, provisional ``, or ``provisional``
inside the parenthesis attributing it) is not taught as a promise, so neither rule reads it there (owner ruling R-c,
2026-11-15: a page must not teach a provisional name as if it were stable).

A name ``MIGRATIONS.toml`` records leaving a public root, and no root exports now, must not be named as code on any
page: spelled in full or attributed, called, decorated with, spelled alone, or used in a block of Python. A bare word
that is still something else (a public name's, or a standard measure's key) is that, not the removed name; one a
page marks removed where it names it (`` `a` and `b`, all removed ``, `` `a` (retired) ``) records its removal.

The import statements the pages show are held by ``test_api_tiers.py``. A name a page mentions that no public root
exports (a module, a logger, a field) is not a name this reads.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from threetears.evals.kernel import METRIC_DESCRIPTORS

from packages.evals.tests.script_loader import load_script

surface = load_script("public_surface.py")

PACKAGE = Path(__file__).resolve().parents[1]

#: The host guide, whose readers this was first written for.
GUIDE = PACKAGE / "docs" / "adopting-a-host.md"

#: Every page the package ships: each page under ``docs/``, the README and the examples' README.
PAGES: tuple[Path, ...] = (
    *sorted((PACKAGE / "docs").glob("*.md")),
    PACKAGE / "README.md",
    PACKAGE / "examples" / "README.md",
)

#: The released snapshot the bare spellings are held against, as a path inside the repository.
SNAPSHOT_IN_REPO = "packages/evals/api/surface.json"

_FENCE = re.compile(r"^```.*?^```", re.M | re.S)
_SPAN = re.compile(r"`([^`\n]+)`")
_QUALIFIED = re.compile(r"^(threetears\.evals(?:\.\w+)+)")
_MODULE = re.compile(r"^threetears\.evals(?:\.\w+)*$")
_BARE = re.compile(r"^@?(\w+)(?:\(.*\))?$")
#: A bare name as prose spells it: alone, called, decorated with, or with a member read off it.
_BARE_SPELLING = re.compile(r"^@?(\w+)(?:\(.*\)|\.\w+.*)?$")
#: The mark a page puts right after a provisional name it names: ``, provisional`` or a parenthesis saying so.
_PROVISIONAL_MARK = re.compile(r"\s*(?:\([^()\n]*\bprovisional\b[^()\n]*\)|,\s*provisional\b)")
#: The mark after a removed name (or the run of names it ends) that says so: ``(removed)``, ``, all retired``.
_REMOVAL_MARK = re.compile(r"(?:(?:,|,? and|,? or)\s*`[^`\n]+`)*\s*[(,]\s*(?:all\s+)?(?:removed|retired)\b")
#: A fenced block of Python on a page, and its body.
_PYTHON_BLOCK = re.compile(r"^```(?:python|py)[^\n]*\n(.*?)^```", re.M | re.S)
#: What a word in a block of Python is not: a string or a comment.
_NOT_CODE = re.compile(r"\"[^\"\n]*\"|'[^'\n]*'|#[^\n]*")
#: A parenthesis naming a module: ``(in `m`)``, ``(from `m`)``, ``(`m`)``, ``(host contract, from `m`; ...)``.
_ATTRIBUTION = re.compile(r"\([^()`\n]*`(threetears\.evals(?:\.\w+)*)`[^()\n]*\)")
#: The run of backticked names a parenthesis qualifies: ``a``, ``b`` and ``c``, directly before it.
_NAMES_BEFORE = re.compile(r"((?:`[^`\n]+`(?:'s)?(?:, | and |, and | or ))*`[^`\n]+`(?:'s)?)\s*$")
#: The names a root is followed by in parentheses: ``threetears.evals.ops`` (`runs_list`, `report_read`).
_NAMES_AFTER = re.compile(r" *\(((?:`[^`\n]+`(?:, | and |, and )?)+)\)")


def _bare_names(spans: str, root: str) -> set[str]:
    return {
        f"{root}.{bare[1]}"
        for span in _SPAN.finditer(spans)
        if (bare := _BARE.match(span[1])) and not _MODULE.match(span[1])
    }


def _prose(text: str) -> str:
    """A page's prose with its fenced blocks blanked and each paragraph on one line, every offset kept."""
    unfenced = _FENCE.sub(lambda block: re.sub(r"[^\n]", " ", block[0]), text)
    return re.sub(r"(?<!\n)\n(?!\n)", " ", unfenced)


def _taught_names(paragraph: str) -> dict[str, int]:
    """Each ``root.name`` one paragraph spells in full or attributes to a root, with where it first does."""
    found: dict[str, int] = {}
    for span in _SPAN.finditer(paragraph):
        if (match := _QUALIFIED.match(span[1])) and not _PROVISIONAL_MARK.match(paragraph, span.end()):
            found.setdefault(match[1], span.start())
        if _MODULE.match(span[1]) and (after := _NAMES_AFTER.match(paragraph[span.end() :])):
            for name in _bare_names(after[1], span[1]):
                found.setdefault(name, span.start())
    for attribution in _ATTRIBUTION.finditer(paragraph):
        if re.search(r"\bprovisional\b", attribution[0]):
            continue
        for before in _NAMES_BEFORE.finditer(paragraph[: attribution.start()]):
            for name in _bare_names(before[1], attribution[1]):
                found.setdefault(name, before.start(1))
    return found


def taught_names(page: Path = GUIDE, *, text: str | None = None) -> dict[str, int]:
    """Every public name one page teaches, spelled qualified or attributed, with the first line that does.

    ``text``, when given, is read in place of the page's own (the hold-back ledger reads a page with the lines that
    only list the declared names blanked)."""
    recorded = surface.load_snapshot()
    text = page.read_text(encoding="utf-8") if text is None else text
    taught: dict[str, int] = {}
    for name, offset in sorted(_taught_names(_prose(text)).items(), key=lambda item: item[1]):
        if name in recorded:
            taught.setdefault(name, text.count("\n", 0, offset) + 1)
    return taught


def bare_names(page: Path, *, text: str | None = None) -> dict[str, int]:
    """Every bare name one page spells in its prose (`` `name` ``, `` `name(...)` ``, `` `name.attr` ``), with the
    first line that does. It is a word, not yet a public name: :func:`test_no_declared_name_a_page_names_goes_provisional`
    resolves it. ``text``, when given, is read in place of the page's own."""
    text = page.read_text(encoding="utf-8") if text is None else text
    prose = _prose(text)
    found: dict[str, int] = {}
    for span in _SPAN.finditer(prose):
        if (
            (bare := _BARE_SPELLING.match(span[1]))
            and not _MODULE.match(span[1])
            and not _PROVISIONAL_MARK.match(prose, span.end())
        ):
            found.setdefault(bare[1], text.count("\n", 0, span.start()) + 1)
    return found


def removed_names() -> dict[str, str]:
    """Each name ``MIGRATIONS.toml`` records leaving a public root that no root exports now, by its word."""
    recorded = surface.load_snapshot()
    return {
        old.rsplit(".", 1)[1]: old
        for release in surface.load_migrations().releases
        for change in release.changes
        for old in (f"{change.source}.{name}" for name in (*change.names, *change.renamed))
        if old not in recorded
    }


def removed_named(text: str, removed: Mapping[str, str], keep: frozenset[str] = frozenset()) -> dict[str, int]:
    """Each removed name one page names as code, with the first line that does. ``removed`` maps a word to the name it
    spells; a bare word ``keep`` holds is something else, and counts only spelled in full or attributed."""
    found: dict[str, int] = {}
    for name, offset in _taught_names(_prose(text)).items():
        if any(name == old or name.startswith(f"{old}.") for old in removed.values()):
            found.setdefault(name, text.count("\n", 0, offset) + 1)
    prose = _prose(text)
    for span in _SPAN.finditer(prose):
        if (
            (bare := _BARE_SPELLING.match(span[1]))
            and bare[1] in removed
            and bare[1] not in keep
            and not _REMOVAL_MARK.match(prose, span.end())
        ):
            found.setdefault(removed[bare[1]], text.count("\n", 0, span.start()) + 1)
    for block in _PYTHON_BLOCK.finditer(text):
        for word in re.finditer(r"\w+", _NOT_CODE.sub(lambda match: " " * len(match[0]), block[1])):
            if word[0] in removed and word[0] not in keep:
                found.setdefault(removed[word[0]], text.count("\n", 0, block.start(1) + word.start()) + 1)
    return found


def test_the_reader_finds_the_names_it_is_written_for() -> None:
    taught = taught_names()
    for name in (
        "threetears.evals.kernel.strict_response_format",
        "threetears.evals.run.price_subject_call_floor",
        "threetears.evals.schema.DOC_TYPE_FACTS",
        "threetears.evals.ops.runs_list",
    ):
        assert name in taught, f"the guide reader missed {name}, so the test below says less than it should"


def test_no_name_a_page_teaches_is_provisional() -> None:
    recorded = surface.load_snapshot()
    provisional = {
        f"{name} <- {page.relative_to(PACKAGE)} line {line}"
        for page in PAGES
        for name, line in taught_names(page).items()
        if not surface.declared(recorded[name].tier)
    }
    assert not provisional, (
        "a shipped page teaches these names, so a reader writes against them: add each to HOST_CONTRACT or STABLE in "
        f"packages/evals/scripts/public_surface.py and run `{surface.REGENERATE_SNAPSHOT}`:\n  "
        + "\n  ".join(sorted(provisional))
    )


def _released_declared() -> set[str]:
    tag = surface.newest_tag()
    assert tag is not None and surface.tag_exists(tag), (
        "no release tag in this checkout, so what the last release declared cannot be read: fetch the tags "
        "(git fetch --tags, or fetch-depth: 0 in CI)."
    )
    source = surface.source_at(tag, SNAPSHOT_IN_REPO)
    if source is None:
        return set()
    names = json.loads(source)["names"]
    return {name for name, entry in names.items() if surface.declared(entry.get("tier", "provisional"))}


def test_no_declared_name_a_page_names_goes_provisional() -> None:
    """A bare name resolves to every public name it spells; when the newest release declared any of them, one must
    still be declared."""
    recorded = surface.load_snapshot()
    released = _released_declared()
    assert released, "the newest release declared nothing, so this test says nothing"
    by_bare: dict[str, list[str]] = {}
    for name in recorded:
        by_bare.setdefault(name.rsplit(".", 1)[1], []).append(name)
    demoted = {
        f"{', '.join(sorted(set(spelt) & released))} <- {page.relative_to(PACKAGE)} line {line}"
        for page in PAGES
        for word, line in bare_names(page).items()
        if (spelt := by_bare.get(word, []))
        and set(spelt) & released
        and not any(surface.declared(recorded[name].tier) for name in spelt)
    }
    assert not demoted, (
        "the last release declared these names and a shipped page still names them, so a reader writes against "
        "them: keep each in HOST_CONTRACT or STABLE in packages/evals/scripts/public_surface.py, or stop the page "
        f"naming it, and run `{surface.REGENERATE_SNAPSHOT}`:\n  " + "\n  ".join(sorted(demoted))
    )


def test_the_bare_reader_finds_the_names_it_is_written_for() -> None:
    serving = bare_names(PACKAGE / "docs" / "serving.md")
    assert {"EXIT_OK", "run_cli", "REFUSAL_KINDS"} <= serving.keys(), "the bare reader missed a name serving.md names"


def test_no_page_names_a_removed_name() -> None:
    """A page that tells a reader to call a removed name hands them an ImportError."""
    removed = removed_names()
    assert removed, "MIGRATIONS.toml records no removed name, so this test says nothing"
    keep = frozenset(name.rsplit(".", 1)[1] for name in surface.load_snapshot()) | frozenset(METRIC_DESCRIPTORS)
    named = {
        f"{name} <- {page.relative_to(PACKAGE)} line {line}"
        for page in PAGES
        for name, line in removed_named(page.read_text(encoding="utf-8"), removed, keep).items()
    }
    assert not named, (
        "a shipped page names these as code, and MIGRATIONS.toml records each leaving its root: write the replacement "
        "its record names instead\n  " + "\n  ".join(sorted(named))
    )


def test_the_readers_see_each_spelling_of_a_name_as_code() -> None:
    """Planted: a removed name in each spelling a page names code by, a decorator a page teaches, and the marks and
    words that are not a promise or not the name."""
    removed = {"judge_config_of": "threetears.evals.schema.judge_config_of"}
    for spelling in (
        "Call `threetears.evals.schema.judge_config_of` on it.",
        "Read it with `judge_config_of` (`threetears.evals.schema`).",
        "Call `judge_config_of(recorded)` on it.",
        "Decorate it with `@judge_config_of(...)`.",
        "Read it with `judge_config_of`.",
        "```python\nconfig = judge_config_of(recorded)\n```\n",
    ):
        assert removed_named(spelling, removed), f"the removed-name reader missed {spelling!r}"
    for words in (
        "The measure `judge_config_of` reads it.",
        '```python\nprint(point["judge_config_of"])  # judge_config_of\n```\n',
        "```text\njudge_config_of(recorded)\n```\n",
    ):
        assert not removed_named(words, removed, frozenset({"judge_config_of"})), f"a word read as the name: {words!r}"
    for marked in ("`judge_config_of` (removed) is gone.", "Its names (`x`, `y` and `judge_config_of`, all retired)."):
        assert not removed_named(marked, removed), f"a record of the removal read as the name: {marked!r}"
    assert bare_names(GUIDE, text="Declare it on the function (`@measure(...)`).") == {"measure": 1}
    assert not bare_names(GUIDE, text="Declare it on the function (`@measure(...)`, provisional).")
    taught = "`@measure(...)` (`threetears.evals.quick`) declares it."
    assert "threetears.evals.quick.measure" in _taught_names(_prose(taught))
    assert not _taught_names(_prose(taught.replace("quick`)", "quick`, provisional)")))
