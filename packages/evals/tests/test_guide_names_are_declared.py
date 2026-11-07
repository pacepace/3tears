"""Every name a shipped page tells a reader to use is declared: ``host-contract`` or ``stable``, never ``provisional``.

The pages are every page under ``docs/``, the package README and the examples' README. Each names things in prose
that a reader is to call, build, catch or match, and a package that teaches a name promises it.

Two spellings are read:

- **Qualified or attributed.** A name spelled in full (`` `threetears.evals.run.definition_drift` ``) or attributed
  to a module in parentheses (`` `strict_response_format` (`threetears.evals.kernel`) ``, `` `A` and `B` (in
  `threetears.evals.kernel`) ``) must be declared.
- **Bare.** A name spelled alone (`` `run_cli(argv, ...)` ``, `` `EXIT_OK` ``) may also be a field, a parameter or
  another root's name. So a bare name is held to stay declared rather than to be declared: if the newest release
  declared it, it must not go provisional while a page still names it.

The import statements the pages show are held by ``test_api_tiers.py``. A name a page mentions that no public root
exports (a module, a logger, a field) is not a name this reads.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

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
_BARE = re.compile(r"^(\w+)(?:\(.*\))?$")
#: A bare name as prose spells it: alone, called, or with a member read off it.
_BARE_SPELLING = re.compile(r"^(\w+)(?:\(.*\)|\.\w+.*)?$")
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
        if match := _QUALIFIED.match(span[1]):
            found.setdefault(match[1], span.start())
        if _MODULE.match(span[1]) and (after := _NAMES_AFTER.match(paragraph[span.end() :])):
            for name in _bare_names(after[1], span[1]):
                found.setdefault(name, span.start())
    for attribution in _ATTRIBUTION.finditer(paragraph):
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
    found: dict[str, int] = {}
    for span in _SPAN.finditer(_prose(text)):
        if (bare := _BARE_SPELLING.match(span[1])) and not _MODULE.match(span[1]):
            found.setdefault(bare[1], text.count("\n", 0, span.start()) + 1)
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
