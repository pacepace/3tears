"""Every name the host guide tells a host to use is declared: ``host-contract`` or ``stable``, never ``provisional``.

``docs/adopting-a-host.md`` names things in prose that a host is to call, build, catch or match. A name it spells in
full (`` `threetears.evals.run.definition_drift` ``) or attributes to a module in parentheses
(`` `strict_response_format` (`threetears.evals.kernel`) ``, `` `A` and `B` (in `threetears.evals.kernel`) ``) is one
the guide teaches, and a package that teaches a name promises it. The import statements the guide shows are held
to ``host-contract`` by ``test_api_tiers.py``; this holds the names its prose carries.

A name the guide mentions that no public root exports (a module, a logger, a field) is not a name this reads.
"""

from __future__ import annotations

import re
from pathlib import Path

from packages.evals.tests.script_loader import load_script

surface = load_script("public_surface.py")

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "adopting-a-host.md"

_FENCE = re.compile(r"^```.*?^```", re.M | re.S)
_SPAN = re.compile(r"`([^`\n]+)`")
_QUALIFIED = re.compile(r"^(threetears\.evals(?:\.\w+)+)")
_MODULE = re.compile(r"^threetears\.evals(?:\.\w+)*$")
_BARE = re.compile(r"^(\w+)(?:\(.*\))?$")
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
    """The guide's prose with its fenced blocks blanked and each paragraph on one line, every offset kept."""
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


def taught_names() -> dict[str, int]:
    """Every public name the guide teaches, with the first line that does."""
    recorded = surface.load_snapshot()
    text = GUIDE.read_text(encoding="utf-8")
    taught: dict[str, int] = {}
    for name, offset in sorted(_taught_names(_prose(text)).items(), key=lambda item: item[1]):
        if name in recorded:
            taught.setdefault(name, text.count("\n", 0, offset) + 1)
    return taught


def test_the_reader_finds_the_names_it_is_written_for() -> None:
    taught = taught_names()
    for name in (
        "threetears.evals.kernel.strict_response_format",
        "threetears.evals.run.price_subject_call_floor",
        "threetears.evals.schema.DOC_TYPE_FACTS",
        "threetears.evals.ops.runs_list",
    ):
        assert name in taught, f"the guide reader missed {name}, so the test below says less than it should"


def test_no_name_the_guide_teaches_is_provisional() -> None:
    recorded = surface.load_snapshot()
    provisional = {name: line for name, line in taught_names().items() if not surface.declared(recorded[name].tier)}
    assert not provisional, (
        "docs/adopting-a-host.md teaches these names, so a host writes against them: add each to HOST_CONTRACT in "
        f"packages/evals/scripts/public_surface.py and run `{surface.REGENERATE_SNAPSHOT}`:\n  "
        + "\n  ".join(f"{name} <- docs/adopting-a-host.md line {line}" for name, line in sorted(provisional.items()))
    )
