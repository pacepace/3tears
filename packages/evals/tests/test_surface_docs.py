"""``docs/surface.md`` and ``llms.txt`` are generated, committed, and held equal to what the generator makes today.

``scripts/generate_surface.py`` builds both from each declared name's kind (``scripts/surface_kinds.py``) and its own
docstring: the page lists every declared name once, by role and then rung; the agent's file is what an AI agent
adopting the package reads first. A change to a name, its kind or its docstring moves them, and this fails until they
are regenerated, so neither describes an older package.
"""

from __future__ import annotations

import re
from pathlib import Path

from packages.evals.tests.script_loader import load_script

generator = load_script("generate_surface.py")
surface = load_script("public_surface.py")

PACKAGE = Path(__file__).resolve().parents[1]

#: One declared name's line on the page: the name, ``threetears.evals.`` dropped, then its docstring's first sentence.
_NAME_LINE = re.compile(r"^- `([\w.]+)`: ", re.M)


def test_the_committed_surface_and_agent_file_are_what_the_package_generates() -> None:
    for path, text in generator.render().items():
        committed = path.read_text(encoding="utf-8") if path.exists() else ""
        assert committed == text, (
            f"{path.relative_to(PACKAGE)} is stale: the package changed under it. Regenerate it with "
            f"`{generator.REGENERATE}` and commit the result."
        )


def test_the_page_lists_every_declared_name_once() -> None:
    page = generator.SURFACE_PATH.read_text(encoding="utf-8")
    names_part = page.split("\n## The action catalogue")[0] + page.split("\n## A host's own tests")[1]
    listed = [
        name if name.startswith("threetears.evals") else f"threetears.evals.{name}"
        for name in _NAME_LINE.findall(names_part)
    ]
    declared = {name for name, entry in surface.load_snapshot().items() if surface.declared(entry.tier)}
    twice = sorted({name for name in listed if listed.count(name) > 1})
    assert not twice, f"docs/surface.md lists these more than once: {twice}"
    assert set(listed) == declared, (
        f"docs/surface.md is not the declared surface. Missing: {sorted(declared - set(listed))}. "
        f"Not declared: {sorted(set(listed) - declared)}."
    )


def test_the_page_opens_with_the_first_eval_and_ends_with_the_counts() -> None:
    """The first screen is the newcomer's: the names a first eval needs, then each rung's; the counts come last."""
    page = generator.SURFACE_PATH.read_text(encoding="utf-8").splitlines()
    first = page.index(f"### Your first eval: these {load_script('surface_kinds.py').typed_through(1)}")
    assert first < 10
    assert page[first + 2].startswith("- `Eval`: ")
    assert page.index("## The counts") > page.index("## Near-duplicate shapes: 2, each saying why both exist")
    # The totals are reported as a trend, never held to a number: the counts' tables have no budget or ceiling column.
    headers = [line for line in page[page.index("## The counts") :] if line.startswith("| ") and "Names" in line]
    assert headers[:2] == ["| Kind | Names | Who meets it |", "| Typed names | Names | Goal |"]
    assert not [header for header in headers if "Budget" in header or "Ceiling" in header]


def test_the_agent_file_is_short_true_and_sufficient() -> None:
    """Read whole by an agent that has never seen the package: the first example runs as shown, and the file says
    how to get it, that it replays for free, how to record live, how to run it and how to read what comes back."""
    text = generator.AGENT_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert len(lines) < 150
    program = generator.example_program()
    assert "\n".join(program) in generator.FIRST_EXAMPLE.read_text(encoding="utf-8")
    compile("\n".join(program), str(generator.FIRST_EXAMPLE), "exec")
    assert all((f"  {line}" if line else "") in lines for line in program)
    flat = " ".join(" ".join(lines).split())
    for needed in (
        "calls nothing and costs nothing",
        "`RECORD=1` with `OPENROUTER_API_KEY` set records it again",
        "git sparse-checkout set packages/evals/examples",
        "python -m threetears.evals 1-triage/triage.py",
        "The first line printed is the answer",
        "refused: and a remedy:",
        "0 Every run completed",
    ):
        assert needed in flat, f"llms.txt does not say {needed!r}"
    assert len(generator.what_it_is()) == 3
    # The wheel ships this file without the docs or the examples: each is named in the repository, by URL.
    for match in re.finditer(r"\S*(?:docs|examples)/\S*", text):
        assert match[0].startswith(("https://", "packages/evals/examples")) or match[0].startswith("(https://"), (
            f"llms.txt names {match[0]!r}, which the installed package does not hold"
        )


def test_a_registered_name_is_listed_only_in_its_register_s_section() -> None:
    """The debt registers (issue #1199) are shown as such: a name awaiting demotion or needing a shipped example is
    listed in its register's section and among no healthy names, and the agent's file offers only the second, marked."""
    kinds = load_script("surface_kinds.py")
    page = generator.SURFACE_PATH.read_text(encoding="utf-8")
    sections = {part.split("\n", 1)[0]: part for part in page.split("\n## ")[1:]}
    demotion = f"Awaiting demotion ({kinds.REGISTER_ISSUE}): {len(kinds.AWAITING_DEMOTION)}"
    shipped = f"Needs a shipped example ({kinds.REGISTER_ISSUE}): {len(kinds.AWAITING_SHIPPED_USE)}"
    for title, register in ((demotion, set(kinds.AWAITING_DEMOTION)), (shipped, set(kinds.AWAITING_SHIPPED_USE))):
        assert title in sections, f"docs/surface.md has no section {title!r}"
        listed = {f"threetears.evals.{name}" for name in _NAME_LINE.findall(sections[title])}
        assert listed == register, f"{title} lists {sorted(listed ^ register)} wrongly"
        elsewhere = {
            f"threetears.evals.{name}"
            for other, text in sections.items()
            if other not in (demotion, shipped)
            for name in _NAME_LINE.findall(text)
        }
        assert not elsewhere & register, f"listed among healthy names too: {sorted(elsewhere & register)}"
    agent = generator.AGENT_PATH.read_text(encoding="utf-8")
    offered = [
        name for name in kinds.AWAITING_DEMOTION if re.search(rf"(?<![\w.]){re.escape(generator.short(name))}\b", agent)
    ]
    assert not offered, f"llms.txt offers names awaiting demotion: {offered}"
    marked = agent.split(f"NEEDS A SHIPPED EXAMPLE ({kinds.REGISTER_ISSUE})")
    assert len(marked) == 2, "llms.txt does not mark the names that need a shipped example"
    assert all(f"  {generator.short(name)}: " in marked[1] for name in kinds.AWAITING_SHIPPED_USE)
