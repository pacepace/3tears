"""``docs/surface.md`` and ``llms.txt`` are generated, committed, and held equal to what the generator makes today.

``scripts/generate_surface.py`` builds both from each declared name's kind (``scripts/surface_kinds.py``) and its own
docstring: the page lists every declared name once, by role and then rung; the agent's file is what an AI agent
adopting the package reads first. A change to a name, its kind or its docstring moves them, and this fails until they
are regenerated, so neither describes an older package.
"""

from __future__ import annotations

import re
import tomllib
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


def test_the_page_opens_with_the_first_eval() -> None:
    """The first screen is the kinds' counts and the names a first eval needs."""
    page = generator.SURFACE_PATH.read_text(encoding="utf-8").splitlines()
    first = page.index("### Your first eval: these 8")
    assert first < 25
    assert page[first + 2].startswith("- `Eval`: ")


def test_the_agent_file_is_short_and_shows_the_first_example_verbatim() -> None:
    text = generator.AGENT_PATH.read_text(encoding="utf-8")
    assert len(text.splitlines()) < 200
    example = set(generator.FIRST_EXAMPLE.read_text(encoding="utf-8").splitlines())
    shown = generator.example_lines()
    assert len(shown) == 5 and all(line in example for line in shown)
    assert all(f"  {line}" in text.splitlines() for line in shown)
    assert len(generator.what_it_is()) == 3


def test_the_wheel_ships_the_agent_file() -> None:
    config = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    shipped = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert shipped["llms.txt"] == "threetears/evals/llms.txt"
