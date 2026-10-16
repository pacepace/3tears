"""free text a model reads sits under the heading it belongs to, every line of it.

A search result's snippet, a memory's content, a skill's notes: written as
``f"   {snippet}"``, only the first line of a many-line text is indented, and the
rest reads as the next item's (metallm acceptance run, 2026-10-16: an agent gave
one note's size and tag to the note above it). Every such value goes through
``threetears.observe.text_block.under`` or ``one_line``, or says on its line why
it is one line (``# one line: <why>``). The checker is
:func:`threetears.enforcement.text_blocks.unbroken_text`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.text_blocks import unbroken_text, unbroken_text_in

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The release and enforcement tooling prints to a person at a terminal, not to a model.
_NOT_FOR_A_MODEL = ("enforcement",)


def _roots() -> list[Path]:
    return [
        path
        for path in sorted((_REPO_ROOT / "packages").glob("**/src"))
        if path.is_dir() and not any(part in _NOT_FOR_A_MODEL for part in path.relative_to(_REPO_ROOT).parts)
    ]


def test_no_free_text_is_indented_at_its_first_line_only() -> None:
    found = unbroken_text_in(_roots())
    assert not found, "free text after an indent goes through under() or one_line():\n" + "\n".join(found)


def test_the_scan_reaches_the_packages() -> None:
    roots = {path.relative_to(_REPO_ROOT).as_posix() for path in _roots()}
    assert {"packages/search/src", "packages/agent/memory/src", "packages/agent/skills/src"} <= roots, roots


@pytest.mark.parametrize(
    "line",
    [
        'lines.append(f"      {excerpt}")',
        'lines.append(f"   {candidate.snippet}")',
        'lines.append(f"  - {handle}: {summary}")',
        'lines.append(f"  why: {use.notes}")',
        'lines.append(f"  [chunk:{cid}]\\n  {preview}")',
        'block = "  " + body',
        'lines.append(f"  {cut_whole(excerpt, 200)}")',
        "lines.append(f\"  {row['content']}\")",
        'lines.append(f"  {excerpt.strip()}")',
        'items.append(f"- [memory:{mem_id}] {text}")',
        'lines.append(f"{position}. {candidate.title}")',
        'listing = "\\n".join(f"- {name}: {description}" for name, description in rows)',
        'lines.append(f"- {when}: {value}")',
        'lines.append(f"  {shown[key]}")',
    ],
)
def test_the_check_finds_a_text_indented_at_its_first_line_only(line: str) -> None:
    assert unbroken_text(line) == [1], line


@pytest.mark.parametrize(
    "line",
    [
        'lines.extend(under(excerpt, "      "))',
        'lines.append(f"  {one_line(summary)}")',
        'lines.append(f"  {path}  ({chars} chars)")',
        'lines.append(f"  [msg:{message_id_end}]")',
        'lines.append(f"  - {untrusted_fence(nonce, preview)}")',
        'log.info(f"  {excerpt}")',
        'await pool.fetch(f"  SELECT {content}")',
        'lines.append(f"  {note}")  # one line: a note here is a single-line label',
        'lines.append(f"{excerpt}")',
        'text = one_line(f"{facts}. {snippet}")',
        'lines.append(f"  {arg_name}: {value!r}")',
        'lines.append(f"- {when} {link.kind.value}: ok")',
    ],
)
def test_the_check_leaves_what_is_one_line_or_marked(line: str) -> None:
    assert unbroken_text(line) == [], line
