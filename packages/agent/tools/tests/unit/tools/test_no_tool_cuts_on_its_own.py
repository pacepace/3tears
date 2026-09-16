"""No tool writes its own truncation phrase: long results go through one mechanism.

Every tool that returns text meets the same wall, and each one used to solve it
alone -- cut at some number, append a phrase, discard the rest. The phrases said
something was missing and never how to get it, and what was cut was gone.
``text_window`` is the one way now: a slice, a note naming the call for the next
part, and nothing discarded.

This fails when a tool grows its own phrase again, which is the only way this
can come back.
"""

from __future__ import annotations

import pathlib
import re

_TOOLS = pathlib.Path(__file__).resolve().parents[3] / "src" / "threetears" / "agent" / "tools"

#: Phrases a tool used to append when it cut. The window's own note is prose the
#: module owns, so it is exempt by living in text_window.py.
_HAND_ROLLED = re.compile(r"\[(Content truncated|Truncated|truncated)[^\]]*\]")


def test_the_mechanism_is_there_to_use() -> None:
    assert (_TOOLS / "text_window.py").is_file()


def test_no_tool_appends_a_truncation_phrase_of_its_own() -> None:
    offenders: list[str] = []
    for path in _TOOLS.rglob("*.py"):
        if path.name == "text_window.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue  # a comment may name the old phrase to explain the history
            if _HAND_ROLLED.search(line):
                offenders.append(f"{path.relative_to(_TOOLS)}:{number}: {line.strip()}")
    assert not offenders, (
        "these cut a result and wrote their own phrase for it; use "
        "threetears.agent.tools.text_window.window_text, whose note names the call "
        "that returns the next part:\n  " + "\n  ".join(offenders)
    )
