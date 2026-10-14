"""One way to hand back a long result: a window over it, and the call for the next one.

Every tool that returns text meets the same wall -- a page, a mail body, a
transcript, a device list longer than anything a prompt should carry whole. Each
one had solved it alone, and all of them solved it the same wrong way: cut at
some number, append a phrase, discard the rest. What was cut was then gone, and
the phrase told the model only that something was missing, never how to get it.

Found in production (metallm conv 01a097bc, 2026-09-16): a whitepaper came back
cut at exactly 15,000 characters, ending mid-sentence, and was discussed as if
whole; a smart-home device list was cut before the room the person asked about,
and the agent reported that the room did not exist.

So this is the one mechanism. A tool that can return more text than it should
send at once takes an ``offset`` (see :class:`WindowedInput`), slices with
:func:`window_text`, and hands back :meth:`TextWindow.rendered` -- the slice
and a note naming the exact call that returns the next part. A tool does not
write its own truncation phrase, and nothing is discarded silently: text past
the window is still there, one call away.

The note goes first (``rendered(note_first=True)``). A reader that keeps only
the head of a long result -- a prompt budget, a stored copy -- then keeps the
line saying it is a part; with the note last, that line was the first thing
lost. Every tool in this workspace renders note-first
(``tests/enforcement/test_a_tool_states_its_notes_first.py``).

The note is prose because its reader is a model; the same facts are on the
:class:`TextWindow` for a caller that wants them typed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

__all__ = ["Found", "TextWindow", "WindowedInput", "find_in_text", "window_text"]

#: Longest note a window can append, so a caller sizing a budget can allow for it.
NOTE_ALLOWANCE: Final[int] = 200


@dataclass(frozen=True)
class TextWindow:
    """A slice of a longer text, and what it takes to read the rest.

    :ivar text: the slice itself, without any note
    :ivar offset: where the slice starts, in characters
    :ivar total: how many characters the whole text has
    :ivar next_offset: the offset that returns the next slice, or ``None`` at the end
    """

    text: str
    offset: int
    total: int
    next_offset: int | None

    @property
    def is_whole(self) -> bool:
        """Whether this window is the entire text."""
        return self.offset == 0 and self.next_offset is None

    def note(self, *, tool: str | None = None, how: str | None = None) -> str:
        """The sentence the model reads, or ``""`` when the window is the whole text.

        ``how`` says where the rest is, for a window nobody can ask for again --
        a copy kept in memory, say. A tool passes its own name instead and the
        note names the call to repeat.

        :param tool: the tool's own name, so the note names the call to repeat
        :ptype tool: str | None
        :param how: where the rest is, when it is not another call to ``tool``
        :ptype how: str | None
        :return: the note, or ``""``
        :rtype: str
        """
        if self.is_whole:
            return ""
        end = self.offset + len(self.text)
        if self.next_offset is None:
            return f"[characters {self.offset:,}-{end:,} of {self.total:,}: the end of it]"
        if how is None:
            if tool is None:
                raise ValueError("a window that is not the whole text needs a tool name or a `how`")
            how = f"call {tool} again with offset={self.next_offset} for the next part"
        return (
            f"[characters {self.offset:,}-{end:,} of {self.total:,}. This is a part, not the whole: "
            f"{how}. Do not report that something is absent from it.]"
        )

    def rendered(
        self,
        *,
        tool: str | None = None,
        how: str | None = None,
        note_first: bool = False,
        notes: Sequence[str] = (),
    ) -> str:
        """The slice and its note, which is what a tool returns.

        With ``note_first`` the window line comes first, then each of ``notes``
        on a line of its own, then a blank line and the slice. Without it the
        window line follows the slice, which is the form a reader that keeps
        only the head of a result loses the note from.

        :param tool: the tool's own name, for the note
        :ptype tool: str | None
        :param how: where the rest is, when it is not another call to ``tool``
        :ptype how: str | None
        :param note_first: put the window line and ``notes`` before the slice
        :ptype note_first: bool
        :param notes: the tool's other notes about this result, each one line;
            only with ``note_first``
        :ptype notes: Sequence[str]
        :return: what the model reads
        :rtype: str
        :raises ValueError: when ``notes`` are given without ``note_first``
        """
        note = self.note(tool=tool, how=how)
        if not note_first:
            if notes:
                raise ValueError("notes are stated before the text: pass note_first=True")
            return f"{self.text}\n\n{note}" if note else self.text
        head = "\n".join(line for line in (note, *notes) if line)
        if not head or not self.text:
            return head or self.text
        return f"{head}\n\n{self.text}"


def window_text(text: str, *, offset: int = 0, max_chars: int) -> TextWindow:
    """The window of ``text`` starting at ``offset``, at most ``max_chars`` long.

    An offset past the end returns an empty window rather than an error: a model
    that walks one part too far is told it reached the end, which is true, and
    is not made to handle a failure it cannot fix.

    :param text: the whole text
    :ptype text: str
    :param offset: where to start, clamped to zero
    :ptype offset: int
    :param max_chars: the most to return; a value below 1 is treated as 1
    :ptype max_chars: int
    :return: the window
    :rtype: TextWindow
    """
    total = len(text)
    start = min(max(offset, 0), total)
    size = max(max_chars, 1)
    end = min(start + size, total)
    return TextWindow(text=text[start:end], offset=start, total=total, next_offset=end if end < total else None)


#: How much text before a match a window opens with, so the match is read in its sentence.
FIND_LEAD_CHARS: Final[int] = 300

#: The most match offsets a find names; the count says how many there are in all.
FIND_LISTED: Final[int] = 20


@dataclass(frozen=True)
class Found:
    """Where a word or phrase appears in a text, from a starting offset on.

    :ivar term: what was looked for, as asked
    :ivar after: the offset the search started at
    :ivar offsets: where the first :data:`FIND_LISTED` matches start, in characters
    :ivar total: how many matches there are from ``after`` on
    """

    term: str
    after: int
    offsets: tuple[int, ...]
    total: int

    @property
    def start(self) -> int | None:
        """Where a window showing the first match opens, or ``None`` when nothing matched."""
        if not self.offsets:
            return None
        return max(self.after, self.offsets[0] - FIND_LEAD_CHARS)

    def note(self) -> str:
        """The line the model reads: how many matches, and the offset of each it can read next.

        :return: one line
        :rtype: str
        """
        where = f" from character {self.after:,}" if self.after else ""
        if not self.offsets:
            return f"[find {self.term!r}: no match{where}.]"
        listed = ", ".join(f"{o:,}" for o in self.offsets)
        more = f" (the first {len(self.offsets)} shown)" if self.total > len(self.offsets) else ""
        word = "match" if self.total == 1 else "matches"
        return (
            f"[find {self.term!r}: {self.total} {word}{where}, at characters {listed}{more}. "
            "The text below opens just before the first; pass offset to read at another.]"
        )


def find_in_text(text: str, term: str, *, after: int = 0) -> Found:
    """Every place ``term`` appears in ``text`` at or after ``after``, ignoring case.

    :param text: the whole text
    :ptype text: str
    :param term: the word or phrase to look for; surrounding spaces are ignored
    :ptype term: str
    :param after: where to start looking, clamped to the text
    :ptype after: int
    :return: the matches
    :rtype: Found
    :raises ValueError: when ``term`` is blank
    """
    # lower(), not casefold(): casefold can change a string's length (the German sharp
    # s), which would move every offset after it away from the text the model reads.
    needle = term.strip().lower()
    if not needle:
        raise ValueError("find needs a word or phrase to look for")
    start = min(max(after, 0), len(text))
    haystack = text.lower()
    offsets: list[int] = []
    total = 0
    at = haystack.find(needle, start)
    while at != -1:
        total += 1
        if len(offsets) < FIND_LISTED:
            offsets.append(at)
        at = haystack.find(needle, at + len(needle))
    return Found(term=term.strip(), after=start, offsets=tuple(offsets), total=total)


class WindowedInput(BaseModel):
    """The ``offset`` argument, worded once, for any tool that windows its result."""

    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Where to start reading, in characters. Omit for the beginning. "
            "When a result says it is a part, call again with the offset it names to continue."
        ),
    )
