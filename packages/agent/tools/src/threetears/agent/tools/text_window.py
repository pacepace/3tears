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
plus a note naming the exact call that returns the next part. A tool does not
write its own truncation phrase, and nothing is discarded silently: text past
the window is still there, one call away.

The note is prose because its reader is a model; the same facts are on the
:class:`TextWindow` for a caller that wants them typed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

__all__ = ["TextWindow", "WindowedInput", "window_text"]

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

    def rendered(self, *, tool: str | None = None, how: str | None = None) -> str:
        """The slice with its note appended, which is what a tool returns.

        :param tool: the tool's own name, for the note
        :ptype tool: str | None
        :param how: where the rest is, when it is not another call to ``tool``
        :ptype how: str | None
        :return: what the model reads
        :rtype: str
        """
        note = self.note(tool=tool, how=how)
        return f"{self.text}\n\n{note}" if note else self.text


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
