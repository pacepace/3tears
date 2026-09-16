"""The one way a long result is handed back: a window, and the call for the next one."""

from __future__ import annotations

import pytest

from threetears.agent.tools.text_window import WindowedInput, window_text


class TestTheWindow:
    def test_a_short_text_comes_back_whole_and_says_nothing(self) -> None:
        window = window_text("short", max_chars=100)
        assert window.text == "short" and window.is_whole
        assert window.note(tool="web_fetch") == "" and window.rendered(tool="web_fetch") == "short"

    def test_a_long_text_returns_its_first_part_and_where_the_next_one_starts(self) -> None:
        window = window_text("x" * 250, max_chars=100)
        assert window.text == "x" * 100
        assert (window.offset, window.total, window.next_offset) == (0, 250, 100)

    def test_the_note_names_the_tool_the_offset_and_the_whole_size(self) -> None:
        note = window_text("x" * 250, max_chars=100).note(tool="web_fetch")
        assert "call web_fetch again with offset=100" in note
        assert "characters 0-100 of 250" in note
        assert "Do not report that something is absent from it." in note

    def test_walking_the_offsets_reads_the_whole_text_once(self) -> None:
        text = "".join(str(n % 10) for n in range(1000))
        seen, offset = "", 0
        while True:
            window = window_text(text, offset=offset, max_chars=256)
            seen += window.text
            if window.next_offset is None:
                break
            offset = window.next_offset
        assert seen == text

    def test_the_last_part_says_it_is_the_end_and_names_no_further_call(self) -> None:
        note = window_text("x" * 250, offset=200, max_chars=100).note(tool="email_read")
        assert "the end of it" in note and "offset=" not in note

    def test_an_offset_past_the_end_says_so_instead_of_failing(self) -> None:
        window = window_text("x" * 50, offset=9999, max_chars=100)
        assert window.text == "" and window.next_offset is None and window.total == 50

    @pytest.mark.parametrize("offset", [-5, 0])
    def test_a_negative_offset_reads_from_the_beginning(self, offset: int) -> None:
        assert window_text("abcdef", offset=offset, max_chars=3).text == "abc"

    @pytest.mark.parametrize("max_chars", [0, -1])
    def test_a_useless_limit_still_makes_progress(self, max_chars: int) -> None:
        """A limit of zero would hand back nothing forever; one character still advances."""
        window = window_text("abc", max_chars=max_chars)
        assert window.text == "a" and window.next_offset == 1

    def test_nothing_is_lost_between_parts(self) -> None:
        text = "abcdefghij"
        first = window_text(text, max_chars=4)
        second = window_text(text, offset=first.next_offset or 0, max_chars=4)
        assert first.text + second.text == "abcdefgh"


class TestTheSharedArgument:
    def test_the_offset_argument_defaults_to_the_beginning_and_refuses_a_negative(self) -> None:
        assert WindowedInput().offset == 0
        with pytest.raises(ValueError):
            WindowedInput(offset=-1)

    def test_its_description_tells_the_model_what_to_do_with_it(self) -> None:
        description = WindowedInput.model_fields["offset"].description or ""
        assert "call again with the offset it names" in description
