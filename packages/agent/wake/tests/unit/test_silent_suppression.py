"""Unit tests for :func:`threetears.agent.wake.dispatch.detect_silent_prefix`.

Pins the ``[SILENT]`` marker contract per PLACEMENT.md §1.4:

- case-insensitive
- tolerates leading whitespace
- tolerates trailing whitespace + newline
- empty / whitespace-only / non-matching text returns ``False``

These cases protect the audit-row + suppression-flag invariant that
:func:`threetears.agent.wake.dispatch.dispatch_wake` relies on to set
``WakeDispatchResult.display_suppressed`` and to skip delivery
routing on silent fires.
"""

from __future__ import annotations

import pytest

from threetears.agent.wake.dispatch import detect_silent_prefix


class TestDetectSilentPrefix:
    """Cover every documented marker-detection branch."""

    @pytest.mark.parametrize(
        "content",
        [
            "[SILENT]",
            "[SILENT] no change observed",
            "[SILENT]\nbatching watchdog: 0 anomalies",
            "  [SILENT]",
            "\n\n[SILENT]\n",
            "[silent] lowercased",
            "[Silent]",
            "[SiLeNt] mixed case",
            "\t[SILENT]\t trailing tab",
        ],
    )
    def test_match(self, content: str) -> None:
        assert detect_silent_prefix(content) is True

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "   ",
            "\n\n\n",
            "ok the watchdog observed something",
            "Not silent: [SILENT] appears later in the body",
            "[SLIENT] typo, do not match",
            "SILENT without brackets",
            "[ SILENT ] inner whitespace not allowed",
        ],
    )
    def test_no_match(self, content: str) -> None:
        assert detect_silent_prefix(content) is False
