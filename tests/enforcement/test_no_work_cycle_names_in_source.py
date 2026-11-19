"""Enforcement: a shipped comment carries its why and its numbers, never a work cycle's name.

A comment or docstring that anchors its reason to a lane of a release night (``SG1``, ``TF1``, ``IT3``) names
something no reader of the package can resolve once that night's records are gone: the measurement the name stood for
(its date, its host, its numbers) is what stays. The review rule ("review and finding ids, chunk numbers and review
history never belong in a shipped comment") had no check; this is it, for the lane names the release nights used.
A dated measurement is fine and stays.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]

#: a release night's lane name: its letters, then its number (``TF1``, ``SG1``, ``IT3``, ``FR4``)
_LANE: Final = re.compile(r"\b(?:SG|TF|TL|TT|FR|WS|IT|RP|LM|LG|NS|PL|HB|SDKR|OPS|RF|IG|LT|TC|RC|SYN|GATE)\d+[a-z]?\b")


def lane_names(text: str) -> list[str]:
    """the lane names ``text`` holds.

    :param text: a module's source
    :ptype text: str
    :return: each one, in order
    :rtype: list[str]
    """
    return [found.group() for found in _LANE.finditer(text)]


def test_the_rule_sees_a_lane_name_and_not_a_dated_measurement() -> None:
    assert lane_names("# 1.70 ms on base (TF1, 2026-11-18)") == ["TF1"]
    assert lane_names('"""moved from reports (SG1 2026-11-18)"""') == ["SG1"]
    assert lane_names("# 2026-11-16, the plan read with and without the literals; UTF8, utf-16") == []


def test_no_package_source_names_a_work_cycle() -> None:
    found = {
        str(path.relative_to(_REPO_ROOT)): names
        for root in ("packages/*/src", "packages/agent/*/src")
        for path in sorted(_REPO_ROOT.glob(f"{root}/**/*.py"))
        if (names := lane_names(path.read_text(encoding="utf-8")))
    }
    assert not found, (
        f"shipped source names a work cycle: {found}. Keep the measurement's date, host and numbers inline and drop "
        "the lane name"
    )
