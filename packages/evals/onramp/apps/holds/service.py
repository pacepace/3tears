"""The library's holds desk: answers requests about the books patrons have asked the library to keep for them.

Every request is a dict with an ``action`` and a ``today`` (``YYYY-MM-DD``), and gets a :class:`Reply`:

- ``{"action": "list", "patron": "p-104", "today": "2026-03-10"}`` lists a patron's holds, each flagged
  ``expired`` once its pickup date has passed, so the desk knows to return it to the shelf.
- ``{"action": "count", "today": ...}`` counts every hold and how many have expired.

Holds are read from ``data/holds.json`` beside this file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

HOLDS_FILE = Path(__file__).with_name("data") / "holds.json"

#: Days a hold stays on the shelf after its pickup date before the desk is told.
GRACE_DAYS = 7


@dataclass
class Hold:
    id: str
    patron: str
    title: str
    pickup_by: str
    expired: bool = False


@dataclass
class Reply:
    status: int
    holds: list[Hold] = field(default_factory=list)
    message: str = ""
    counts: dict[str, int] = field(default_factory=dict)


def _load() -> list[Hold]:
    return [Hold(**row) for row in json.loads(HOLDS_FILE.read_text(encoding="utf-8"))]


def _flag(hold: Hold, today: date) -> Hold:
    late = (today - date.fromisoformat(hold.pickup_by)).days
    hold.expired = late > GRACE_DAYS
    return hold


def handle(request: dict) -> Reply:
    """Answer one request. An unknown action, or a request without ``today``, is a 400."""
    try:
        today = date.fromisoformat(request["today"])
    except KeyError, ValueError:
        return Reply(status=400, message="a request names today as YYYY-MM-DD")
    holds = [_flag(hold, today) for hold in _load()]
    if request.get("action") == "list":
        mine = [hold for hold in holds if hold.patron == request.get("patron")]
        if not mine:
            return Reply(status=404, message=f"no holds for {request.get('patron')}")
        return Reply(status=200, holds=mine)
    if request.get("action") == "count":
        return Reply(status=200, counts={"holds": len(holds), "expired": sum(hold.expired for hold in holds)})
    return Reply(status=400, message=f"unknown action {request.get('action')!r}")
