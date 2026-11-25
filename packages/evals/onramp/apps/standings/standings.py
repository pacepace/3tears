"""The results table for the Thursday quiz league at the Fox & Anchor.

``table(entries)`` ranks one night's teams:

- more points ranks higher;
- teams level on points are separated by their answer sheet for the last round: the team that handed it in earlier
  ranks higher (``handed_in``, a ``HH:MM:SS`` time on the night);
- teams level on both share a place, and the places after them are skipped, as in a sports table: 1, 2, 2, 4.

``season(nights)`` adds up league points across nights: 10 for a win, 8 for second, then 6, 5, 4, 3, 2 and 1 for
eighth; a shared place earns that place's points for every team in it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

#: League points by place, first to eighth; ninth and below score nothing.
LEAGUE_POINTS = (10, 8, 6, 5, 4, 3, 2, 1)


@dataclass(frozen=True)
class Entry:
    team: str
    points: int
    handed_in: str  # HH:MM:SS

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Entry:
        return cls(row["team"].strip(), int(row["points"]), row["handed_in"].strip())


@dataclass(frozen=True)
class Place:
    place: int
    team: str
    points: int


def _seconds(clock: str) -> int:
    hours, minutes, seconds = (int(part) for part in clock.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def table(entries: list[Entry]) -> list[Place]:
    """One night's results table, top first."""
    ordered = sorted(entries, key=lambda e: (e.points, _seconds(e.handed_in)), reverse=True)
    places: list[Place] = []
    for i, entry in enumerate(ordered):
        previous = ordered[i - 1] if i else None
        level = previous is not None and (previous.points, previous.handed_in) == (entry.points, entry.handed_in)
        places.append(Place(places[-1].place if level else i + 1, entry.team, entry.points))
    return places


def season(nights: list[list[Entry]]) -> list[tuple[str, int]]:
    """League points per team across ``nights``, most first, then by team name."""
    totals: Counter[str] = Counter()
    for night in nights:
        for row in table(night):
            totals[row.team] += LEAGUE_POINTS[row.place - 1] if row.place <= len(LEAGUE_POINTS) else 0
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))
