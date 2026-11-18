"""Delivery estimates for the booking page: when a parcel booked now should reach its customer.

A parcel's journey, as our network runs it (every time here is UK time, and we run seven days a week):

1. **Collection.** Parcels booked before the 15:00 cut-off leave the origin depot on that evening's trunk run at
   18:00; parcels booked at or after 15:00 leave on the next evening's.
2. **Trunking.** The trunk lorry averages :data:`TRUNK_KMH` over the lane's road distance, and every hub transfer on
   the lane adds :data:`HUB_TRANSFER_HOURS`.
3. **Customs.** A lane that crosses a border waits for customs clearance at the destination; the lane table gives the
   usual wait in days (``customs_days``).
4. **Final mile.** At the destination depot a parcel takes :data:`SORT_HOURS` to sort, then goes out on the first
   delivery round that leaves after that (rounds leave at 08:00 and 13:00) and is delivered on average
   :data:`ROUND_HOURS` after its round leaves.

The lanes are in ``lanes.json`` next to this module: origin, destination, road distance in kilometres, the number of
hub transfers and the customs days.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

CUTOFF = time(15, 0)
TRUNK_DEPARTS = time(18, 0)
TRUNK_KMH = 72.0
HUB_TRANSFER_HOURS = 2.5
SORT_HOURS = 1.5
ROUND_DEPARTS = (time(8, 0), time(13, 0))
ROUND_HOURS = 2.0
LANES_FILE = Path(__file__).with_name("lanes.json")


@dataclass(frozen=True)
class Lane:
    origin: str
    destination: str
    km: float
    hubs: int
    customs_days: int


def load_lanes(path: Path = LANES_FILE) -> dict[tuple[str, str], Lane]:
    """The lane table, keyed by (origin, destination) depot code."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(row["origin"], row["destination"]): Lane(**row) for row in rows}


LANES = load_lanes()


def departure(booked: datetime) -> datetime:
    """When a parcel booked at ``booked`` leaves the origin depot."""
    day = booked.date() if booked.time() < CUTOFF else booked.date() + timedelta(days=1)
    return datetime.combine(day, TRUNK_DEPARTS)


def transit_hours(lane: Lane) -> float:
    """Hours from leaving the origin depot to reaching the destination depot, cleared."""
    hours = lane.km / TRUNK_KMH + lane.hubs * HUB_TRANSFER_HOURS
    hours += lane.customs_days
    return hours


def next_round(ready: datetime) -> datetime:
    """The first delivery round leaving at or after ``ready``."""
    rounds = (datetime.combine(ready.date() + timedelta(days=d), at) for d in (0, 1) for at in ROUND_DEPARTS)
    return next(when for when in rounds if when >= ready)


def estimate(booked: datetime, origin: str, destination: str) -> datetime:
    """The estimated delivery time, to the minute, for a parcel booked at ``booked`` on the lane given."""
    lane = LANES[(origin, destination)]
    arrives = departure(booked) + timedelta(hours=transit_hours(lane))
    ready = arrives + timedelta(hours=SORT_HOURS)
    delivered = next_round(ready) + timedelta(hours=ROUND_HOURS)
    return delivered.replace(second=0, microsecond=0)


def hours_between(a: datetime, b: datetime) -> float:
    """How far apart two times are, in hours."""
    return math.fabs((a - b).total_seconds()) / 3600
