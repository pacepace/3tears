"""Business-travel emissions for the quarterly sustainability report.

A trip is a list of legs, each a mode of travel and a distance in kilometres, and the nights spent in hotels. Its
footprint, in kg CO2e, is:

- for each leg, the distance times the mode's conversion factor for the 2026 reporting year (kg CO2e per
  passenger-km), with a flight leg also multiplied by its cabin's weighting (business class takes up more of the
  aircraft than economy);
- plus, for each hotel night, the factor for a night in the country where it was spent.

Factors are per passenger, so a trip is one traveller's trip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: kg CO2e per passenger-km, by mode.
FACTORS = {
    "car": 0.16844,
    "electric_car": 0.04695,
    "taxi": 0.14861,
    "bus": 0.10846,
    "coach": 0.02733,
    "rail": 0.03546,
    "eurostar": 0.00446,
    "underground": 0.02781,
    "ferry": 0.11286,
    "flight_domestic": 0.24587,
    "flight_short_haul": 0.15102,
    "flight_long_haul": 0.14787,
}
#: Weighting of a flight leg by cabin.
CABINS = {"economy": 1.0, "premium_economy": 1.6, "business": 2.9, "first": 4.0}
#: kg CO2e per hotel room night, by country.
HOTEL_NIGHT = {"GB": 10.4, "FR": 5.6, "DE": 13.2, "NL": 11.9, "IE": 12.1, "ES": 9.8, "US": 16.1}


@dataclass(frozen=True)
class Leg:
    mode: str
    km: float
    cabin: str = "economy"


@dataclass(frozen=True)
class Trip:
    trip_id: str
    legs: tuple[Leg, ...]
    hotel_nights: tuple[tuple[str, int], ...] = ()

    @classmethod
    def from_row(cls, row: dict) -> Trip:
        legs = tuple(Leg(leg["mode"], float(leg["km"]), leg.get("cabin", "economy")) for leg in row["legs"])
        nights = tuple((country, int(count)) for country, count in row.get("hotel_nights", {}).items())
        return cls(row["trip_id"], legs, nights)


def leg_kg(leg: Leg) -> float:
    """One leg's emissions."""
    if leg.mode not in FACTORS:
        raise ValueError(f"no conversion factor for mode {leg.mode!r}")
    weight = CABINS[leg.cabin] if leg.mode.startswith("flight") else 1.0
    return leg.km * FACTORS[leg.mode] * weight


def footprint_kg(trip: Trip) -> float:
    """A trip's emissions in kg CO2e, to two decimal places."""
    travel = sum(leg_kg(leg) for leg in trip.legs)
    stays = sum(HOTEL_NIGHT[country] * nights for country, nights in trip.hotel_nights)
    return round(travel + stays, 2)


def footprint_file(path: str | Path) -> dict[str, float]:
    """Every trip's footprint in a file of trips (one JSON trip per line), by trip id."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["trip_id"]: footprint_kg(Trip.from_row(row)) for row in rows}
