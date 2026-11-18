"""The harbour weather board: turns station readings into the whole degrees Celsius the board shows.

The older stations on the network still report in Fahrenheit. ``display_c(temp_f)`` is the figure the board shows
for one reading: the temperature in Celsius, rounded to the nearest whole degree, with halves rounded away from zero
(2.5 °C shows as 3, and -2.5 °C as -3). ``board_line(station, readings)`` is one station's line on the board: its
latest reading and the low and high of the readings given, each as the board shows it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class ReadingError(ValueError):
    """A reading the board cannot show: not a number, or outside what a station can measure."""


#: The sensors' range in Fahrenheit; anything outside it is a fault, not weather.
SENSOR_RANGE_F = (-40.0, 140.0)


def to_celsius(temp_f: float) -> float:
    """Fahrenheit to Celsius, unrounded."""
    return (temp_f - 32.0) * 5.0 / 9.0


def nearest_degree(celsius: float) -> int:
    """Celsius to the nearest whole degree, halves away from zero."""
    return int(math.copysign(math.floor(abs(celsius) + 0.5), celsius))


def display_c(temp_f: float | str) -> int:
    """The whole degree Celsius the board shows for one Fahrenheit reading."""
    try:
        value = float(temp_f)
    except TypeError, ValueError:
        raise ReadingError(f"reading {temp_f!r} is not a number") from None
    low, high = SENSOR_RANGE_F
    if not math.isfinite(value) or not low <= value <= high:
        raise ReadingError(f"reading {temp_f!r} is outside the sensor's range")
    return nearest_degree(to_celsius(value))


@dataclass(frozen=True)
class BoardLine:
    station: str
    now_c: int
    low_c: int
    high_c: int

    def text(self) -> str:
        return f"{self.station:<12} {self.now_c:>3}°C  (low {self.low_c}°C, high {self.high_c}°C)"


def board_line(station: str, readings: list[dict[str, Any]]) -> BoardLine:
    """One station's board line from its readings, each ``{"time": "2026-01-14T06:00", "temp_f": 27.32}``."""
    if not readings:
        raise ReadingError(f"no readings for {station}")
    ordered = sorted(readings, key=lambda reading: reading["time"])
    shown = [display_c(reading["temp_f"]) for reading in ordered]
    return BoardLine(station=station, now_c=shown[-1], low_c=min(shown), high_c=max(shown))
