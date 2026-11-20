# weatherboard

The software behind the weather board on the harbour wall (`board.py`). Most of the stations on our network are old
and report in Fahrenheit; the board shows whole degrees Celsius. `display_c(temp_f)` is the figure the board shows for
one reading, and `board_line(station, readings)` builds a station's line (latest reading, low and high).

Two of the stations, Quay and Lighthouse, were replaced last year with loggers that record both scales.
`data/dual_logged.jsonl` is a week of their readings: the Fahrenheit value the board receives and the Celsius value
the logger recorded beside it, to a tenth of a degree. Since the board rounds to whole degrees, what it shows should
never be more than half a degree from the logger's Celsius. `data/readings_2026-01-19.jsonl` is a day of readings
from the three old stations, as the board receives them.

Tests: `python -m unittest discover tests` from this directory.
