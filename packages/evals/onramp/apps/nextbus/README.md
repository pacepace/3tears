# nextbus

The "next buses" answer for route 412, used by the stop signs, the journey app and the SMS line. Given a stop and the
moment someone asked, `next_departures(timetable, stop, asked)` gives the next three departures from that stop, as
local times to the minute (`2026-03-02T08:04`). `load_timetable("data/timetable.json")` reads the timetable.

The rules, also in the module docstring of `departures.py`:

- The timetable is in Brisbane local time (UTC+10 all year; Queensland has no daylight saving), with separate lists
  for weekdays, Saturdays and Sundays.
- A question's time is an ISO time. With an offset (`+10:00`, or `Z` for UTC) it is that moment. Without an offset,
  as the SMS gateway sends it, it is the local clock time at the stop.
- A bus leaving in the minute asked still counts. After the last bus of the day, the answer carries on into the next
  day's timetable.

`data/answers.jsonl` is a fortnight of questions logged from all three channels in March, with `shown`, the
departures the old journey planner answered each one with. The old planner was retired because of its hosting costs,
not its answers: the operations team checked these against the timetable, and the new code should answer every
question with exactly the same departures.

Tests: `python -m unittest discover tests` from this directory.
