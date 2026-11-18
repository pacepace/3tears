# slotfinder

We are one team spread over nine office cities, and finding a time for a call used to mean Ines and a world clock.
`slots.py` does it: `free_slots(day, attendees, minutes)` lists every half-hour start in the UTC day `day` when a
meeting of `minutes` fits inside every attendee's working hours, and `first_slot` gives the earliest. An `Attendee`
has a name, an office city and working hours in that city's local time. `from_request(request)` reads a request as
`data/requests.jsonl` stores one.

Local times follow each city's own clocks through the year, summer time included, with the rules in the module
docstring and `CITIES`: London, Lisbon and Berlin change on the last Sundays of March and October; New York and
Toronto on the second Sunday of March and the first of November; Sydney's summer time starts on the first Sunday of
October and ends on the first Sunday of April; Singapore, Bangalore and Sao Paulo keep one time all year.

`data/requests.jsonl` is this year's log of requests that Ines worked out by hand: who was invited, the UTC day, the
meeting's length, and `offered`, the UTC start times (`HH:MM`) she found where everybody was free. The tool should
offer exactly those slots for each request.

Tests: `python -m unittest discover tests` from this directory.
