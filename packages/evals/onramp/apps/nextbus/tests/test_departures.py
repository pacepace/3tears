import unittest
from datetime import date, datetime

from departures import LOCAL, asked_at, departures_on, next_departures

TIMETABLE = {
    "Depot": {
        "weekday": ["06:00", "06:30", "07:00", "23:00"],
        "saturday": ["08:00", "09:00"],
        "sunday": ["10:00"],
    }
}


class AskedAtTests(unittest.TestCase):
    def test_a_local_offset(self):
        self.assertEqual(asked_at("2026-03-02T06:10:00+10:00"), datetime(2026, 3, 2, 6, 10, tzinfo=LOCAL))

    def test_utc(self):
        self.assertEqual(asked_at("2026-03-01T20:10:00Z"), datetime(2026, 3, 2, 6, 10, tzinfo=LOCAL))


class TimetableTests(unittest.TestCase):
    def test_the_day_picks_its_list(self):
        self.assertEqual([d.hour for d in departures_on(TIMETABLE["Depot"], date(2026, 3, 7))], [8, 9])
        self.assertEqual([d.hour for d in departures_on(TIMETABLE["Depot"], date(2026, 3, 8))], [10])

    def test_the_next_three(self):
        self.assertEqual(
            next_departures(TIMETABLE, "Depot", "2026-03-02T06:10:00+10:00"),
            ["2026-03-02T06:30", "2026-03-02T07:00", "2026-03-02T23:00"],
        )

    def test_a_bus_in_the_minute_asked_counts(self):
        self.assertEqual(next_departures(TIMETABLE, "Depot", "2026-03-02T06:30:40+10:00", 1), ["2026-03-02T06:30"])

    def test_after_the_last_bus(self):
        # Friday night: the next buses are Saturday's.
        self.assertEqual(
            next_departures(TIMETABLE, "Depot", "2026-03-06T13:30:00Z"),
            ["2026-03-07T08:00", "2026-03-07T09:00", "2026-03-08T10:00"],
        )

    def test_an_unknown_stop(self):
        with self.assertRaises(KeyError):
            next_departures(TIMETABLE, "Nowhere", "2026-03-02T06:10:00+10:00")


if __name__ == "__main__":
    unittest.main()
