import unittest
from datetime import date, datetime, timedelta, timezone

from slots import Attendee, first_slot, free_slots, in_summer_time, sunday, utc_offset, CITIES


def utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


class ClockTests(unittest.TestCase):
    def test_changeover_sundays(self):
        self.assertEqual(sunday(2026, CITIES["London"].summer_starts), date(2026, 3, 29))
        self.assertEqual(sunday(2026, CITIES["London"].summer_ends), date(2026, 10, 25))
        self.assertEqual(sunday(2026, CITIES["New York"].summer_starts), date(2026, 3, 8))
        self.assertEqual(sunday(2026, CITIES["New York"].summer_ends), date(2026, 11, 1))

    def test_northern_summer_time(self):
        self.assertFalse(in_summer_time("London", date(2026, 1, 15)))
        self.assertTrue(in_summer_time("London", date(2026, 7, 15)))
        self.assertTrue(in_summer_time("New York", date(2026, 3, 20)))
        self.assertFalse(in_summer_time("Singapore", date(2026, 7, 15)))

    def test_offsets(self):
        self.assertEqual(utc_offset("Berlin", utc("2026-07-01T12:00")), timedelta(hours=2))
        self.assertEqual(utc_offset("New York", utc("2026-01-12T15:00")), timedelta(hours=-5))
        self.assertEqual(utc_offset("Bangalore", utc("2026-01-12T15:00")), timedelta(hours=5, minutes=30))
        self.assertEqual(utc_offset("Sydney", utc("2026-07-01T00:00")), timedelta(hours=10))


class SlotTests(unittest.TestCase):
    def test_london_and_new_york_in_winter(self):
        people = [Attendee("Callum", "London", "08:30-17:30"), Attendee("Marisol", "New York", "08:00-17:00")]
        slots = free_slots(date(2026, 1, 14), people, 60)
        self.assertEqual(slots[0], utc("2026-01-14T13:00"))
        self.assertEqual(slots[-1], utc("2026-01-14T16:30"))

    def test_london_and_new_york_in_summer(self):
        people = [Attendee("Callum", "London", "08:30-17:30"), Attendee("Marisol", "New York", "08:00-17:00")]
        self.assertEqual(first_slot(date(2026, 6, 10), people, 30), utc("2026-06-10T12:00"))

    def test_sydney_and_singapore_in_july(self):
        people = [Attendee("Liam", "Sydney", "08:00-17:30"), Attendee("Wei Ling", "Singapore", "09:00-18:00")]
        slots = free_slots(date(2026, 7, 8), people, 30)
        self.assertEqual((slots[0], slots[-1]), (utc("2026-07-08T01:00"), utc("2026-07-08T07:00")))

    def test_quarter_hour_steps(self):
        people = [Attendee("Callum", "London", "08:30-17:30"), Attendee("Marisol", "New York", "08:00-17:00")]
        slots = free_slots(date(2026, 1, 14), people, 30, step=15)
        self.assertEqual(slots[:3], [utc("2026-01-14T13:00"), utc("2026-01-14T13:15"), utc("2026-01-14T13:30")])
        self.assertEqual(slots[-1], utc("2026-01-14T17:00"))

    def test_a_step_must_divide_the_day(self):
        with self.assertRaises(ValueError):
            free_slots(date(2026, 1, 14), [Attendee("Callum", "London")], step=25)

    def test_no_overlap_gives_nothing(self):
        people = [Attendee("Beatriz", "Sao Paulo", "09:00-12:00"), Attendee("Wei Ling", "Singapore", "09:00-18:00")]
        self.assertEqual(free_slots(date(2026, 2, 3), people), [])
        self.assertIsNone(first_slot(date(2026, 2, 3), people))
        self.assertEqual(free_slots(date(2026, 2, 3), []), [])


if __name__ == "__main__":
    unittest.main()
