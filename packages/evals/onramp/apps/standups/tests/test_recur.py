import unittest
from datetime import date, datetime, time, timedelta, timezone

from recur import LONDON, Series, invites, load_series, occurrences


def series(**changes):
    fields = dict(title="Stand-up", first=date(2026, 1, 12), at=time(9, 30), every="weekday", count=5)
    fields.update(changes)
    return Series(**fields)


class LondonTimeTests(unittest.TestCase):
    def test_winter_is_gmt_and_summer_is_bst(self):
        self.assertEqual(datetime(2026, 1, 15, 12, 0, tzinfo=LONDON).utcoffset(), timedelta(0))
        self.assertEqual(datetime(2026, 7, 15, 12, 0, tzinfo=LONDON).utcoffset(), timedelta(hours=1))
        self.assertEqual(datetime(2026, 7, 15, 12, 0, tzinfo=LONDON).tzname(), "BST")

    def test_utc_converts_to_london_on_both_sides_of_the_change(self):
        before = datetime(2026, 3, 29, 0, 30, tzinfo=timezone.utc).astimezone(LONDON)
        after = datetime(2026, 3, 29, 1, 30, tzinfo=timezone.utc).astimezone(LONDON)
        self.assertEqual((before.hour, before.minute), (0, 30))
        self.assertEqual((after.hour, after.minute), (2, 30))


class OccurrenceTests(unittest.TestCase):
    def test_a_weekday_series_skips_the_weekend(self):
        days = [when.date() for when in occurrences(series(first=date(2026, 1, 15), count=4))]
        self.assertEqual(days, [date(2026, 1, 15), date(2026, 1, 16), date(2026, 1, 19), date(2026, 1, 20)])

    def test_a_skipped_date_does_not_count(self):
        skipped = series(every="week", count=3, skip=frozenset({date(2026, 1, 19)}))
        self.assertEqual(
            [when.date() for when in occurrences(skipped)], [date(2026, 1, 12), date(2026, 1, 26), date(2026, 2, 2)]
        )

    def test_a_fortnightly_series(self):
        days = [when.date() for when in occurrences(series(every="fortnight", count=3, first=date(2026, 6, 3)))]
        self.assertEqual(days, [date(2026, 6, 3), date(2026, 6, 17), date(2026, 7, 1)])

    def test_meetings_are_at_the_series_time(self):
        self.assertTrue(all(when.time() == time(9, 30) for when in occurrences(series(count=10))))

    def test_an_unknown_repeat_or_an_empty_series_is_refused(self):
        with self.assertRaises(ValueError):
            series(every="month")
        with self.assertRaises(ValueError):
            series(count=0)


class InviteTests(unittest.TestCase):
    def test_winter_invites_are_the_wall_clock_time(self):
        self.assertEqual(invites(series(count=2)), ["2026-01-12T09:30Z", "2026-01-13T09:30Z"])

    def test_summer_invites_are_an_hour_earlier_in_utc(self):
        summer = series(first=date(2026, 6, 1), every="week", count=3, at=time(14, 0))
        self.assertEqual(invites(summer), ["2026-06-01T13:00Z", "2026-06-08T13:00Z", "2026-06-15T13:00Z"])

    def test_a_series_reads_from_json(self):
        row = {"title": "Retro", "first": "2026-02-04", "at": "16:00", "every": "week", "count": 2, "skip": []}
        self.assertEqual(invites(Series.from_json(row)), ["2026-02-04T16:00Z", "2026-02-11T16:00Z"])

    def test_the_booked_series_load(self):
        booked = load_series()
        self.assertTrue(booked)
        self.assertTrue(all(len(invites(one)) == one.count for one in booked))


if __name__ == "__main__":
    unittest.main()
