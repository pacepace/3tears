import unittest
from datetime import datetime

from eta import LANES, departure, estimate, hours_between, next_round, transit_hours


class StepTests(unittest.TestCase):
    def test_the_cutoff(self):
        self.assertEqual(departure(datetime(2026, 9, 1, 14, 59)), datetime(2026, 9, 1, 18, 0))
        self.assertEqual(departure(datetime(2026, 9, 1, 15, 0)), datetime(2026, 9, 2, 18, 0))

    def test_rounds(self):
        self.assertEqual(next_round(datetime(2026, 9, 2, 6, 10)), datetime(2026, 9, 2, 8, 0))
        self.assertEqual(next_round(datetime(2026, 9, 2, 8, 0)), datetime(2026, 9, 2, 8, 0))
        self.assertEqual(next_round(datetime(2026, 9, 2, 9, 30)), datetime(2026, 9, 2, 13, 0))
        self.assertEqual(next_round(datetime(2026, 9, 2, 22, 0)), datetime(2026, 9, 3, 8, 0))

    def test_a_domestic_lane(self):
        lane = LANES[("MAN", "LDN")]
        self.assertAlmostEqual(transit_hours(lane), 335 / 72)


class EstimateTests(unittest.TestCase):
    def test_overnight_to_london(self):
        # leaves 18:00, in at 22:39, sorted by 00:09, out on the 08:00 round
        self.assertEqual(estimate(datetime(2026, 9, 1, 10, 0), "MAN", "LDN"), datetime(2026, 9, 2, 10, 0))

    def test_booked_after_the_cutoff(self):
        self.assertEqual(estimate(datetime(2026, 9, 1, 16, 0), "MAN", "LDN"), datetime(2026, 9, 3, 10, 0))

    def test_a_long_lane_with_a_hub(self):
        # 665 km and one hub: in at 05:44, sorted by 07:14, out at 08:00
        self.assertEqual(estimate(datetime(2026, 9, 1, 9, 0), "LDN", "GLA"), datetime(2026, 9, 2, 10, 0))

    def test_hours_between(self):
        self.assertEqual(hours_between(datetime(2026, 9, 2, 10, 0), datetime(2026, 9, 2, 8, 30)), 1.5)


if __name__ == "__main__":
    unittest.main()
