import unittest
from datetime import date

from proration import Adjustment, bill_day, cycle, prorate, share


class CycleTests(unittest.TestCase):
    def test_bill_days(self):
        self.assertEqual(bill_day(2026, 2, 31), date(2026, 2, 28))
        self.assertEqual(bill_day(2026, 3, 15), date(2026, 3, 15))

    def test_cycles(self):
        self.assertEqual(cycle(1, date(2026, 6, 10)), (date(2026, 6, 1), date(2026, 7, 1)))
        self.assertEqual(cycle(15, date(2026, 6, 10)), (date(2026, 5, 15), date(2026, 6, 15)))
        self.assertEqual(cycle(31, date(2026, 2, 28)), (date(2026, 2, 28), date(2026, 3, 31)))
        self.assertEqual(cycle(20, date(2026, 12, 25)), (date(2026, 12, 20), date(2027, 1, 20)))


class ProrateTests(unittest.TestCase):
    def test_share(self):
        self.assertEqual(share(3000, 10, 30), 1000)
        self.assertEqual(share(2899, 10, 30), 966)

    def test_an_upgrade_a_third_of_the_way_through(self):
        # 1 June to 1 July is 30 days; from 21 June, 10 are left
        adjustment = prorate("team", "business", 1, date(2026, 6, 21))
        self.assertEqual(adjustment, Adjustment(966, 2483))
        self.assertEqual(adjustment.net_pence, 1517)

    def test_a_downgrade_leaves_credit(self):
        self.assertLess(prorate("business", "team", 1, date(2026, 6, 21)).net_pence, 0)

    def test_a_change_on_the_bill_day_is_a_whole_cycle(self):
        self.assertEqual(prorate("starter", "plus", 12, date(2026, 7, 12)), Adjustment(599, 1199))


if __name__ == "__main__":
    unittest.main()
