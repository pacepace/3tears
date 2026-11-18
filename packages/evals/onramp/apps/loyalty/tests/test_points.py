import unittest

from points import Line, eligible_spend, load_export, points_for, statement


def basket(*pence, category="produce"):
    return [Line(item=f"item {n}", category=category, pence=p) for n, p in enumerate(pence)]


class SpendTests(unittest.TestCase):
    def test_excluded_categories_earn_nothing(self):
        lines = basket(1000) + [Line("Lottery ticket", "lottery", 200), Line("Cigarettes", "Tobacco", 1650)]
        self.assertEqual(eligible_spend(lines), 1000)

    def test_a_refund_takes_its_amount_off(self):
        self.assertEqual(eligible_spend(basket(1000, 450, -450)), 1000)


class PointsTests(unittest.TestCase):
    def test_a_point_per_whole_pound(self):
        self.assertEqual(points_for(basket(2399), "silver").base, 23)

    def test_gold_is_always_double(self):
        self.assertEqual(points_for(basket(1250), "gold").total, 24)

    def test_silver_is_double_from_forty_pounds(self):
        self.assertEqual(points_for(basket(3999), "silver").total, 39)
        self.assertEqual(points_for(basket(4000), "silver").total, 80)
        self.assertEqual(points_for(basket(5550), "silver").total, 110)

    def test_bronze_at_the_threshold_gets_the_bonus(self):
        self.assertEqual(points_for(basket(6000), "bronze").total, 60 + 25)

    def test_nothing_spent_earns_nothing(self):
        self.assertEqual(points_for(basket(500, -500), "bronze").total, 0)
        self.assertEqual(points_for([], "gold").total, 0)

    def test_an_unknown_tier_is_refused(self):
        with self.assertRaises(ValueError):
            points_for(basket(1000), "platinum")
        with self.assertRaises(ValueError):
            points_for([], "platinum")


class ExportTests(unittest.TestCase):
    def test_the_export_reads_and_every_basket_earns(self):
        baskets = load_export()
        self.assertTrue(baskets)
        self.assertGreater(statement(baskets), sum(points_for([], row["tier"]).total for row in baskets))


if __name__ == "__main__":
    unittest.main()
