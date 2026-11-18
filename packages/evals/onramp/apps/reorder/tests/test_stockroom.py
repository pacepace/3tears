import unittest

from stockroom import Line, daily_demand, load_lines, order_quantity, order_sheet, reorder_point


def line(**changes):
    fields = dict(
        sku="T-1", name="test line", sold=56, window_days=28, lead_days=5, safety_days=3, on_hand=10, on_order=0, pack=1
    )
    fields.update(changes)
    return Line(**fields)


class DemandTests(unittest.TestCase):
    def test_a_steady_seller(self):
        self.assertEqual(daily_demand(56, 28), 2)
        self.assertEqual(daily_demand(0, 28), 0)

    def test_a_window_must_have_days_and_sales_cannot_be_negative(self):
        with self.assertRaises(ValueError):
            daily_demand(10, 0)
        with self.assertRaises(ValueError):
            daily_demand(-1, 28)


class ReorderTests(unittest.TestCase):
    def test_the_reorder_point_covers_lead_time_and_safety_stock(self):
        self.assertEqual(reorder_point(line()), 2 * (5 + 3))

    def test_a_line_above_its_reorder_point_orders_nothing(self):
        self.assertEqual(order_quantity(line(on_hand=17)), 0)

    def test_a_due_line_orders_up_to_the_point_plus_cover(self):
        # Reorder point 16, plus 14 days at 2 a day: 44 wanted, 10 held.
        self.assertEqual(order_quantity(line()), 34)

    def test_what_is_on_order_counts_towards_the_position(self):
        self.assertEqual(order_quantity(line(on_hand=4, on_order=6)), 34)

    def test_orders_come_in_whole_packs(self):
        self.assertEqual(order_quantity(line(pack=12)), 36)

    def test_a_line_that_does_not_sell_orders_nothing(self):
        self.assertEqual(order_quantity(line(sold=0, on_hand=0)), 0)


class SheetTests(unittest.TestCase):
    def test_the_review_file_loads(self):
        lines = load_lines()
        self.assertTrue(lines)
        self.assertEqual(len(order_sheet(lines)), len(lines))
        self.assertTrue(all(isinstance(point, int) and order >= 0 for _, point, order in order_sheet(lines)))


if __name__ == "__main__":
    unittest.main()
