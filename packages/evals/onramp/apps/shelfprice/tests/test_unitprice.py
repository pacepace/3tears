import unittest
from decimal import Decimal

from unitprice import SizeError, best_value, load_audit, pack_quantity, unit_price


class PackTests(unittest.TestCase):
    def test_weights_and_volumes(self):
        self.assertEqual(pack_quantity("500g"), (Decimal("0.500"), "kg"))
        self.assertEqual(pack_quantity("75cl"), (Decimal("0.75"), "l"))
        self.assertEqual(pack_quantity("1.5 KG"), (Decimal("1.5"), "kg"))

    def test_multipacks_and_counts(self):
        self.assertEqual(pack_quantity("6 x 330ml"), (Decimal("1.980"), "l"))
        self.assertEqual(pack_quantity("12 pack"), (Decimal(12), "each"))
        self.assertEqual(pack_quantity("4"), (Decimal(4), "each"))

    def test_unreadable_sizes_are_refused(self):
        for bad in ("", "a lot", "500 stone", "0g"):
            with self.assertRaises(SizeError):
                pack_quantity(bad)


class UnitPriceTests(unittest.TestCase):
    def test_per_kilogram(self):
        self.assertEqual(unit_price("1.35", "500g").label, "£2.70/kg")

    def test_per_litre(self):
        self.assertEqual(unit_price("8.99", "75cl").label, "£11.99/l")

    def test_per_item(self):
        self.assertEqual(unit_price("2.85", "12 pack").label, "£0.24 each")

    def test_rounded_half_up_to_a_penny(self):
        self.assertEqual(unit_price("0.25", "200g").pence, 125)
        self.assertEqual(unit_price("1.00", "3 each").pence, 33)

    def test_a_bad_price_is_refused(self):
        with self.assertRaises(SizeError):
            unit_price("free", "1kg")


class BestValueTests(unittest.TestCase):
    def test_the_cheapest_per_unit(self):
        products = [
            {"name": "small", "price": "1.20", "size": "250g"},
            {"name": "large", "price": "3.80", "size": "1kg"},
        ]
        self.assertEqual(best_value(products)["name"], "large")

    def test_different_units_are_not_compared(self):
        with self.assertRaises(SizeError):
            best_value([{"name": "a", "price": "1", "size": "1kg"}, {"name": "b", "price": "1", "size": "1l"}])

    def test_every_audited_line_prices(self):
        for product in load_audit():
            self.assertTrue(unit_price(product["price"], product["size"]).label.startswith("£"))


if __name__ == "__main__":
    unittest.main()
