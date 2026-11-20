import unittest
from decimal import Decimal

from rates import QuoteError, quote, quote_many


class QuoteTests(unittest.TestCase):
    def test_a_light_domestic_parcel_is_a_small_parcel(self):
        q = quote("0.4", "domestic")
        self.assertEqual((q.band, q.pence, q.price), ("small parcel", 355, "£3.55"))

    def test_weights_in_the_middle_of_a_band(self):
        self.assertEqual(quote("1.4", "domestic").pence, 495)
        self.assertEqual(quote("3.2", "europe").pence, 2100)
        self.assertEqual(quote(7.5, "world").pence, 5250)

    def test_signed_for_adds_the_surcharge(self):
        self.assertEqual(quote("3.2", "europe", signed=True).pence, 2100 + 150)

    def test_the_weight_is_read_as_a_decimal(self):
        self.assertEqual(quote(" 0.75 ", "domestic").weight_kg, Decimal("0.75"))

    def test_too_heavy_is_refused(self):
        with self.assertRaises(QuoteError):
            quote("24.5", "domestic")
        with self.assertRaises(QuoteError):
            quote("12", "world")

    def test_an_unknown_zone_or_a_bad_weight_is_refused(self):
        with self.assertRaises(QuoteError):
            quote("1.5", "mars")
        for bad in ("", "heavy", "-1", "0", "nan"):
            with self.assertRaises(QuoteError):
                quote(bad, "domestic")

    def test_a_batch_is_priced_in_order(self):
        parcels = [{"weight_kg": "0.3", "zone": "europe"}, {"weight_kg": "6.1", "zone": "domestic", "signed": True}]
        self.assertEqual([q.pence for q in quote_many(parcels)], [995, 1049 + 150])


if __name__ == "__main__":
    unittest.main()
