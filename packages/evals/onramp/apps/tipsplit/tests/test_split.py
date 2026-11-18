import json
import tempfile
import unittest
from pathlib import Path

from split import Bill, split, split_file, total_pence


def bill(subtotal, tip=0, voucher=0, diners=("Ana", "Ben")):
    return Bill("T1", subtotal, tip, voucher, tuple(diners))


class TotalTests(unittest.TestCase):
    def test_a_tip(self):
        self.assertEqual(total_pence(bill(4000, tip=12)), 4480)

    def test_no_tip(self):
        self.assertEqual(total_pence(bill(4000)), 4000)

    def test_a_voucher_and_no_tip(self):
        self.assertEqual(total_pence(bill(4000, voucher=1500)), 2500)

    def test_a_half_penny_rounds_up(self):
        self.assertEqual(total_pence(bill(1005, tip=10)), 1106)

    def test_never_below_zero(self):
        self.assertEqual(total_pence(bill(800, voucher=1000)), 0)


class SplitTests(unittest.TestCase):
    def test_an_even_split(self):
        self.assertEqual(split(bill(4000, tip=10)), {"Ana": 2200, "Ben": 2200})

    def test_the_first_diners_take_the_odd_pennies(self):
        self.assertEqual(split(bill(1000, diners=("Ana", "Ben", "Cy"))), {"Ana": 334, "Ben": 333, "Cy": 333})

    def test_a_bill_needs_a_diner(self):
        with self.assertRaises(ValueError):
            split(bill(1000, diners=()))

    def test_a_file_of_bills(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bills.jsonl"
            rows = [
                {"table": "T1", "subtotal_pence": 5000, "tip_percent": 15, "diners": ["Ana", "Ben"]},
                {"table": "T2", "subtotal_pence": 3000, "voucher_pence": 500, "diners": ["Cy"]},
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            self.assertEqual(split_file(path), {"T1": {"Ana": 2875, "Ben": 2875}, "T2": {"Cy": 2500}})


if __name__ == "__main__":
    unittest.main()
