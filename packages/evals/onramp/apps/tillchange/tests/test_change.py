import json
import tempfile
import unittest
from pathlib import Path

from change import CannotMakeChange, change_for, load_drawer, make_change, pence

FULL = {"5": 10, "2": 10, "1": 10, "0.50": 10, "0.20": 10, "0.10": 10, "0.05": 10, "0.02": 10, "0.01": 10}


class PenceTests(unittest.TestCase):
    def test_values(self):
        self.assertEqual([pence(d) for d in ("5", "0.50", "0.05", "0.01")], [500, 50, 5, 1])


class ChangeTests(unittest.TestCase):
    def test_coins(self):
        self.assertEqual(make_change(88, FULL), {"0.50": 1, "0.20": 1, "0.10": 1, "0.05": 1, "0.02": 1, "0.01": 1})

    def test_notes_and_coins(self):
        self.assertEqual(make_change(870, FULL), {"5": 1, "2": 1, "1": 1, "0.50": 1, "0.20": 1})

    def test_nothing_owed(self):
        self.assertEqual(make_change(0, FULL), {})

    def test_a_short_drawer_uses_smaller_coins(self):
        self.assertEqual(make_change(40, {"0.20": 1, "0.10": 5}), {"0.20": 1, "0.10": 2})

    def test_a_drawer_that_cannot_make_it(self):
        with self.assertRaises(CannotMakeChange):
            make_change(3, {"0.02": 5})

    def test_a_sale(self):
        self.assertEqual(change_for(1290, 1500, FULL), {"2": 1, "0.10": 1})

    def test_not_enough_tendered(self):
        with self.assertRaises(ValueError):
            change_for(1290, 1000, FULL)

    def test_a_float_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "float.json"
            path.write_text(json.dumps({"0.50": 2, "0.20": 3}), encoding="utf-8")
            self.assertEqual(make_change(90, load_drawer(path)), {"0.50": 1, "0.20": 2})


if __name__ == "__main__":
    unittest.main()
