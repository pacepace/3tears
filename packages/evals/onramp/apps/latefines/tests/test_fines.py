import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from fines import Closure, Loan, chargeable_days, closed_days, fine, fines_file, late_days

WINTER = Closure(date(2026, 12, 24), date(2026, 12, 28))


def loan(item_type="book", reader="adult", due="2026-03-02", returned="2026-03-05"):
    return Loan("L1", item_type, reader, date.fromisoformat(due), date.fromisoformat(returned))


class LateDayTests(unittest.TestCase):
    def test_late_days_run_up_to_the_return(self):
        self.assertEqual(late_days(date(2026, 3, 2), date(2026, 3, 4)), [date(2026, 3, 3), date(2026, 3, 4)])

    def test_back_on_time(self):
        self.assertEqual(late_days(date(2026, 3, 2), date(2026, 3, 2)), [])

    def test_sundays_are_free(self):
        # Friday 6 March to Tuesday 10 March: Saturday, Sunday, Monday, Tuesday are late, and Sunday is closed.
        self.assertEqual(chargeable_days(loan(due="2026-03-06", returned="2026-03-10"), []), 3)

    def test_the_winter_break_is_closed(self):
        closed = closed_days([WINTER])
        self.assertIn(date(2026, 12, 24), closed)
        self.assertIn(date(2026, 12, 26), closed)

    def test_a_loan_back_during_the_break(self):
        # Due Tuesday 22 December, back Saturday 26 December: only the 23rd was open.
        self.assertEqual(chargeable_days(loan(due="2026-12-22", returned="2026-12-26"), [WINTER]), 1)


class FineTests(unittest.TestCase):
    def test_a_late_book(self):
        self.assertEqual(fine(loan(), []), 75)

    def test_the_cap(self):
        self.assertEqual(fine(loan(item_type="dvd", returned="2026-03-31"), []), 1200)

    def test_junior_and_concession_readers_pay_nothing(self):
        self.assertEqual(fine(loan(reader="junior"), []), 0)
        self.assertEqual(fine(loan(reader="concession"), []), 0)

    def test_a_returns_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            returns = Path(tmp) / "returns.jsonl"
            closures = Path(tmp) / "closures.jsonl"
            rows = [
                {
                    "loan_id": "a",
                    "item_type": "magazine",
                    "reader": "adult",
                    "due": "2026-12-21",
                    "returned": "2026-12-27",
                },
                {
                    "loan_id": "b",
                    "item_type": "laptop",
                    "reader": "adult",
                    "due": "2026-03-02",
                    "returned": "2026-03-03",
                },
            ]
            returns.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            closures.write_text(json.dumps({"first": "2026-12-24", "last": "2026-12-28"}) + "\n", encoding="utf-8")
            self.assertEqual(fines_file(returns, closures), {"a": 20, "b": 500})


if __name__ == "__main__":
    unittest.main()
