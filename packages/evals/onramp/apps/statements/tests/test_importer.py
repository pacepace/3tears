import unittest
from datetime import date

from importer import StatementError, Transaction, parse_date, parse_pence, read_statement


class ParseTests(unittest.TestCase):
    def test_amounts(self):
        self.assertEqual(parse_pence("12.50"), 1250)
        self.assertEqual(parse_pence("-12.50"), -1250)
        self.assertEqual(parse_pence("(12.50)"), -1250)
        self.assertEqual(parse_pence("£1,234.56"), 123456)
        with self.assertRaises(StatementError):
            parse_pence("twelve")

    def test_dates(self):
        self.assertEqual(parse_date("2026-03-04"), date(2026, 3, 4))
        self.assertEqual(parse_date("25/03/2026"), date(2026, 3, 25))
        self.assertEqual(parse_date("31/01/2026"), date(2026, 1, 31))
        with self.assertRaises(StatementError):
            parse_date("March 4th")


class StatementTests(unittest.TestCase):
    def test_one_amount_column(self):
        text = 'Date,Description,Amount\n2026-03-14,TESCO STORES,-23.10\n2026-03-25,SALARY,"2,450.00"\n'
        self.assertEqual(
            read_statement(text),
            [Transaction(date(2026, 3, 14), "TESCO STORES", -2310), Transaction(date(2026, 3, 25), "SALARY", 245000)],
        )

    def test_paid_out_and_paid_in_columns_in_any_order(self):
        text = "Narrative,Date,Paid in,Paid out,Balance\nVET CLINIC,18/02/2026,,84.19,2828.11\nINTEREST,28/02/2026,3.87,,2831.98\n"
        self.assertEqual([t.pence for t in read_statement(text)], [-8419, 387])

    def test_blank_rows_are_skipped(self):
        self.assertEqual(read_statement(""), [])
        self.assertEqual(len(read_statement("Date,Details,Amount\n\n2026-01-20,BOOTS,(3.50)\n,,\n")), 1)

    def test_an_unknown_layout_is_refused(self):
        with self.assertRaises(StatementError):
            read_statement("When,What,How much\n2026-01-20,BOOTS,3.50\n")


if __name__ == "__main__":
    unittest.main()
