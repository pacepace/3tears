import unittest

from mdtable import TableError, clean, format_table


class CleanTests(unittest.TestCase):
    def test_whitespace_and_pipes(self):
        self.assertEqual(clean("  two   words\n"), "two words")
        self.assertEqual(clean("a | b"), "a \\| b")
        self.assertEqual(clean(None), "")
        self.assertEqual(clean(42), "42")

    def test_invisible_characters_go(self):
        self.assertEqual(clean("﻿Office​"), "Office")


class TableTests(unittest.TestCase):
    def test_columns_line_up(self):
        self.assertEqual(
            format_table(["Name", "Ext."], [["Sam Hughes", 214], ["Jo", 3001]], ["l", "r"]),
            "| Name       | Ext. |\n| :--------- | ---: |\n| Sam Hughes |  214 |\n| Jo         | 3001 |",
        )

    def test_centred_and_unaligned(self):
        self.assertEqual(
            format_table(["Office", "Days"], [["Leeds", "28"]], ["c"]),
            "| Office | Days |\n| :----: | ---- |\n| Leeds  | 28   |",
        )

    def test_narrow_columns_are_three_wide(self):
        self.assertEqual(format_table(["A"], [["1"]]), "| A   |\n| --- |\n| 1   |")

    def test_accented_names_written_as_one_character(self):
        self.assertEqual(
            format_table(["Lead"], [["Zoë Clarke"], ["Tom"]]),
            "| Lead       |\n| ---------- |\n| Zoë Clarke |\n| Tom        |",
        )

    def test_short_rows_are_filled_and_long_ones_refused(self):
        self.assertEqual(format_table(["A", "B"], [["x"]]), "| A   | B   |\n| --- | --- |\n| x   |     |")
        with self.assertRaises(TableError):
            format_table(["A"], [["x", "y"]])
        with self.assertRaises(TableError):
            format_table(["A"], [["x"]], ["left"])


if __name__ == "__main__":
    unittest.main()
