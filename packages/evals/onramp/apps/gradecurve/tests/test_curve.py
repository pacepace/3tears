import tempfile
import unittest
from pathlib import Path

from curve import Result, curve_file, curve_section, letter, parse_score, shift


class PartsTests(unittest.TestCase):
    def test_scores(self):
        self.assertEqual(parse_score(" 71.5 "), 71.5)
        self.assertIsNone(parse_score(""))

    def test_letters(self):
        self.assertEqual([letter(s) for s in (95, 90, 89.5, 80, 70, 60, 59.5)], ["A", "A", "B", "B", "C", "D", "F"])
        self.assertEqual(letter(None), "I")

    def test_shift_moves_up_to_the_target(self):
        self.assertEqual(shift([60.0, 64.0, 68.0]), 8.0)

    def test_shift_never_moves_down(self):
        self.assertEqual(shift([80.0, 90.0]), 0.0)


class SectionTests(unittest.TestCase):
    def test_a_section_is_curved(self):
        results = curve_section([("s1", 50.0), ("s2", 60.0), ("s3", 70.0), ("s4", 98.0)])
        self.assertEqual([r.curved for r in results], [52.5, 62.5, 72.5, 100.0])
        self.assertEqual([r.letter for r in results], ["F", "D", "C", "A"])

    def test_a_strong_section_keeps_its_scores_and_absentees_are_incomplete(self):
        results = curve_section([("s1", 95.0), ("s2", 90.0), ("s3", None), ("s4", 88.0), ("s5", 92.0)])
        self.assertEqual(results[2], Result("s3", None, None, "I"))
        self.assertEqual([r.letter for r in results], ["A", "A", "I", "B", "A"])

    def test_nobody_sat_it(self):
        self.assertEqual([r.letter for r in curve_section([("s1", None)])], ["I"])

    def test_a_marks_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            sheet = Path(tmp) / "section-Z.csv"
            sheet.write_text("student,score\ns1,70\ns2,64.5\ns3,75.5\n", encoding="utf-8")
            self.assertEqual([r.letter for r in curve_file(sheet)], ["C", "D", "C"])


if __name__ == "__main__":
    unittest.main()
