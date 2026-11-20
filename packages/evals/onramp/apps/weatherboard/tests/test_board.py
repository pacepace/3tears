import unittest

from board import ReadingError, board_line, display_c, to_celsius


class DisplayTests(unittest.TestCase):
    def test_landmarks(self):
        self.assertEqual(display_c(32), 0)
        self.assertEqual(display_c(50), 10)
        self.assertEqual(display_c(98.6), 37)
        self.assertEqual(display_c("68"), 20)

    def test_rounds_to_the_nearest_degree(self):
        self.assertEqual(display_c(37.94), 3)  # 3.3 °C
        self.assertEqual(display_c(38.66), 4)  # 3.7 °C
        self.assertEqual(display_c(45.5), 8)  # 7.5 °C: halves go up

    def test_unrounded_conversion(self):
        self.assertAlmostEqual(to_celsius(41.0), 5.0)

    def test_a_faulty_reading_is_refused(self):
        for bad in ("n/a", None, float("nan"), 451.0, -80):
            with self.assertRaises(ReadingError):
                display_c(bad)


class BoardLineTests(unittest.TestCase):
    def test_latest_low_and_high(self):
        readings = [
            {"time": "2026-01-19T14:00", "temp_f": 46.4},
            {"time": "2026-01-19T08:00", "temp_f": 39.2},
            {"time": "2026-01-19T11:00", "temp_f": 43.7},
        ]
        line = board_line("Market Hill", readings)
        self.assertEqual((line.now_c, line.low_c, line.high_c), (8, 4, 8))
        self.assertIn("Market Hill", line.text())

    def test_no_readings_is_refused(self):
        with self.assertRaises(ReadingError):
            board_line("Quay", [])


if __name__ == "__main__":
    unittest.main()
