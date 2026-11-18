import unittest

from standings import Entry, Place, season, table


class TableTests(unittest.TestCase):
    def test_more_points_ranks_higher(self):
        night = [
            Entry("Smarty Pints", 31, "21:44:10"),
            Entry("Bad Hombres", 40, "21:47:00"),
            Entry("Quiz Khalifa", 35, "21:41:00"),
        ]
        self.assertEqual(
            table(night),
            [Place(1, "Bad Hombres", 40), Place(2, "Quiz Khalifa", 35), Place(3, "Smarty Pints", 31)],
        )

    def test_level_on_both_shares_a_place(self):
        night = [
            Entry("A", 40, "21:45:00"),
            Entry("B", 38, "21:46:30"),
            Entry("C", 38, "21:46:30"),
            Entry("D", 30, "21:40:00"),
        ]
        self.assertEqual([(p.place, p.team) for p in table(night)], [(1, "A"), (2, "B"), (2, "C"), (4, "D")])

    def test_an_empty_night(self):
        self.assertEqual(table([]), [])

    def test_rows_are_read(self):
        self.assertEqual(
            Entry.from_row({"team": " The Anchormen ", "points": "33", "handed_in": "21:50:02"}),
            Entry("The Anchormen", 33, "21:50:02"),
        )


class SeasonTests(unittest.TestCase):
    def test_league_points_add_up(self):
        first = [Entry("A", 40, "21:45:00"), Entry("B", 30, "21:46:00")]
        second = [Entry("A", 20, "21:45:00"), Entry("B", 30, "21:46:00"), Entry("C", 25, "21:42:00")]
        self.assertEqual(season([first, second]), [("B", 18), ("A", 16), ("C", 8)])

    def test_a_shared_place_scores_for_both(self):
        night = [Entry("A", 40, "21:45:00"), Entry("B", 40, "21:45:00"), Entry("C", 10, "21:41:00")]
        self.assertEqual(season([night]), [("A", 10), ("B", 10), ("C", 6)])


if __name__ == "__main__":
    unittest.main()
