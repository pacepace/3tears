import unittest
from pathlib import Path

from digest import digest, digest_file, score, sentences

REPORT = (
    "The search page was down for ten minutes on Monday. Customers saw an error page instead of results. "
    "The team met afterwards. The root cause was a full disk on the search cluster. "
    "Disk was freed and search was restored at 10:12. Lunch was late."
)


class SentenceTests(unittest.TestCase):
    def test_a_report_splits_into_its_sentences(self):
        self.assertEqual(len(sentences(REPORT)), 6)
        self.assertEqual(sentences("One here. Two there!\nThree?"), ["One here.", "Two there!", "Three?"])

    def test_cues_figures_and_the_lead_score(self):
        self.assertGreater(score("Customers saw errors for twenty minutes today.", 3), 0)
        self.assertGreater(
            score("The outage began at 09:10 for everyone.", 0), score("The outage began for everyone.", 4)
        )
        self.assertLess(score("Lunch was late.", 5), 0)


class DigestTests(unittest.TestCase):
    def test_the_telling_sentences_in_report_order(self):
        self.assertEqual(
            digest(REPORT),
            "Customers saw an error page instead of results. The root cause was a full disk on the search cluster. "
            "Disk was freed and search was restored at 10:12.",
        )

    def test_fewer_sentences(self):
        self.assertEqual(len(sentences(digest(REPORT, max_sentences=2))), 2)

    def test_a_short_report_is_its_own_digest(self):
        self.assertEqual(digest("Nothing broke this week."), "Nothing broke this week.")

    def test_a_report_file(self):
        reports = sorted((Path(__file__).resolve().parents[1] / "data" / "reports").glob("*.txt"))
        self.assertTrue(reports)
        self.assertTrue(digest_file(reports[0]))


if __name__ == "__main__":
    unittest.main()
