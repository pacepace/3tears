import unittest
from datetime import date

from naming import MAX_SLUG, slug, stored_name


class SlugTests(unittest.TestCase):
    def test_words_are_lower_case_and_hyphenated(self):
        self.assertEqual(slug("Spring Menu Costings"), "spring-menu-costings")

    def test_punctuation_separates_words(self):
        self.assertEqual(slug("Health & Safety policy (rev. 7)"), "health-safety-policy-rev-7")
        self.assertEqual(slug("Wine list 2025/26"), "wine-list-2025-26")
        self.assertEqual(slug("--Q1 VAT return--"), "q1-vat-return")

    def test_a_long_title_is_cut_back_to_a_whole_word(self):
        title = "Minutes of the quarterly operations meeting with the area managers and the finance team"
        cut = slug(title)
        self.assertLessEqual(len(cut), MAX_SLUG)
        self.assertTrue(title.lower().replace(" ", "-").startswith(cut))
        self.assertFalse(cut.endswith("-"))

    def test_nothing_left_is_untitled(self):
        self.assertEqual(slug(""), "untitled")
        self.assertEqual(slug("   ...  "), "untitled")


class StoredNameTests(unittest.TestCase):
    def test_date_slug_and_extension(self):
        self.assertEqual(stored_name("Q1 VAT return", "2026-04-02", "PDF"), "2026-04-02-q1-vat-return.pdf")
        self.assertEqual(stored_name("Rota", date(2026, 1, 5), ".xlsx"), "2026-01-05-rota.xlsx")

    def test_a_bad_extension_is_refused(self):
        with self.assertRaises(ValueError):
            stored_name("Rota", "2026-01-05", "tar.gz")


if __name__ == "__main__":
    unittest.main()
