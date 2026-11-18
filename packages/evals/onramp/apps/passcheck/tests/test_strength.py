import unittest

from strength import band_of, rate


class BandTests(unittest.TestCase):
    def test_band_edges(self):
        self.assertEqual(
            [band_of(s) for s in (0, 39, 40, 59, 60, 79, 80, 100)],
            ["weak", "weak", "fair", "fair", "good", "good", "strong", "strong"],
        )


class RateTests(unittest.TestCase):
    def test_a_long_mixed_password_is_strong(self):
        rating = rate("Velvet^Otter^88", "ines")
        self.assertEqual((rating.score, rating.band, rating.reasons), (90, "strong", ()))

    def test_length_and_variety(self):
        self.assertEqual(rate("kqzmwvtp").score, 32)
        self.assertEqual(rate("kqzmwvtP").score, 42)
        self.assertEqual(rate("kqzmwvT8").score, 52)

    def test_a_common_password_is_weak(self):
        rating = rate("letmein")
        self.assertEqual(rating.band, "weak")
        self.assertIn("common word", rating.reasons[0])

    def test_a_common_word_inside_a_longer_password(self):
        self.assertEqual(rate("dragon-kq7").score, rate("plover-kq7").score - 25)

    def test_the_username_counts_against(self):
        self.assertEqual(rate("Fatima1990", "fatima").band, "weak")
        self.assertEqual(rate("Fatima1990", "jo").band, "good")

    def test_runs_and_repeats(self):
        self.assertEqual(rate("kqzm123wvt").score, rate("kqzm183wvt").score - 10)
        self.assertEqual(rate("kqzm888wvt").score, rate("kqzm818wvt").score - 10)
        self.assertIn("1 run(s) like abc or 123", rate("xyz-pqmw-41").reasons)

    def test_the_score_stays_in_range(self):
        self.assertEqual(rate("qwerty").score, 0)
        self.assertEqual(rate("Zq!8" * 20).score, 90)


if __name__ == "__main__":
    unittest.main()
