import unittest

from sentiment import mood, raw_score, score, tokens


class TokenTests(unittest.TestCase):
    def test_words_and_clause_breaks(self):
        self.assertEqual(tokens("Didn't LOVE it, sorry!"), ["didn't", "love", "it", ",", "sorry", "!"])

    def test_a_curly_apostrophe_is_a_straight_one(self):
        self.assertEqual(tokens("wasn’t"), ["wasn't"])


class ScoreTests(unittest.TestCase):
    def test_weights_add_up(self):
        self.assertEqual(raw_score("good food, friendly staff"), 4.0)

    def test_a_negator_turns_round_and_shrinks_the_next_sentiment_word(self):
        self.assertEqual(raw_score("not good"), -1.5)
        self.assertEqual(raw_score("wasn't very good"), -2.25)

    def test_negation_reaches_three_words(self):
        self.assertEqual(raw_score("not at all good"), -1.5)
        self.assertEqual(raw_score("not in the slightest good"), 2.0)

    def test_a_clause_break_ends_a_negation(self):
        self.assertEqual(raw_score("not hot, good though"), 2.0)
        self.assertEqual(raw_score("not hot but good"), 2.0)

    def test_an_intensifier_counts_half_again(self):
        self.assertEqual(raw_score("really friendly"), 3.0)

    def test_the_score_is_scaled_into_minus_one_to_one(self):
        self.assertEqual(score("amazing amazing amazing amazing amazing"), 0.968)
        self.assertEqual(score("awful"), -0.612)
        self.assertEqual(score("we ordered at six"), 0.0)

    def test_mood(self):
        self.assertEqual([mood(t) for t in ("lovely", "rude", "ok then")], ["positive", "negative", "positive"])
        self.assertEqual(mood("arrived on time"), "neutral")


if __name__ == "__main__":
    unittest.main()
