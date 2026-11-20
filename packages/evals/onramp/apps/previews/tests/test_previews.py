import unittest

from previews import ELLIPSIS, LIMITS, flatten, preview, previews


class PreviewTests(unittest.TestCase):
    def test_a_short_message_is_shown_whole(self):
        self.assertEqual(preview("Priya: running late", "watch"), "Priya: running late")

    def test_whitespace_is_flattened(self):
        self.assertEqual(flatten("a\n\n b\t c  "), "a b c")
        self.assertEqual(preview("Calendar:\n  standup   now", "banner"), "Calendar: standup now")

    def test_a_long_message_is_cut_with_an_ellipsis(self):
        body = "Tom: can someone look at the staging deploy, the health check has been red all morning"
        shown = preview(body, "watch")
        self.assertTrue(shown.endswith(ELLIPSIS))
        self.assertTrue(body.startswith(shown[:-1]))

    def test_no_punctuation_before_the_ellipsis(self):
        shown = preview("Joy: lunch? thinking the noodle place, or the new one on Ash Street", "watch")
        self.assertNotIn(shown[-2], ",.;: ")

    def test_every_channel(self):
        self.assertEqual(set(previews("hi")), set(LIMITS))

    def test_an_unknown_channel_is_refused(self):
        with self.assertRaises(ValueError):
            preview("hi", "pager")


if __name__ == "__main__":
    unittest.main()
