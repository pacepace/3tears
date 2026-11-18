import unittest

from messages import TemplateError, choose, number, render


class PartsTests(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(number(1204), "1,204")
        self.assertEqual(number(3.0), "3")
        self.assertEqual(number(1.5), "1.5")
        self.assertEqual(number("Drafts"), "Drafts")

    def test_forms(self):
        self.assertEqual(choose(1, ["one", "other"]), "one")
        self.assertEqual(choose(0, ["one", "other"]), "other")
        self.assertEqual(choose(0, ["zero", "one", "other"]), "zero")
        with self.assertRaises(TemplateError):
            choose("3", ["one", "other"])
        with self.assertRaises(TemplateError):
            choose(3, ["one"])


class RenderTests(unittest.TestCase):
    def test_a_plain_field(self):
        self.assertEqual(render("Hello {name}!", {"name": "Priya"}), "Hello Priya!")
        self.assertEqual(render("{n} views", {"n": 20481}), "20,481 views")

    def test_two_fields(self):
        self.assertEqual(
            render("{owner} shared {count|a file|# files}", {"owner": "Tom", "count": 1}), "Tom shared a file"
        )

    def test_plural_fields(self):
        template = "{count|no messages|one message|# messages}"
        self.assertEqual(
            [render(template, {"count": n}) for n in (0, 1, 2, 15)],
            ["no messages", "one message", "2 messages", "15 messages"],
        )

    def test_a_plural_in_a_sentence(self):
        self.assertEqual(
            render("Sync finished: {count|one file|# files} updated", {"count": 7}), "Sync finished: 7 files updated"
        )

    def test_a_missing_value_is_left_as_written(self):
        self.assertEqual(render("Moved to {folder}", {}), "Moved to {folder}")

    def test_literal_braces(self):
        self.assertEqual(render("Write {{name}} for {who}", {"who": "everyone"}), "Write {name} for everyone")


if __name__ == "__main__":
    unittest.main()
