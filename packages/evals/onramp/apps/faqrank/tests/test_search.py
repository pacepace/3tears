import unittest

from search import Index, search, stem, terms


class TermTests(unittest.TestCase):
    def test_endings_and_stopwords(self):
        self.assertEqual(stem("refunds"), "refund")
        self.assertEqual(stem("billing"), "bill")
        self.assertEqual(stem("address"), "address")
        self.assertEqual(terms("How do I export my Invoices"), ["export", "invoice"])


class SearchTests(unittest.TestCase):
    def test_the_answer_comes_first(self):
        self.assertEqual(search("reset password")[0].id, "kb-101")
        self.assertEqual(search("refund")[0].id, "kb-201")
        self.assertEqual(search("cancel subscription")[0].id, "kb-202")
        self.assertEqual(search("export csv")[0].id, "kb-301")

    def test_best_first_and_at_most_k(self):
        hits = search("invoice", k=3)
        self.assertEqual(len(hits), 3)
        self.assertEqual([h.score for h in hits], sorted((h.score for h in hits), reverse=True))

    def test_nothing_matches(self):
        self.assertEqual(search("the of and"), [])
        self.assertEqual(search("zebra"), [])

    def test_a_title_counts_more_than_the_body(self):
        index = Index([
            {"id": "a", "title": "Webhooks", "body": "Set them up in settings."},
            {"id": "b", "title": "Settings", "body": "Webhooks and other settings."},
        ])
        self.assertEqual([h.id for h in index.search("webhooks")], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
