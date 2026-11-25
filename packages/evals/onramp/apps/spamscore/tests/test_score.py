import tempfile
import unittest
from pathlib import Path

from score import Message, load_contacts, score, shouted, to_review

CONTACTS = frozenset({"orders@brightlane.co.uk"})


class SignalTests(unittest.TestCase):
    def test_a_plain_message_from_a_customer(self):
        self.assertEqual(score(Message("ana@gmail.com", "Order 12", "Where is my order?"), CONTACTS), 0.0)

    def test_a_link_from_a_stranger(self):
        self.assertEqual(score(Message("ana@gmail.com", "Photos", "See https://imgur.com/a/1"), CONTACTS), 3.0)

    def test_a_link_from_a_contact(self):
        message = Message("orders@brightlane.co.uk", "Tracking", "https://track.example.com/1")
        self.assertEqual(score(message, CONTACTS), 0.0)

    def test_a_contact_whose_replies_go_elsewhere_with_a_link(self):
        message = Message("orders@brightlane.co.uk", "Docs", "www.files.example/x", reply_to="x@other.example")
        self.assertEqual(score(message, CONTACTS), 1.0)

    def test_a_risky_attachment(self):
        message = Message("ana@gmail.com", "Invoice", "Please see attached.", attachments=("invoice.PDF.exe",))
        self.assertEqual(score(message, CONTACTS), 4.0)

    def test_money_words_count_once_each_and_at_most_three(self):
        body = "free free cash prize winner crypto"
        self.assertEqual(score(Message("x@spam.example", "Hello", body), CONTACTS), 3.0)

    def test_shouting(self):
        self.assertTrue(shouted("CLAIM YOUR PRIZE"))
        self.assertFalse(shouted("OK THEN"))
        self.assertFalse(shouted("Invoice 4471 query"))

    def test_spam_goes_to_review(self):
        message = Message("x@spam.example", "YOU ARE A WINNER!!!", "Claim your cash at http://spam.example")
        self.assertEqual(score(message, CONTACTS), 7.5)
        self.assertTrue(to_review(message, CONTACTS))

    def test_never_above_ten(self):
        message = Message("x@spam.example", "FREE CASH PRIZE!!!", "http://x.example", attachments=("a.exe",))
        self.assertEqual(score(message, CONTACTS), 10.0)


class ContactsTests(unittest.TestCase):
    def test_contacts_ignore_case_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contacts.txt"
            path.write_text("Orders@Brightlane.co.uk\n\n", encoding="utf-8")
            contacts = load_contacts(path)
        self.assertEqual(contacts, CONTACTS)
        self.assertEqual(score(Message("ORDERS@brightlane.co.uk", "Hi", "Thanks"), contacts), 0.0)


if __name__ == "__main__":
    unittest.main()
