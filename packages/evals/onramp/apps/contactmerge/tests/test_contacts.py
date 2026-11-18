import tempfile
import unittest
from pathlib import Path

from contacts import Contact, merge, merge_exports, phone_key, same_person


class KeyTests(unittest.TestCase):
    def test_phone_numbers_compare_by_digits(self):
        self.assertEqual(phone_key("07700 900123"), "447700900123")
        self.assertEqual(phone_key("+44 7700 900123"), "447700900123")
        self.assertEqual(phone_key("0044 7700-900-123"), "447700900123")
        self.assertEqual(phone_key(""), "")

    def test_rows_columns_are_read_in_any_case(self):
        contact = Contact.from_row({"Full Name": "Ada Price", "E-mail": " ada@price.co.uk ", "Mobile": "07700 900001"})
        self.assertEqual((contact.name, contact.email, contact.phone), ("Ada Price", "ada@price.co.uk", "07700 900001"))


class SamePersonTests(unittest.TestCase):
    def test_a_shared_email_is_one_person(self):
        self.assertTrue(same_person(Contact("Ada Price", "ada@price.co.uk"), Contact("A. Price", "ada@price.co.uk ")))

    def test_a_shared_phone_is_one_person(self):
        self.assertTrue(same_person(Contact("Ada", phone="07700 900001"), Contact("Ada Price", phone="+447700900001")))

    def test_a_shared_name_is_not(self):
        self.assertFalse(same_person(Contact("Sam Hughes", "sam@a.com"), Contact("Sam Hughes", "sam@b.com")))

    def test_blank_details_never_match(self):
        self.assertFalse(same_person(Contact("Ada"), Contact("Bo")))


class MergeTests(unittest.TestCase):
    def test_people_are_joined_transitively(self):
        rows = [
            Contact("Ada Price", "ada@price.co.uk", source="crm"),
            Contact("Bo Lin", "bo@lin.com", source="crm"),
            Contact("Ada", "ada@price.co.uk", "07700 900001", source="events"),
            Contact("Ada P", "ada.p@gmail.com", "+44 7700 900001", source="newsletter"),
        ]
        people = merge(rows)
        self.assertEqual([len(p.rows) for p in people], [3, 1])
        self.assertEqual(people[0].name, "Ada Price")
        self.assertEqual(people[0].emails, ["ada.p@gmail.com", "ada@price.co.uk"])

    def test_exports_are_read_from_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            crm, events = Path(tmp) / "crm.csv", Path(tmp) / "events.csv"
            crm.write_text("Full Name,Email,Phone\nAda Price,ada@price.co.uk,\n,,\n", encoding="utf-8")
            events.write_text("Name,E-mail,Mobile\nAda,ada@price.co.uk,07700 900001\n", encoding="utf-8")
            people = merge_exports([crm, events])
        self.assertEqual(len(people), 1)
        self.assertEqual([row.source for row in people[0].rows], ["crm", "events"])


if __name__ == "__main__":
    unittest.main()
