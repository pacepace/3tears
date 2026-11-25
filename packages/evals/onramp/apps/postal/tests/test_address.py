import unittest

from address import AddressError, label, label_lines, load_addresses


class LabelTests(unittest.TestCase):
    def test_a_house_address(self):
        self.assertEqual(label_lines("27 Church St, Ripon, HG4 1DS"), ["27 CHURCH STREET", "RIPON", "HG4 1DS"])

    def test_case_spaces_and_full_stops_are_tidied(self):
        self.assertEqual(label_lines("9  oak gro.,  york , yo10 4hh"), ["9 OAK GROVE", "YORK", "YO10 4HH"])

    def test_the_postcode_gets_its_space(self):
        self.assertEqual(label_lines("21 Fern Dr, Swindon, sn252ab")[-1], "SN25 2AB")
        self.assertEqual(label_lines("1 Deansgate, Manchester, m11ae")[-1], "M1 1AE")

    def test_a_named_house_keeps_its_name(self):
        self.assertEqual(
            label_lines("Rose Cottage, Lower Ln, Ashford, TN26 3AP"),
            ["ROSE COTTAGE", "LOWER LANE", "ASHFORD", "TN26 3AP"],
        )

    def test_a_flat_is_on_its_own_line(self):
        self.assertEqual(label_lines("Flat 2, Leeds, LS6 2AB"), ["FLAT 2", "LEEDS", "LS6 2AB"])

    def test_an_apartment_is_printed_as_a_flat(self):
        self.assertEqual(label_lines("Apt. 7, London, N1 7GR")[0], "FLAT 7")

    def test_a_street_type_is_only_spelt_out_at_the_end(self):
        self.assertEqual(label_lines("4 St Andrews Sq, St Ives, TR26 1AA")[:2], ["4 ST ANDREWS SQUARE", "ST IVES"])

    def test_no_postcode_is_refused(self):
        with self.assertRaises(AddressError):
            label_lines("27 Church St, Ripon")

    def test_the_label_is_one_line_under_another(self):
        self.assertEqual(label("6 Willow Grn, Sale, M33 2ST"), "6 WILLOW GREEN\nSALE\nM33 2ST")

    def test_every_order_address_labels(self):
        addresses = load_addresses()
        self.assertTrue(addresses)
        for raw in addresses:
            self.assertGreaterEqual(len(label_lines(raw)), 3, raw)


if __name__ == "__main__":
    unittest.main()
