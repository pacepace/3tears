import json
import tempfile
import unittest
from pathlib import Path

from label import band, label, labels_file

LOAF = {
    "sku": "1",
    "form": "food",
    "category": "bakery",
    "energy_kj": 1046,
    "fat": 3.0,
    "saturates": 1.3,
    "sugars": 22.6,
    "salt": 0.8,
}
COLA = {
    "sku": "2",
    "form": "drink",
    "category": "soft drinks",
    "energy_kj": 180,
    "fat": 0.0,
    "saturates": 0.0,
    "sugars": 10.6,
    "salt": 0.01,
}


class BandTests(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(band(3.0, 3.0, 17.5), "LOW")
        self.assertEqual(band(3.1, 3.0, 17.5), "MED")
        self.assertEqual(band(17.5, 3.0, 17.5), "MED")
        self.assertEqual(band(17.6, 3.0, 17.5), "HIGH")


class LabelTests(unittest.TestCase):
    def test_a_food(self):
        self.assertEqual(
            label(LOAF),
            "Per 100g: Energy 1046kJ/250kcal | Fat 3.0g LOW | Saturates 1.3g LOW | Sugars 22.6g HIGH | Salt 0.80g MED",
        )

    def test_a_drink(self):
        self.assertEqual(
            label(COLA),
            "Per 100ml: Energy 180kJ/43kcal | Fat 0.0g LOW | Saturates 0.0g LOW | Sugars 10.6g MED | Salt 0.01g LOW",
        )

    def test_the_form_must_be_known(self):
        with self.assertRaises(ValueError):
            label(dict(LOAF, form="powder"))

    def test_a_file_of_sheets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "specs.jsonl"
            path.write_text(json.dumps(LOAF) + "\n" + json.dumps(COLA) + "\n", encoding="utf-8")
            labels = labels_file(path)
        self.assertEqual(sorted(labels), ["1", "2"])
        self.assertTrue(labels["2"].startswith("Per 100ml: "))


if __name__ == "__main__":
    unittest.main()
